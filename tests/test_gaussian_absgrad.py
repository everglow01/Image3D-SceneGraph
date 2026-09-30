from __future__ import annotations

import copy
import sys
import types

import pytest

from image3d_scenegraph.gaussian.config import (
    GaussianConfigError,
    assert_single_field_ablation,
    resolve_internal_config,
    resolve_public_config,
    resolved_config_record,
    validate_effective_config,
)


def test_absgrad_profile_preserves_public_config_and_hashes_single_leaf():
    legacy = resolve_public_config("standard_v1")
    assert legacy.effective_config_hash == "0130a3f0b087b4aaa5eb06f85d451b8fc812eb6dbe85723c740086c8eced653d"
    baseline = resolve_internal_config("absgrad_ablation_v1")
    candidate = resolve_internal_config(
        "absgrad_ablation_v1", {"densification": {"absgrad": True}}
    )
    assert assert_single_field_ablation(
        baseline.effective_config, candidate.effective_config
    ) == "densification.absgrad"
    restored = copy.deepcopy(baseline.effective_config)
    restored["schema_version"] = 10
    assert restored["densification"].pop("absgrad") is False
    assert restored == legacy.effective_config
    assert baseline.effective_config_hash != candidate.effective_config_hash
    assert resolved_config_record(candidate)["schema_version"] == 11
    with pytest.raises(GaussianConfigError, match="unknown"):
        resolve_internal_config(overrides={"densification": {"absgrad": True}})
    invalid = copy.deepcopy(candidate.effective_config)
    invalid["strategy"]["name"] = "mcmc_v1"
    with pytest.raises(GaussianConfigError, match="requires default_v1"):
        validate_effective_config(invalid)
    with pytest.raises(GaussianConfigError, match="boolean"):
        resolve_internal_config("absgrad_ablation_v1", {"densification": {"absgrad": 1}})


def test_unpatched_distributed_absgrad_fails_closed(monkeypatch):
    pytest.importorskip("torch")
    from image3d_scenegraph.gaussian.render import require_distributed_absgrad

    monkeypatch.setitem(sys.modules, "gsplat", types.SimpleNamespace(rendering=types.SimpleNamespace()))
    with pytest.raises(RuntimeError, match="isolated gsplat overlay"):
        require_distributed_absgrad()


@pytest.mark.parametrize("rank", [0, 1])
def test_absgrad_reverse_exchange_preserves_owner_camera_order(monkeypatch, rank):
    torch = pytest.importorskip("torch")
    from image3d_scenegraph.gaussian.render import collect_distributed_absgrad

    counts = [2, 3]
    projected = torch.zeros(2, counts[rank], 2)
    rasterized = torch.zeros(1, 5, 2)
    rasterized.absgrad = torch.arange(10.0).reshape(1, 5, 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)

    def exchange(incoming, outgoing):
        assert [list(v.shape) for v in outgoing] == [[2, 2], [3, 2]]
        assert torch.equal(torch.cat(outgoing), rasterized.absgrad[0])
        for camera, tensor in enumerate(incoming):
            tensor.fill_(camera + 1)

    monkeypatch.setattr(torch.distributed, "all_to_all", exchange)
    collect_distributed_absgrad({
        "means2d": projected,
        "project_absgrad_means2d": rasterized,
        "project_absgrad_counts": counts,
    })
    assert torch.equal(projected.absgrad[0], torch.ones(counts[rank], 2))
    assert torch.equal(projected.absgrad[1], torch.full((counts[rank], 2), 2.0))
    del rasterized.absgrad
    with pytest.raises(RuntimeError, match="missing, invalid or misaligned"):
        collect_distributed_absgrad({
            "means2d": projected,
            "project_absgrad_means2d": rasterized,
            "project_absgrad_counts": counts,
        })


def test_strategy_and_training_render_share_absgrad_flag(monkeypatch):
    torch = pytest.importorskip("torch")
    from image3d_scenegraph.gaussian import trainer

    config = resolve_internal_config(
        "absgrad_ablation_v1", {"densification": {"absgrad": True}}
    ).effective_config
    strategy = trainer._build_strategy(types.SimpleNamespace, config)
    assert strategy.absgrad is True
    assert strategy.grow_grad2d == 0.0002
    assert strategy.refine_stop_iter == 15000
    calls = []

    def render(*args, **kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(metadata={"radii": torch.ones(1)})

    monkeypatch.setattr(trainer, "render_gaussians", render)
    view = types.SimpleNamespace(camera="camera")
    wrapped = types.SimpleNamespace(to=lambda device: view)
    model = types.SimpleNamespace(means=torch.zeros(1))
    for distributed in (False, True):
        trainer._render_visible_training_view(
            model, [wrapped], 0, 0,
            distributed=distributed, gradient_statistics=True,
        )
    assert all(call["gradient_statistics"] is True for call in calls)
