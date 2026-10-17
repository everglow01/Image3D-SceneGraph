from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image


torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.evaluation import (
    GaussianEvaluationError,
    evaluate_model,
    write_frozen_candidate,
    _authorize_test,
)
from image3d_scenegraph.gaussian.model import GaussianModel


def model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.full((2,), 0.1),
    )


def test_distribution_numpy_path_exactly_matches_float_list():
    from image3d_scenegraph.gaussian.evaluation import _distribution

    values = torch.linspace(-2, 3, 10001, dtype=torch.float32)
    assert _distribution(values.numpy()) == _distribution(values.tolist())
    assert _distribution(values[::3].numpy()) == _distribution(values[::3].tolist())


def test_evaluation_emits_view_and_health_progress(monkeypatch):
    from image3d_scenegraph.gaussian import evaluation

    events = []
    monkeypatch.setattr(evaluation, "record_training_phase",
                        lambda phase, iteration, **kw: events.append((phase, iteration, kw)))
    view = SimpleNamespace(camera=SimpleNamespace(image_id="a"), image=torch.full((4, 4, 3), .5))
    evaluate_model(model(), [view, view], split="validation", sh_degree=0, iteration=10000,
                   renderer=lambda *_args, **_kw: SimpleNamespace(image=view.image))
    assert events == [
        ("validation_view", 10000, {"completed_views": 0, "total_views": 2}),
        ("validation_view", 10000, {"completed_views": 1, "total_views": 2}),
        ("validation_health", 10000, {"completed_views": 2, "total_views": 2}),
    ]


def test_evaluation_reports_distributions_resources_and_topology(tmp_path, monkeypatch):
    gaussian = model()
    views = [
        SimpleNamespace(
            camera=SimpleNamespace(image_id=image_id),
            image=torch.full((4, 4, 3), 0.5),
        )
        for image_id in ("a", "b")
    ]
    renderer = lambda *_args, **_kwargs: SimpleNamespace(  # noqa: E731
        image=torch.full((4, 4, 3), 0.5)
    )

    result = evaluate_model(
        gaussian,
        views,
        split="validation",
        sh_degree=0,
        preview_dir=tmp_path / "previews",
        progress_events=[
            {"topology_net_growth": 3},
            {"topology_net_growth": -1},
            {"opacity_reset": True},
        ],
        renderer=renderer,
    )

    assert result["schema_version"] == 2
    assert result["split"] == "validation"
    assert result["quality_role"] == "held_out_model_selection"
    assert result["selection_eligible"] is True
    assert result["quality_profiles"]["primary"] == "raw_float_v1"
    assert "display_clamped_uint8_v1" in result["quality_profiles"]
    assert result["psnr"]["mean"] == pytest.approx(120.0)
    assert result["display_psnr"]["mean"] == pytest.approx(120.0)
    assert result["ssim"]["p50"] == pytest.approx(1.0)
    assert result["display_ssim"]["p50"] == pytest.approx(1.0)
    assert result["topology"] == {
        "strategy_updates": 2,
        "net_growth": 2,
        "opacity_resets": 1,
    }
    assert result["health"]["visible_gaussian_count"] == 0
    assert result["health"]["screen_radius"] is None
    assert result["health"]["max_scale"]["max"] == pytest.approx(0.1)
    assert result["lpips"]["status"] == "not_run"
    assert len(list((tmp_path / "previews").glob("*.png"))) == 2


@pytest.mark.parametrize(("split", "role"), [
    ("fit_validation", "in_sample_after_train_validation_fit"),
    ("control_validation", "held_out_after_train_only_control"),
])
def test_fit_validation_metrics_are_not_selection_eligible(split, role):
    gaussian = model()
    view = SimpleNamespace(
        camera=SimpleNamespace(image_id="fit"),
        image=torch.full((4, 4, 3), 0.5),
    )
    renderer = lambda *_args, **_kwargs: SimpleNamespace(  # noqa: E731
        image=torch.full((4, 4, 3), 0.5)
    )

    result = evaluate_model(
        gaussian,
        [view],
        split=split,
        sh_degree=0,
        renderer=renderer,
    )

    assert result["split"] == split
    assert result["quality_role"] == role
    assert result["selection_eligible"] is False


def test_evaluation_separates_raw_float_and_display_clamped_metrics(tmp_path):
    gaussian = model()
    views = [
        SimpleNamespace(
            camera=SimpleNamespace(image_id="over-range"),
            image=torch.full((4, 4, 3), 0.5),
        )
    ]
    renderer = lambda *_args, **_kwargs: SimpleNamespace(  # noqa: E731
        image=torch.full((4, 4, 3), 1.5)
    )

    result = evaluate_model(
        gaussian,
        views,
        split="validation",
        sh_degree=0,
        preview_dir=tmp_path / "previews",
        renderer=renderer,
    )

    assert result["psnr"]["mean"] == pytest.approx(0.0)
    assert result["display_psnr"]["mean"] > result["psnr"]["mean"]
    assert result["per_view"][0]["psnr"] == result["psnr"]["mean"]
    assert result["per_view"][0]["display_psnr"] == result["display_psnr"]["mean"]
    assert Image.open(tmp_path / "previews" / "over-range.png").getpixel((0, 0)) == (
        255,
        255,
        255,
    )


def test_evaluation_reports_screen_and_scale_health():
    gaussian = model()
    gaussian.log_scales.data[0] = torch.log(torch.full((3,), 0.2))
    gaussian.opacity_logits.data[0] = torch.logit(torch.tensor(0.5))
    views = [
        SimpleNamespace(
            camera=SimpleNamespace(image_id="a"),
            image=torch.full((4, 4, 3), 0.5),
        )
    ]

    def renderer(*_args, **_kwargs):
        return SimpleNamespace(
            image=torch.full((4, 4, 3), 0.5),
            metadata={
                "gaussian_ids": torch.tensor([0, 1]),
                "radii": torch.tensor([[1.0, 1.0], [0.2, 0.2]]),
                "width": 4,
                "height": 4,
            },
        )

    result = evaluate_model(
        gaussian,
        views,
        split="validation",
        sh_degree=0,
        renderer=renderer,
        health_thresholds={
            "split_screen_fraction": 0.05,
            "max_screen_fraction": 0.15,
            "max_world_scale": 0.1,
            "opacity_threshold": 0.005,
        },
    )

    assert result["health"]["visible_gaussian_count"] == 2
    assert result["health"]["screen_split_count"] == 1
    assert result["health"]["screen_prune_count"] == 1
    assert result["health"]["high_opacity_screen_prune_count"] == 1
    assert result["health"]["world_scale_prune_count"] == 1
    assert result["scale"]["max"] == pytest.approx(0.2)


def test_test_authorization_is_hash_bound_and_consumed_once(tmp_path):
    frozen = tmp_path / "candidate.json"
    hashes = ("a" * 64, "b" * 64, "c" * 64)
    write_frozen_candidate(
        frozen,
        candidate_id="room-standard-v1",
        dataset_hash=hashes[0],
        effective_config_hash=hashes[1],
        model_sha256=hashes[2],
    )
    assert json.loads(frozen.read_text())["schema_version"] == 1

    consumption = _authorize_test(
        frozen,
        dataset_hash=hashes[0],
        config_hash=hashes[1],
        model_hash=hashes[2],
    )

    assert json.loads(consumption.read_text())["status"] == "running"
    with pytest.raises(GaussianEvaluationError, match="already consumed"):
        _authorize_test(
            frozen,
            dataset_hash=hashes[0],
            config_hash=hashes[1],
            model_hash=hashes[2],
        )


def test_test_authorization_rejects_changed_model(tmp_path):
    frozen = tmp_path / "candidate.json"
    write_frozen_candidate(
        frozen,
        candidate_id="room-standard-v1",
        dataset_hash="a" * 64,
        effective_config_hash="b" * 64,
        model_sha256="c" * 64,
    )

    with pytest.raises(GaussianEvaluationError, match="model_sha256"):
        _authorize_test(
            frozen,
            dataset_hash="a" * 64,
            config_hash="b" * 64,
            model_hash="d" * 64,
        )


def test_unpacked_screen_health_uses_local_shard_across_cameras():
    from image3d_scenegraph.gaussian.evaluation import _accumulate_screen_health

    maximum = torch.zeros(3)
    visible = torch.zeros(3, dtype=torch.bool)
    radii = torch.tensor([[[0, 0], [3, 2], [9, 0]], [[4, 1], [0, 0], [0, 0]]])
    before = radii.clone()
    rng = torch.get_rng_state().clone()
    metadata = {"gaussian_ids": None, "radii": radii, "width": 20, "height": 10}
    assert _accumulate_screen_health(metadata, maximum, visible)
    torch.testing.assert_close(maximum, torch.tensor([.2, .15, 0.]))
    assert visible.tolist() == [True, True, False]
    assert torch.equal(radii, before) and torch.equal(rng, torch.get_rng_state())
    assert _accumulate_screen_health({**metadata, "radii": torch.zeros_like(radii)}, maximum, visible)
    torch.testing.assert_close(maximum, torch.tensor([.2, .15, 0.]))
    assert not _accumulate_screen_health({}, maximum, visible)
    with pytest.raises(GaussianEvaluationError, match="local shard"):
        _accumulate_screen_health({**metadata, "radii": radii[:, :2]}, maximum, visible)


def test_packed_screen_health_reduces_duplicates_and_ignores_zero_radii():
    from image3d_scenegraph.gaussian.evaluation import _accumulate_screen_health

    maximum = torch.zeros(3)
    visible = torch.zeros(3, dtype=torch.bool)
    metadata = {"gaussian_ids": torch.tensor([1, 1, 2]),
                "radii": torch.tensor([[2, 2], [5, 4], [0, 0]]), "width": 10, "height": 10}
    assert _accumulate_screen_health(metadata, maximum, visible)
    torch.testing.assert_close(maximum, torch.tensor([0., .5, 0.]))
    assert visible.tolist() == [False, True, False]
    empty = {**metadata, "gaussian_ids": torch.empty(0, dtype=torch.long),
             "radii": torch.empty((0, 2), dtype=torch.int32)}
    assert _accumulate_screen_health(empty, maximum, visible)


def test_missing_screen_metadata_is_reported_as_unavailable():
    view = SimpleNamespace(camera=SimpleNamespace(image_id="a"), image=torch.full((4, 4, 3), .5))
    result = evaluate_model(model(), [view], split="validation", sh_degree=0,
                            renderer=lambda *_args, **_kw: SimpleNamespace(image=view.image))
    assert result["health"]["screen_health_views"] == 0
    assert result["health"]["screen_health_scope"] == "local_gaussians_across_projected_cameras"
