from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


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

    assert result["schema_version"] == 1
    assert result["split"] == "validation"
    assert result["psnr"]["mean"] == pytest.approx(120.0)
    assert result["ssim"]["p50"] == pytest.approx(1.0)
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
