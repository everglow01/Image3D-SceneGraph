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
        progress_events=[{"densified": 3, "pruned": 1}, {"opacity_reset": True}],
        renderer=renderer,
    )

    assert result["schema_version"] == 1
    assert result["split"] == "validation"
    assert result["psnr"]["mean"] == pytest.approx(120.0)
    assert result["ssim"]["p50"] == pytest.approx(1.0)
    assert result["topology"] == {"densified": 3, "pruned": 1, "opacity_resets": 1}
    assert result["lpips"]["status"] == "not_run"
    assert len(list((tmp_path / "previews").glob("*.png"))) == 2


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
