from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance, create_attempt, write_checkpoint
from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.trainer import (
    _checkpoint_state,
    _load_model,
    _maybe_update_topology,
    _torch_load,
    evaluate_views,
)
from image3d_scenegraph.gaussian.config import resolve_internal_config
from types import SimpleNamespace


def model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.2, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.full((3,), 0.1),
    )


def test_topology_update_prunes_and_honors_budget():
    gaussian = model()
    config = resolve_internal_config().effective_config
    config["gaussian_budget"]["max_count"] = 4
    config["densification"].update(
        start_iteration=1,
        end_iteration=10,
        every_iterations=1,
        gradient_threshold=0.5,
    )
    gaussian.opacity_logits.data[2] = -20

    event = _maybe_update_topology(
        gaussian,
        torch.tensor([1.0, 1.0, 1.0]),
        torch.ones(3),
        torch.zeros(3),
        config,
        1,
    )

    assert event == {"densified": 2, "pruned": 1, "gaussian_count": 4}
    gaussian.validate(max_count=4)


def test_validation_payload_marks_lpips_not_run(monkeypatch):
    gaussian = model()
    config = resolve_internal_config().effective_config
    view = SimpleNamespace(
        camera=SimpleNamespace(image_id="validation-1"),
        image=torch.full((4, 4, 3), 0.5),
    )
    monkeypatch.setattr(
        "image3d_scenegraph.gaussian.trainer.render_gaussians",
        lambda *_args, **_kwargs: SimpleNamespace(image=torch.full((4, 4, 3), 0.5)),
    )

    payload = evaluate_views(gaussian, [view], config)

    assert payload["split"] == "validation"
    assert payload["mean_psnr"] == pytest.approx(120.0)
    assert payload["lpips"] == {
        "status": "not_run",
        "reason": "pretrained_weight_license_and_hash_not_audited",
    }


def test_checkpoint_state_round_trips_real_model_and_optimizer(tmp_path):
    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizer = torch.optim.Adam(gaussian.parameter_groups(config["learning_rate"]))
    loss = sum(parameter.square().sum() for parameter in gaussian.parameters())
    loss.backward()
    optimizer.step()
    provenance = CheckpointProvenance("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    create_attempt(tmp_path, attempt_id="train-001", kind="fresh", provenance=provenance)
    state = _checkpoint_state(
        gaussian,
        optimizer,
        torch.ones(gaussian.count),
        torch.ones(gaussian.count),
        torch.ones(gaussian.count),
        [{"iteration": 1, "loss": float(loss)}],
        1,
    )

    write_checkpoint(
        tmp_path,
        attempt_id="train-001",
        iteration=1,
        purpose="final",
        provenance=provenance,
        state=state,
    )
    restored = _load_model(state.model, torch.device("cpu"))
    restored_optimizer = torch.optim.Adam(restored.parameter_groups(config["learning_rate"]))
    restored_optimizer.load_state_dict(_torch_load(state.optimizer, torch.device("cpu")))

    assert restored.count == gaussian.count
    for expected, actual in zip(gaussian.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    assert restored_optimizer.state_dict()["state"]
