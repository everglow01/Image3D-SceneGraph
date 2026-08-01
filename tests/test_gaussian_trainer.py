from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.checkpoint import (
    CheckpointProvenance,
    create_attempt,
    load_checkpoint,
    write_checkpoint,
)
from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.render import RenderCamera
from image3d_scenegraph.gaussian.runtime import TrainingView
from image3d_scenegraph.gaussian.trainer import (
    _accumulate_statistics,
    _bounded_growth_capacity,
    _checkpoint_state,
    _final_checkpoint_due,
    _load_model,
    _maybe_update_topology,
    _torch_load,
    _write_latest_checkpoint,
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


def test_topology_update_prunes_splits_duplicates_and_honors_budget():
    gaussian = model()
    gaussian.log_scales.data[0] = torch.log(torch.full((3,), 0.001))
    config = resolve_internal_config().effective_config
    config["gaussian_budget"]["max_count"] = 5
    config["densification"].update(
        start_iteration=1,
        end_iteration=10,
        every_iterations=1,
        gradient_threshold=0.5,
        duplicate_scale_threshold=0.01,
        split_children=2,
    )
    gaussian.opacity_logits.data[2] = -20
    optimizer = torch.optim.Adam(gaussian.parameter_groups(config["learning_rate"]))
    sum(parameter.square().sum() for parameter in gaussian.parameters()).backward()
    optimizer.step()

    result = _maybe_update_topology(
        gaussian,
        optimizer,
        torch.tensor([1.0, 1.0, 1.0]),
        torch.ones(3),
        torch.zeros(3),
        config,
        1,
    )

    assert result is not None
    event, remapped, *_ = result
    assert event["duplicated"] == 1
    assert event["split_parents"] == 1
    assert event["pruned_low_opacity"] == 1
    assert event["gaussian_count"] == 4
    assert remapped.state_dict()["state"]
    gaussian.validate(max_count=5)


def test_full_budget_prioritizes_split_over_duplicate_after_pruning():
    gaussian = model()
    gaussian.log_scales.data[0] = torch.log(torch.full((3,), 0.001))
    gaussian.opacity_logits.data[2] = -20
    config = resolve_internal_config().effective_config
    config["gaussian_budget"]["max_count"] = 3
    config["densification"].update(
        start_iteration=1,
        end_iteration=10,
        every_iterations=1,
        gradient_threshold=0.5,
        duplicate_scale_threshold=0.01,
        screen_size_end_iteration=10,
    )
    optimizer = torch.optim.Adam(gaussian.parameter_groups(config["learning_rate"]))

    result = _maybe_update_topology(
        gaussian,
        optimizer,
        torch.tensor([1.0, 1.0, 0.0]),
        torch.ones(3),
        torch.zeros(3),
        config,
        1,
    )

    assert result is not None
    event = result[0]
    assert event["split_candidates"] == 1
    assert event["split_selected"] == 1
    assert event["duplicate_candidates"] == 1
    assert event["duplicate_selected"] == 0
    assert event["pruned_low_opacity"] == 1
    assert event["gaussian_count"] == 3


def test_full_budget_without_prunable_rows_skips_growth():
    gaussian = model()
    gaussian.log_scales.data[0] = torch.log(torch.full((3,), 0.001))
    config = resolve_internal_config().effective_config
    config["gaussian_budget"]["max_count"] = 3
    config["densification"].update(
        start_iteration=1,
        end_iteration=10,
        every_iterations=1,
        gradient_threshold=0.5,
        duplicate_scale_threshold=0.01,
        screen_size_end_iteration=10,
    )
    optimizer = torch.optim.Adam(gaussian.parameter_groups(config["learning_rate"]))

    result = _maybe_update_topology(
        gaussian,
        optimizer,
        torch.tensor([1.0, 1.0, 0.0]),
        torch.ones(3),
        torch.zeros(3),
        config,
        1,
    )

    assert result is not None
    event = result[0]
    assert event["split_selected"] == 0
    assert event["duplicate_selected"] == 0
    assert event["budget_skipped"] == 2
    assert event["gaussian_count"] == 3


def test_screen_threshold_splits_then_prunes():
    config = resolve_internal_config().effective_config
    config["gaussian_budget"]["max_count"] = 5
    config["densification"].update(start_iteration=1, end_iteration=10, every_iterations=1)

    split_model = model()
    split_optimizer = torch.optim.Adam(split_model.parameter_groups(config["learning_rate"]))
    split = _maybe_update_topology(
        split_model,
        split_optimizer,
        torch.zeros(3),
        torch.ones(3),
        torch.tensor([0.06, 0.0, 0.0]),
        config,
        1,
    )
    assert split is not None
    assert split[0]["split_selected"] == 1
    assert split[0]["pruned_screen_size"] == 0

    prune_model = model()
    prune_optimizer = torch.optim.Adam(prune_model.parameter_groups(config["learning_rate"]))
    pruned = _maybe_update_topology(
        prune_model,
        prune_optimizer,
        torch.zeros(3),
        torch.ones(3),
        torch.tensor([0.16, 0.0, 0.0]),
        config,
        1,
        cleanup_only=True,
    )
    assert pruned is not None
    assert pruned[0]["split_selected"] == 0
    assert pruned[0]["pruned_screen_size"] == 1
    assert pruned[0]["gaussian_count"] == 2


def test_bounded_growth_shares_saturated_capacity():
    assert _bounded_growth_capacity(100, 100, 100, 1, 0.25) == (25, 75)
    assert _bounded_growth_capacity(100, 10, 100, 1, 0.25) == (10, 90)
    assert _bounded_growth_capacity(100, 100, 10, 1, 0.25) == (90, 10)
    assert _bounded_growth_capacity(100, 100, 100, 2, 0.25) == (12, 76)
    assert _bounded_growth_capacity(0, 100, 100, 1, 0.25) == (0, 0)


def test_gradient_statistics_use_screen_units_and_normalized_radius():
    means2d = torch.zeros((1, 2))
    means2d.absgrad = torch.tensor([[0.01, 0.02]])
    gradient_sum = torch.zeros(1)
    gradient_count = torch.zeros(1)
    max_radius = torch.zeros(1)

    _accumulate_statistics(
        {
            "means2d": means2d,
            "gaussian_ids": torch.tensor([0]),
            "radii": torch.tensor([[20.0, 10.0]]),
            "width": 200,
            "height": 100,
        },
        gradient_sum,
        gradient_count,
        max_radius,
    )

    assert gradient_sum.item() == pytest.approx(2**0.5)
    assert gradient_count.item() == 1
    assert max_radius.item() == pytest.approx(0.1)


def test_validation_payload_marks_lpips_not_run(monkeypatch):
    gaussian = model()
    config = resolve_internal_config().effective_config
    view = TrainingView(
        RenderCamera(
            image_id="validation-1",
            camera_from_normalized=torch.eye(4),
            intrinsic=torch.eye(3),
            width=4,
            height=4,
        ),
        torch.full((4, 4, 3), 0.5),
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


def test_only_final_iteration_writes_a_checkpoint():
    assert not _final_checkpoint_due(500, 3_000)
    assert not _final_checkpoint_due(2_999, 3_000)
    assert _final_checkpoint_due(3_000, 3_000)


def test_latest_checkpoint_replaces_intermediate_checkpoint(tmp_path):
    expected_provenance = CheckpointProvenance("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    create_attempt(tmp_path, attempt_id="train-001", kind="fresh", provenance=expected_provenance)
    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizer = torch.optim.Adam(gaussian.parameter_groups(config["learning_rate"]))
    checkpoint_state = _checkpoint_state(
        gaussian,
        optimizer,
        torch.zeros(gaussian.count),
        torch.zeros(gaussian.count),
        torch.zeros(gaussian.count),
        [{"iteration": 1, "loss": 1.0}],
        1,
    )

    _write_latest_checkpoint(
        tmp_path,
        attempt_id="train-001",
        iteration=1,
        purpose="periodic",
        validation_score=None,
        provenance=expected_provenance,
        state=checkpoint_state,
    )
    _write_latest_checkpoint(
        tmp_path,
        attempt_id="train-001",
        iteration=2,
        purpose="final",
        validation_score=None,
        provenance=expected_provenance,
        state=checkpoint_state,
    )

    checkpoints = tmp_path / "attempts" / "train-001" / "checkpoints"
    assert [path.name for path in checkpoints.iterdir()] == ["iteration_000000002"]
    assert load_checkpoint(tmp_path, "train-001", 2).record.purpose == "final"


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
