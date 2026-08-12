from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.checkpoint import (
    CheckpointProvenance,
    create_attempt,
    load_checkpoint,
)
from image3d_scenegraph.gaussian.config import resolve_internal_config
from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.render import RenderCamera
from image3d_scenegraph.gaussian.runtime import TrainingView
from image3d_scenegraph.gaussian.trainer import (
    _build_strategy,
    _checkpoint_state,
    _load_model,
    _next_camera,
    _opacity_reset_due,
    _torch_load,
    _training_scene_scale,
    _update_position_learning_rate,
    _write_latest_checkpoint,
    evaluate_views,
)


def model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.2, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.full((3,), 0.1),
    )


def test_v7_strategy_disables_regressive_screen_pruning():
    from gsplat.strategy import DefaultStrategy

    config = resolve_internal_config().effective_config
    strategy = _build_strategy(DefaultStrategy, config)

    assert strategy.refine_start_iter == 500
    assert strategy.refine_stop_iter == 15_000
    assert strategy.refine_every == 100
    assert strategy.grow_grad2d == pytest.approx(0.0002)
    assert strategy.grow_scale3d == pytest.approx(0.01)
    assert strategy.prune_opa == pytest.approx(0.005)
    assert strategy.prune_scale3d == pytest.approx(0.1)
    assert strategy.reset_every == 3_000
    assert strategy.refine_scale2d_stop_iter == 0
    assert strategy.grow_scale2d == pytest.approx(1e10)
    assert strategy.prune_scale2d == pytest.approx(1e10)
    assert strategy.absgrad is False


def test_explicit_screen_pruning_uses_training_resolution():
    from gsplat.strategy import DefaultStrategy

    config = resolve_internal_config(
        overrides={"pruning": {"screen_radius_enabled": True}}
    ).effective_config
    view = TrainingView(
        RenderCamera("train", torch.eye(4), torch.eye(3), 1000, 600),
        torch.zeros((600, 1000, 3)),
    )

    strategy = _build_strategy(DefaultStrategy, config, [view])

    assert strategy.prune_scale2d == pytest.approx(0.02)


def test_strategy_can_update_project_model_and_optimizers():
    from gsplat.strategy import DefaultStrategy

    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])
    strategy = _build_strategy(DefaultStrategy, config)

    strategy.check_sanity(gaussian.params, optimizers)
    assert set(optimizers) == set(gaussian.params)


def test_camera_sampling_visits_each_view_before_reshuffle():
    torch.manual_seed(7)
    order: list[int] = []
    cursor = 0
    selected = []
    for _ in range(5):
        order, cursor, index = _next_camera(5, order, cursor)
        selected.append(index)

    assert sorted(selected) == list(range(5))
    old_order = order
    order, cursor, _ = _next_camera(5, order, cursor)
    assert cursor == 1
    assert order is not old_order


def test_only_position_learning_rate_decays():
    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])
    feature_before = optimizers["sh0"].param_groups[0]["lr"]

    _update_position_learning_rate(optimizers["means"], config, 30_000, 30_000, 1.1)

    assert optimizers["means"].param_groups[0]["lr"] == pytest.approx(0.0000016 * 1.1)
    assert optimizers["sh0"].param_groups[0]["lr"] == feature_before


def test_training_scene_scale_matches_graphdeco_train_camera_extent():
    contract = {
        "splits": {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
        "normalization": {"normalized_from_world": torch.eye(4).tolist()},
        "images": [
            {"image_id": "a", "world_from_camera": torch.eye(4).tolist()},
            {
                "image_id": "b",
                "world_from_camera": torch.tensor(
                    [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=torch.float32,
                ).tolist(),
            },
            {"image_id": "c", "world_from_camera": torch.eye(4).tolist()},
            {"image_id": "d", "world_from_camera": torch.eye(4).tolist()},
        ],
    }

    assert _training_scene_scale(contract) == pytest.approx(1.1)


def test_reset_schedule_stops_at_refinement_boundary():
    config = resolve_internal_config().effective_config
    assert _opacity_reset_due(config, 3_000)
    assert _opacity_reset_due(config, 12_000)
    assert not _opacity_reset_due(config, 15_000)
    assert not _opacity_reset_due(config, 30_000)

    config["densification"]["enabled"] = False
    assert not _opacity_reset_due(config, 3_000)


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
        lambda *_args, **_kwargs: SimpleNamespace(
            image=torch.full((4, 4, 3), 0.5), metadata={}
        ),
    )

    payload = evaluate_views(gaussian, [view], config)

    assert payload["split"] == "validation"
    assert payload["mean_psnr"] == pytest.approx(120.0)
    assert payload["lpips"] == {
        "status": "not_run",
        "reason": "pretrained_weight_license_and_hash_not_audited",
    }


def test_checkpoint_state_round_trips_model_and_per_parameter_optimizers(tmp_path):
    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])
    loss = sum(parameter.square().sum() for parameter in gaussian.params.values())
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
    strategy_state = {
        "grad2d": torch.ones(gaussian.count),
        "count": torch.ones(gaussian.count),
        "scene_scale": 1.0,
    }
    state = _checkpoint_state(
        gaussian,
        optimizers,
        strategy_state,
        [2, 0, 1],
        1,
        [{"iteration": 1, "loss": float(loss)}],
        1,
    )

    restored = _load_model(state.model, torch.device("cpu"))
    optimizer_payload = _torch_load(state.optimizer, torch.device("cpu"))
    dense_payload = _torch_load(state.densification, torch.device("cpu"))

    assert restored.count == gaussian.count
    for expected, actual in zip(gaussian.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(expected, actual)
    assert set(optimizer_payload) == set(optimizers)
    assert dense_payload["camera_order"] == [2, 0, 1]
    assert dense_payload["camera_cursor"] == 1
    assert torch.equal(dense_payload["strategy_state"]["grad2d"], torch.ones(3))


def test_latest_checkpoint_replaces_intermediate_checkpoint(tmp_path):
    provenance = CheckpointProvenance("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    create_attempt(tmp_path, attempt_id="train-001", kind="fresh", provenance=provenance)
    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])
    state = _checkpoint_state(
        gaussian,
        optimizers,
        {"grad2d": None, "count": None, "scene_scale": 1.0},
        [],
        0,
        [{"iteration": 1, "loss": 1.0}],
        1,
    )

    for iteration, purpose in ((1, "periodic"), (2, "final")):
        _write_latest_checkpoint(
            tmp_path,
            attempt_id="train-001",
            iteration=iteration,
            purpose=purpose,
            validation_score=None,
            provenance=provenance,
            state=state,
        )

    checkpoints = tmp_path / "attempts" / "train-001" / "checkpoints"
    assert [path.name for path in checkpoints.iterdir()] == ["iteration_000000002"]
    assert load_checkpoint(tmp_path, "train-001", 2).record.purpose == "final"
