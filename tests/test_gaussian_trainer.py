from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.checkpoint import (
    CheckpointProvenance,
    create_attempt,
    load_checkpoint,
)
from image3d_scenegraph.gaussian.config import (
    resolve_internal_config,
    resolve_mcmc_config,
)
from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.render import RenderCamera
from image3d_scenegraph.gaussian.runtime import TrainingView, load_training_views
from image3d_scenegraph.gaussian.trainer import (
    TrainingError,
    _build_mcmc_strategy,
    _build_strategy,
    _checkpoint_rank_bytes,
    _checkpoint_state,
    _load_model,
    _local_gaussian_cap,
    _mcmc_refinement_due,
    _mcmc_regularizers,
    _merge_model_shards,
    _model_bytes,
    _next_camera,
    _next_camera_batch,
    _opacity_reset_due,
    _recovery_prune_due,
    _pack_checkpoint_shards,
    _torch_load,
    _training_scene_scale,
    _update_position_learning_rate,
    _validate_mcmc_alive_set,
    _write_latest_checkpoint,
    evaluate_views,
)


def model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.2, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.full((3,), 0.1),
    )


def test_cpu_training_views_keep_uint8_until_device_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "image3d_scenegraph.gaussian.runtime.validate_contract", lambda *_args: None
    )
    image_path = tmp_path / "train.png"
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (0, 127, 255))
    image.putpixel((1, 0), (255, 64, 0))
    image.save(image_path)
    contract = {
        "splits": {"train": ["train"], "validation": [], "test": []},
        "normalization": {"normalized_from_world": torch.eye(4).tolist()},
        "images": [
            {
                "image_id": "train",
                "path": image_path.name,
                "distortion": {"state": "none"},
                "intrinsic": torch.eye(3).tolist(),
                "camera_from_world": torch.eye(4).tolist(),
            }
        ],
    }

    view = load_training_views(
        contract,
        tmp_path,
        split="train",
        longest_edge=3072,
        device=torch.device("cpu"),
    )[0]

    assert view.image.dtype == torch.uint8
    assert view.image.element_size() == 1
    transferred = view.to(torch.device("cpu"))
    assert transferred.image.dtype == torch.float32
    assert torch.allclose(
        transferred.image,
        torch.tensor([[[0, 127, 255], [255, 64, 0]]], dtype=torch.float32)
        / 255.0,
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


def test_mcmc_strategy_uses_frozen_global_budget_method_settings():
    from gsplat.strategy import MCMCStrategy

    config = resolve_mcmc_config().effective_config
    strategy = _build_mcmc_strategy(MCMCStrategy, config, 1_500_000)

    assert strategy.cap_max == 1_500_000
    assert strategy.noise_lr == pytest.approx(500_000.0)
    assert strategy.refine_start_iter == 500
    assert strategy.refine_stop_iter == 25_000
    assert strategy.refine_every == 100
    assert strategy.min_opacity == pytest.approx(0.005)


def test_mcmc_global_cap_is_split_exactly_across_ranks():
    assert [_local_gaussian_cap(3_000_001, rank, 2) for rank in range(2)] == [
        1_500_001,
        1_500_000,
    ]
    assert sum(_local_gaussian_cap(3_000_000, rank, 4) for rank in range(4)) == 3_000_000
    with pytest.raises(TrainingError, match="smaller than distributed world size"):
        _local_gaussian_cap(1, 0, 2)


def test_mcmc_refinement_matches_gsplat_schedule():
    config = resolve_mcmc_config().effective_config

    assert not _mcmc_refinement_due(config, 500)
    assert _mcmc_refinement_due(config, 600)
    assert _mcmc_refinement_due(config, 24_900)
    assert not _mcmc_refinement_due(config, 25_000)


def test_mcmc_regularizers_use_activated_opacity_and_scale():
    gaussian = model()
    config = resolve_mcmc_config().effective_config

    opacity, scale = _mcmc_regularizers(gaussian, config)

    assert float(opacity.detach()) == pytest.approx(0.001)
    assert float(scale.detach()) == pytest.approx(0.001)


def test_mcmc_all_dead_guard_fails_before_upstream_multinomial():
    gaussian = model()
    with torch.no_grad():
        gaussian.opacity_logits.fill_(-20.0)

    with pytest.raises(TrainingError, match="no live Gaussian"):
        _validate_mcmc_alive_set(
            gaussian,
            iteration=600,
            min_opacity=0.005,
            device=torch.device("cpu"),
            world_size=1,
        )


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


def test_distributed_camera_batch_visits_distinct_views_before_reshuffle():
    torch.manual_seed(7)
    order, cursor, first = _next_camera_batch(5, [], 0, 2)
    order, cursor, second = _next_camera_batch(5, order, cursor, 2)

    assert len(set(first + second)) == 4
    order, cursor, third = _next_camera_batch(5, order, cursor, 2)
    assert cursor == 1
    assert set(first + second + third[:1]) == set(range(5))


def test_distributed_checkpoint_selects_matching_rank_and_world_size():
    packed = _pack_checkpoint_shards([b"rank-0", b"rank-1"], 2)

    assert _checkpoint_rank_bytes(packed, 1, 2) == b"rank-1"
    with pytest.raises(TrainingError, match="world size mismatch"):
        _checkpoint_rank_bytes(packed, 0, 1)
    with pytest.raises(TrainingError, match="single-GPU checkpoint"):
        _checkpoint_rank_bytes(_model_bytes(model()), 0, 2)


def test_distributed_model_shards_merge_into_single_snapshot(tmp_path):
    gaussian = model()
    paths = [tmp_path / "rank-0.pt", tmp_path / "rank-1.pt"]
    for rank, path in enumerate(paths):
        shard = GaussianModel(
            means=gaussian.means.detach()[rank::2],
            log_scales=gaussian.log_scales.detach()[rank::2],
            quats=gaussian.quats.detach()[rank::2],
            opacity_logits=gaussian.opacity_logits.detach()[rank::2],
            sh_coeffs=gaussian.sh_coeffs.detach()[rank::2],
            max_sh_degree=gaussian.max_sh_degree,
        )
        path.write_bytes(_model_bytes(shard))

    destination = tmp_path / "model.pt"
    merged = _merge_model_shards(paths, destination)

    assert merged.count == gaussian.count
    assert _load_model(destination.read_bytes(), torch.device("cpu")).count == gaussian.count
    assert not any(path.exists() for path in paths)


def test_only_position_learning_rate_decays():
    gaussian = model()
    config = resolve_internal_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])
    feature_before = optimizers["sh0"].param_groups[0]["lr"]

    _update_position_learning_rate(optimizers["means"], config, 30_000, 30_000, 1.1)

    assert optimizers["means"].param_groups[0]["lr"] == pytest.approx(0.0000016 * 1.1)
    assert optimizers["sh0"].param_groups[0]["lr"] == feature_before


def test_mcmc_position_learning_rate_uses_frozen_delay_multiplier():
    gaussian = model()
    config = resolve_mcmc_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])

    _update_position_learning_rate(optimizers["means"], config, 0, 30_000, 1.0)

    assert optimizers["means"].param_groups[0]["lr"] == pytest.approx(0.00016 * 0.01)


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


def test_recovery_prune_schedule_fires_once_per_reset_window():
    config = resolve_internal_config(
        overrides={"opacity_reset": {"recovery_prune": {"enabled": True}}}
    ).effective_config

    for iteration in (3_500, 6_500, 9_500, 12_500):
        assert _recovery_prune_due(config, iteration) == 0.05
    for iteration in (3_000, 3_499, 3_501, 13_000, 15_000, 15_500, 30_000):
        assert _recovery_prune_due(config, iteration) is None

    config["opacity_reset"]["recovery_prune"]["enabled"] = False
    assert _recovery_prune_due(config, 3_500) is None

    config["opacity_reset"]["recovery_prune"]["enabled"] = True
    config["opacity_reset"]["enabled"] = False
    assert _recovery_prune_due(config, 3_500) is None
    config["opacity_reset"]["enabled"] = True

    # A window longer than the run never lands inside the training loop.
    config["opacity_reset"]["recovery_prune"]["window_iterations"] = 20_000
    assert _recovery_prune_due(config, 23_000) == 0.05
    assert _recovery_prune_due(config, 20_000) is None  # origin 0 is not a reset


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


def test_checkpoint_state_round_trips_mcmc_strategy_state():
    from gsplat.strategy import MCMCStrategy

    gaussian = model()
    config = resolve_mcmc_config().effective_config
    optimizers = gaussian.optimizers(config["learning_rate"])
    strategy_state = _build_mcmc_strategy(
        MCMCStrategy, config, 3_000_000
    ).initialize_state()

    state = _checkpoint_state(
        gaussian,
        optimizers,
        strategy_state,
        [],
        0,
        [],
        1,
    )
    restored = _torch_load(state.densification, torch.device("cpu"))

    assert torch.equal(
        restored["strategy_state"]["binoms"], strategy_state["binoms"]
    )


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
