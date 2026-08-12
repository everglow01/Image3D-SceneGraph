"""Project-owned 3D Gaussian training lifecycle."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from .checkpoint import (
    CheckpointProvenance,
    CheckpointState,
    create_attempt,
    load_checkpoint,
    prune_attempt_checkpoints,
    write_checkpoint,
)
from .config import ResolvedGaussianConfig, validate_effective_config
from .evaluation import evaluate_model
from .initialization import InitializationResult
from .model import GaussianModel, GaussianModelError
from .render import render_gaussians
from .runtime import TrainingView, load_training_views
from .training_math import active_sh_degree, exponential_learning_rate, l1_ssim_loss


class TrainingError(RuntimeError):
    """Raised when project-owned Gaussian training cannot complete."""


class TrainingCancelled(TrainingError):
    """Raised when cancellation is requested between training iterations."""


class TrainingOutOfMemory(TrainingError):
    """Raised when CUDA reports an out-of-memory condition."""


@dataclass(frozen=True)
class TrainingResult:
    iteration: int
    candidate_iteration: int
    gaussian_count: int
    initial_loss: float
    final_loss: float
    validation: dict[str, Any]
    final_validation: dict[str, Any]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    elapsed_seconds: float
    final_checkpoint_hash: str
    final_checkpoint_path: str
    model_path: str
    result_path: str
    progress_path: str


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_provenance(
    *, dataset_hash: str, effective_config_hash: str
) -> CheckpointProvenance:
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "model.py",
        "render.py",
        "runtime.py",
        "trainer.py",
        "training_math.py",
        "initialization.py",
    ):
        digest.update(name.encode())
        digest.update((source_root / name).read_bytes())
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": getattr(torch.version, "cuda", None),
        "gsplat": _gsplat_version(),
    }
    return CheckpointProvenance(
        dataset_hash=dataset_hash,
        effective_config_hash=effective_config_hash,
        code_hash=digest.hexdigest(),
        environment_hash=hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def train_gaussians(
    *,
    contract: dict[str, Any],
    dataset_root: Path,
    initialization: InitializationResult,
    resolved_config: ResolvedGaussianConfig,
    run_dir: Path,
    attempt_id: str = "train-001",
    attempt_kind: str = "fresh",
    parent_attempt_id: str | None = None,
    resume_iteration: int | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    try:
        from gsplat.strategy import DefaultStrategy
        from gsplat.strategy.ops import reset_opa
    except ImportError as exc:
        raise TrainingError("project Gaussian training requires pinned gsplat") from exc

    config = resolved_config.effective_config
    validate_effective_config(config)
    if not torch.cuda.is_available():
        raise TrainingError("project Gaussian training requires CUDA")
    device = torch.device("cuda")
    seed_training(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = training_provenance(
        dataset_hash=str(contract["dataset_hash"]),
        effective_config_hash=resolved_config.effective_config_hash,
    )
    create_attempt(
        run_dir,
        attempt_id=attempt_id,
        kind=attempt_kind,
        provenance=provenance,
        parent_attempt_id=parent_attempt_id,
        resume_iteration=resume_iteration,
    )
    artifact_dir = run_dir / "attempts" / attempt_id / "artifacts"
    artifact_dir.mkdir()
    progress_path = artifact_dir / "progress.jsonl"

    train_views = load_training_views(
        contract,
        dataset_root,
        split="train",
        longest_edge=int(config["resolution"]["longest_edge"]),
        device=torch.device("cpu"),
    )
    validation_views = load_training_views(
        contract,
        dataset_root,
        split="validation",
        longest_edge=int(config["resolution"]["longest_edge"]),
        device=torch.device("cpu"),
    )
    test_ids = set(str(value) for value in contract["splits"]["test"])
    loaded_ids = {view.camera.image_id for view in (*train_views, *validation_views)}
    if loaded_ids & test_ids:
        raise TrainingError("held-out test views entered the trainer runtime")

    model = GaussianModel.from_points(
        torch.from_numpy(initialization.points).to(device=device, dtype=torch.float32),
        torch.from_numpy(initialization.colors).to(device=device, dtype=torch.float32) / 255.0,
        torch.from_numpy(initialization.scales).to(device=device, dtype=torch.float32),
        max_sh_degree=int(config["sh_schedule"]["max_degree"]),
    )
    scene_scale = _training_scene_scale(contract)
    optimizers = model.optimizers(config["learning_rate"], position_scale=scene_scale)
    strategy = _build_strategy(DefaultStrategy, config, train_views)
    strategy.check_sanity(model.params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    history: list[dict[str, Any]] = []
    start_iteration = 1
    camera_order: list[int] = []
    camera_cursor = 0

    if attempt_kind == "resume":
        if parent_attempt_id is None or resume_iteration is None:
            raise TrainingError("resume requires parent attempt and iteration")
        loaded = load_checkpoint(
            run_dir,
            parent_attempt_id,
            resume_iteration,
            expected_provenance=provenance,
        )
        model = _load_model(loaded.state.model, device)
        optimizers = model.optimizers(
            config["learning_rate"], position_scale=scene_scale
        )
        optimizer_payload = _torch_load(loaded.state.optimizer, device)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(optimizer_payload[name])
        dense_payload = _torch_load(loaded.state.densification, device)
        strategy_state = dense_payload["strategy_state"]
        camera_order = list(dense_payload["camera_order"])
        camera_cursor = int(dense_payload["camera_cursor"])
        history = list(loaded.state.metric_history)
        _restore_rng(loaded.state.rng)
        start_iteration = resume_iteration + 1
        strategy.check_sanity(model.params, optimizers)

    total_iterations = int(config["iterations"])
    if start_iteration > total_iterations:
        raise TrainingError("resume checkpoint already reached configured iterations")
    started = time.perf_counter()
    initial_loss = float(history[0]["loss"]) if history else float("nan")
    last_validation: dict[str, Any] = {"status": "not_run"}
    best_validation = -float("inf")
    best_validation_iteration: int | None = None
    best_validation_payload: dict[str, Any] | None = None
    completed_iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, total_iterations + 1):
            if cancel_requested is not None and cancel_requested():
                if completed_iteration > 0:
                    _write_latest_checkpoint(
                        run_dir,
                        attempt_id=attempt_id,
                        iteration=completed_iteration,
                        purpose="periodic",
                        validation_score=None,
                        provenance=provenance,
                        state=_checkpoint_state(
                            model,
                            optimizers,
                            strategy_state,
                            camera_order,
                            camera_cursor,
                            history,
                            completed_iteration,
                        ),
                    )
                raise TrainingCancelled("Gaussian training cancellation requested")

            camera_order, camera_cursor, view_index = _next_camera(
                len(train_views), camera_order, camera_cursor
            )
            _update_position_learning_rate(
                optimizers["means"], config, iteration, total_iterations, scene_scale
            )
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            view, rendered = _render_visible_training_view(
                model,
                train_views,
                view_index,
                active_sh_degree(iteration, config["sh_schedule"]),
            )
            strategy.step_pre_backward(
                model.params, optimizers, strategy_state, iteration, rendered.metadata
            )
            loss, terms = l1_ssim_loss(
                rendered.image.clamp(0, 1) if config["loss"]["clamp_render"] else rendered.image,
                view.image,
                l1_weight=float(config["loss"]["l1_weight"]),
                ssim_weight=float(config["loss"]["ssim_weight"]),
            )
            if not torch.isfinite(loss):
                raise TrainingError(f"non-finite training loss at iteration {iteration}")
            loss.backward()
            model.validate_gradients()
            for optimizer in optimizers.values():
                optimizer.step()

            before_count = model.count
            strategy.step_post_backward(
                model.params,
                optimizers,
                strategy_state,
                iteration,
                rendered.metadata,
                packed=True,
            )
            if _opacity_reset_due(config, iteration):
                reset_opa(
                    params=model.params,
                    optimizers=optimizers,
                    state=strategy_state,
                    value=float(config["pruning"]["opacity_threshold"]) * 2.0,
                )
            model.validate()
            after_count = model.count
            value = float(loss.detach())
            if not history:
                initial_loss = value
            event: dict[str, Any] = {
                "iteration": iteration,
                "view_id": view.camera.image_id,
                "loss": value,
                "l1": terms["l1"],
                "ssim": terms["ssim"],
                "sh_degree": active_sh_degree(iteration, config["sh_schedule"]),
                "gaussian_count": after_count,
            }
            if after_count != before_count:
                event.update(
                    {
                        "topology_count_before": before_count,
                        "topology_count_after": after_count,
                        "topology_net_growth": after_count - before_count,
                    }
                )
            if _opacity_reset_due(config, iteration):
                event["opacity_reset"] = True
            history.append(event)
            _publish_event(progress_path, event, progress_callback)
            completed_iteration = iteration

            validation_due = iteration in config["evaluation"]["validation_iterations"]
            if validation_due:
                last_validation = evaluate_views(
                    model,
                    validation_views,
                    config,
                    preview_dir=artifact_dir / "validation" / f"iteration_{iteration:09d}",
                    progress_events=history,
                )
                validation_event = {
                    "iteration": iteration,
                    "event": "validation",
                    "optimizer_updates": iteration,
                    "nominal_iterations": total_iterations,
                    "attempt_elapsed_seconds": time.perf_counter() - started,
                    **last_validation,
                }
                history.append(validation_event)
                _publish_event(progress_path, validation_event, progress_callback)
                candidate = float(last_validation["mean_psnr"])
                if candidate > best_validation:
                    best_validation = candidate
                    best_validation_iteration = iteration
                    best_validation_payload = last_validation
                    (artifact_dir / ".best-model.pt").write_bytes(_model_bytes(model))

            if iteration == total_iterations:
                _write_latest_checkpoint(
                    run_dir,
                    attempt_id=attempt_id,
                    iteration=iteration,
                    purpose="final",
                    validation_score=None,
                    provenance=provenance,
                    state=_checkpoint_state(
                        model,
                        optimizers,
                        strategy_state,
                        camera_order,
                        camera_cursor,
                        history,
                        iteration,
                    ),
                )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise TrainingOutOfMemory("CUDA out of memory during Gaussian training") from exc
    except GaussianModelError as exc:
        raise TrainingError(str(exc)) from exc

    elapsed = time.perf_counter() - started
    if best_validation_iteration is None or best_validation_payload is None:
        raise TrainingError("training completed without a Validation candidate")
    final_checkpoint = load_checkpoint(
        run_dir, attempt_id, total_iterations, expected_provenance=provenance
    )
    candidate_path = artifact_dir / ".best-model.pt"
    candidate_model = _load_model(candidate_path.read_bytes(), device)
    _validate_model_health(candidate_model, len(initialization.points), best_validation_payload)
    model_path = artifact_dir / "model.pt"
    model_path.write_bytes(candidate_path.read_bytes())
    candidate_path.unlink()
    result_path = artifact_dir / "result.json"
    result = TrainingResult(
        iteration=total_iterations,
        candidate_iteration=best_validation_iteration,
        gaussian_count=candidate_model.count,
        initial_loss=initial_loss,
        final_loss=float(next(item["loss"] for item in reversed(history) if "loss" in item)),
        validation=best_validation_payload,
        final_validation=last_validation,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        elapsed_seconds=elapsed,
        final_checkpoint_hash=final_checkpoint.record.checkpoint_hash,
        final_checkpoint_path=(
            Path("attempts") / attempt_id / "checkpoints" / f"iteration_{total_iterations:09d}"
        ).as_posix(),
        model_path=model_path.relative_to(run_dir).as_posix(),
        result_path=result_path.relative_to(run_dir).as_posix(),
        progress_path=progress_path.relative_to(run_dir).as_posix(),
    )
    result_path.write_text(
        json.dumps(result.__dict__, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


@torch.no_grad()
def evaluate_views(
    model: GaussianModel,
    views: list[TrainingView],
    config: dict[str, Any],
    *,
    preview_dir: Path | None = None,
    progress_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    evaluated = evaluate_model(
        model,
        views,
        split="validation",
        sh_degree=active_sh_degree(int(config["iterations"]), config["sh_schedule"]),
        preview_dir=preview_dir,
        progress_events=progress_events,
        renderer=render_gaussians,
        health_thresholds={
            "split_screen_fraction": 0.05,
            "max_screen_fraction": 0.15,
            "max_world_scale": float(config["pruning"]["max_world_scale"]),
            "opacity_threshold": float(config["pruning"]["opacity_threshold"]),
        },
    )
    return {
        **evaluated,
        "mean_psnr": evaluated["psnr"]["mean"],
        "mean_ssim": evaluated["ssim"]["mean"],
    }


def save_validation_previews(
    model: GaussianModel,
    views: list[TrainingView],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    degree = active_sh_degree(int(config["iterations"]), config["sh_schedule"])
    with torch.no_grad():
        for view in views:
            rendered = render_gaussians(model, view.camera, sh_degree=degree, background=None)
            pixels = rendered.image.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
            Image.fromarray(pixels).save(output_dir / f"validation_{view.camera.image_id}.png")


def _build_strategy(
    strategy_type,
    config: dict[str, Any],
    train_views: list[TrainingView] | tuple[TrainingView, ...] = (),
):
    densify = config["densification"]
    pruning = config["pruning"]
    max_dimension = max(
        (max(view.camera.width, view.camera.height) for view in train_views),
        default=int(config["resolution"]["longest_edge"]),
    )
    screen_pruning = bool(pruning["enabled"]) and bool(pruning["screen_radius_enabled"])
    screen_fraction = (
        float(pruning["max_screen_radius_pixels"]) / max_dimension
        if screen_pruning
        else 1e10
    )
    return strategy_type(
        prune_opa=float(pruning["opacity_threshold"]) if pruning["enabled"] else 0.0,
        grow_grad2d=float(densify["gradient_threshold"]),
        grow_scale3d=float(densify["scale_threshold"]),
        grow_scale2d=1e10,
        prune_scale3d=float(pruning["max_world_scale"]) if pruning["enabled"] else 1e10,
        prune_scale2d=screen_fraction,
        refine_scale2d_stop_iter=(
            int(densify["end_iteration"])
            if densify["enabled"] and screen_pruning
            else 0
        ),
        refine_start_iter=int(densify["start_iteration"]),
        refine_stop_iter=int(densify["end_iteration"]) if densify["enabled"] else 0,
        reset_every=int(config["opacity_reset"]["every_iterations"]),
        refine_every=int(densify["every_iterations"]),
        absgrad=False,
        revised_opacity=False,
        verbose=False,
    )


def _next_camera(
    count: int, order: list[int], cursor: int
) -> tuple[list[int], int, int]:
    if cursor >= len(order):
        order = torch.randperm(count, device="cpu").tolist()
        cursor = 0
    index = int(order[cursor])
    return order, cursor + 1, index


def _render_visible_training_view(
    model: GaussianModel,
    views: list[TrainingView],
    first_index: int,
    sh_degree: int,
):
    for offset in range(len(views)):
        view = views[(first_index + offset) % len(views)].to(model.means.device)
        rendered = render_gaussians(
            model,
            view.camera,
            sh_degree=sh_degree,
            background=None,
        )
        radii = rendered.metadata.get("radii")
        if radii is not None and bool((radii > 0).any()):
            return view, rendered
    raise TrainingError("no training view sees any initialized Gaussian")


def _opacity_reset_due(config: dict[str, Any], iteration: int) -> bool:
    reset = config["opacity_reset"]
    return (
        bool(reset["enabled"])
        and bool(config["densification"]["enabled"])
        and iteration > 0
        and iteration < int(config["densification"]["end_iteration"])
        and iteration % int(reset["every_iterations"]) == 0
    )


def _training_scene_scale(contract: dict[str, Any]) -> float:
    train_ids = {str(value) for value in contract["splits"]["train"]}
    normalized_from_world = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    centers = []
    for image in contract["images"]:
        if str(image["image_id"]) not in train_ids:
            continue
        center = np.asarray(image["world_from_camera"], dtype=np.float64)[:3, 3]
        centers.append(
            normalized_from_world[:3, :3] @ center + normalized_from_world[:3, 3]
        )
    if not centers:
        raise TrainingError("training split contains no camera centers")
    values = np.stack(centers)
    radius = float(np.linalg.norm(values - values.mean(axis=0), axis=1).max()) * 1.1
    if not np.isfinite(radius) or radius <= 0:
        raise TrainingError("training cameras do not define a usable scene extent")
    return radius


def _update_position_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    iteration: int,
    total_iterations: int,
    scene_scale: float = 1.0,
) -> None:
    settings = config["learning_rate"]["position"]
    optimizer.param_groups[0]["lr"] = scene_scale * exponential_learning_rate(
        float(settings["initial"]),
        float(settings["final"]),
        iteration,
        total_iterations,
        delay_multiplier=1.0,
    )


def _validate_model_health(
    model: GaussianModel, initial_count: int, validation: dict[str, Any]
) -> dict[str, Any]:
    model.validate()
    opacity = model.opacity_logits.detach().sigmoid()
    effective = int((opacity > 0.01).sum())
    minimum_count = max(1, int(initial_count * 0.05))
    mean_psnr = float(validation.get("mean_psnr", float("nan")))
    mean_ssim = float(validation.get("mean_ssim", float("nan")))
    if model.count < minimum_count:
        raise TrainingError(f"Gaussian model collapsed to {model.count} rows; minimum is {minimum_count}")
    if effective == 0:
        raise TrainingError("Gaussian model has no effective-opacity rows")
    if not np.isfinite(mean_psnr) or not np.isfinite(mean_ssim):
        raise TrainingError("Gaussian Validation metrics are non-finite")
    return {
        "gaussian_count": model.count,
        "effective_opacity_count": effective,
        "opacity_p50": float(torch.quantile(opacity, 0.5)),
        "opacity_p90": float(torch.quantile(opacity, 0.9)),
    }


def _write_latest_checkpoint(
    run_dir: Path,
    *,
    attempt_id: str,
    iteration: int,
    purpose: str,
    validation_score: float | None,
    provenance: CheckpointProvenance,
    state: CheckpointState,
) -> None:
    write_checkpoint(
        run_dir,
        attempt_id=attempt_id,
        iteration=iteration,
        purpose=purpose,
        validation_score=validation_score,
        provenance=provenance,
        state=state,
    )
    prune_attempt_checkpoints(run_dir, attempt_id, keep_iterations=(iteration,))


def _checkpoint_state(
    model: GaussianModel,
    optimizers: dict[str, torch.optim.Optimizer],
    strategy_state: dict[str, Any],
    camera_order: list[int],
    camera_cursor: int,
    history: list[dict[str, Any]],
    iteration: int,
) -> CheckpointState:
    return CheckpointState(
        model=_model_bytes(model),
        optimizer=_torch_bytes({name: value.state_dict() for name, value in optimizers.items()}),
        scheduler=json.dumps({"iteration": iteration}).encode(),
        densification=_torch_bytes(
            {
                "strategy_state": strategy_state,
                "camera_order": camera_order,
                "camera_cursor": camera_cursor,
            }
        ),
        rng=_rng_bytes(),
        metric_history=tuple(history),
    )


def _model_bytes(model: GaussianModel) -> bytes:
    snapshot = model.snapshot()
    return _torch_bytes(
        {
            "max_sh_degree": int(snapshot["max_sh_degree"]),
            "state_dict": {
                name: value.cpu()
                for name, value in snapshot.items()
                if name != "max_sh_degree"
            },
        }
    )


def _load_model(content: bytes, device: torch.device) -> GaussianModel:
    payload = _torch_load(content, device)
    state = payload["state_dict"]
    if "log_scales" not in state:
        state = {
            "means": state["params.means"],
            "log_scales": state["params.scales"],
            "quats": state["params.quats"],
            "opacity_logits": state["params.opacities"],
            "sh_coeffs": torch.cat((state["params.sh0"], state["params.shN"]), dim=1),
        }
    model = GaussianModel(
        means=state["means"],
        log_scales=state["log_scales"],
        quats=state["quats"],
        opacity_logits=state["opacity_logits"],
        sh_coeffs=state["sh_coeffs"],
        max_sh_degree=int(payload["max_sh_degree"]),
    ).to(device)
    model.validate()
    return model


def _rng_bytes() -> bytes:
    return _torch_bytes(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        }
    )


def _restore_rng(content: bytes) -> None:
    payload = _torch_load(content, torch.device("cpu"))
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda"]])


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _torch_load(content: bytes, device: torch.device) -> Any:
    return torch.load(io.BytesIO(content), map_location=device, weights_only=False)


def _publish_event(
    path: Path,
    event: dict[str, Any],
    callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
    if callback is not None:
        callback(event)


def _gsplat_version() -> str:
    try:
        import gsplat
    except ImportError:
        return "missing"
    return str(gsplat.__version__)
