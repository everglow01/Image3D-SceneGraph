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
from .training_math import (
    active_sh_degree,
    exponential_learning_rate,
    l1_ssim_loss,
)


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
    *,
    dataset_hash: str,
    effective_config_hash: str,
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
    optimizer = torch.optim.Adam(model.parameter_groups(config["learning_rate"]), eps=1e-15)
    history: list[dict[str, Any]] = []
    start_iteration = 1
    gradient_sum = torch.zeros(model.count, device=device)
    gradient_count = torch.zeros(model.count, device=device)
    max_radius = torch.zeros(model.count, device=device)

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
        optimizer = torch.optim.Adam(model.parameter_groups(config["learning_rate"]), eps=1e-15)
        optimizer.load_state_dict(_torch_load(loaded.state.optimizer, device))
        densification = _torch_load(loaded.state.densification, device)
        gradient_sum = densification["gradient_sum"]
        gradient_count = densification["gradient_count"]
        max_radius = densification["max_radius"]
        history = list(loaded.state.metric_history)
        _restore_rng(loaded.state.rng)
        start_iteration = resume_iteration + 1

    total_iterations = int(config["iterations"])
    if start_iteration > total_iterations:
        raise TrainingError("resume checkpoint already reached configured iterations")
    started = time.perf_counter()
    initial_loss = float(history[0]["loss"]) if history else float("nan")
    last_validation: dict[str, Any] = {"status": "not_run"}
    best_validation = -float("inf")
    best_validation_iteration: int | None = None
    best_validation_payload: dict[str, Any] | None = None

    try:
        for iteration in range(start_iteration, total_iterations + 1):
            if cancel_requested is not None and cancel_requested():
                raise TrainingCancelled("Gaussian training cancellation requested")
            _update_learning_rates(optimizer, config, iteration, total_iterations)
            first_view_index = int(
                torch.randint(
                    len(train_views),
                    (1,),
                    generator=_iteration_generator(int(config["seed"]), iteration),
                )
            )
            optimizer.zero_grad(set_to_none=True)
            view, rendered = _render_visible_training_view(
                model,
                train_views,
                first_view_index,
                active_sh_degree(iteration, config["sh_schedule"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss, terms = l1_ssim_loss(
                rendered.image,
                view.image,
                l1_weight=float(config["loss"]["l1_weight"]),
                ssim_weight=float(config["loss"]["ssim_weight"]),
            )
            if not torch.isfinite(loss):
                raise TrainingError(f"non-finite training loss at iteration {iteration}")
            loss.backward()
            model.validate_gradients()
            _accumulate_statistics(
                rendered.metadata,
                gradient_sum,
                gradient_count,
                max_radius,
            )
            optimizer.step()
            model.validate(max_count=int(config["gaussian_budget"]["max_count"]))
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
                "gaussian_count": model.count,
            }

            validation_improved = False
            validation_due = (
                iteration % int(config["evaluation"]["validation_every_iterations"]) == 0
                or iteration == total_iterations
            )
            final_iteration = iteration == total_iterations
            cleanup_iteration_due = iteration == int(config["pruning"]["cleanup_iteration"])
            boundary_cleanup_due = iteration == int(config["densification"]["end_iteration"])
            if boundary_cleanup_due or cleanup_iteration_due:
                _collect_screen_statistics(
                    model,
                    [*train_views, *validation_views] if cleanup_iteration_due else train_views,
                    active_sh_degree(iteration, config["sh_schedule"]),
                    max_radius,
                )
            if boundary_cleanup_due or cleanup_iteration_due:
                cleanup = _maybe_update_topology(
                    model,
                    optimizer,
                    gradient_sum,
                    gradient_count,
                    max_radius,
                    config,
                    iteration,
                    cleanup_only=True,
                )
                if cleanup is not None:
                    cleanup_event, optimizer, gradient_sum, gradient_count, max_radius = cleanup
                    cleanup_event.update(
                        {
                            "iteration": iteration,
                            "event": "final_cleanup" if cleanup_iteration_due else "boundary_cleanup",
                        }
                    )
                    history.append(cleanup_event)
                    _publish_event(progress_path, cleanup_event, progress_callback)

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
                    **last_validation,
                }
                history.append(validation_event)
                _publish_event(progress_path, validation_event, progress_callback)
                candidate = float(last_validation["mean_psnr"])
                selection_eligible = boundary_cleanup_due or iteration >= int(
                    config["pruning"]["cleanup_iteration"]
                )
                if selection_eligible and candidate > best_validation:
                    validation_improved = True
                    best_validation = candidate
                    best_validation_iteration = iteration
                    best_validation_payload = last_validation
                    (artifact_dir / ".best-model.pt").write_bytes(
                        _checkpoint_state(
                            model,
                            optimizer,
                            gradient_sum,
                            gradient_count,
                            max_radius,
                            history,
                            iteration,
                        ).model
                    )

            topology = None
            if not final_iteration and not boundary_cleanup_due and not cleanup_iteration_due:
                topology = _maybe_update_topology(
                    model,
                    optimizer,
                    gradient_sum,
                    gradient_count,
                    max_radius,
                    config,
                    iteration,
                    train_view_count=len(train_views),
                )
            if topology is not None:
                topology_event, optimizer, gradient_sum, gradient_count, max_radius = topology
                event.update(topology_event)
            reset = config["opacity_reset"]
            densification = config["densification"]
            reset_due = (
                reset["enabled"]
                and iteration % int(reset["every_iterations"]) == 0
                and iteration < int(densification["end_iteration"])
                and iteration < total_iterations
            )
            if reset_due:
                model.reset_opacity(float(reset["value"]))
                _clear_optimizer_parameter_state(optimizer, model.opacity_logits)
                event["opacity_reset"] = True
            history.append(event)
            _publish_event(progress_path, event, progress_callback)

            if _final_checkpoint_due(iteration, total_iterations):
                _write_latest_checkpoint(
                    run_dir,
                    attempt_id=attempt_id,
                    iteration=iteration,
                    purpose="final",
                    validation_score=None,
                    provenance=provenance,
                    state=_checkpoint_state(
                        model,
                        optimizer,
                        gradient_sum,
                        gradient_count,
                        max_radius,
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
        run_dir,
        attempt_id,
        total_iterations,
        expected_provenance=provenance,
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
            Path("attempts")
            / attempt_id
            / "checkpoints"
            / f"iteration_{total_iterations:09d}"
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
            "split_screen_fraction": float(config["densification"]["split_screen_fraction"]),
            "max_screen_fraction": float(config["pruning"]["max_screen_fraction"]),
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
            rendered = render_gaussians(
                model,
                view.camera,
                sh_degree=degree,
                background=torch.ones(3, device=model.means.device),
            )
            pixels = (
                rendered.image.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
            )
            Image.fromarray(pixels).save(output_dir / f"validation_{view.camera.image_id}.png")


def _maybe_update_topology(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    gradient_sum: torch.Tensor,
    gradient_count: torch.Tensor,
    max_radius: torch.Tensor,
    config: dict[str, Any],
    iteration: int,
    *,
    train_view_count: int = 0,
    cleanup_only: bool = False,
) -> tuple[
    dict[str, int | float],
    torch.optim.Optimizer,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    densify = config["densification"]
    scheduled = (
        densify["enabled"]
        and int(densify["start_iteration"]) <= iteration <= int(densify["end_iteration"])
        and iteration % int(densify["every_iterations"]) == 0
    )
    reset_interval = int(config["opacity_reset"]["every_iterations"])
    recovery_iterations = train_view_count + int(densify["every_iterations"])
    recovering_from_reset = (
        config["opacity_reset"]["enabled"]
        and iteration >= reset_interval
        and iteration % reset_interval < recovery_iterations
    )
    if not cleanup_only and (not scheduled or recovering_from_reset):
        return None

    average = gradient_sum / gradient_count.clamp_min(1)
    scales = model.log_scales.detach().exp().amax(dim=1)
    high_gradient = average > float(densify["gradient_threshold"])
    large = scales > float(densify["duplicate_scale_threshold"])
    screen_active = iteration <= int(densify["screen_size_end_iteration"])
    screen_split = max_radius > float(densify["split_screen_fraction"])

    pruning = config["pruning"]
    finite = torch.stack(
        [
            torch.isfinite(parameter).flatten(1).all(dim=1)
            if parameter.ndim > 1
            else torch.isfinite(parameter)
            for parameter in model.parameters()
        ]
    ).all(dim=0)
    low_opacity = (
        model.opacity_logits.detach().sigmoid() < float(pruning["opacity_threshold"])
        if pruning["enabled"]
        else torch.zeros(model.count, dtype=torch.bool, device=model.means.device)
    )
    screen_size = (
        max_radius > float(pruning["max_screen_fraction"])
        if pruning["enabled"] and (screen_active or cleanup_only)
        else torch.zeros(model.count, dtype=torch.bool, device=model.means.device)
    )
    world_size = (
        scales > float(pruning["max_world_scale"])
        if pruning["enabled"] and iteration > reset_interval
        else torch.zeros(model.count, dtype=torch.bool, device=model.means.device)
    )
    prune = ~finite | low_opacity | screen_size | world_size
    if prune.all():
        prune[torch.argmax(model.opacity_logits.detach())] = False

    split_mask = ((high_gradient & large) | (screen_split & screen_active)) & ~prune
    duplicate_mask = high_gradient & ~large & ~prune
    split_candidates = _rank_topology_candidates(
        split_mask,
        screen_split,
        average,
    )
    duplicate_candidates = _rank_topology_candidates(
        duplicate_mask,
        torch.zeros_like(duplicate_mask),
        average,
    )
    keep_indices = torch.where(~prune)[0]
    children = int(densify["split_children"])
    budget = int(config["gaussian_budget"]["max_count"])
    available = max(0, budget - len(keep_indices))
    if cleanup_only:
        split_indices = split_candidates[:0]
        duplicate_indices = duplicate_candidates[:0]
    else:
        split_capacity, duplicate_capacity = _bounded_growth_capacity(
            available,
            len(split_candidates),
            len(duplicate_candidates),
            children - 1,
            float(densify["split_budget_fraction"]),
        )
        split_indices = split_candidates[:split_capacity]
        duplicate_indices = duplicate_candidates[:duplicate_capacity]

    old_parameters = dict(model.named_group_parameters())
    topology_map = model.update_topology(
        keep_indices,
        duplicate_indices,
        split_indices,
        split_children=children,
    )
    source_indices = topology_map[0]
    new_rows = topology_map[1].bool()
    optimizer = _remap_optimizer(
        optimizer,
        old_parameters,
        model,
        source_indices,
        new_rows,
        config["learning_rate"],
    )
    device = model.means.device
    active = average[gradient_count > 0]
    quantiles = (
        torch.quantile(active, torch.tensor([0.5, 0.9, 0.99], device=device))
        if len(active)
        else torch.zeros(3, device=device)
    )
    event: dict[str, int | float] = {
        "duplicate_candidates": int(len(duplicate_candidates)),
        "duplicate_selected": int(len(duplicate_indices)),
        "split_candidates": int(len(split_candidates)),
        "split_selected": int(len(split_indices)),
        "budget_skipped": int(
            len(split_candidates) - len(split_indices)
            + len(duplicate_candidates) - len(duplicate_indices)
        ),
        "budget_evicted": 0,
        "duplicated": int(len(duplicate_indices)),
        "split_parents": int(len(split_indices)),
        "split_children": int(len(split_indices) * children),
        "densified": int(len(duplicate_indices) + len(split_indices) * (children - 1)),
        "pruned": int(prune.sum()),
        "pruned_non_finite": int((~finite).sum()),
        "pruned_low_opacity": int((low_opacity & finite).sum()),
        "pruned_screen_size": int((screen_size & finite & ~low_opacity).sum()),
        "pruned_world_size": int(
            (world_size & finite & ~low_opacity & ~screen_size).sum()
        ),
        "gradient_candidates": int(high_gradient.sum()),
        "gradient_p50": float(quantiles[0]),
        "gradient_p90": float(quantiles[1]),
        "gradient_p99": float(quantiles[2]),
        "gaussian_count": model.count,
    }
    return (
        event,
        optimizer,
        torch.zeros(model.count, device=device),
        torch.zeros(model.count, device=device),
        torch.zeros(model.count, device=device),
    )


def _bounded_growth_capacity(
    available: int,
    split_candidates: int,
    duplicate_candidates: int,
    split_cost: int,
    split_fraction: float,
) -> tuple[int, int]:
    if available <= 0 or split_cost <= 0:
        return 0, 0
    total_demand = split_candidates * split_cost + duplicate_candidates
    if total_demand <= available:
        return split_candidates, duplicate_candidates
    split_budget = min(available, max(split_cost, int(available * split_fraction)))
    selected_splits = min(split_candidates, split_budget // split_cost)
    remaining = available - selected_splits * split_cost
    selected_duplicates = min(duplicate_candidates, remaining)
    remaining -= selected_duplicates
    if remaining:
        selected_splits += min(split_candidates - selected_splits, remaining // split_cost)
    return selected_splits, selected_duplicates


def _rank_topology_candidates(
    mask: torch.Tensor,
    screen_priority: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    indices = torch.where(mask)[0]
    if not len(indices):
        return indices
    indices = indices[torch.argsort(gradient[indices], descending=True, stable=True)]
    return indices[
        torch.argsort(screen_priority[indices].to(torch.int8), descending=True, stable=True)
    ]


def _remap_optimizer(
    optimizer: torch.optim.Optimizer,
    old_parameters: dict[str, torch.nn.Parameter],
    model: GaussianModel,
    source_indices: torch.Tensor,
    new_rows: torch.Tensor,
    learning_rates: dict[str, Any],
) -> torch.optim.Optimizer:
    remapped = torch.optim.Adam(model.parameter_groups(learning_rates), eps=1e-15)
    for group in remapped.param_groups:
        name = group["name"]
        old_parameter = old_parameters[name]
        state = optimizer.state.get(old_parameter)
        if not state:
            continue
        new_parameter = group["params"][0]
        new_state: dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == old_parameter.shape:
                remapped_value = value[source_indices].clone()
                remapped_value[new_rows] = 0
                new_state[key] = remapped_value
            elif torch.is_tensor(value):
                new_state[key] = value.clone()
            else:
                new_state[key] = value
        remapped.state[new_parameter] = new_state
    return remapped


def _clear_optimizer_parameter_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> None:
    for key, value in optimizer.state.get(parameter, {}).items():
        if key != "step" and torch.is_tensor(value):
            value.zero_()


def _validate_model_health(
    model: GaussianModel,
    initial_count: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    model.validate()
    opacity = model.opacity_logits.detach().sigmoid()
    effective = int((opacity > 0.01).sum())
    minimum_count = max(1, int(initial_count * 0.05))
    mean_psnr = float(validation.get("mean_psnr", float("nan")))
    mean_ssim = float(validation.get("mean_ssim", float("nan")))
    if model.count < minimum_count:
        raise TrainingError(
            f"Gaussian model collapsed to {model.count} rows; minimum is {minimum_count}"
        )
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


@torch.no_grad()
def _collect_screen_statistics(
    model: GaussianModel,
    views: list[TrainingView],
    sh_degree: int,
    max_radius: torch.Tensor,
) -> None:
    for stored_view in views:
        view = stored_view.to(model.means.device)
        rendered = render_gaussians(
            model,
            view.camera,
            sh_degree=sh_degree,
            background=torch.ones(3, device=model.means.device),
        )
        ids = rendered.metadata.get("gaussian_ids")
        radii = rendered.metadata.get("radii")
        if ids is None or radii is None or not len(ids):
            continue
        normalized = radii.to(torch.float32).amax(dim=-1) / float(
            max(rendered.metadata["width"], rendered.metadata["height"])
        )
        max_radius.scatter_reduce_(
            0,
            ids.long(),
            normalized,
            reduce="amax",
            include_self=True,
        )


def _accumulate_statistics(
    metadata: dict[str, Any],
    gradient_sum: torch.Tensor,
    gradient_count: torch.Tensor,
    max_radius: torch.Tensor,
) -> None:
    means2d = metadata.get("means2d")
    ids = metadata.get("gaussian_ids")
    radii = metadata.get("radii")
    absolute = getattr(means2d, "absgrad", None)
    if absolute is None or ids is None or radii is None or not len(ids):
        return
    gradients = absolute.clone()
    gradients[..., 0] *= float(metadata["width"]) / 2.0
    gradients[..., 1] *= float(metadata["height"]) / 2.0
    values = gradients.norm(dim=-1)
    ids = ids.long()
    gradient_sum.index_add_(0, ids, values)
    gradient_count.index_add_(0, ids, torch.ones_like(values))
    radius_values = radii.to(torch.float32).amax(dim=-1) / float(
        max(metadata["width"], metadata["height"])
    )
    max_radius.scatter_reduce_(0, ids, radius_values, reduce="amax", include_self=True)


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
            background=torch.ones(3, device=model.means.device),
            gradient_statistics=True,
        )
        radii = rendered.metadata.get("radii")
        if radii is not None and bool((radii > 0).any()):
            return view, rendered
    raise TrainingError("no training view sees any initialized Gaussian")


def _iteration_generator(seed: int, iteration: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + iteration)
    return generator


def _update_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    iteration: int,
    total_iterations: int,
) -> None:
    learning_rate = config["learning_rate"]
    for group in optimizer.param_groups:
        settings = learning_rate[group["name"]]
        group["lr"] = exponential_learning_rate(
            float(settings["initial"]),
            float(settings["final"]),
            iteration,
            total_iterations,
            delay_multiplier=float(learning_rate["delay_multiplier"]),
        )


def _final_checkpoint_due(iteration: int, total_iterations: int) -> bool:
    return iteration == total_iterations


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
    optimizer: torch.optim.Optimizer,
    gradient_sum: torch.Tensor,
    gradient_count: torch.Tensor,
    max_radius: torch.Tensor,
    history: list[dict[str, Any]],
    iteration: int,
) -> CheckpointState:
    model_payload = {
        "max_sh_degree": model.max_sh_degree,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    densification = {
        "gradient_sum": gradient_sum.detach().cpu(),
        "gradient_count": gradient_count.detach().cpu(),
        "max_radius": max_radius.detach().cpu(),
    }
    return CheckpointState(
        model=_torch_bytes(model_payload),
        optimizer=_torch_bytes(optimizer.state_dict()),
        scheduler=json.dumps({"iteration": iteration}).encode(),
        densification=_torch_bytes(densification),
        rng=_rng_bytes(),
        metric_history=tuple(history),
    )


def _load_model(content: bytes, device: torch.device) -> GaussianModel:
    payload = _torch_load(content, device)
    state = payload["state_dict"]
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
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    return _torch_bytes(payload)


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
