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
    write_checkpoint,
)
from .config import ResolvedGaussianConfig, validate_effective_config
from .initialization import InitializationResult
from .model import GaussianModel, GaussianModelError
from .render import render_gaussians
from .runtime import TrainingView, load_training_views
from .training_math import (
    active_sh_degree,
    exponential_learning_rate,
    l1_ssim_loss,
    psnr,
    structural_similarity,
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
    gaussian_count: int
    initial_loss: float
    final_loss: float
    validation: dict[str, Any]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    elapsed_seconds: float
    checkpoint_hash: str
    checkpoint_path: str
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
        device=device,
    )
    validation_views = load_training_views(
        contract,
        dataset_root,
        split="validation",
        longest_edge=int(config["resolution"]["longest_edge"]),
        device=device,
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

            topology = _maybe_update_topology(
                model,
                gradient_sum,
                gradient_count,
                max_radius,
                config,
                iteration,
            )
            if topology is not None:
                event.update(topology)
                optimizer = torch.optim.Adam(
                    model.parameter_groups(config["learning_rate"]), eps=1e-15
                )
                gradient_sum = torch.zeros(model.count, device=device)
                gradient_count = torch.zeros(model.count, device=device)
                max_radius = torch.zeros(model.count, device=device)
            reset = config["opacity_reset"]
            if reset["enabled"] and iteration % int(reset["every_iterations"]) == 0:
                model.reset_opacity(float(reset["value"]))
                event["opacity_reset"] = True
            history.append(event)
            _publish_event(progress_path, event, progress_callback)

            validation_due = (
                iteration % int(config["evaluation"]["validation_every_iterations"]) == 0
                or iteration == total_iterations
            )
            if validation_due:
                last_validation = evaluate_views(model, validation_views, config)
                validation_event = {
                    "iteration": iteration,
                    "event": "validation",
                    **last_validation,
                }
                history.append(validation_event)
                _publish_event(progress_path, validation_event, progress_callback)

            checkpoint_due = (
                iteration % int(config["checkpoint"]["every_iterations"]) == 0
                or iteration == total_iterations
            )
            if checkpoint_due:
                purpose = "final" if iteration == total_iterations else "periodic"
                score = None
                if validation_due and iteration != total_iterations:
                    candidate = float(last_validation["mean_psnr"])
                    if candidate > best_validation:
                        purpose = "best_validation"
                        score = candidate
                        best_validation = candidate
                write_checkpoint(
                    run_dir,
                    attempt_id=attempt_id,
                    iteration=iteration,
                    purpose=purpose,
                    validation_score=score,
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
    final_checkpoint = load_checkpoint(
        run_dir,
        attempt_id,
        total_iterations,
        expected_provenance=provenance,
    )
    model_path = artifact_dir / "model.pt"
    model_path.write_bytes(final_checkpoint.state.model)
    result_path = artifact_dir / "result.json"
    result = TrainingResult(
        iteration=total_iterations,
        gaussian_count=model.count,
        initial_loss=initial_loss,
        final_loss=float(next(item["loss"] for item in reversed(history) if "loss" in item)),
        validation=last_validation,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        elapsed_seconds=elapsed,
        checkpoint_hash=final_checkpoint.record.checkpoint_hash,
        checkpoint_path=(
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
) -> dict[str, Any]:
    per_view = []
    degree = active_sh_degree(int(config["iterations"]), config["sh_schedule"])
    for view in views:
        rendered = render_gaussians(
            model,
            view.camera,
            sh_degree=degree,
            background=torch.ones(3, device=model.means.device),
        )
        per_view.append(
            {
                "image_id": view.camera.image_id,
                "psnr": psnr(rendered.image, view.image),
                "ssim": float(structural_similarity(rendered.image, view.image)),
            }
        )
    return {
        "status": "complete",
        "split": "validation",
        "num_views": len(per_view),
        "mean_psnr": float(np.mean([item["psnr"] for item in per_view])),
        "mean_ssim": float(np.mean([item["ssim"] for item in per_view])),
        "lpips": {"status": "not_run", "reason": "dependency_not_audited_in_r2_10"},
        "per_view": per_view,
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
    gradient_sum: torch.Tensor,
    gradient_count: torch.Tensor,
    max_radius: torch.Tensor,
    config: dict[str, Any],
    iteration: int,
) -> dict[str, int] | None:
    densify = config["densification"]
    if not densify["enabled"]:
        return None
    if not (
        int(densify["start_iteration"]) <= iteration <= int(densify["end_iteration"])
        and iteration % int(densify["every_iterations"]) == 0
    ):
        return None
    average = gradient_sum / gradient_count.clamp_min(1)
    clone_mask = average > float(densify["gradient_threshold"])
    pruning = config["pruning"]
    keep = torch.ones(model.count, dtype=torch.bool, device=model.means.device)
    if pruning["enabled"]:
        keep &= model.opacity_logits.sigmoid() >= float(pruning["opacity_threshold"])
        keep &= max_radius <= float(pruning["max_screen_size"])
    keep &= torch.isfinite(model.means).all(dim=1)
    if not keep.any():
        keep[torch.argmax(model.opacity_logits)] = True
    kept_indices = torch.where(keep)[0]
    clone_indices = torch.where(clone_mask & keep)[0]
    budget = int(config["gaussian_budget"]["max_count"])
    clone_indices = clone_indices[: max(0, budget - len(kept_indices))]
    before = model.count
    model.replace_rows(kept_indices, clone_indices)
    return {
        "densified": int(len(clone_indices)),
        "pruned": int(before - len(kept_indices)),
        "gaussian_count": model.count,
    }


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
    values = absolute.norm(dim=-1)
    gradient_sum.index_add_(0, ids.long(), values)
    gradient_count.index_add_(0, ids.long(), torch.ones_like(values))
    radius_values = radii.to(torch.float32).amax(dim=-1)
    max_radius.scatter_reduce_(0, ids.long(), radius_values, reduce="amax", include_self=True)


def _render_visible_training_view(
    model: GaussianModel,
    views: list[TrainingView],
    first_index: int,
    sh_degree: int,
):
    for offset in range(len(views)):
        view = views[(first_index + offset) % len(views)]
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
