"""Project-owned Gaussian validation and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image

from .config import ResolvedGaussianConfig, resolved_config_record
from .dataset import sha256_file, validate_contract
from .model import GaussianModel
from .render import render_gaussians
from .runtime import TrainingView, load_evaluation_views
from .training_math import psnr, structural_similarity

EVALUATION_SCHEMA_VERSION = 1
LPIPS_NOT_RUN = {
    "status": "not_run",
    "reason": "pretrained_weight_license_and_hash_not_audited",
}


class GaussianEvaluationError(RuntimeError):
    """Raised when Gaussian evaluation cannot produce an auditable result."""


def load_model_snapshot(path: Path, device: torch.device) -> GaussianModel:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
        state = payload["state_dict"]
        model = GaussianModel(
            means=state["means"],
            log_scales=state["log_scales"],
            quats=state["quats"],
            opacity_logits=state["opacity_logits"],
            sh_coeffs=state["sh_coeffs"],
            max_sh_degree=int(payload["max_sh_degree"]),
        ).to(device)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise GaussianEvaluationError(f"invalid Gaussian model snapshot: {exc}") from exc
    model.validate()
    return model


def evaluate_model(
    model: GaussianModel,
    views: list[TrainingView],
    *,
    split: str,
    sh_degree: int,
    preview_dir: Path | None = None,
    progress_events: Iterable[dict[str, Any]] = (),
    renderer: Callable[..., Any] = render_gaussians,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise GaussianEvaluationError("evaluation split must be validation or test")
    if not views:
        raise GaussianEvaluationError(f"evaluation split is empty: {split}")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=False)

    per_view: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    peak_allocated = 0
    peak_reserved = 0
    if model.means.is_cuda:
        torch.cuda.reset_peak_memory_stats(model.means.device)

    with torch.no_grad():
        for view in views:
            try:
                if model.means.is_cuda:
                    torch.cuda.synchronize(model.means.device)
                started = time.perf_counter()
                rendered = renderer(
                    model,
                    view.camera,
                    sh_degree=sh_degree,
                    background=torch.ones(3, device=model.means.device),
                )
                if model.means.is_cuda:
                    torch.cuda.synchronize(model.means.device)
                render_ms = (time.perf_counter() - started) * 1000.0
                entry = {
                    "image_id": view.camera.image_id,
                    "psnr": psnr(rendered.image, view.image),
                    "ssim": float(structural_similarity(rendered.image, view.image)),
                    "render_milliseconds": render_ms,
                }
                if not all(np.isfinite(float(entry[key])) for key in ("psnr", "ssim", "render_milliseconds")):
                    raise GaussianEvaluationError("non-finite view metric")
                per_view.append(entry)
                if preview_dir is not None:
                    pixels = rendered.image.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
                    Image.fromarray(pixels).save(
                        preview_dir / f"{_safe_name(view.camera.image_id)}.png"
                    )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise GaussianEvaluationError("CUDA out of memory during Gaussian evaluation") from exc
            except GaussianEvaluationError:
                raise
            except (RuntimeError, ValueError) as exc:
                failures.append({"image_id": view.camera.image_id, "reason": str(exc)})

    if not per_view:
        raise GaussianEvaluationError(f"every {split} view failed")
    if model.means.is_cuda:
        peak_allocated = int(torch.cuda.max_memory_allocated(model.means.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(model.means.device))
    render_values = [float(item["render_milliseconds"]) for item in per_view]
    total_seconds = sum(render_values) / 1000.0
    opacity = model.opacity_logits.detach().sigmoid().cpu().numpy()
    scales = model.log_scales.detach().exp().reshape(-1).cpu().numpy()
    topology = _topology_summary(progress_events)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete" if not failures else "complete_with_failures",
        "split": split,
        "num_views": len(views),
        "successful_views": len(per_view),
        "failed_views": failures,
        "psnr": _distribution([float(item["psnr"]) for item in per_view]),
        "ssim": _distribution([float(item["ssim"]) for item in per_view]),
        "lpips": dict(LPIPS_NOT_RUN),
        "render_milliseconds": _distribution(render_values),
        "render_fps": len(per_view) / total_seconds if total_seconds > 0 else 0.0,
        "gaussian_count": model.count,
        "opacity": _distribution(opacity.tolist()),
        "scale": _distribution(scales.tolist()),
        "topology": topology,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "per_view": per_view,
        "geometry_evaluation": {
            "status": "not_run",
            "reason": "rendering_evaluation_is_reported_separately",
        },
    }


def run_evaluation(
    *,
    contract: dict[str, Any],
    dataset_root: Path,
    model_path: Path,
    resolved_config: ResolvedGaussianConfig,
    split: str,
    output_dir: Path,
    frozen_candidate_path: Path | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    validate_contract(contract, dataset_root)
    config_record = resolved_config_record(resolved_config)
    if split not in {"validation", "test"}:
        raise GaussianEvaluationError("evaluation split must be validation or test")
    if output_dir.exists():
        raise GaussianEvaluationError(f"evaluation output already exists: {output_dir}")
    model_hash = sha256_file(model_path)
    consumption_path: Path | None = None
    if split == "test":
        if frozen_candidate_path is None:
            raise GaussianEvaluationError("test evaluation requires a frozen candidate record")
        consumption_path = _authorize_test(
            frozen_candidate_path,
            dataset_hash=str(contract["dataset_hash"]),
            config_hash=resolved_config.effective_config_hash,
            model_hash=model_hash,
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_snapshot(model_path, device)
        views = load_evaluation_views(
            contract,
            dataset_root,
            split=split,
            longest_edge=int(resolved_config.effective_config["resolution"]["longest_edge"]),
            device=device,
        )
        progress = _read_progress(progress_path)
        result = evaluate_model(
            model,
            views,
            split=split,
            sh_degree=int(resolved_config.effective_config["sh_schedule"]["max_degree"]),
            preview_dir=output_dir / "previews",
            progress_events=progress,
        )
        result["provenance"] = {
            "dataset_hash": contract["dataset_hash"],
            "effective_config_hash": resolved_config.effective_config_hash,
            "model_sha256": model_hash,
            "model_bytes": model_path.stat().st_size,
        }
        (output_dir / "evaluation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as handle:
            for entry in result["per_view"]:
                handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        record = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "split": split,
            "evaluation_sha256": sha256_file(output_dir / "evaluation.json"),
            "metrics_sha256": sha256_file(output_dir / "metrics.jsonl"),
            "config": config_record,
        }
        (output_dir / "record.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if consumption_path is not None:
            _finish_test_consumption(consumption_path, "complete", record["evaluation_sha256"])
        return result
    except Exception:
        if consumption_path is not None:
            _finish_test_consumption(consumption_path, "failed", None)
        raise


def write_frozen_candidate(
    path: Path,
    *,
    candidate_id: str,
    dataset_hash: str,
    effective_config_hash: str,
    model_sha256: str,
) -> None:
    for name, value in {
        "dataset_hash": dataset_hash,
        "effective_config_hash": effective_config_hash,
        "model_sha256": model_sha256,
    }.items():
        if not _is_sha256(value):
            raise GaussianEvaluationError(f"invalid frozen candidate {name}")
    if not candidate_id or Path(candidate_id).name != candidate_id:
        raise GaussianEvaluationError("candidate ID must be a non-empty filename-safe value")
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "frozen",
        "candidate_id": candidate_id,
        "dataset_hash": dataset_hash,
        "effective_config_hash": effective_config_hash,
        "model_sha256": model_sha256,
        "selection_split": "validation",
        "test_metrics_seen": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _authorize_test(
    path: Path,
    *,
    dataset_hash: str,
    config_hash: str,
    model_hash: str,
) -> Path:
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianEvaluationError(f"cannot read frozen candidate: {exc}") from exc
    expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "frozen",
        "dataset_hash": dataset_hash,
        "effective_config_hash": config_hash,
        "model_sha256": model_hash,
        "selection_split": "validation",
        "test_metrics_seen": False,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise GaussianEvaluationError(f"frozen candidate mismatch: {key}")
    consumption = path.with_name(f"{path.stem}.test-consumed.json")
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "candidate_id": frozen.get("candidate_id"),
        "status": "running",
        "dataset_hash": dataset_hash,
        "effective_config_hash": config_hash,
        "model_sha256": model_hash,
        "evaluation_sha256": None,
    }
    try:
        with consumption.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise GaussianEvaluationError("frozen candidate test evaluation was already consumed") from exc
    return consumption


def _finish_test_consumption(path: Path, status: str, evaluation_hash: str | None) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["evaluation_sha256"] = evaluation_hash
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_progress(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    events = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianEvaluationError(f"cannot read trainer progress: {exc}") from exc
    return events


def _topology_summary(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    densified = 0
    pruned = 0
    opacity_resets = 0
    for event in events:
        densified += int(event.get("densified", 0))
        pruned += int(event.get("pruned", 0))
        opacity_resets += int(event.get("opacity_reset") is True)
    return {
        "densified": densified,
        "pruned": pruned,
        "opacity_resets": opacity_resets,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise GaussianEvaluationError("metric distribution must contain finite values")
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
