"""Deterministic canonical and browser export for project-owned Gaussians."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .dataset import sha256_file, validate_contract
from .evaluation import load_model_snapshot

EXPORT_SCHEMA_VERSION = 1
PLY_FIELDS = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


class GaussianExportError(RuntimeError):
    """Raised when a Gaussian export is unsafe or invalid."""


def export_gaussians(
    *,
    model_path: Path,
    contract: dict[str, Any],
    config_record: dict[str, Any],
    evaluation_path: Path,
    output_dir: Path,
    checkpoint_hash: str | None = None,
    postprocess_record_path: Path | None = None,
    postprocess_mask_path: Path | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    if output_dir.exists():
        raise GaussianExportError(f"export output already exists: {output_dir}")
    config_hash = str(config_record.get("effective_config_hash", ""))
    if not _is_sha256(config_hash):
        raise GaussianExportError("effective config hash is missing or invalid")
    evaluation = _read_json(evaluation_path)
    provenance = evaluation.get("provenance", {})
    if provenance.get("dataset_hash") != contract["dataset_hash"]:
        raise GaussianExportError("evaluation dataset hash mismatch")
    if provenance.get("effective_config_hash") != config_hash:
        raise GaussianExportError("evaluation config hash mismatch")
    model_hash = sha256_file(model_path)
    if provenance.get("model_sha256") != model_hash:
        raise GaussianExportError("evaluation model hash mismatch")
    postprocess = None
    if postprocess_record_path is not None or postprocess_mask_path is not None:
        if postprocess_record_path is None or postprocess_mask_path is None:
            raise GaussianExportError(
                "postprocess record and mask must be provided together"
            )
        if not postprocess_record_path.is_file() or not postprocess_mask_path.is_file():
            raise GaussianExportError("postprocess provenance assets are missing")
        postprocess = _read_json(postprocess_record_path)
        if postprocess.get("filtered_model_sha256") != model_hash:
            raise GaussianExportError("postprocess filtered model hash mismatch")
        if postprocess.get("mask_sha256") != sha256_file(postprocess_mask_path):
            raise GaussianExportError("postprocess mask hash mismatch")

    import torch

    model = load_model_snapshot(model_path, torch.device("cpu"))
    opacity = model.opacity_logits.detach().sigmoid()
    effective_opacity_count = int((opacity > 0.01).sum())
    if effective_opacity_count == 0:
        raise GaussianExportError("Gaussian model has no browser-visible opacity rows")
    rows = _model_rows(model)
    output_dir.mkdir(parents=True, exist_ok=False)
    canonical_path = output_dir / "canonical.ply"
    browser_path = output_dir / "scene.ply"
    _write_binary_ply(canonical_path, rows)
    _write_binary_ply(browser_path, rows)
    camera_path = _camera_path(contract)
    (output_dir / "camera_path.json").write_text(
        json.dumps(camera_path, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    scene_center, scene_radius = _scene_frame(model)
    health = evaluation.get("health")
    if not isinstance(health, dict):
        health = None
    metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": "project_gaussian_ply_v1",
        "browser_derivative_format": "inria_v1_binary_little_endian",
        "coordinate_frame": "normalized",
        "world_units": "arbitrary",
        "camera_axes": contract["coordinate_system"]["camera_axes"],
        "world_from_normalized": contract["normalization"]["world_from_normalized"],
        "gaussian_count": model.count,
        "scene_center": scene_center,
        "scene_radius_p95": scene_radius,
        "effective_opacity_count": effective_opacity_count,
        "viewer_minimum_opacity": 0.005,
        "opacity_p50": float(torch.quantile(opacity, 0.5)),
        "opacity_p90": float(torch.quantile(opacity, 0.9)),
        "health": health,
        "sh_degree": model.max_sh_degree,
        "sh_layout": "dc_rgb_then_rest_channel_major",
        "opacity": "logit",
        "scale": "natural_log",
        "quaternion": "wxyz",
        "dataset_hash": contract["dataset_hash"],
        "effective_config_hash": config_hash,
        "model_sha256": model_hash,
        "checkpoint_hash": checkpoint_hash,
        "canonical_sha256": sha256_file(canonical_path),
        "browser_sha256": sha256_file(browser_path),
        "camera_path_sha256": sha256_file(output_dir / "camera_path.json"),
        "evaluation_sha256": sha256_file(evaluation_path),
        "media": {
            "status": "not_generated",
            "reason": "camera_path_descriptor_is_the_stage2_baseline",
        },
    }
    if postprocess is not None:
        metadata["postprocess"] = {
            "profile": postprocess.get("profile"),
            "source_model_sha256": postprocess.get("source_model_sha256"),
            "filtered_model_sha256": postprocess.get("filtered_model_sha256"),
            "mask_sha256": postprocess.get("mask_sha256"),
            "counts": postprocess.get("counts"),
        }
    metadata_path = output_dir / "export.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    bundle_entries = {
        "gaussian/canonical.ply": canonical_path,
        "gaussian/scene.ply": browser_path,
        "gaussian/export.json": metadata_path,
        "gaussian/camera_path.json": output_dir / "camera_path.json",
        "contracts/dataset.json": _write_json_asset(output_dir / ".dataset.json", contract),
        "contracts/effective_config.json": _write_json_asset(
            output_dir / ".effective_config.json", config_record
        ),
        "evaluation/evaluation.json": evaluation_path,
    }
    if postprocess_record_path is not None and postprocess_mask_path is not None:
        bundle_entries.update(
            {
                "postprocess/diagnostics.json": postprocess_record_path,
                "postprocess/filter-mask.npz": postprocess_mask_path,
            }
        )
    bundle_path = output_dir / "result.zip"
    write_deterministic_zip(bundle_path, bundle_entries)
    bundle_record = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "note": "bundle hash is external to avoid a self-referential archive hash",
    }
    (output_dir / "bundle.json").write_text(
        json.dumps(bundle_record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {**metadata, **bundle_record}


def read_gaussian_ply(path: Path) -> dict[str, np.ndarray]:
    content = path.read_bytes()
    marker = b"end_header\n"
    end = content.find(marker)
    if end < 0:
        raise GaussianExportError("Gaussian PLY header is incomplete")
    header = content[: end + len(marker)].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise GaussianExportError("Gaussian PLY must be binary little-endian")
    match = next(
        (line for line in header.splitlines() if line.startswith("element vertex ")),
        None,
    )
    if match is None:
        raise GaussianExportError("Gaussian PLY has no vertex count")
    count = int(match.split()[-1])
    properties = [line.split()[-1] for line in header.splitlines() if line.startswith("property float ")]
    if tuple(properties) != PLY_FIELDS:
        raise GaussianExportError("Gaussian PLY property layout mismatch")
    values = np.frombuffer(content, dtype="<f4", offset=end + len(marker))
    if values.size != count * len(PLY_FIELDS):
        raise GaussianExportError("Gaussian PLY payload size mismatch")
    matrix = values.reshape(count, len(PLY_FIELDS)).copy()
    if not np.isfinite(matrix).all():
        raise GaussianExportError("Gaussian PLY contains non-finite values")
    return {name: matrix[:, index] for index, name in enumerate(PLY_FIELDS)}


def write_deterministic_zip(
    path: Path,
    entries: dict[str, Path],
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise GaussianExportError(f"bundle already exists: {path}")
    for name, source in entries.items():
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or not name:
            raise GaussianExportError(f"unsafe bundle path: {name}")
        if not source.is_file():
            raise GaussianExportError(f"bundle source is missing: {source}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, entries[name].read_bytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scene_frame(model) -> tuple[list[float], float]:
    import torch

    center = model.means.detach().median(dim=0).values
    radius = torch.quantile(torch.linalg.vector_norm(model.means.detach() - center, dim=1), 0.95)
    return center.tolist(), max(float(radius), 1e-6)


def _model_rows(model) -> np.ndarray:
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    count = model.count
    sh = state["sh_coeffs"]
    if sh.shape != (count, 16, 3):
        raise GaussianExportError("canonical export requires degree-3 SH tensors")
    quats = state["quats"] / np.maximum(np.linalg.norm(state["quats"], axis=1, keepdims=True), 1e-12)
    rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(count, 45)
    rows = np.concatenate(
        (
            state["means"],
            np.zeros((count, 3), dtype=np.float32),
            sh[:, 0, :],
            rest,
            state["opacity_logits"][:, None],
            state["log_scales"],
            quats,
        ),
        axis=1,
    ).astype("<f4", copy=False)
    if rows.shape != (count, len(PLY_FIELDS)) or not np.isfinite(rows).all():
        raise GaussianExportError("canonical Gaussian rows are invalid")
    return rows


def _write_binary_ply(path: Path, rows: np.ndarray) -> None:
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Image3D-SceneGraph canonical Gaussian schema v1",
        f"element vertex {len(rows)}",
    ]
    header.extend(f"property float {name}" for name in PLY_FIELDS)
    header.extend(("end_header", ""))
    with path.open("xb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(rows.astype("<f4", copy=False).tobytes(order="C"))


def _camera_path(contract: dict[str, Any]) -> dict[str, Any]:
    validation_ids = set(str(value) for value in contract["splits"]["validation"])
    entries = [entry for entry in contract["images"] if str(entry["image_id"]) in validation_ids]
    if len(entries) < 2:
        raise GaussianExportError("camera path requires at least two validation cameras")
    normalized_from_world = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    radius_limit = 2.0
    keyframes = []
    for entry in entries:
        world_from_camera = np.asarray(entry["world_from_camera"], dtype=np.float64)
        center = normalized_from_world @ np.append(world_from_camera[:3, 3], 1.0)
        if not np.isfinite(center).all() or np.linalg.norm(center[:3]) > radius_limit:
            raise GaussianExportError("camera path exits the trusted normalized scene bound")
        keyframes.append(
            {
                "image_id": str(entry["image_id"]),
                "center_normalized": center[:3].tolist(),
                "world_from_camera": entry["world_from_camera"],
            }
        )
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "coordinate_frame": "normalized",
        "world_units": "arbitrary",
        "trusted_radius": radius_limit,
        "interpolation": "linear_between_validation_keyframes",
        "keyframes": keyframes,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianExportError(f"cannot read JSON asset {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GaussianExportError(f"JSON asset must contain an object: {path}")
    return value


def _write_json_asset(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
