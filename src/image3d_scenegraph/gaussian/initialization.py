"""Deterministic sparse and dense initialization for project-owned 3DGS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import sha256_file


@dataclass(frozen=True)
class InitializationResult:
    points: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    diagnostics: dict[str, Any]


class InitializationError(ValueError):
    """Raised when an initialization source cannot be selected safely."""


def sparse_initialization(
    points_path: Path,
    normalized_from_world: np.ndarray,
    *,
    max_points: int,
    min_track_length: int = 3,
    max_reprojection_error: float = 4.0,
) -> InitializationResult:
    points, colors, errors, tracks = read_colmap_points_text(points_path)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(errors)
    supported = tracks >= min_track_length
    accurate = errors <= max_reprojection_error
    accepted = finite & supported & accurate
    indices = np.flatnonzero(accepted)
    if len(indices) > max_points:
        order = np.lexsort((indices, -tracks[indices], errors[indices]))
        indices = np.sort(indices[order[:max_points]])
    if not len(indices):
        raise InitializationError("sparse initialization rejected every point")
    selected_points = transform_points(points[indices], normalized_from_world)
    selected_colors = colors[indices]
    settings = {
        "kind": "colmap_sparse_v1",
        "min_track_length": min_track_length,
        "max_reprojection_error": max_reprojection_error,
        "max_points": max_points,
        "scale_initialization": "graphdeco_3nn_rms_v1",
    }
    diagnostics = _diagnostics(
        source=points_path,
        settings=settings,
        points=selected_points,
        colors=selected_colors,
        counts={
            "input": len(points),
            "rejected_non_finite": int((~finite).sum()),
            "rejected_track_support": int((finite & ~supported).sum()),
            "rejected_reprojection_error": int((finite & supported & ~accurate).sum()),
            "accepted_before_budget": int(accepted.sum()),
            "accepted": len(indices),
            "rejected_budget": int(accepted.sum() - len(indices)),
        },
    )
    return InitializationResult(
        selected_points,
        selected_colors,
        graphdeco_nearest_neighbor_scales(selected_points),
        diagnostics,
    )


def dense_initialization(
    points_path: Path,
    normalized_from_world: np.ndarray,
    *,
    max_points: int,
    voxel_size: float,
    diagnostics_path: Path | None = None,
    min_support: int = 1,
    min_confidence: float = 0.0,
    outlier_quantile: float = 0.005,
) -> InitializationResult:
    points, colors = read_rgb_ply(points_path)
    if colors is None:
        raise InitializationError("dense initialization PLY must contain RGB colors")
    count = len(points)
    support = np.ones(count, dtype=np.int64)
    confidence = np.ones(count, dtype=np.float32)
    sidecar_hash = None
    if diagnostics_path is not None:
        try:
            with np.load(diagnostics_path) as payload:
                support = np.asarray(payload["support_counts"])
                confidence = np.asarray(payload["confidence"])
        except (OSError, ValueError, KeyError) as exc:
            raise InitializationError(f"cannot read dense support diagnostics: {exc}") from exc
        if support.shape != (count,) or confidence.shape != (count,):
            raise InitializationError("dense support diagnostics do not match PLY row count")
        sidecar_hash = sha256_file(diagnostics_path)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    supported = support >= min_support
    confident = confidence >= min_confidence
    candidate = finite & supported & confident
    candidate_indices = np.flatnonzero(candidate)
    if not len(candidate_indices):
        raise InitializationError("dense initialization rejected every point before outlier filtering")
    candidate_points = transform_points(points[candidate_indices], normalized_from_world)
    lower = np.quantile(candidate_points, outlier_quantile, axis=0)
    upper = np.quantile(candidate_points, 1.0 - outlier_quantile, axis=0)
    inlier = np.logical_and(candidate_points >= lower, candidate_points <= upper).all(axis=1)
    inlier_indices = candidate_indices[inlier]
    normalized = candidate_points[inlier]
    if not len(inlier_indices):
        raise InitializationError("dense initialization rejected every point as an outlier")

    cells = np.floor(normalized / voxel_size).astype(np.int64)
    order = np.lexsort(
        (
            inlier_indices,
            -confidence[inlier_indices],
            -support[inlier_indices],
            cells[:, 2],
            cells[:, 1],
            cells[:, 0],
        )
    )
    ordered_cells = cells[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1)
    chosen = order[first]
    voxel_indices = inlier_indices[chosen]
    voxel_points = normalized[chosen]
    if len(voxel_indices) > max_points:
        budget = morton_stratified_indices(voxel_points, max_points)
        voxel_indices = voxel_indices[budget]
        voxel_points = voxel_points[budget]
    sort_order = np.argsort(voxel_indices, kind="stable")
    voxel_indices = voxel_indices[sort_order]
    voxel_points = voxel_points[sort_order]
    selected_colors = colors[voxel_indices]
    settings = {
        "kind": "filtered_dense_v1",
        "min_support": min_support,
        "min_confidence": min_confidence,
        "outlier_quantile": outlier_quantile,
        "voxel_size_normalized": voxel_size,
        "max_points": max_points,
        "diagnostics_sha256": sidecar_hash,
        "scale_initialization": "graphdeco_3nn_rms_v1",
    }
    diagnostics = _diagnostics(
        source=points_path,
        settings=settings,
        points=voxel_points,
        colors=selected_colors,
        counts={
            "input": count,
            "rejected_non_finite": int((~finite).sum()),
            "rejected_support": int((finite & ~supported).sum()),
            "rejected_confidence": int((finite & supported & ~confident).sum()),
            "accepted_before_outlier": len(candidate_indices),
            "rejected_outlier": int((~inlier).sum()),
            "accepted_before_voxel": len(inlier_indices),
            "rejected_voxel": len(inlier_indices) - len(chosen),
            "accepted_before_budget": len(chosen),
            "rejected_budget": len(chosen) - len(voxel_indices),
            "accepted": len(voxel_indices),
        },
    )
    return InitializationResult(
        voxel_points,
        selected_colors,
        graphdeco_nearest_neighbor_scales(voxel_points),
        diagnostics,
    )


def read_colmap_points_text(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = []
    colors = []
    errors = []
    tracks = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InitializationError(f"cannot read COLMAP sparse points: {exc}") from exc
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or (len(parts) - 8) % 2:
            raise InitializationError("invalid COLMAP points3D row")
        points.append([float(value) for value in parts[1:4]])
        colors.append([int(value) for value in parts[4:7]])
        errors.append(float(parts[7]))
        tracks.append((len(parts) - 8) // 2)
    if not points:
        raise InitializationError("COLMAP sparse source contains no points")
    color_array = np.asarray(colors, dtype=np.int64)
    if ((color_array < 0) | (color_array > 255)).any():
        raise InitializationError("COLMAP sparse colors must be uint8 RGB")
    return (
        np.asarray(points, dtype=np.float32),
        color_array.astype(np.uint8),
        np.asarray(errors, dtype=np.float32),
        np.asarray(tracks, dtype=np.int32),
    )


def read_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise InitializationError("Open3D is required to read initialization PLY files") from exc
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if not len(points):
        raise InitializationError("dense initialization PLY contains no points")
    colors = None
    if cloud.has_colors():
        colors = np.clip(np.rint(np.asarray(cloud.colors) * 255.0), 0, 255).astype(np.uint8)
    return points, colors


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise InitializationError("normalization transform must be a finite 4 x 4 matrix")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    transformed = homogeneous @ matrix.T
    if np.any(np.abs(transformed[:, 3]) <= 1e-12):
        raise InitializationError("normalization produced an invalid homogeneous coordinate")
    result = (transformed[:, :3] / transformed[:, 3:]).astype(np.float32)
    if not np.isfinite(result).all():
        raise InitializationError("normalization produced non-finite points")
    return result


def graphdeco_nearest_neighbor_scales(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.full(1, 0.01, dtype=np.float32)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise InitializationError("Open3D is required to estimate Gaussian scales") from exc
    values = o3d.core.Tensor(points.astype(np.float32))
    search = o3d.core.nns.NearestNeighborSearch(values)
    search.knn_index()
    neighbor_count = min(4, len(points))
    indices, squared_distances = search.knn_search(values, neighbor_count)
    neighbor_indices = indices.numpy()
    distances = squared_distances.numpy()
    distances[neighbor_indices == np.arange(len(points))[:, None]] = np.inf
    count = min(3, len(points) - 1)
    distances = np.partition(distances, count - 1, axis=1)[:, :count]
    scales = np.sqrt(np.maximum(distances.mean(axis=1), 1e-7))
    return scales.astype(np.float32)


def morton_stratified_indices(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        raise InitializationError("Morton budget requires 0 < max_points < point count")
    bits = 21
    maximum = (1 << bits) - 1
    lower = points.min(axis=0).astype(np.float64)
    side = float(np.max(points.max(axis=0).astype(np.float64) - lower))
    quantized = np.zeros(points.shape, dtype=np.uint64)
    if side > 0:
        quantized = np.floor((points.astype(np.float64) - lower) * (maximum / side)).astype(
            np.uint64
        )
        np.minimum(quantized, maximum, out=quantized)
    codes = _spread_bits(quantized[:, 0]) | (_spread_bits(quantized[:, 1]) << np.uint64(1)) | (
        _spread_bits(quantized[:, 2]) << np.uint64(2)
    )
    order = np.argsort(codes, kind="stable")
    ranks = (2 * np.arange(max_points, dtype=np.int64) + 1) * len(points) // (2 * max_points)
    return np.sort(order[ranks].astype(np.int64))


def _spread_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True) & np.uint64(0x1FFFFF)
    values = (values | (values << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    values = (values | (values << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    values = (values | (values << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    values = (values | (values << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    return (values | (values << np.uint64(2))) & np.uint64(0x1249249249249249)


def _diagnostics(
    *,
    source: Path,
    settings: dict[str, Any],
    points: np.ndarray,
    colors: np.ndarray,
    counts: dict[str, int],
) -> dict[str, Any]:
    selected_hash = hashlib.sha256(points.tobytes() + colors.tobytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "source_sha256": sha256_file(source),
        "settings": settings,
        "counts": counts,
        "selected_rows_sha256": selected_hash,
    }
    payload["selection_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def load_frozen_initialization(
    asset_path: Path,
    diagnostics_path: Path,
    *,
    expected_sha256: str,
) -> InitializationResult:
    """Load a previously selected initialization without recomputing it."""
    if sha256_file(asset_path) != expected_sha256:
        raise InitializationError("frozen initialization asset hash mismatch")
    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        with np.load(asset_path, allow_pickle=False) as payload:
            if set(payload.files) != {"points", "colors", "scales"}:
                raise InitializationError("frozen initialization must contain points, colors, and scales")
            points = np.asarray(payload["points"])
            colors = np.asarray(payload["colors"])
            scales = np.asarray(payload["scales"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise InitializationError(f"cannot read frozen initialization: {exc}") from exc

    count = len(points)
    if points.dtype != np.float32 or points.shape != (count, 3):
        raise InitializationError("frozen initialization points must be float32 N x 3")
    if colors.dtype != np.uint8 or colors.shape != (count, 3):
        raise InitializationError("frozen initialization colors must be uint8 N x 3")
    if scales.dtype != np.float32 or scales.shape != (count,):
        raise InitializationError("frozen initialization scales must be float32 N")
    if count == 0 or not np.isfinite(points).all():
        raise InitializationError("frozen initialization points must be non-empty and finite")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise InitializationError("frozen initialization scales must be finite and positive")
    if diagnostics.get("asset_sha256") != expected_sha256:
        raise InitializationError("frozen initialization diagnostics asset hash mismatch")
    if diagnostics.get("counts", {}).get("accepted") != count:
        raise InitializationError("frozen initialization accepted count mismatch")
    selected_hash = hashlib.sha256(points.tobytes() + colors.tobytes()).hexdigest()
    if diagnostics.get("selected_rows_sha256") != selected_hash:
        raise InitializationError("frozen initialization selected rows hash mismatch")
    selection_payload = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"asset_sha256", "selection_hash"}
    }
    selection_hash = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if diagnostics.get("selection_hash") != selection_hash:
        raise InitializationError("frozen initialization selection hash mismatch")
    return InitializationResult(points, colors, scales, diagnostics)


def write_initialization(
    output_path: Path,
    diagnostics_path: Path,
    result: InitializationResult,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, points=result.points, colors=result.colors, scales=result.scales)
    diagnostics = dict(result.diagnostics)
    diagnostics["asset_sha256"] = sha256_file(output_path)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
