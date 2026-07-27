from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pointcloud import find_dominant_planes, fit_plane_svd, plane_basis, read_ply_points, sample_points, vector_to_json


DEFAULT_METRICS = {
    "sample_size": 50_000,
    "ransac_iterations": 400,
    "plane_distance": 0.03,
    "min_points": 1_000,
    "grid_size": 20,
    "local_min_points": 20,
    "layer_bin_width": 0.02,
    "layer_min_peak_ratio": 0.1,
    "layer_min_separation": 0.05,
    "seed": 42,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure fixed-ROI point-cloud geometry quality.")
    parser.add_argument("--input", required=True, type=Path, help="Input PLY point cloud.")
    parser.add_argument("--rois", required=True, type=Path, help="Fixed ROI JSON definition.")
    parser.add_argument("--selection-transform", type=Path, help="JSON file with a 4x4 transform used only for ROI selection.")
    parser.add_argument("--output", type=Path, help="Output JSON. Prints JSON when omitted.")
    args = parser.parse_args()

    diagnostics = evaluate_fixed_roi_quality(args.input, args.rois, args.selection_transform)
    text = json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def evaluate_fixed_roi_quality(input_path: Path, roi_path: Path, selection_transform_path: Path | None = None) -> dict[str, Any]:
    definition = read_definition(roi_path)
    metrics = {**DEFAULT_METRICS, **definition.get("metrics", {})}
    validate_metrics(metrics)
    points = read_ply_points(input_path)
    finite_points = points[np.isfinite(points).all(axis=1)]
    selection_points = transform_points(finite_points, selection_transform_path)

    return {
        "schema_version": 1,
        "input": str(input_path),
        "roi_definition": str(roi_path),
        "selection_transform": str(selection_transform_path) if selection_transform_path else None,
        "input_points": int(len(points)),
        "finite_input_points": int(len(finite_points)),
        "metrics": metrics,
        "rois": [evaluate_roi(finite_points, selection_points, roi, metrics) for roi in definition["rois"]],
    }


def transform_points(points: np.ndarray, path: Path | None) -> np.ndarray:
    if path is None:
        return points
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read selection transform {path}: {exc}") from exc
    matrix = np.asarray(value.get("transform") if isinstance(value, dict) else value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("selection transform must be a finite 4x4 affine matrix")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def read_definition(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ROI definition {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("ROI definition must be a schema_version 1 object")
    rois = value.get("rois")
    if not isinstance(rois, list) or not rois:
        raise ValueError("ROI definition must contain a non-empty rois list")
    names: set[str] = set()
    for roi in rois:
        if not isinstance(roi, dict):
            raise ValueError("each ROI must be an object")
        name = roi.get("name")
        lower = roi.get("min")
        upper = roi.get("max")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("ROI names must be unique non-empty strings")
        names.add(name)
        if not valid_corner(lower) or not valid_corner(upper) or any(a >= b for a, b in zip(lower, upper)):
            raise ValueError(f"ROI {name!r} must have finite min < max bounds")
    return value


def valid_corner(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)


def validate_metrics(metrics: dict[str, Any]) -> None:
    for name in ("sample_size", "ransac_iterations", "min_points", "grid_size", "local_min_points", "seed"):
        if not isinstance(metrics.get(name), int) or metrics[name] <= 0:
            raise ValueError(f"metric {name} must be a positive integer")
    for name in ("plane_distance", "layer_bin_width", "layer_min_peak_ratio", "layer_min_separation"):
        if not isinstance(metrics.get(name), (int, float)) or not math.isfinite(metrics[name]) or metrics[name] <= 0:
            raise ValueError(f"metric {name} must be a positive finite number")


def evaluate_roi(points: np.ndarray, selection_points: np.ndarray, roi: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    lower = np.asarray(roi["min"], dtype=np.float64)
    upper = np.asarray(roi["max"], dtype=np.float64)
    selected = points[np.all((selection_points >= lower) & (selection_points <= upper), axis=1)]
    result: dict[str, Any] = {"name": roi["name"], "status": "ok", "selected_points": int(len(selected))}
    if len(selected) == 0:
        result["status"] = "empty_roi"
        return result
    if len(selected) < metrics["min_points"]:
        result["status"] = "insufficient_points"
        return result

    sampled = sample_points(selected, metrics["sample_size"], metrics["seed"])
    planes = find_dominant_planes(
        sampled,
        iterations=metrics["ransac_iterations"],
        max_planes=1,
        distance_threshold=float(metrics["plane_distance"]),
        seed=metrics["seed"],
    )
    if not planes:
        result["status"] = "no_dominant_plane"
        return result

    plane = planes[0]
    normal = np.asarray(plane["normal"], dtype=np.float64)
    offset = float(plane["offset"])
    distances = selected @ normal + offset
    inlier_mask = np.abs(distances) <= metrics["plane_distance"]
    inliers = selected[inlier_mask]
    if len(inliers) < 3:
        result["status"] = "no_dominant_plane"
        return result

    normal, offset = fit_plane_svd(inliers)
    distances = selected @ normal + offset
    inlier_mask = np.abs(distances) <= metrics["plane_distance"]
    inliers = selected[inlier_mask]
    inlier_distances = distances[inlier_mask]
    basis_u, basis_v = plane_basis(normal)
    projected = np.column_stack(((inliers - inliers.mean(axis=0)) @ basis_u, (inliers - inliers.mean(axis=0)) @ basis_v))

    result.update(
        robust_plane={
            "normal": vector_to_json(normal),
            "offset": float(offset),
            "inlier_count": int(len(inliers)),
            "inlier_ratio": float(len(inliers) / len(selected)),
            "rms_distance": float(math.sqrt(np.mean(inlier_distances**2))),
        },
        normal_dispersion=normal_dispersion(inliers, normal, projected, metrics),
        thickness=thickness(distances),
        parallel_layer_count=count_layers(distances, metrics),
        coverage=coverage(projected, metrics["grid_size"]),
    )
    return result


def normal_dispersion(inliers: np.ndarray, normal: np.ndarray, projected: np.ndarray, metrics: dict[str, Any]) -> dict[str, Any]:
    indices = grid_indices(projected, metrics["grid_size"])
    angles: list[float] = []
    for cell in np.unique(indices, axis=0):
        local = inliers[np.all(indices == cell, axis=1)]
        if len(local) < metrics["local_min_points"]:
            continue
        local_normal, _ = fit_plane_svd(local)
        cosine = min(1.0, abs(float(np.dot(local_normal, normal))))
        angles.append(math.degrees(math.acos(cosine)))
    if not angles:
        return {"status": "insufficient_local_support", "local_plane_count": 0, "p50_degrees": None, "p90_degrees": None}
    return {
        "status": "ok",
        "local_plane_count": len(angles),
        "p50_degrees": float(np.percentile(angles, 50)),
        "p90_degrees": float(np.percentile(angles, 90)),
    }


def thickness(distances: np.ndarray) -> dict[str, float]:
    lower, upper = np.percentile(distances, [5, 95])
    return {"p05": float(lower), "p95": float(upper), "p05_p95": float(upper - lower)}


def count_layers(distances: np.ndarray, metrics: dict[str, Any]) -> int:
    lower, upper = np.percentile(distances, [0.5, 99.5])
    if upper - lower < metrics["layer_bin_width"]:
        return 1
    edges = np.arange(lower, upper + metrics["layer_bin_width"], metrics["layer_bin_width"])
    counts, edges = np.histogram(distances, bins=edges)
    if len(counts) == 0 or counts.max() == 0:
        return 0
    candidates = [
        index
        for index, count in enumerate(counts)
        if count >= counts.max() * metrics["layer_min_peak_ratio"]
        and (index == 0 or count >= counts[index - 1])
        and (index == len(counts) - 1 or count >= counts[index + 1])
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: (-counts[item], item)):
        center = (edges[index] + edges[index + 1]) / 2
        if all(abs(center - (edges[other] + edges[other + 1]) / 2) >= metrics["layer_min_separation"] for other in selected):
            selected.append(index)
    return len(selected)


def coverage(projected: np.ndarray, grid_size: int) -> float:
    indices = grid_indices(projected, grid_size)
    return float(len(np.unique(indices, axis=0)) / (grid_size * grid_size))


def grid_indices(projected: np.ndarray, grid_size: int) -> np.ndarray:
    lower = np.percentile(projected, 5, axis=0)
    upper = np.percentile(projected, 95, axis=0)
    span = np.maximum(upper - lower, 1e-12)
    normalized = np.clip((projected - lower) / span, 0, 1 - np.finfo(float).eps)
    return np.floor(normalized * grid_size).astype(np.int32)


if __name__ == "__main__":
    main()
