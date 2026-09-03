"""CPU-only readiness checks for Gaussian training geometry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import camera_from_normalized_transform
from .initialization import InitializationResult
from .render import NEAR_PLANE_NORMALIZED


PROFILE_ID = "gaussian_geometry_readiness_v1"
CAMERA_MAX_TO_MEDIAN_LIMIT = 100.0
CAMERA_MAX_TO_P99_LIMIT = 10.0
INITIAL_SCALE_FLOOR = float(np.sqrt(1e-7))
INITIAL_SCALE_FLOOR_FRACTION_LIMIT = 0.5
PROJECTION_SAMPLE_LIMIT = 4096
PROJECTION_VIEW_LIMIT = 5


class GeometryReadinessError(ValueError):
    """Raised before CUDA when Gaussian inputs are geometrically unusable."""


def build_geometry_readiness(
    contract: dict[str, Any],
    initialization: InitializationResult,
    effective_config: dict[str, Any],
    *,
    trainer_id: str,
) -> dict[str, Any]:
    images = contract["images"]
    centers = np.stack(
        [
            np.asarray(image["world_from_camera"], dtype=np.float64)[:3, 3]
            for image in images
        ]
    )
    robust_center = np.median(centers, axis=0)
    center_distances = np.linalg.norm(centers - robust_center, axis=1)
    center_distribution = _distribution(center_distances)
    maximum_index = int(np.argmax(center_distances))
    maximum = center_distribution["max"]
    median = center_distribution["p50"]
    p99 = center_distribution["p99"]
    max_to_median = _finite_ratio(maximum, median)
    max_to_p99 = _finite_ratio(maximum, p99)
    camera_outlier = (
        _at_least_ratio(maximum, median, CAMERA_MAX_TO_MEDIAN_LIMIT)
        and _at_least_ratio(maximum, p99, CAMERA_MAX_TO_P99_LIMIT)
    )

    points = np.asarray(initialization.points, dtype=np.float32)
    scales = np.asarray(initialization.scales, dtype=np.float32)
    scale_multiplier = float(effective_config["initialization"]["scale_multiplier"])
    effective_scales = scales.astype(np.float64) * scale_multiplier
    floor_count = int(
        (scales <= INITIAL_SCALE_FLOOR * (1.0 + 1e-6)).sum()
    )
    floor_fraction = floor_count / len(scales)
    pruning = effective_config["pruning"]
    project_pruning = (
        trainer_id == "project"
        and str(effective_config["strategy"]["name"]) == "default_v1"
        and bool(pruning["enabled"])
    )
    maximum_scale = float(pruning["max_world_scale"])
    oversized = effective_scales > maximum_scale * (1.0 + 1e-6)
    oversized_count = int(oversized.sum()) if project_pruning else 0

    reasons = []
    if camera_outlier:
        reasons.append("unusable_camera_pose_outlier")
    if floor_fraction >= INITIAL_SCALE_FLOOR_FRACTION_LIMIT:
        reasons.append("initialization_scale_floor_collapse")
    if project_pruning and oversized_count == len(scales):
        reasons.append("initialization_all_oversized")

    point_center = np.median(points.astype(np.float64), axis=0)
    point_distances = np.linalg.norm(points - point_center, axis=1)
    record: dict[str, Any] = {
        "schema_version": 1,
        "profile": PROFILE_ID,
        "status": "failed" if reasons else "passed",
        "reason_codes": reasons,
        "policy": {
            "camera_center_reference": "componentwise_median",
            "camera_p99_method": "lower_order_statistic",
            "camera_max_to_median_limit": CAMERA_MAX_TO_MEDIAN_LIMIT,
            "camera_max_to_p99_limit": CAMERA_MAX_TO_P99_LIMIT,
            "initial_scale_floor": INITIAL_SCALE_FLOOR,
            "initial_scale_floor_fraction_limit": INITIAL_SCALE_FLOOR_FRACTION_LIMIT,
            "near_plane_normalized": NEAR_PLANE_NORMALIZED,
            "test_rgb_loaded": False,
        },
        "trainer": {
            "id": trainer_id,
            "strategy": str(effective_config["strategy"]["name"]),
            "scale_multiplier": scale_multiplier,
        },
        "camera_centers": {
            "count": len(images),
            "robust_center_world": robust_center.tolist(),
            "distance_world": center_distribution,
            "max_to_median": max_to_median,
            "max_to_p99": max_to_p99,
            "farthest_image_id": str(images[maximum_index]["image_id"]),
            "farthest_image_path": str(images[maximum_index]["path"]),
        },
        "normalization": {
            "method": contract["normalization"].get("method"),
            "radius_world": float(contract["normalization"]["radius_world"]),
        },
        "initialization": {
            "count": len(points),
            "point_distance_from_median_normalized": _distribution(point_distances),
            "point_axis_p01_normalized": np.quantile(points, 0.01, axis=0).tolist(),
            "point_axis_p99_normalized": np.quantile(points, 0.99, axis=0).tolist(),
            "scale_normalized": _distribution(scales),
            "effective_scale_normalized": _distribution(effective_scales),
            "scale_floor_count": floor_count,
            "scale_floor_fraction": floor_fraction,
            "pre_render_world_scale_pruning": {
                "applied": project_pruning,
                "maximum_effective_scale": (
                    float(pruning["max_world_scale"]) if project_pruning else None
                ),
                "before": len(scales),
                "removed": oversized_count,
                "after": len(scales) - oversized_count,
            },
        },
        "projection_risk": {"status": "not_run_geometry_passed"},
    }
    if reasons:
        record["projection_risk"] = _projection_risk(
            contract,
            points,
            effective_scales,
        )
    return record


def project_initialization_keep_mask(
    initialization: InitializationResult,
    effective_config: dict[str, Any],
) -> np.ndarray:
    count = len(initialization.points)
    pruning = effective_config["pruning"]
    if (
        str(effective_config["strategy"]["name"]) != "default_v1"
        or not bool(pruning["enabled"])
    ):
        return np.ones(count, dtype=bool)
    effective_scales = np.asarray(initialization.scales, dtype=np.float64) * float(
        effective_config["initialization"]["scale_multiplier"]
    )
    maximum_scale = float(pruning["max_world_scale"])
    return effective_scales <= maximum_scale * (1.0 + 1e-6)


def require_geometry_readiness(record: dict[str, Any]) -> None:
    reasons = record.get("reason_codes")
    if record.get("status") != "passed":
        joined = ",".join(str(reason) for reason in reasons or ["unknown"])
        raise GeometryReadinessError(f"Gaussian geometry readiness failed: {joined}")


def write_geometry_readiness(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _projection_risk(
    contract: dict[str, Any],
    points: np.ndarray,
    effective_scales: np.ndarray,
) -> dict[str, Any]:
    if len(points) > PROJECTION_SAMPLE_LIMIT:
        sample_indices = np.linspace(
            0, len(points) - 1, PROJECTION_SAMPLE_LIMIT, dtype=np.int64
        )
        sample_points = points[sample_indices]
        sample_scales = effective_scales[sample_indices]
    else:
        sample_points = points
        sample_scales = effective_scales
    selected_ids = {
        str(image_id)
        for split in ("train", "validation")
        for image_id in contract["splits"][split]
    }
    radius_world = float(contract["normalization"]["radius_world"])
    legacy_near = NEAR_PLANE_NORMALIZED / radius_world
    views = []
    for image in contract["images"]:
        if str(image["image_id"]) not in selected_ids:
            continue
        camera = camera_from_normalized_transform(
            image["camera_from_world"], contract["normalization"]
        )
        camera_points = sample_points @ camera[:3, :3].T + camera[:3, 3]
        depth = camera_points[:, 2]
        positive = depth > 0
        legacy_visible = depth >= legacy_near
        near_rejected = legacy_visible & (depth < NEAR_PLANE_NORMALIZED)
        projected = np.empty(0, dtype=np.float64)
        if legacy_visible.any():
            focal = max(
                float(image["intrinsic"][0][0]),
                float(image["intrinsic"][1][1]),
            )
            projected = (
                focal * sample_scales[legacy_visible] / depth[legacy_visible]
            )
        views.append(
            {
                "image_id": str(image["image_id"]),
                "path": str(image["path"]),
                "positive_depth_count": int(positive.sum()),
                "legacy_near_visible_count": int(legacy_visible.sum()),
                "normalized_near_rejected_count": int(near_rejected.sum()),
                "projected_radius_pixels": (
                    _distribution(projected) if len(projected) else None
                ),
            }
        )
    views.sort(
        key=lambda view: (
            -(
                view["projected_radius_pixels"]["p99"]
                if view["projected_radius_pixels"] is not None
                else -1.0
            ),
            view["image_id"],
        )
    )
    return {
        "status": "diagnostic_only",
        "sampled_gaussian_count": len(sample_points),
        "evaluated_train_validation_view_count": len(views),
        "test_view_count": len(contract["splits"]["test"]),
        "test_rgb_loaded": False,
        "normalized_near_plane": NEAR_PLANE_NORMALIZED,
        "legacy_effective_near_plane_normalized": legacy_near,
        "highest_risk_views": views[:PROJECTION_VIEW_LIMIT],
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise GeometryReadinessError("readiness distributions require finite values")
    return {
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99, method="lower")),
        "max": float(array.max()),
    }


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def _at_least_ratio(numerator: float, denominator: float, limit: float) -> bool:
    return numerator > 1e-12 and (
        denominator <= 1e-12 or numerator / denominator >= limit
    )
