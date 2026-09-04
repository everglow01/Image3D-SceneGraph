"""Scale-invariant health checks for raw COLMAP sparse camera poses."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from image3d_scenegraph.geometry.grouping import (
    ColmapImage,
    colmap_camera_center,
    parse_colmap_images_with_points,
    parse_colmap_points3d,
    qvec_to_rotmat,
)
from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT,
    MIN_VIDEO_REGISTRATION_RATE,
    MIN_VIDEO_TEMPORAL_COVERAGE,
    analyze_registration_timeline,
)


PROFILE_ID = "sfm_pose_health_v1"
CENTER_MAX_TO_MEDIAN_LIMIT = 100.0
CENTER_MAX_TO_P99_LIMIT = 10.0
CENTER_P99_TO_MEDIAN_LIMIT = 100.0
TEMPORAL_BOUNDARY_SPEED_TO_P90_LIMIT = 100.0
MAX_AUTOMATIC_REPAIR_FRACTION = 0.10
BRIDGE_PAIR_LIMIT = 32
OUTLIER_LIMIT = 64
MAX_IMAGE_ID = 2_147_483_647


class SfmPoseHealthError(ValueError):
    """Raised when raw sparse pose evidence is invalid or unhealthy."""


def build_sfm_pose_health(
    *,
    images: list[ColmapImage],
    points3d: Mapping[int, np.ndarray],
    selected_timestamps: Mapping[str, float] | None = None,
    database_path: Path | None = None,
) -> dict[str, Any]:
    if len(images) < 2:
        raise SfmPoseHealthError("SfM pose health requires at least two registered cameras")
    image_by_id = {image.image_id: image for image in images}
    if len(image_by_id) != len(images):
        raise SfmPoseHealthError("SfM pose health received duplicate image IDs")

    centers = np.stack([colmap_camera_center(image) for image in images])
    if not np.isfinite(centers).all():
        raise SfmPoseHealthError("SfM camera centers must be finite")
    robust_center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - robust_center, axis=1)
    distance_distribution = _distribution(distances)
    median = float(distance_distribution["p50"])
    p99 = float(distance_distribution["p99"])
    maximum = float(distance_distribution["max"])
    max_to_median = _finite_ratio(maximum, median)
    max_to_p99 = _finite_ratio(maximum, p99)
    p99_to_median = _finite_ratio(p99, median)
    isolated = _at_least_ratio(
        maximum, median, CENTER_MAX_TO_MEDIAN_LIMIT
    ) and _at_least_ratio(maximum, p99, CENTER_MAX_TO_P99_LIMIT)
    branch = _at_least_ratio(p99, median, CENTER_P99_TO_MEDIAN_LIMIT)
    reasons: list[str] = []
    if isolated:
        reasons.append("isolated_camera_pose_outlier")
    if branch:
        reasons.append("multiscale_camera_pose_branch")
    if maximum <= 1e-12:
        reasons.append("degenerate_camera_extent")

    outlier_indices = [
        index
        for index, distance in enumerate(distances)
        if _at_least_ratio(float(distance), median, CENTER_MAX_TO_MEDIAN_LIMIT)
    ]
    outlier_ids = {images[index].image_id for index in outlier_indices}
    timestamps = _validated_timestamps(selected_timestamps)
    temporal = _temporal_health(images, centers, outlier_ids, timestamps)
    support = _image_support(images, points3d)
    bridges = _bridge_pairs(
        images,
        outlier_ids,
        database_path=database_path,
        timestamps=timestamps,
    )
    covisibility = _covisibility_summary(images, outlier_ids, bridges)
    outliers = [
        {
            "image_id": images[index].image_id,
            "name": images[index].name,
            "time_seconds": timestamps.get(images[index].name),
            "distance_world": float(distances[index]),
            "distance_to_median_ratio": _finite_ratio(
                float(distances[index]), median
            ),
            **support[images[index].image_id],
        }
        for index in sorted(
            outlier_indices,
            key=lambda value: (-float(distances[value]), images[value].image_id),
        )[:OUTLIER_LIMIT]
    ]
    repair = _repair_eligibility(
        images,
        outlier_ids,
        timestamps,
        temporal,
        required=bool(reasons),
    )
    return {
        "schema_version": 1,
        "profile": PROFILE_ID,
        "status": "failed" if reasons else "passed",
        "reason_codes": reasons,
        "policy": {
            "camera_center_reference": "componentwise_median",
            "camera_p99_method": "lower_order_statistic",
            "camera_max_to_median_limit": CENTER_MAX_TO_MEDIAN_LIMIT,
            "camera_max_to_p99_limit": CENTER_MAX_TO_P99_LIMIT,
            "camera_p99_to_median_limit": CENTER_P99_TO_MEDIAN_LIMIT,
            "temporal_boundary_speed_to_p90_limit": (
                TEMPORAL_BOUNDARY_SPEED_TO_P90_LIMIT
            ),
            "maximum_automatic_repair_fraction": MAX_AUTOMATIC_REPAIR_FRACTION,
            "test_rgb_loaded": False,
            "world_units": "arbitrary",
        },
        "camera_centers": {
            "count": len(images),
            "robust_center_world": robust_center.tolist(),
            "distance_world": distance_distribution,
            "max_to_median": max_to_median,
            "max_to_p99": max_to_p99,
            "p99_to_median": p99_to_median,
        },
        "temporal": temporal,
        "observation_support": {
            "registered_observation_count": sum(
                int(value["registered_observation_count"])
                for value in support.values()
            ),
            "positive_depth_fraction": _weighted_positive_depth_fraction(support),
        },
        "outlier_candidates": outliers,
        "covisibility": covisibility,
        "bridge_pairs": bridges,
        "automatic_repair": repair,
    }


def build_sfm_pose_health_from_text(
    *,
    model_dir: Path,
    selected_timestamps: Mapping[str, float] | None = None,
    database_path: Path | None = None,
) -> dict[str, Any]:
    return build_sfm_pose_health(
        images=parse_colmap_images_with_points(model_dir / "images.txt"),
        points3d=parse_colmap_points3d(model_dir / "points3D.txt"),
        selected_timestamps=selected_timestamps,
        database_path=database_path,
    )


def require_sfm_pose_health(record: Mapping[str, Any]) -> None:
    if record.get("status") != "passed":
        reasons = record.get("reason_codes")
        joined = ",".join(str(value) for value in reasons or ["unknown"])
        raise SfmPoseHealthError(f"SfM pose health failed: {joined}")


def write_sfm_pose_health(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selected_timestamps_from_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise SfmPoseHealthError("video selection has no selected frame list")
    result: dict[str, float] = {}
    for item in selected:
        if not isinstance(item, Mapping):
            raise SfmPoseHealthError("video selection frame is invalid")
        name = Path(str(item.get("path", ""))).name
        try:
            timestamp = float(item["time_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SfmPoseHealthError("video selection timestamp is invalid") from exc
        if not name or name in result or not math.isfinite(timestamp) or timestamp < 0:
            raise SfmPoseHealthError("video selection timestamps are invalid")
        result[name] = timestamp
    return result


def _validated_timestamps(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    if values is None:
        return {}
    result = {str(name): float(value) for name, value in values.items()}
    if any(
        not name or not math.isfinite(value) or value < 0
        for name, value in result.items()
    ):
        raise SfmPoseHealthError("video timestamps must be finite and non-negative")
    return result


def _temporal_health(
    images: list[ColmapImage],
    centers: np.ndarray,
    outlier_ids: set[int],
    timestamps: Mapping[str, float],
) -> dict[str, Any]:
    if not timestamps:
        return {"status": "not_available", "catastrophic_boundaries": []}
    indexed = [
        (timestamps[image.name], image, centers[index])
        for index, image in enumerate(images)
        if image.name in timestamps
    ]
    indexed.sort(key=lambda value: (value[0], value[1].image_id))
    timeline = analyze_registration_timeline(
        timestamps, [image.name for image in images]
    )
    steps: list[dict[str, Any]] = []
    core_speeds: list[float] = []
    for (left_time, left, left_center), (right_time, right, right_center) in zip(
        indexed, indexed[1:]
    ):
        elapsed = right_time - left_time
        if elapsed <= 0:
            continue
        speed = float(np.linalg.norm(right_center - left_center) / elapsed)
        relative_rotation = qvec_to_rotmat(right.qvec) @ qvec_to_rotmat(
            left.qvec
        ).T
        rotation_jump = float(
            np.degrees(
                np.arccos(
                    np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
                )
            )
        )
        boundary = (left.image_id in outlier_ids) != (right.image_id in outlier_ids)
        steps.append(
            {
                "left_image_id": left.image_id,
                "right_image_id": right.image_id,
                "left_name": left.name,
                "right_name": right.name,
                "left_time_seconds": left_time,
                "right_time_seconds": right_time,
                "translation_speed_world_per_second": speed,
                "rotation_jump_degrees": rotation_jump,
                "outlier_boundary": boundary,
            }
        )
        if left.image_id not in outlier_ids and right.image_id not in outlier_ids:
            core_speeds.append(speed)
    baseline = (
        float(np.quantile(core_speeds, 0.9, method="lower"))
        if core_speeds
        else 0.0
    )
    catastrophic = []
    for step in steps:
        ratio = _finite_ratio(
            float(step["translation_speed_world_per_second"]), baseline
        )
        step["speed_to_core_p90_ratio"] = ratio
        if step["outlier_boundary"] and _at_least_ratio(
            float(step["translation_speed_world_per_second"]),
            baseline,
            TEMPORAL_BOUNDARY_SPEED_TO_P90_LIMIT,
        ):
            catastrophic.append(step)
    return {
        "status": "available",
        "registered_timestamp_count": len(indexed),
        "registration_timeline": {
            key: timeline[key]
            for key in (
                "registered_count",
                "registration_rate",
                "temporal_coverage",
                "gap_violation_count",
                "maximum_registered_gap_seconds",
            )
        },
        "translation_speed_world_per_second": _optional_distribution(
            np.asarray([step["translation_speed_world_per_second"] for step in steps])
        ),
        "rotation_jump_degrees": _optional_distribution(
            np.asarray([step["rotation_jump_degrees"] for step in steps])
        ),
        "core_translation_speed_p90_world_per_second": baseline,
        "catastrophic_boundaries": catastrophic[:BRIDGE_PAIR_LIMIT],
    }


def _image_support(
    images: list[ColmapImage], points3d: Mapping[int, np.ndarray]
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for image in images:
        rotation = qvec_to_rotmat(image.qvec)
        depths: list[float] = []
        positive = 0
        valid = 0
        for _x, _y, point_id in image.observations:
            point = points3d.get(point_id)
            if point is None:
                continue
            depth = float((rotation @ np.asarray(point) + image.tvec)[2])
            if not math.isfinite(depth):
                raise SfmPoseHealthError("SfM observation depth is not finite")
            valid += 1
            depths.append(depth)
            positive += depth > 0
        result[image.image_id] = {
            "registered_observation_count": len(image.observations),
            "resolved_point_count": valid,
            "positive_depth_fraction": positive / valid if valid else None,
            "observed_depth_world": _optional_distribution(
                np.asarray(depths, dtype=np.float64)
            ),
        }
    return result


def _weighted_positive_depth_fraction(
    support: Mapping[int, Mapping[str, Any]],
) -> float | None:
    resolved = sum(int(value["resolved_point_count"]) for value in support.values())
    if not resolved:
        return None
    positive = sum(
        int(value["resolved_point_count"])
        * float(value["positive_depth_fraction"] or 0.0)
        for value in support.values()
    )
    return positive / resolved


def _bridge_pairs(
    images: list[ColmapImage],
    outlier_ids: set[int],
    *,
    database_path: Path | None,
    timestamps: Mapping[str, float],
) -> list[dict[str, Any]]:
    if not outlier_ids:
        return []
    point_images: dict[int, set[int]] = {}
    for image in images:
        for _x, _y, point_id in image.observations:
            point_images.setdefault(point_id, set()).add(image.image_id)
    shared: dict[tuple[int, int], int] = {}
    all_ids = {image.image_id for image in images}
    core_ids = all_ids - outlier_ids
    for image_ids in point_images.values():
        for left in image_ids & outlier_ids:
            for right in image_ids & core_ids:
                pair = tuple(sorted((left, right)))
                shared[pair] = shared.get(pair, 0) + 1
    database = _database_pair_counts(database_path, set(shared))
    names = {image.image_id: image.name for image in images}
    records = []
    for pair, shared_points in shared.items():
        left, right = pair
        evidence = database.get(pair, {})
        records.append(
            {
                "image_ids": [left, right],
                "image_names": [names[left], names[right]],
                "time_span_seconds": (
                    abs(timestamps[names[left]] - timestamps[names[right]])
                    if names[left] in timestamps and names[right] in timestamps
                    else None
                ),
                "shared_final_tracks": shared_points,
                "candidate_match_count": evidence.get("candidate_match_count"),
                "verified_inlier_count": evidence.get("verified_inlier_count"),
                "geometric_config": evidence.get("geometric_config"),
            }
        )
    return sorted(
        records,
        key=lambda value: (
            -int(value["verified_inlier_count"] or 0),
            -int(value["shared_final_tracks"]),
            value["image_ids"],
        ),
    )[:BRIDGE_PAIR_LIMIT]


def _covisibility_summary(
    images: list[ColmapImage],
    outlier_ids: set[int],
    bridges: list[dict[str, Any]],
) -> dict[str, Any]:
    parent = {image.image_id: image.image_id for image in images}

    def find(image_id: int) -> int:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    point_images: dict[int, list[int]] = {}
    for image in images:
        for _x, _y, point_id in image.observations:
            point_images.setdefault(point_id, []).append(image.image_id)
    for image_ids in point_images.values():
        unique_ids = sorted(set(image_ids))
        for image_id in unique_ids[1:]:
            union(unique_ids[0], image_id)

    grouped: dict[int, list[int]] = {}
    for image_id in parent:
        grouped.setdefault(find(image_id), []).append(image_id)
    components = []
    for image_ids in grouped.values():
        ordered = sorted(image_ids)
        outlier_count = len(set(ordered) & outlier_ids)
        components.append(
            {
                "image_count": len(ordered),
                "core_image_count": len(ordered) - outlier_count,
                "outlier_image_count": outlier_count,
                "image_ids": ordered[:OUTLIER_LIMIT],
                "image_ids_truncated": len(ordered) > OUTLIER_LIMIT,
            }
        )
    components.sort(
        key=lambda value: (-int(value["image_count"]), value["image_ids"])
    )
    verified = [
        int(pair["verified_inlier_count"])
        for pair in bridges
        if pair["verified_inlier_count"] is not None
    ]
    return {
        "component_count": len(components),
        "isolated_component_count": sum(
            int(component["image_count"] == 1) for component in components
        ),
        "largest_component_image_count": int(components[0]["image_count"]),
        "mixed_core_outlier_component_count": sum(
            int(
                component["core_image_count"] > 0
                and component["outlier_image_count"] > 0
            )
            for component in components
        ),
        "components": components[:BRIDGE_PAIR_LIMIT],
        "components_truncated": len(components) > BRIDGE_PAIR_LIMIT,
        "outlier_core_bridges": {
            "reported_pair_count": len(bridges),
            "verified_pair_count": len(verified),
            "maximum_verified_inliers": max(verified) if verified else None,
            "shared_final_track_count": sum(
                int(pair["shared_final_tracks"]) for pair in bridges
            ),
        },
    }


def _database_pair_counts(
    path: Path | None, pairs: set[tuple[int, int]]
) -> dict[tuple[int, int], dict[str, int | None]]:
    if path is None or not pairs:
        return {}
    if not path.is_file():
        raise SfmPoseHealthError(f"COLMAP database is missing: {path}")
    result: dict[tuple[int, int], dict[str, int | None]] = {}
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """
                WITH pair_ids AS (
                  SELECT pair_id FROM matches
                  UNION SELECT pair_id FROM two_view_geometries
                )
                SELECT pair_ids.pair_id, matches.rows,
                       two_view_geometries.rows, two_view_geometries.config
                FROM pair_ids
                LEFT JOIN matches ON matches.pair_id = pair_ids.pair_id
                LEFT JOIN two_view_geometries
                  ON two_view_geometries.pair_id = pair_ids.pair_id
                """
            )
            for pair_id, candidates, verified, config in rows:
                pair = _pair_image_ids(int(pair_id))
                if pair in pairs:
                    result[pair] = {
                        "candidate_match_count": (
                            int(candidates) if candidates is not None else 0
                        ),
                        "verified_inlier_count": (
                            int(verified) if verified is not None else 0
                        ),
                        "geometric_config": (
                            int(config) if config is not None else None
                        ),
                    }
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise SfmPoseHealthError(f"cannot read COLMAP pair evidence: {exc}") from exc
    return result


def _repair_eligibility(
    images: list[ColmapImage],
    outlier_ids: set[int],
    timestamps: Mapping[str, float],
    temporal: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {"eligible": False, "reason": "pose_health_passed"}
    if not outlier_ids:
        return {"eligible": False, "reason": "no_bounded_outlier_set"}
    if not timestamps:
        return {"eligible": False, "reason": "video_timestamps_unavailable"}
    fraction = len(outlier_ids) / len(images)
    if fraction > MAX_AUTOMATIC_REPAIR_FRACTION:
        return {
            "eligible": False,
            "reason": "outlier_fraction_exceeds_limit",
            "outlier_fraction": fraction,
        }
    retained = [image.name for image in images if image.image_id not in outlier_ids]
    timeline = analyze_registration_timeline(timestamps, retained)
    if int(timeline["registered_count"]) < MIN_VIDEO_REGISTERED_COUNT:
        reason = "retained_registered_count_below_gate"
    elif float(timeline["registration_rate"]) < MIN_VIDEO_REGISTRATION_RATE:
        reason = "retained_registration_rate_below_gate"
    elif float(timeline["temporal_coverage"]) < MIN_VIDEO_TEMPORAL_COVERAGE:
        reason = "retained_temporal_coverage_below_gate"
    elif not temporal.get("catastrophic_boundaries"):
        reason = "catastrophic_temporal_boundary_missing"
    else:
        reason = "eligible"
    return {
        "eligible": reason == "eligible",
        "reason": reason,
        "outlier_fraction": fraction,
        "excluded_image_ids": sorted(outlier_ids),
        "retained_timeline": {
            key: timeline[key]
            for key in (
                "registered_count",
                "registration_rate",
                "temporal_coverage",
                "gap_violation_count",
                "maximum_registered_gap_seconds",
            )
        },
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise SfmPoseHealthError("SfM pose distributions require finite values")
    return {
        "count": len(array),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9, method="lower")),
        "p99": float(np.quantile(array, 0.99, method="lower")),
        "max": float(array.max()),
    }


def _optional_distribution(values: np.ndarray) -> dict[str, float | int] | None:
    return _distribution(values) if len(values) else None


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def _at_least_ratio(numerator: float, denominator: float, limit: float) -> bool:
    return numerator > 1e-12 and (
        denominator <= 1e-12 or numerator / denominator >= limit
    )


def _pair_image_ids(pair_id: int) -> tuple[int, int]:
    second = pair_id % MAX_IMAGE_ID
    first = (pair_id - second) // MAX_IMAGE_ID
    return (first, second) if first < second else (second, first)
