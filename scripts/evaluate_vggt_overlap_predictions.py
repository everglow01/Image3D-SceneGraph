#!/usr/bin/env python3
"""Measure disagreement between retained VGGT predictions of the same image."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from run_colmap_vggt_dense import FusionCamera, unproject_depth_with_colmap_pose


MIN_SCALE_DOMINANCE_PAIRS = 10
MIN_SCALE_DOMINANCE_REDUCTION = 0.5
MIN_SCALE_DOMINANCE_P90_REDUCTION = 0.3
MIN_SCALE_DOMINANCE_IMPROVED_FRACTION = 0.75


def valid_canvas_mask(record: dict[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in record["image_shape"])
    transform = record["canvas_transform"]
    top = int(transform["pad_top"])
    left = int(transform["pad_left"])
    bottom = top + int(transform["resized_height"])
    right = left + int(transform["resized_width"])
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError(f"invalid canvas transform for {record['image']}")
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def residual_summary(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    absolute = np.abs(values)
    deviations = np.abs(values - median)
    return {
        "signed_median": median,
        "mad": float(np.median(deviations)),
        "absolute_p50": float(np.percentile(absolute, 50)),
        "absolute_p90": float(np.percentile(absolute, 90)),
    }


def inverse_depth_summary(
    first_depth: np.ndarray,
    second_depth: np.ndarray,
) -> dict[str, float]:
    first_inverse = 1.0 / first_depth
    second_inverse = 1.0 / second_depth
    relative = 2.0 * np.abs(first_inverse - second_inverse) / (
        first_inverse + second_inverse
    )
    return {
        "symmetric_relative_p50": float(np.percentile(relative, 50)),
        "symmetric_relative_p90": float(np.percentile(relative, 90)),
    }


def spatial_residual_grid(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    grid_size: int,
) -> np.ndarray:
    height, width = residual.shape
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for row in range(grid_size):
        y0 = row * height // grid_size
        y1 = (row + 1) * height // grid_size
        for column in range(grid_size):
            x0 = column * width // grid_size
            x1 = (column + 1) * width // grid_size
            values = np.abs(residual[y0:y1, x0:x1][mask[y0:y1, x0:x1]])
            if values.size:
                grid[row, column] = np.median(values)
    return grid


def load_prediction(
    prediction_dir: Path,
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    path = prediction_dir / record["prediction_file"]
    with np.load(path) as payload:
        depth = payload["depth"].astype(np.float64)
        confidence = payload["confidence"].astype(np.float64)
    expected_shape = tuple(int(value) for value in record["image_shape"])
    if depth.shape != expected_shape or confidence.shape != expected_shape:
        raise ValueError(f"prediction shape does not match index: {path}")
    return depth, confidence


def points_in_aligned_rois(
    *,
    depth: np.ndarray,
    record: dict[str, Any],
    camera: dict[str, Any],
    image: dict[str, Any],
    alignment_transform: np.ndarray,
    rois: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    intrinsic = np.asarray(record["fusion_intrinsic"], dtype=np.float32)
    points = unproject_depth_with_colmap_pose(
        depth=depth.astype(np.float32),
        camera=FusionCamera(
            model=str(camera["model"]),
            intrinsic=intrinsic,
            radial_distortion=tuple(float(value) for value in record["radial_distortion"]),
        ),
        qvec=np.asarray(image["qvec"], dtype=np.float64),
        tvec=np.asarray(image["tvec"], dtype=np.float64),
    ).reshape(-1, 3)
    aligned = (
        points @ alignment_transform[:3, :3].T
        + alignment_transform[:3, 3]
    )
    return {
        roi["name"]: np.all(
            (aligned >= np.asarray(roi["min"], dtype=np.float32))
            & (aligned <= np.asarray(roi["max"], dtype=np.float32)),
            axis=1,
        ).reshape(depth.shape)
        for roi in rois
    }


def compare_prediction_pair(
    *,
    first_record: dict[str, Any],
    second_record: dict[str, Any],
    prediction_dir: Path,
    heatmap_dir: Path,
    confidence_percentile: float,
    min_common_pixels: int,
    grid_size: int,
    roi_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_name = (
        f"image_{int(first_record['image_id'])}_groups_"
        f"{int(first_record['group_index']):04d}_{int(second_record['group_index']):04d}"
    )
    base = {
        "pair_id": pair_name,
        "image": first_record["image"],
        "image_id": first_record["image_id"],
        "first_group_index": first_record["group_index"],
        "second_group_index": second_record["group_index"],
        "first_group_position": first_record["group_position"],
        "second_group_position": second_record["group_position"],
        "first_role": first_record["role"],
        "second_role": second_record["role"],
        "includes_first_wins": bool(
            first_record["selected_for_first_wins"]
            or second_record["selected_for_first_wins"]
        ),
    }
    if (
        first_record["image"] != second_record["image"]
        or first_record["image_shape"] != second_record["image_shape"]
        or first_record["original_size"] != second_record["original_size"]
        or first_record["canvas_transform"] != second_record["canvas_transform"]
    ):
        return {**base, "status": "incompatible_canvas"}

    first_depth, first_confidence = load_prediction(prediction_dir, first_record)
    second_depth, second_confidence = load_prediction(prediction_dir, second_record)
    canvas = valid_canvas_mask(first_record)
    first_depth_valid = np.isfinite(first_depth) & (first_depth > 0)
    second_depth_valid = np.isfinite(second_depth) & (second_depth > 0)
    first_confidence_valid = np.isfinite(first_confidence)
    second_confidence_valid = np.isfinite(second_confidence)
    first_threshold_values = first_confidence[
        canvas & first_depth_valid & first_confidence_valid
    ]
    second_threshold_values = second_confidence[
        canvas & second_depth_valid & second_confidence_valid
    ]
    counts: dict[str, int | None] = {
        "canvas_pixels": int(np.count_nonzero(canvas)),
        "padding_pixels": int(canvas.size - np.count_nonzero(canvas)),
        "first_invalid_depth_pixels": int(np.count_nonzero(canvas & ~first_depth_valid)),
        "second_invalid_depth_pixels": int(np.count_nonzero(canvas & ~second_depth_valid)),
    }
    if not first_threshold_values.size or not second_threshold_values.size:
        return {**base, "status": "no_valid_depth", "pixel_counts": counts}

    first_threshold = float(np.percentile(first_threshold_values, confidence_percentile))
    second_threshold = float(np.percentile(second_threshold_values, confidence_percentile))
    common = (
        canvas
        & first_depth_valid
        & second_depth_valid
        & first_confidence_valid
        & second_confidence_valid
        & (first_confidence >= first_threshold)
        & (second_confidence >= second_threshold)
    )
    common_count = int(np.count_nonzero(common))
    counts.update(
        {
            "first_low_confidence_pixels": int(
                np.count_nonzero(canvas & first_confidence_valid & (first_confidence < first_threshold))
            ),
            "second_low_confidence_pixels": int(
                np.count_nonzero(canvas & second_confidence_valid & (second_confidence < second_threshold))
            ),
            "common_reliable_pixels": common_count,
            "occluded_pixels": None,
        }
    )
    if common_count < min_common_pixels:
        return {
            **base,
            "status": "insufficient_common_reliable_pixels",
            "pixel_counts": counts,
            "confidence_thresholds": [first_threshold, second_threshold],
        }

    y, x = np.indices(first_depth.shape)
    train = common & (((x + y) % 2) == 0)
    held_out = common & ~train
    if np.count_nonzero(train) < min_common_pixels // 2 or np.count_nonzero(held_out) < min_common_pixels // 2:
        return {
            **base,
            "status": "insufficient_train_or_held_out_pixels",
            "pixel_counts": counts,
            "confidence_thresholds": [first_threshold, second_threshold],
        }

    first_anchor = first_record.get("sparse_scale_anchor")
    second_anchor = second_record.get("sparse_scale_anchor")
    if first_anchor is None or second_anchor is None:
        return {
            **base,
            "status": "missing_sparse_scale_anchor",
            "pixel_counts": counts,
            "confidence_thresholds": [first_threshold, second_threshold],
        }

    raw_difference = np.log(first_depth) - np.log(second_depth)
    first_scale = float(first_anchor["scale"])
    second_scale = float(second_anchor["scale"])
    anchored_difference = raw_difference + np.log(first_scale) - np.log(second_scale)
    fitted_log_offset = float(np.median(anchored_difference[train]))
    fitted_difference = anchored_difference - fitted_log_offset
    joint_confidence = np.minimum(first_confidence, second_confidence)
    held_out_confidence = joint_confidence[held_out]
    confidence_split = float(np.median(held_out_confidence))
    lower_confidence = held_out & (joint_confidence <= confidence_split)
    higher_confidence = held_out & (joint_confidence > confidence_split)

    heatmaps = np.stack(
        [
            spatial_residual_grid(raw_difference, common, grid_size=grid_size),
            spatial_residual_grid(anchored_difference, common, grid_size=grid_size),
            spatial_residual_grid(fitted_difference, held_out, grid_size=grid_size),
        ]
    )
    heatmap_path = heatmap_dir / f"{pair_name}.npy"
    np.save(heatmap_path, heatmaps, allow_pickle=False)

    fitted_grid = heatmaps[2]
    finite_grid = fitted_grid[np.isfinite(fitted_grid)]
    spatial_summary = {
        "valid_grid_cells": int(finite_grid.size),
        "absolute_log_depth_grid_p10": (
            float(np.percentile(finite_grid, 10)) if finite_grid.size else None
        ),
        "absolute_log_depth_grid_p90": (
            float(np.percentile(finite_grid, 90)) if finite_grid.size else None
        ),
        "absolute_log_depth_grid_p90_minus_p10": (
            float(np.percentile(finite_grid, 90) - np.percentile(finite_grid, 10))
            if finite_grid.size
            else None
        ),
    }

    first_anchored_depth = first_depth[held_out] * first_scale
    second_anchored_depth = second_depth[held_out] * second_scale
    second_fitted_depth = second_anchored_depth * np.exp(fitted_log_offset)
    roi_metrics: dict[str, Any] = {}
    if roi_context is not None:
        fusion_record = roi_context["fusion_by_image"][first_record["image"]]
        pose_record = roi_context["pose_by_image"][first_record["image"]]
        roi_masks = points_in_aligned_rois(
            depth=first_depth * first_scale,
            record=fusion_record,
            camera=roi_context["camera_by_id"][pose_record["camera_id"]],
            image=pose_record,
            alignment_transform=roi_context["alignment_transform"],
            rois=roi_context["rois"],
        )
        for roi_name, roi_mask in roi_masks.items():
            roi_held_out = held_out & roi_mask
            pixel_count = int(np.count_nonzero(roi_held_out))
            roi_metrics[roi_name] = (
                {
                    "status": "evaluated",
                    "held_out_pixels": pixel_count,
                    "anchored_log_depth": residual_summary(
                        anchored_difference[roi_held_out]
                    ),
                    "scale_fitted_log_depth": residual_summary(
                        fitted_difference[roi_held_out]
                    ),
                }
                if pixel_count
                else {"status": "not_observed", "held_out_pixels": 0}
            )
    return {
        **base,
        "status": "evaluated",
        "pixel_counts": {
            **counts,
            "train_pixels": int(np.count_nonzero(train)),
            "held_out_pixels": int(np.count_nonzero(held_out)),
        },
        "confidence_thresholds": [first_threshold, second_threshold],
        "confidence_strata": {
            "joint_confidence_median": confidence_split,
            "lower_held_out_pixels": int(np.count_nonzero(lower_confidence)),
            "higher_held_out_pixels": int(np.count_nonzero(higher_confidence)),
            "lower_anchored_absolute_p50": percentile_or_none(
                np.abs(anchored_difference[lower_confidence]), 50
            ),
            "higher_anchored_absolute_p50": percentile_or_none(
                np.abs(anchored_difference[higher_confidence]), 50
            ),
        },
        "sparse_scale_anchors": [first_anchor, second_anchor],
        "raw_log_depth": residual_summary(raw_difference[held_out]),
        "anchored_log_depth": residual_summary(anchored_difference[held_out]),
        "scale_only_fit": {
            "train_log_offset": fitted_log_offset,
            "second_depth_multiplier": float(np.exp(fitted_log_offset)),
            "held_out_log_depth": residual_summary(fitted_difference[held_out]),
        },
        "raw_inverse_depth": inverse_depth_summary(
            first_depth[held_out],
            second_depth[held_out],
        ),
        "anchored_inverse_depth": inverse_depth_summary(
            first_anchored_depth,
            second_anchored_depth,
        ),
        "fitted_inverse_depth": inverse_depth_summary(
            first_anchored_depth,
            second_fitted_depth,
        ),
        "roi_metrics": roi_metrics,
        "spatial_summary": spatial_summary,
        "heatmap_file": heatmap_path.name,
        "heatmap_channels": [
            "raw_absolute_log_depth",
            "anchored_absolute_log_depth",
            "scale_fitted_held_out_absolute_log_depth",
        ],
        "heatmap_shape": list(heatmaps.shape),
    }


def aggregate_pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [record for record in records if record["status"] == "evaluated"]
    status_counts = Counter(record["status"] for record in records)
    if not evaluated:
        return {
            "pair_count": len(records),
            "evaluated_pair_count": 0,
            "status_counts": dict(sorted(status_counts.items())),
        }

    def values(path: tuple[str, ...]) -> np.ndarray:
        result = []
        for record in evaluated:
            value: Any = record
            for key in path:
                value = value[key]
            result.append(float(value))
        return np.asarray(result)

    anchored_p50 = values(("anchored_log_depth", "absolute_p50"))
    anchored_p90 = values(("anchored_log_depth", "absolute_p90"))
    fitted_p50 = values(("scale_only_fit", "held_out_log_depth", "absolute_p50"))
    fitted_p90 = values(("scale_only_fit", "held_out_log_depth", "absolute_p90"))
    first_wins = [record for record in evaluated if record["includes_first_wins"]]
    role_counts = Counter(
        "|".join(sorted((record["first_role"], record["second_role"])))
        for record in evaluated
    )
    lower_confidence_p50 = [
        record["confidence_strata"]["lower_anchored_absolute_p50"]
        for record in evaluated
        if record["confidence_strata"]["lower_anchored_absolute_p50"] is not None
    ]
    higher_confidence_p50 = [
        record["confidence_strata"]["higher_anchored_absolute_p50"]
        for record in evaluated
        if record["confidence_strata"]["higher_anchored_absolute_p50"] is not None
    ]

    def subgroup_medians(subgroup: list[dict[str, Any]]) -> dict[str, float] | None:
        if not subgroup:
            return None
        return {
            "anchored_absolute_p50": float(
                np.median(
                    [record["anchored_log_depth"]["absolute_p50"] for record in subgroup]
                )
            ),
            "scale_fitted_absolute_p50": float(
                np.median(
                    [
                        record["scale_only_fit"]["held_out_log_depth"]["absolute_p50"]
                        for record in subgroup
                    ]
                )
            ),
        }

    return {
        "pair_count": len(records),
        "evaluated_pair_count": len(evaluated),
        "status_counts": dict(sorted(status_counts.items())),
        "first_wins_pair_count": len(first_wins),
        "non_first_wins_pair_count": len(evaluated) - len(first_wins),
        "first_wins_pair_metric_medians": subgroup_medians(first_wins),
        "non_first_wins_pair_metric_medians": subgroup_medians(
            [record for record in evaluated if not record["includes_first_wins"]]
        ),
        "role_pair_counts": dict(sorted(role_counts.items())),
        "confidence_strata_pair_medians": {
            "lower_anchored_absolute_p50": (
                float(np.median(lower_confidence_p50))
                if lower_confidence_p50
                else None
            ),
            "higher_anchored_absolute_p50": (
                float(np.median(higher_confidence_p50))
                if higher_confidence_p50
                else None
            ),
        },
        "pair_metric_medians": {
            "raw_absolute_p50": float(
                np.median(values(("raw_log_depth", "absolute_p50")))
            ),
            "anchored_absolute_p50": float(np.median(anchored_p50)),
            "anchored_absolute_p90": float(np.median(anchored_p90)),
            "scale_fitted_absolute_p50": float(np.median(fitted_p50)),
            "scale_fitted_absolute_p90": float(np.median(fitted_p90)),
            "scale_fitted_spatial_grid_p90_minus_p10": float(
                np.median(
                    [
                        record["spatial_summary"][
                            "absolute_log_depth_grid_p90_minus_p10"
                        ]
                        for record in evaluated
                        if record["spatial_summary"][
                            "absolute_log_depth_grid_p90_minus_p10"
                        ]
                        is not None
                    ]
                )
            ),
        },
        "scale_fit_reduction": {
            "median_p50_fraction": float(
                1.0 - np.median(fitted_p50) / max(np.median(anchored_p50), 1e-12)
            ),
            "median_p90_fraction": float(
                1.0 - np.median(fitted_p90) / max(np.median(anchored_p90), 1e-12)
            ),
            "p50_improved_pair_fraction": float(np.mean(fitted_p50 < anchored_p50)),
            "p90_improved_pair_fraction": float(np.mean(fitted_p90 < anchored_p90)),
        },
    }


def build_roi_context(
    *,
    roi_path: Path,
    cameras_path: Path,
    fusion_path: Path,
    alignment_path: Path,
) -> dict[str, Any]:
    roi_payload = json.loads(roi_path.read_text(encoding="utf-8"))
    cameras_payload = json.loads(cameras_path.read_text(encoding="utf-8"))
    fusion_payload = json.loads(fusion_path.read_text(encoding="utf-8"))
    alignment_payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    transform = np.asarray(alignment_payload["transform"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("alignment transform must be 4x4")
    return {
        "rois": roi_payload["rois"],
        "camera_by_id": {
            int(camera["camera_id"]): camera for camera in cameras_payload["cameras"]
        },
        "pose_by_image": {
            image["name"]: image for image in cameras_payload["images"]
        },
        "fusion_by_image": {
            record["image"]: record for record in fusion_payload["images"]
        },
        "alignment_transform": transform,
    }


def evaluate_overlap_predictions(
    *,
    index_path: Path,
    heatmap_dir: Path,
    confidence_percentile: float,
    min_common_pixels: int,
    grid_size: int,
    roi_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not index.get("capture_enabled"):
        raise ValueError("unsupported or disabled VGGT window prediction index")
    prediction_dir = index_path.parent / "vggt_window_predictions"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in index["predictions"]:
        by_image[record["image"]].append(record)
    records: list[dict[str, Any]] = []
    for image_name in sorted(by_image):
        predictions = sorted(
            by_image[image_name],
            key=lambda record: (record["group_index"], record["group_position"]),
        )
        for first, second in combinations(predictions, 2):
            records.append(
                compare_prediction_pair(
                    first_record=first,
                    second_record=second,
                    prediction_dir=prediction_dir,
                    heatmap_dir=heatmap_dir,
                    confidence_percentile=confidence_percentile,
                    min_common_pixels=min_common_pixels,
                    grid_size=grid_size,
                    roi_context=roi_context,
                )
            )

    aggregate = aggregate_pair_metrics(records)
    roi_aggregate: dict[str, Any] = {}
    roi_names = sorted(
        {
            roi_name
            for record in records
            for roi_name in record.get("roi_metrics", {})
        }
    )
    for roi_name in roi_names:
        observed = [
            record["roi_metrics"][roi_name]
            for record in records
            if record.get("roi_metrics", {}).get(roi_name, {}).get("status")
            == "evaluated"
        ]
        roi_aggregate[roi_name] = {
            "observed_pair_count": len(observed),
            "held_out_pixel_count": sum(
                record["held_out_pixels"] for record in observed
            ),
            "anchored_absolute_p50_median": (
                float(
                    np.median(
                        [
                            record["anchored_log_depth"]["absolute_p50"]
                            for record in observed
                        ]
                    )
                )
                if observed
                else None
            ),
            "scale_fitted_absolute_p50_median": (
                float(
                    np.median(
                        [
                            record["scale_fitted_log_depth"]["absolute_p50"]
                            for record in observed
                        ]
                    )
                )
                if observed
                else None
            ),
        }
    reduction = aggregate.get("scale_fit_reduction")
    scene_supports_scale_dominance = bool(
        reduction
        and aggregate["evaluated_pair_count"] >= MIN_SCALE_DOMINANCE_PAIRS
        and reduction["median_p50_fraction"] >= MIN_SCALE_DOMINANCE_REDUCTION
        and reduction["median_p90_fraction"] >= MIN_SCALE_DOMINANCE_P90_REDUCTION
        and reduction["p50_improved_pair_fraction"]
        >= MIN_SCALE_DOMINANCE_IMPROVED_FRACTION
    )
    return {
        "schema_version": 1,
        "source_index": index_path.as_posix(),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "comparison": {
            "same_image_same_canvas": True,
            "confidence_percentile_per_prediction": confidence_percentile,
            "minimum_common_reliable_pixels": min_common_pixels,
            "train_held_out_split": "canvas_checkerboard_parity_even_odd",
            "scale_fit": "median_anchored_log_depth_offset_on_train_pixels",
            "reported_fit_residual": "held_out_pixels_only",
            "heatmap_grid_size": grid_size,
            "occlusion_handling": "not_applicable_same_image_same_canvas",
            "distortion_handling": "not_applied_direct_canvas_comparison",
        },
        "source_summary": {
            key: index[key]
            for key in (
                "fusion_policy",
                "grouping",
                "batch_size",
                "requested_overlap_size",
                "frames_chunk_size",
                "group_count",
                "registered_image_count",
                "prediction_count",
                "unique_image_count",
                "overlap_image_count",
                "max_predictions_per_image",
            )
        },
        "aggregate": aggregate,
        "roi_aggregate": roi_aggregate,
        "decision_gate": {
            "scene_supports_scale_dominance": scene_supports_scale_dominance,
            "criteria": {
                "minimum_evaluated_pairs": MIN_SCALE_DOMINANCE_PAIRS,
                "minimum_median_p50_reduction_fraction": MIN_SCALE_DOMINANCE_REDUCTION,
                "minimum_median_p90_reduction_fraction": MIN_SCALE_DOMINANCE_P90_REDUCTION,
                "minimum_p50_improved_pair_fraction": MIN_SCALE_DOMINANCE_IMPROVED_FRACTION,
            },
            "g1_10_status": "requires_consistent_support_across_multiple_frozen_scenes",
        },
        "pairs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heatmap-dir", required=True, type=Path)
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--min-common-pixels", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--rois", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--fusion", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.confidence_percentile < 100:
        raise SystemExit("--confidence-percentile must be between 0 and 100")
    if args.min_common_pixels <= 0:
        raise SystemExit("--min-common-pixels must be positive")
    if args.grid_size <= 0:
        raise SystemExit("--grid-size must be positive")

    roi_paths = (args.rois, args.cameras, args.fusion, args.alignment)
    if any(path is not None for path in roi_paths) and not all(
        path is not None for path in roi_paths
    ):
        raise SystemExit("--rois, --cameras, --fusion, and --alignment must be supplied together")
    roi_context = (
        build_roi_context(
            roi_path=args.rois,
            cameras_path=args.cameras,
            fusion_path=args.fusion,
            alignment_path=args.alignment,
        )
        if all(path is not None for path in roi_paths)
        else None
    )
    payload = evaluate_overlap_predictions(
        index_path=args.index,
        heatmap_dir=args.heatmap_dir,
        confidence_percentile=args.confidence_percentile,
        min_common_pixels=args.min_common_pixels,
        grid_size=args.grid_size,
        roi_context=roi_context,
    )
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"overlap diagnostics differ: {args.output}")
        print(f"verified {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
