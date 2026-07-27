#!/usr/bin/env python3
"""Compare diagnostic intrinsics candidates on frozen COLMAP sparse points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_vggt_canvas_coordinates import (
    unique_prediction_records,
    validate_canvas_record,
)
from image3d_scenegraph.geometry.grouping import (
    parse_colmap_images_with_points,
    parse_colmap_points3d,
)
from run_colmap_sparse import parse_colmap_cameras
from run_colmap_vggt_dense import (
    FusionCamera,
    build_fusion_camera,
    project_world_points_to_depth_canvas,
)


EDGE_NORMALIZED_RADIUS = 0.75
MIN_EDGE_P90_IMPROVED_SCENES = 3
MIN_CROSS_SCENE_MEDIAN_EDGE_P90_REDUCTION = 0.20
MAX_SCENE_ALL_P50_REGRESSION = 0.05
MAX_FALLBACK_FRACTION = 0.05
CANDIDATE_NAMES = (
    "production_colmap",
    "pixel_center_colmap",
    "vggt",
    "vggt_focal_colmap_center",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_summary(values: np.ndarray) -> dict[str, float | int | None]:
    if not values.size:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def valid_intrinsic(value: Any) -> np.ndarray | None:
    intrinsic = np.asarray(value, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.all(np.isfinite(intrinsic))
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.isclose(intrinsic[2, 2], 1.0)
    ):
        return None
    return intrinsic.astype(np.float32)


def build_candidate_cameras(
    *,
    production: FusionCamera,
    vggt_intrinsic: Any,
    scale_x: float,
    scale_y: float,
) -> tuple[dict[str, FusionCamera], dict[str, str | None]]:
    pixel_center = production.intrinsic.copy()
    pixel_center[0, 2] += 0.5 * (scale_x - 1.0)
    pixel_center[1, 2] += 0.5 * (scale_y - 1.0)
    vggt = valid_intrinsic(vggt_intrinsic)
    fallback: dict[str, str | None] = {name: None for name in CANDIDATE_NAMES}
    if vggt is None:
        vggt = production.intrinsic.copy()
        fallback["vggt"] = "invalid_vggt_intrinsic_to_production_colmap"
        fallback["vggt_focal_colmap_center"] = (
            "invalid_vggt_intrinsic_to_pixel_center_colmap"
        )
    reconciled = pixel_center.copy()
    if fallback["vggt_focal_colmap_center"] is None:
        reconciled[0, 0] = vggt[0, 0]
        reconciled[1, 1] = vggt[1, 1]
    return (
        {
            "production_colmap": production,
            "pixel_center_colmap": FusionCamera(
                production.model, pixel_center, production.radial_distortion
            ),
            "vggt": FusionCamera(
                production.model, vggt, production.radial_distortion
            ),
            "vggt_focal_colmap_center": FusionCamera(
                production.model, reconciled, production.radial_distortion
            ),
        },
        fallback,
    )


def relative_change(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline if baseline > 0 else 0.0


def evaluate_scene(
    *,
    name: str,
    job_dir: Path,
    index_path: Path,
) -> dict[str, Any]:
    sparse_dir = job_dir / "colmap_vggt" / "sparse_txt"
    fusion_path = job_dir / "diagnostics" / "fusion.json"
    required = [
        index_path,
        fusion_path,
        sparse_dir / "cameras.txt",
        sparse_dir / "images.txt",
        sparse_dir / "points3D.txt",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing G1.19 inputs: {missing}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    predictions = unique_prediction_records(index)
    selected = {
        record["image"]: record
        for record in index["predictions"]
        if record.get("selected_for_first_wins")
    }
    if set(selected) != set(predictions):
        raise ValueError(f"scene {name} does not have one first-wins prediction per image")
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    fusion_by_name = {record["image"]: record for record in fusion["images"]}
    cameras = {
        camera["camera_id"]: camera
        for camera in parse_colmap_cameras(sparse_dir / "cameras.txt")
    }
    images = parse_colmap_images_with_points(sparse_dir / "images.txt")
    points3d = parse_colmap_points3d(sparse_dir / "points3D.txt")
    registered_names = {image.name for image in images}
    if set(predictions) != registered_names or set(fusion_by_name) != registered_names:
        raise ValueError(f"scene {name} inventories differ")

    errors: dict[str, dict[str, list[np.ndarray]]] = {
        candidate: {"all": [], "center": [], "edge": []}
        for candidate in CANDIDATE_NAMES
    }
    fallback_counts = {candidate: 0 for candidate in CANDIDATE_NAMES}
    focal_ratios: dict[str, list[float]] = {"fx": [], "fy": []}
    image_records: list[dict[str, Any]] = []

    for image in sorted(images, key=lambda item: item.image_id):
        prediction = selected[image.name]
        transform, _ = validate_canvas_record(prediction)
        original_size = tuple(int(value) for value in prediction["original_size"])
        image_shape = tuple(int(value) for value in prediction["image_shape"])
        production = build_fusion_camera(
            colmap_camera=cameras[image.camera_id],
            original_size=original_size,
            image_shape=image_shape,
        )
        fusion_record = fusion_by_name[image.name]
        if not np.allclose(
            production.intrinsic,
            np.asarray(fusion_record["fusion_intrinsic"], dtype=np.float32),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(f"production intrinsic mismatch for {image.name}")
        candidates, fallback = build_candidate_cameras(
            production=production,
            vggt_intrinsic=prediction.get("intrinsic"),
            scale_x=transform.scale_x,
            scale_y=transform.scale_y,
        )
        for candidate, reason in fallback.items():
            fallback_counts[candidate] += int(reason is not None)
        if fallback["vggt"] is None:
            focal_ratios["fx"].append(
                float(candidates["vggt"].intrinsic[0, 0] / production.intrinsic[0, 0])
            )
            focal_ratios["fy"].append(
                float(candidates["vggt"].intrinsic[1, 1] / production.intrinsic[1, 1])
            )

        observations = [
            (x, y, point_id)
            for x, y, point_id in image.observations
            if point_id in points3d
        ]
        if not observations:
            continue
        world = np.stack([points3d[point_id] for _, _, point_id in observations])
        observed_x = np.asarray([item[0] for item in observations], dtype=np.float64)
        observed_y = np.asarray([item[1] for item in observations], dtype=np.float64)
        target_u = (observed_x + 0.5) * transform.scale_x - 0.5 + transform.pad_left
        target_v = (observed_y + 0.5) * transform.scale_y - 0.5 + transform.pad_top
        half_width = max((transform.resized_width - 1) / 2, 1.0)
        half_height = max((transform.resized_height - 1) / 2, 1.0)
        center_u = transform.pad_left + (transform.resized_width - 1) / 2
        center_v = transform.pad_top + (transform.resized_height - 1) / 2
        radius = np.maximum(
            np.abs(target_u - center_u) / half_width,
            np.abs(target_v - center_v) / half_height,
        )
        edge = radius >= EDGE_NORMALIZED_RADIUS
        valid_target = (
            np.isfinite(target_u)
            & np.isfinite(target_v)
            & (target_u >= transform.pad_left)
            & (target_u <= transform.pad_left + transform.resized_width - 1)
            & (target_v >= transform.pad_top)
            & (target_v <= transform.pad_top + transform.resized_height - 1)
        )
        per_image: dict[str, Any] = {}
        for candidate_name, candidate in candidates.items():
            u, v, depth = project_world_points_to_depth_canvas(
                world,
                camera=candidate,
                qvec=image.qvec,
                tvec=image.tvec,
            )
            valid = valid_target & np.isfinite(u) & np.isfinite(v) & (depth > 1e-6)
            candidate_errors = np.hypot(u[valid] - target_u[valid], v[valid] - target_v[valid])
            edge_errors = candidate_errors[edge[valid]]
            center_errors = candidate_errors[~edge[valid]]
            errors[candidate_name]["all"].append(candidate_errors)
            errors[candidate_name]["center"].append(center_errors)
            errors[candidate_name]["edge"].append(edge_errors)
            per_image[candidate_name] = {
                "fallback": fallback[candidate_name],
                "all": error_summary(candidate_errors),
                "edge": error_summary(edge_errors),
            }
        image_records.append(
            {
                "image": image.name,
                "image_id": image.image_id,
                "valid_sparse_point_count": per_image["production_colmap"]["all"]["count"],
                "candidates": per_image,
            }
        )

    candidate_summary: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        strata = {
            stratum: error_summary(
                np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
            )
            for stratum, parts in errors[candidate].items()
        }
        candidate_summary[candidate] = {
            "fallback_image_count": fallback_counts[candidate],
            "fallback_fraction": fallback_counts[candidate] / len(images),
            **strata,
        }
    baseline = candidate_summary["production_colmap"]
    for candidate in CANDIDATE_NAMES[1:]:
        summary = candidate_summary[candidate]
        summary["change_from_production"] = {
            "all_p50_fraction": relative_change(summary["all"]["p50"], baseline["all"]["p50"]),
            "all_p90_fraction": relative_change(summary["all"]["p90"], baseline["all"]["p90"]),
            "edge_p50_fraction": relative_change(summary["edge"]["p50"], baseline["edge"]["p50"]),
            "edge_p90_fraction": relative_change(summary["edge"]["p90"], baseline["edge"]["p90"]),
        }

    return {
        "name": name,
        "sources": {
            "job_dir": job_dir.as_posix(),
            "prediction_index": index_path.as_posix(),
            "prediction_index_sha256": sha256(index_path),
            "fusion_sha256": sha256(fusion_path),
            "colmap_files_sha256": {
                filename: sha256(sparse_dir / filename)
                for filename in ("cameras.txt", "images.txt", "points3D.txt")
            },
        },
        "inventory": {
            "image_count": len(images),
            "sparse_point_observation_count": candidate_summary["production_colmap"]["all"]["count"],
            "camera_models": sorted({camera["model"] for camera in cameras.values()}),
        },
        "vggt_to_production_focal_ratio": {
            axis: error_summary(np.asarray(values, dtype=np.float64))
            for axis, values in focal_ratios.items()
        },
        "candidates": candidate_summary,
        "images": image_records,
    }


def evaluate_intrinsics_candidates(
    scenes: list[tuple[str, Path, Path]],
) -> dict[str, Any]:
    if len({name for name, _, _ in scenes}) != len(scenes):
        raise ValueError("scene names must be unique")
    scene_records = [
        evaluate_scene(name=name, job_dir=job_dir, index_path=index_path)
        for name, job_dir, index_path in scenes
    ]
    decisions: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES[1:]:
        edge_reductions = [
            -scene["candidates"][candidate]["change_from_production"]["edge_p90_fraction"]
            for scene in scene_records
        ]
        all_p50_changes = [
            scene["candidates"][candidate]["change_from_production"]["all_p50_fraction"]
            for scene in scene_records
        ]
        fallback_fractions = [
            scene["candidates"][candidate]["fallback_fraction"]
            for scene in scene_records
        ]
        improved_count = sum(reduction > 0 for reduction in edge_reductions)
        median_reduction = float(np.median(edge_reductions))
        worst_all_p50_change = max(all_p50_changes)
        worst_fallback = max(fallback_fractions)
        decisions[candidate] = {
            "edge_p90_improved_scene_count": improved_count,
            "median_edge_p90_reduction_fraction": median_reduction,
            "worst_scene_all_p50_change_fraction": worst_all_p50_change,
            "maximum_fallback_fraction": worst_fallback,
            "opens_g1_20": bool(
                improved_count >= MIN_EDGE_P90_IMPROVED_SCENES
                and median_reduction >= MIN_CROSS_SCENE_MEDIAN_EDGE_P90_REDUCTION
                and worst_all_p50_change <= MAX_SCENE_ALL_P50_REGRESSION
                and worst_fallback <= MAX_FALLBACK_FRACTION
            ),
        }
    opened = [candidate for candidate, decision in decisions.items() if decision["opens_g1_20"]]
    return {
        "schema_version": 1,
        "evaluation": "g1_19_intrinsics_candidate_sparse_reprojection",
        "protocol": {
            "target_pixels": "COLMAP sparse observations mapped through PIL resize pixel-center semantics onto the retained VGGT canvas",
            "same_points_pose_distortion_canvas_for_all_candidates": True,
            "edge_normalized_radius_minimum": EDGE_NORMALIZED_RADIUS,
            "candidate_definitions": {
                "production_colmap": "current resized/padded COLMAP intrinsic",
                "pixel_center_colmap": "production COLMAP focal with principal point shifted by 0.5*(scale-1) per axis",
                "vggt": "retained first-wins VGGT intrinsic",
                "vggt_focal_colmap_center": "VGGT fx/fy with pixel-center-corrected COLMAP cx/cy",
            },
            "distortion": "frozen COLMAP radial coefficients for every candidate",
            "world_points_and_poses": "frozen reconstruction COLMAP sparse points and camera-from-world poses",
            "ground_truth_used": False,
        },
        "decision_gate": {
            "criteria": {
                "minimum_edge_p90_improved_scenes": MIN_EDGE_P90_IMPROVED_SCENES,
                "minimum_median_edge_p90_reduction_fraction": MIN_CROSS_SCENE_MEDIAN_EDGE_P90_REDUCTION,
                "maximum_scene_all_p50_regression_fraction": MAX_SCENE_ALL_P50_REGRESSION,
                "maximum_fallback_fraction": MAX_FALLBACK_FRACTION,
            },
            "candidates": decisions,
            "g1_20_status": "open_for_single_factor_ablation" if opened else "blocked",
            "opened_candidates": opened,
            "production_default_changed": False,
        },
        "scenes": scene_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        nargs=3,
        action="append",
        metavar=("NAME", "JOB_DIR", "INDEX"),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = evaluate_intrinsics_candidates(
        [(name, Path(job_dir), Path(index)) for name, job_dir, index in args.scene]
    )
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"intrinsics diagnostics differ: {args.output}")
        print(f"verified {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
