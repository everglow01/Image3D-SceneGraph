#!/usr/bin/env python3
"""Replay frozen VGGT depths with one diagnostic dense-fusion intrinsics candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from audit_vggt_canvas_coordinates import validate_canvas_record
from image3d_scenegraph.geometry.grouping import (
    build_covisibility_graph,
    parse_colmap_images_with_points,
)
from run_colmap_sparse import parse_colmap_cameras
from run_colmap_vggt_dense import (
    FusionCamera,
    FusionFrame,
    apply_point_budget,
    build_consistency_payload,
    build_fusion_camera,
    filter_points_by_cross_view_consistency,
    point_budget_diagnostics,
    write_support_point_diagnostics,
)
from run_vggt_pointcloud import load_padded_rgb_images, write_json, write_ply


CANDIDATES = ("production_colmap", "pixel_center_colmap")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_candidate_camera(
    production: FusionCamera,
    *,
    scale_x: float,
    scale_y: float,
    candidate: str,
) -> FusionCamera:
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported intrinsics candidate: {candidate}")
    intrinsic = production.intrinsic.copy()
    if candidate == "pixel_center_colmap":
        intrinsic[0, 2] += 0.5 * (scale_x - 1.0)
        intrinsic[1, 2] += 0.5 * (scale_y - 1.0)
    return FusionCamera(production.model, intrinsic, production.radial_distortion)


def selected_prediction_records(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in index.get("predictions", []):
        if not record.get("selected_for_first_wins"):
            continue
        image_name = str(record["image"])
        if image_name in selected:
            raise ValueError(f"multiple first-wins predictions for {image_name}")
        selected[image_name] = record
    if len(selected) != int(index.get("unique_image_count", len(selected))):
        raise ValueError("first-wins prediction inventory is incomplete")
    return selected


def read_runner_log(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def replay_frozen_intrinsics_ablation(
    *,
    source_dir: Path,
    index_path: Path,
    output_dir: Path,
    candidate: str,
    support_diagnostics: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_dir = source_dir.resolve()
    index_path = index_path.resolve()
    output_dir = output_dir.resolve()
    sparse_dir = source_dir / "colmap_vggt" / "sparse_txt"
    fusion_path = source_dir / "diagnostics" / "fusion.json"
    consistency_path = source_dir / "diagnostics" / "consistency.json"
    cameras_path = source_dir / "geometry" / "cameras.json"
    run_log_path = source_dir / "logs" / "run.log"
    required = [
        index_path,
        fusion_path,
        consistency_path,
        cameras_path,
        run_log_path,
        sparse_dir / "cameras.txt",
        sparse_dir / "images.txt",
        sparse_dir / "points3D.txt",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing G1.20 frozen inputs: {missing}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not index.get("capture_enabled"):
        raise ValueError("unsupported or disabled VGGT prediction index")
    selected = selected_prediction_records(index)
    records_by_name: dict[str, list[dict[str, Any]]] = {}
    for record in index["predictions"]:
        records_by_name.setdefault(str(record["image"]), []).append(record)
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    run_log = read_runner_log(run_log_path)
    if fusion.get("fusion_mode") != "points" or consistency.get("fusion_mode") != "points":
        raise ValueError("G1.20 replay supports only frozen points-mode captures")
    if fusion.get("depth_scale_mode") != "per_frame":
        raise ValueError("G1.20 replay requires frozen per-frame depth scales")

    images = parse_colmap_images_with_points(sparse_dir / "images.txt")
    images_by_name = {image.name: image for image in images}
    cameras = {
        camera["camera_id"]: camera
        for camera in parse_colmap_cameras(sparse_dir / "cameras.txt")
    }
    fusion_records = fusion.get("images", [])
    fusion_by_name = {record["image"]: record for record in fusion_records}
    consistency_by_name = {
        record["image"]: record for record in consistency.get("images", [])
    }
    inventories = [set(selected), set(images_by_name), set(fusion_by_name), set(consistency_by_name)]
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise ValueError("capture, COLMAP, fusion, and consistency inventories differ")

    prediction_dir = index_path.parent / "vggt_window_predictions"
    frames: list[FusionFrame] = []
    confidence_thresholds: dict[int, float] = {}
    camera_deltas: list[dict[str, float | str]] = []
    prediction_hashes: list[dict[str, Any]] = []
    for fusion_record in fusion_records:
        image_name = str(fusion_record["image"])
        record = selected[image_name]
        image = images_by_name[image_name]
        transform, _ = validate_canvas_record(record)
        original_size = tuple(int(value) for value in record["original_size"])
        image_shape = tuple(int(value) for value in record["image_shape"])
        production = build_fusion_camera(
            colmap_camera=cameras[image.camera_id],
            original_size=original_size,
            image_shape=image_shape,
        )
        frozen_intrinsic = np.asarray(fusion_record["fusion_intrinsic"], dtype=np.float32)
        if not np.array_equal(production.intrinsic, frozen_intrinsic):
            raise ValueError(f"frozen production intrinsic mismatch for {image_name}")
        camera = build_candidate_camera(
            production,
            scale_x=transform.scale_x,
            scale_y=transform.scale_y,
            candidate=candidate,
        )
        prediction_path = prediction_dir / str(record["prediction_file"])
        if not prediction_path.is_file() or prediction_path.stat().st_size != int(record["file_bytes"]):
            raise ValueError(f"retained prediction file mismatch for {image_name}")
        with np.load(prediction_path) as payload:
            depth = np.asarray(payload["depth"], dtype=np.float32)
            confidence = np.asarray(payload["confidence"], dtype=np.float32)
        overlap_disagreement = None
        if support_diagnostics and len(records_by_name[image_name]) > 1:
            first_scaled = depth * float(fusion_record["depth_scale"])
            overlap_disagreement = np.full(depth.shape, np.nan, dtype=np.float32)
            for other_record in records_by_name[image_name]:
                if other_record is record:
                    continue
                other_path = prediction_dir / str(other_record["prediction_file"])
                with np.load(other_path) as other_payload:
                    other_depth = np.asarray(other_payload["depth"], dtype=np.float32)
                anchor = other_record.get("sparse_scale_anchor")
                if anchor is None:
                    continue
                other_scaled = other_depth * float(anchor["scale"])
                valid = (
                    np.isfinite(first_scaled)
                    & (first_scaled > 0)
                    & np.isfinite(other_scaled)
                    & (other_scaled > 0)
                )
                disagreement = np.full(depth.shape, np.nan, dtype=np.float32)
                disagreement[valid] = np.abs(
                    np.log(first_scaled[valid]) - np.log(other_scaled[valid])
                )
                overlap_disagreement = np.fmax(overlap_disagreement, disagreement)
        if depth.shape != image_shape or confidence.shape != image_shape:
            raise ValueError(f"retained prediction shape mismatch for {image_name}")
        image_path = Path(str(record["image_path"]))
        colors = load_padded_rgb_images([image_path], image_shape)[0]
        frames.append(
            FusionFrame(
                image_path=image_path,
                colmap_image=image,
                camera=camera,
                depth=depth,
                confidence=confidence,
                colors=colors,
                scale=float(fusion_record["depth_scale"]),
                image_shape=image_shape,
                original_size=original_size,
                source_group_index=int(record["group_index"]),
                source_group_position=int(record["group_position"]),
                source_window_role=str(record["role"]),
                scale_observations=int(fusion_record["scale_observations"]),
                scale_log_mad=(
                    float(fusion_record["scale_log_mad"])
                    if fusion_record["scale_log_mad"] is not None
                    else float("nan")
                ),
                overlap_disagreement=overlap_disagreement,
            )
        )
        confidence_thresholds[image.image_id] = float(
            consistency_by_name[image_name]["confidence_threshold"]
        )
        camera_deltas.append(
            {
                "image": image_name,
                "cx": float(camera.intrinsic[0, 2] - production.intrinsic[0, 2]),
                "cy": float(camera.intrinsic[1, 2] - production.intrinsic[1, 2]),
            }
        )
        prediction_hashes.append(
            {
                "image": image_name,
                "path": prediction_path.as_posix(),
                "sha256": sha256(prediction_path),
            }
        )

    cross_view = fusion.get("cross_view_filter")
    if not isinstance(cross_view, dict):
        raise ValueError("frozen cross-view filter diagnostics are missing")
    graph = build_covisibility_graph(
        [frame.colmap_image for frame in frames],
        max_neighbors=int(cross_view["neighbors"]),
        min_shared_points=int(cross_view["min_shared_points"]),
    )
    filtered = filter_points_by_cross_view_consistency(
        frames,
        covisibility_graph=graph,
        confidence_thresholds=confidence_thresholds,
        relative_threshold=float(consistency["relative_threshold"]),
        support_policy=str(consistency["support_policy"]),
        stride=int(consistency["stride"]),
        retain_point_diagnostics=support_diagnostics,
    )
    frozen_budget = fusion["point_budget"]
    budget = apply_point_budget(
        filtered.points,
        filtered.colors,
        int(run_log["max_points"]),
        int(run_log.get("seed", 42)),
        policy=str(frozen_budget["policy"]),
    )
    if not len(budget.points):
        raise RuntimeError("G1.20 replay produced no points")

    geometry_dir = output_dir / "geometry"
    diagnostics_dir = output_dir / "diagnostics"
    logs_dir = output_dir / "logs"
    geometry_dir.mkdir(parents=True)
    diagnostics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    points_path = geometry_dir / "points.ply"
    write_ply(points_path, budget.points, budget.colors)
    support_summary = None
    if support_diagnostics:
        if filtered.point_diagnostics is None:
            raise RuntimeError("Per-point support diagnostics were not retained")
        support_summary = write_support_point_diagnostics(
            diagnostics_dir / "support_points.npz",
            diagnostics=filtered.point_diagnostics,
            selected_indices=budget.selected_indices,
            frames=frames,
            expected_point_count=len(budget.points),
            source_index_path=index_path,
        )
    shutil.copy2(cameras_path, geometry_dir / "cameras.json")
    consistency_payload = {
        "fusion_mode": "points",
        **build_consistency_payload(
            filtered,
            confidence_thresholds=confidence_thresholds,
            confidence_percentile=float(consistency["confidence_percentile"]),
            confidence_threshold_scope=str(consistency["confidence_threshold_scope"]),
            support_policy=str(consistency["support_policy"]),
            relative_threshold=float(consistency["relative_threshold"]),
            stride=int(consistency["stride"]),
        ),
    }
    write_json(diagnostics_dir / "consistency.json", consistency_payload)

    cx = np.asarray([record["cx"] for record in camera_deltas], dtype=np.float64)
    cy = np.asarray([record["cy"] for record in camera_deltas], dtype=np.float64)
    payload = {
        "schema_version": 1,
        "evaluation": "g1_20_frozen_dense_intrinsics_ablation",
        "candidate": candidate,
        "protocol": {
            "prediction_policy": "retained_first_wins_only",
            "inference_rerun": False,
            "frozen": [
                "depth",
                "confidence",
                "rgb",
                "COLMAP pose and radial distortion",
                "per-frame scale",
                "covisibility graph settings",
                "confidence thresholds",
                "support policy",
                "relative threshold",
                "stride",
                "point budget policy and cap",
                "seed",
            ],
            "only_changed_factor": "fusion camera cx/cy" if candidate == "pixel_center_colmap" else None,
            "ground_truth_used_for_reconstruction": False,
            "production_default_changed": False,
        },
        "sources": {
            "source_dir": source_dir.as_posix(),
            "prediction_index": index_path.as_posix(),
            "prediction_index_sha256": sha256(index_path),
            "fusion_sha256": sha256(fusion_path),
            "consistency_sha256": sha256(consistency_path),
            "run_log_sha256": sha256(run_log_path),
            "colmap_files_sha256": {
                name: sha256(sparse_dir / name)
                for name in ("cameras.txt", "images.txt", "points3D.txt")
            },
            "selected_predictions": prediction_hashes,
        },
        "inventory": {
            "image_count": len(frames),
            "selected_prediction_count": len(prediction_hashes),
        },
        "principal_point_delta_pixels": {
            "cx_min": float(np.min(cx)),
            "cx_median": float(np.median(cx)),
            "cx_max": float(np.max(cx)),
            "cy_min": float(np.min(cy)),
            "cy_median": float(np.median(cy)),
            "cy_max": float(np.max(cy)),
        },
        "filter": {
            "candidate_points": filtered.candidate_points,
            "accepted_points": filtered.accepted_points,
            "rejected_points": filtered.rejected_points,
            "unverified_points": filtered.unverified_points,
            "supported_points": filtered.supported_points,
            "multi_visible_points": filtered.multi_visible_points,
            "residual_p50": consistency_payload["residual_p50"],
            "residual_p90": consistency_payload["residual_p90"],
        },
        "point_budget": point_budget_diagnostics(budget),
        "support_point_diagnostics": support_summary,
        "output": {
            "points": points_path.as_posix(),
            "points_sha256": sha256(points_path),
            "point_count": int(len(budget.points)),
        },
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    write_json(diagnostics_dir / "g1_20_ablation.json", payload)
    (logs_dir / "run.log").write_text(
        "\n".join(
            [
                "evaluation=g1_20_frozen_dense_intrinsics_ablation",
                f"candidate={candidate}",
                "inference_rerun=false",
                f"num_images={len(frames)}",
                f"candidate_points={filtered.candidate_points}",
                f"accepted_points={filtered.accepted_points}",
                f"num_points={len(budget.points)}",
                f"residual_p50={consistency_payload['residual_p50']:.9f}",
                f"residual_p90={consistency_payload['residual_p90']:.9f}",
                f"elapsed_seconds={payload['elapsed_seconds']:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--support-diagnostics", action="store_true")
    args = parser.parse_args()
    payload = replay_frozen_intrinsics_ablation(
        source_dir=args.source_dir,
        index_path=args.index,
        output_dir=args.output_dir,
        candidate=args.candidate,
        support_diagnostics=args.support_diagnostics,
    )
    print(f"wrote {payload['output']['points']}")
    print(f"num_points={payload['output']['point_count']}")


if __name__ == "__main__":
    main()
