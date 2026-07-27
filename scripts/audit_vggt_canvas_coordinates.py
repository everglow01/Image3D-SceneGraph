#!/usr/bin/env python3
"""Audit VGGT canvas, COLMAP intrinsics, and projection coordinate conventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image3d_scenegraph.geometry.grouping import (
    parse_colmap_images_with_points,
    parse_colmap_points3d,
)
from run_colmap_sparse import parse_colmap_cameras
from run_colmap_vggt_dense import (
    build_fusion_camera,
    build_vggt_image_transform,
    project_world_points_to_depth_canvas,
    unproject_depth_pixels_with_colmap_pose,
)


ROUND_TRIP_PIXEL_THRESHOLD = 1e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def unique_prediction_records(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    coordinate_keys = ("original_size", "image_shape", "canvas_transform")
    for record in index["predictions"]:
        image_name = str(record["image"])
        previous = records.setdefault(image_name, record)
        if any(previous[key] != record[key] for key in coordinate_keys):
            raise ValueError(f"inconsistent retained canvas metadata for {image_name}")
    return records


def validate_canvas_record(record: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    original_size = tuple(int(value) for value in record["original_size"])
    image_shape = tuple(int(value) for value in record["image_shape"])
    transform = build_vggt_image_transform(original_size, image_shape)
    expected = {
        "scale_x": transform.scale_x,
        "scale_y": transform.scale_y,
        "pad_left": transform.pad_left,
        "pad_top": transform.pad_top,
        "pad_right": image_shape[1] - transform.pad_left - transform.resized_width,
        "pad_bottom": image_shape[0] - transform.pad_top - transform.resized_height,
        "resized_width": transform.resized_width,
        "resized_height": transform.resized_height,
    }
    actual = record["canvas_transform"]
    for key, value in expected.items():
        if isinstance(value, float):
            if not np.isclose(float(actual[key]), value, atol=1e-12, rtol=0):
                raise ValueError(f"canvas {key} mismatch for {record['image']}")
        elif int(actual[key]) != value:
            raise ValueError(f"canvas {key} mismatch for {record['image']}")

    image_path = Path(record["image_path"])
    with Image.open(image_path) as image:
        if image.size != original_size:
            raise ValueError(f"source image size mismatch for {record['image']}")

    # PIL/torchvision resize samples pixel centers. The production calibration path
    # currently uses the simpler x' = scale*x + pad convention; report, do not hide,
    # the resulting constant offset before G1.19 compares intrinsics candidates.
    return transform, {
        "x": 0.5 * (1.0 - transform.scale_x),
        "y": 0.5 * (1.0 - transform.scale_y),
    }


def audit_canvas_coordinates(
    *,
    job_dir: Path,
    index_path: Path,
    round_trip_pixel_threshold: float = ROUND_TRIP_PIXEL_THRESHOLD,
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
        raise FileNotFoundError(f"missing audit inputs: {missing}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not index.get("capture_enabled"):
        raise ValueError("unsupported or disabled VGGT prediction index")
    prediction_by_name = unique_prediction_records(index)
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    fusion_by_name = {record["image"]: record for record in fusion["images"]}
    cameras = {
        camera["camera_id"]: camera
        for camera in parse_colmap_cameras(sparse_dir / "cameras.txt")
    }
    images = parse_colmap_images_with_points(sparse_dir / "images.txt")
    points3d = parse_colmap_points3d(sparse_dir / "points3D.txt")
    registered_names = {image.name for image in images}
    if set(prediction_by_name) != registered_names or set(fusion_by_name) != registered_names:
        raise ValueError("capture, fusion, and COLMAP registered-image inventories differ")

    pixel_round_trip_errors: list[float] = []
    world_round_trip_relative_errors: list[float] = []
    sparse_observation_original_errors: list[float] = []
    sparse_observation_canvas_errors: list[float] = []
    pil_center_offsets_x: list[float] = []
    pil_center_offsets_y: list[float] = []
    image_records: list[dict[str, Any]] = []

    for image in sorted(images, key=lambda item: item.image_id):
        prediction = prediction_by_name[image.name]
        transform, pil_offset = validate_canvas_record(prediction)
        pil_center_offsets_x.append(abs(pil_offset["x"]))
        pil_center_offsets_y.append(abs(pil_offset["y"]))
        original_size = tuple(int(value) for value in prediction["original_size"])
        image_shape = tuple(int(value) for value in prediction["image_shape"])
        camera = build_fusion_camera(
            colmap_camera=cameras[image.camera_id],
            original_size=original_size,
            image_shape=image_shape,
        )
        fusion_record = fusion_by_name[image.name]
        if not np.allclose(
            camera.intrinsic,
            np.asarray(fusion_record["fusion_intrinsic"], dtype=np.float32),
            atol=1e-6,
            rtol=0,
        ) or tuple(camera.radial_distortion) != tuple(fusion_record["radial_distortion"]):
            raise ValueError(f"fusion camera metadata mismatch for {image.name}")

        observations = [
            (x, y, point_id)
            for x, y, point_id in image.observations
            if point_id in points3d
        ]
        if not observations:
            image_records.append({"image": image.name, "sparse_point_count": 0})
            continue
        world = np.stack([points3d[point_id] for _, _, point_id in observations])
        u, v, depth = project_world_points_to_depth_canvas(
            world,
            camera=camera,
            qvec=image.qvec,
            tvec=image.tvec,
        )
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth) & (depth > 1e-6)
        if not np.any(valid):
            image_records.append({"image": image.name, "sparse_point_count": 0})
            continue
        world = world[valid]
        u = u[valid]
        v = v[valid]
        depth = depth[valid]
        recovered_world = unproject_depth_pixels_with_colmap_pose(
            depth=depth,
            u=u,
            v=v,
            camera=camera,
            qvec=image.qvec,
            tvec=image.tvec,
        )
        round_u, round_v, _ = project_world_points_to_depth_canvas(
            recovered_world,
            camera=camera,
            qvec=image.qvec,
            tvec=image.tvec,
        )
        pixel_errors = np.hypot(round_u - u, round_v - v)
        world_errors = np.linalg.norm(recovered_world - world, axis=1) / np.maximum(
            np.linalg.norm(world, axis=1), 1.0
        )
        observed_x = np.asarray([item[0] for item in observations], dtype=np.float64)[valid]
        observed_y = np.asarray([item[1] for item in observations], dtype=np.float64)[valid]
        projected_original_x = (u - transform.pad_left) / transform.scale_x
        projected_original_y = (v - transform.pad_top) / transform.scale_y
        original_errors = np.hypot(
            projected_original_x - observed_x,
            projected_original_y - observed_y,
        )
        canvas_errors = np.hypot(
            (projected_original_x - observed_x) * transform.scale_x,
            (projected_original_y - observed_y) * transform.scale_y,
        )
        pixel_round_trip_errors.extend(pixel_errors.astype(float).tolist())
        world_round_trip_relative_errors.extend(world_errors.astype(float).tolist())
        sparse_observation_original_errors.extend(original_errors.astype(float).tolist())
        sparse_observation_canvas_errors.extend(canvas_errors.astype(float).tolist())
        image_records.append(
            {
                "image": image.name,
                "camera_id": image.camera_id,
                "camera_model": camera.model,
                "original_size": list(original_size),
                "canvas_shape": list(image_shape),
                "resize": {
                    "width": transform.resized_width,
                    "height": transform.resized_height,
                    "scale_x": transform.scale_x,
                    "scale_y": transform.scale_y,
                },
                "padding": {
                    "left": transform.pad_left,
                    "top": transform.pad_top,
                    "right": image_shape[1] - transform.pad_left - transform.resized_width,
                    "bottom": image_shape[0] - transform.pad_top - transform.resized_height,
                },
                "principal_point": {
                    "production_canvas": [
                        float(camera.intrinsic[0, 2]),
                        float(camera.intrinsic[1, 2]),
                    ],
                    "pil_pixel_center_equivalent": [
                        float(camera.intrinsic[0, 2] - pil_offset["x"]),
                        float(camera.intrinsic[1, 2] - pil_offset["y"]),
                    ],
                    "production_minus_pil_pixels": [pil_offset["x"], pil_offset["y"]],
                },
                "sparse_point_count": int(np.count_nonzero(valid)),
                "pixel_round_trip_max": float(np.max(pixel_errors)),
                "sparse_observation_original_p90": float(np.percentile(original_errors, 90)),
            }
        )

    pixel_summary = distribution(pixel_round_trip_errors)
    passed = bool(
        pixel_summary["count"]
        and pixel_summary["max"] is not None
        and pixel_summary["max"] <= round_trip_pixel_threshold
    )
    return {
        "schema_version": 1,
        "audit": "g1_18_vggt_canvas_intrinsics_coordinates",
        "sources": {
            "job_dir": job_dir.as_posix(),
            "prediction_index": index_path.as_posix(),
            "prediction_index_sha256": sha256(index_path),
            "fusion": fusion_path.as_posix(),
            "fusion_sha256": sha256(fusion_path),
            "colmap_model": sparse_dir.as_posix(),
            "colmap_files_sha256": {
                name: sha256(sparse_dir / name)
                for name in ("cameras.txt", "images.txt", "points3D.txt")
            },
        },
        "coordinate_convention": {
            "original_observation": "COLMAP image coordinates (x right, y down)",
            "resize_and_padding": "integer resize divisible by 14, then symmetric white padding to VGGT canvas",
            "production_mapping": "u_canvas = scale_x*x_original + pad_left; v_canvas = scale_y*y_original + pad_top",
            "principal_point_mapping": "cx_canvas = scale_x*cx_original + pad_left; cy_canvas = scale_y*cy_original + pad_top",
            "distortion": "COLMAP radial distortion is applied for projection and iteratively inverted for backprojection",
            "depth": "positive OpenCV camera Z in COLMAP camera-from-world coordinates",
            "world_transform": "X_camera = R(qvec)*X_world + tvec",
            "pixel_center_audit": "PIL/torchvision resize uses center sampling: (x+0.5)*scale-0.5+pad; production omits this constant subpixel correction",
            "production_minus_pil_pixel_center_offset": {
                "x_absolute": distribution(pil_center_offsets_x),
                "y_absolute": distribution(pil_center_offsets_y),
            },
        },
        "inventory": {
            "registered_image_count": len(images),
            "captured_prediction_count": int(index["prediction_count"]),
            "unique_captured_image_count": len(prediction_by_name),
            "camera_models": sorted({camera["model"] for camera in cameras.values()}),
        },
        "round_trip_gate": {
            "pixel_threshold": round_trip_pixel_threshold,
            "sparse_point_count": len(pixel_round_trip_errors),
            "pixel_error": pixel_summary,
            "world_relative_error": distribution(world_round_trip_relative_errors),
            "passed": passed,
        },
        "sparse_observation_reprojection": {
            "role": "reported sanity distribution only; includes COLMAP bundle-adjustment residual and is not the implementation round-trip gate",
            "original_pixel_error": distribution(sparse_observation_original_errors),
            "canvas_pixel_error": distribution(sparse_observation_canvas_errors),
        },
        "conclusion": {
            "production_coordinate_chain_is_self_consistent": passed,
            "pixel_center_convention_requires_candidate_evaluation": bool(
                max(pil_center_offsets_x + pil_center_offsets_y, default=0.0) > 0
            ),
            "production_fusion_changed": False,
        },
        "images": image_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    index_path = args.index or args.job_dir / "diagnostics" / "vggt_window_predictions.json"
    payload = audit_canvas_coordinates(job_dir=args.job_dir, index_path=index_path)
    if not payload["round_trip_gate"]["passed"]:
        raise SystemExit("projection/backprojection round-trip exceeded fixed pixel threshold")
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"canvas audit differs: {args.output}")
        print(f"verified {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
