from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_colmap_sparse import build_camera_payload, discover_images, parse_colmap_cameras, run_command
from run_vggt_pointcloud import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_VGGT_REPO_DIR,
    infer_image_group,
    load_padded_rgb_images,
    load_vggt_model,
    select_device,
    select_dtype,
    validate_local_vggt,
    write_json,
    write_ply,
)


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    observations: list[tuple[float, float, int]]


@dataclass(frozen=True)
class DepthScaleEstimate:
    scale: float
    observation_count: int
    log_mad: float


@dataclass(frozen=True)
class VggtImageTransform:
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int


@dataclass(frozen=True)
class FusionCamera:
    model: str
    intrinsic: np.ndarray
    radial_distortion: tuple[float, ...]


@dataclass(frozen=True)
class FusionFrame:
    image_path: Path
    colmap_image: ColmapImage
    camera: FusionCamera
    depth: np.ndarray
    confidence: np.ndarray
    colors: np.ndarray
    scale: float
    image_shape: tuple[int, int]
    original_size: tuple[int, int]


@dataclass(frozen=True)
class CovisibilityEdge:
    target_image_id: int
    shared_points: int
    baseline: float


@dataclass(frozen=True)
class ConsistencyFilterResult:
    points: np.ndarray
    colors: np.ndarray
    candidate_points: int
    accepted_points: int
    rejected_points: int
    unverified_points: int
    supported_points: int
    residual_samples: np.ndarray
    image_records: list[dict[str, Any]]


@dataclass(frozen=True)
class CrossViewValidation:
    accepted: np.ndarray
    support_counts: np.ndarray
    visible_counts: np.ndarray
    occluded_counts: np.ndarray
    mean_relative_error: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse VGGT dense depth with COLMAP global camera poses.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_VGGT_REPO_DIR)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="exhaustive")
    parser.add_argument("--vggt-batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--conf-percentile", type=float, default=50.0)
    parser.add_argument("--min-scale-observations", type=int, default=20)
    parser.add_argument("--consistency-neighbors", type=int, default=6)
    parser.add_argument("--consistency-min-shared-points", type=int, default=20)
    parser.add_argument("--consistency-relative-threshold", type=float, default=0.08)
    parser.add_argument("--consistency-min-relative-threshold", type=float, default=0.02)
    parser.add_argument("--consistency-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.consistency_neighbors <= 0:
        raise SystemExit("--consistency-neighbors must be positive")
    if args.consistency_min_shared_points <= 0:
        raise SystemExit("--consistency-min-shared-points must be positive")
    if not 0 < args.consistency_relative_threshold < 1:
        raise SystemExit("--consistency-relative-threshold must be between 0 and 1")
    if not 0 < args.consistency_min_relative_threshold <= args.consistency_relative_threshold:
        raise SystemExit(
            "--consistency-min-relative-threshold must be positive and no larger than "
            "--consistency-relative-threshold"
        )
    if args.consistency_stride <= 0:
        raise SystemExit("--consistency-stride must be positive")

    started_at = time.perf_counter()
    colmap = shutil.which("colmap")
    if colmap is None:
        raise SystemExit("COLMAP executable not found. Install COLMAP and ensure `colmap` is on PATH.")

    image_paths = discover_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    validate_local_vggt(args.repo_dir, args.checkpoint_dir)
    sys.path.insert(0, str(args.repo_dir.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = args.output_dir
    geometry_dir = output_dir / "geometry"
    logs_dir = output_dir / "logs"
    work_dir = output_dir / "colmap_vggt"
    sparse_dir = work_dir / "sparse"
    text_dir = work_dir / "sparse_txt"
    database_path = work_dir / "database.db"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    colmap_started_at = time.perf_counter()
    colmap_logs = run_colmap_pipeline(
        colmap=colmap,
        image_dir=args.image_dir,
        database_path=database_path,
        sparse_dir=sparse_dir,
        text_dir=text_dir,
        matcher=args.matcher,
    )
    colmap_seconds = time.perf_counter() - colmap_started_at

    colmap_images = parse_colmap_images_with_points(text_dir / "images.txt")
    points3d = parse_colmap_points3d(text_dir / "points3D.txt")
    colmap_cameras = {
        camera["camera_id"]: camera for camera in parse_colmap_cameras(text_dir / "cameras.txt")
    }
    registered_by_name = {image.name: image for image in colmap_images}
    registered_paths = [path for path in image_paths if path.name in registered_by_name]
    if not registered_paths:
        raise RuntimeError("COLMAP did not register any input images")

    device = select_device(args.device)
    dtype = select_dtype(device, args.precision)
    model = load_vggt_model(
        model_cls=VGGT,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        dtype=dtype,
        enable_point=False,
    )
    model.eval()

    vggt_started_at = time.perf_counter()
    depth_items = run_vggt_depth_batches(
        model=model,
        image_paths=registered_paths,
        load_and_preprocess_images=load_and_preprocess_images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        device=device,
        dtype=dtype,
        batch_size=args.vggt_batch_size,
    )
    vggt_seconds = time.perf_counter() - vggt_started_at

    scale_estimates: dict[str, DepthScaleEstimate] = {}
    for image_path, item in depth_items.items():
        colmap_image = registered_by_name[image_path.name]
        estimate = estimate_depth_scale(
            colmap_image=colmap_image,
            points3d=points3d,
            depth=item["depth"],
            image_shape=item["image_shape"],
            original_size=item["original_size"],
            min_observations=args.min_scale_observations,
        )
        if estimate is not None:
            scale_estimates[image_path.name] = estimate
    if not scale_estimates:
        raise RuntimeError("Could not estimate VGGT-to-COLMAP depth scale from sparse observations")
    fallback_scale = float(np.median([estimate.scale for estimate in scale_estimates.values()]))
    consistency_relative_threshold = derive_consistency_relative_threshold(
        scale_estimates.values(),
        min_threshold=args.consistency_min_relative_threshold,
        max_threshold=args.consistency_relative_threshold,
    )

    used_scales: list[float] = []
    fusion_camera_records: list[dict[str, Any]] = []
    fusion_frames: list[FusionFrame] = []
    for image_path, item in depth_items.items():
        colmap_image = registered_by_name[image_path.name]
        colmap_camera = colmap_cameras.get(colmap_image.camera_id)
        if colmap_camera is None:
            raise RuntimeError(f"COLMAP camera {colmap_image.camera_id} missing for {image_path.name}")
        fusion_camera = build_fusion_camera(
            colmap_camera=colmap_camera,
            original_size=item["original_size"],
            image_shape=item["image_shape"],
        )
        estimate = scale_estimates.get(image_path.name)
        scale = estimate.scale if estimate is not None else fallback_scale
        used_scales.append(scale)
        fusion_frames.append(
            FusionFrame(
                image_path=image_path,
                colmap_image=colmap_image,
                camera=fusion_camera,
                depth=item["depth"],
                confidence=item["confidence"],
                colors=item["colors"],
                scale=scale,
                image_shape=item["image_shape"],
                original_size=item["original_size"],
            )
        )
        fusion_camera_records.append(
            build_fusion_camera_record(
                image_name=image_path.name,
                camera_id=colmap_image.camera_id,
                camera=fusion_camera,
                vggt_intrinsic=item["intrinsic"],
                scale_estimate=estimate,
                fallback_scale=fallback_scale,
            )
        )

    confidence_threshold = compute_global_confidence_threshold(fusion_frames, args.conf_percentile)
    covisibility_graph = build_covisibility_graph(
        fusion_frames,
        max_neighbors=args.consistency_neighbors,
        min_shared_points=args.consistency_min_shared_points,
    )
    filtered = filter_points_by_cross_view_consistency(
        fusion_frames,
        covisibility_graph=covisibility_graph,
        confidence_threshold=confidence_threshold,
        relative_threshold=consistency_relative_threshold,
        stride=args.consistency_stride,
    )
    flat_points, flat_colors = cap_points(filtered.points, filtered.colors, args.max_points, args.seed)
    if len(flat_points) == 0:
        raise RuntimeError("Cross-view consistency rejected every dense point")

    write_ply(geometry_dir / "points.ply", flat_points, flat_colors)
    write_json(geometry_dir / "cameras.json", build_camera_payload(text_dir))
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    visibility_graph_path = diagnostics_dir / "visibility_graph.json"
    consistency_path = diagnostics_dir / "consistency.json"
    fusion_diagnostics_path = diagnostics_dir / "fusion.json"
    write_json(
        visibility_graph_path,
        build_visibility_graph_payload(fusion_frames, covisibility_graph),
    )
    write_json(
        consistency_path,
        build_consistency_payload(
            filtered,
            confidence_threshold=confidence_threshold,
            relative_threshold=consistency_relative_threshold,
            stride=args.consistency_stride,
        ),
    )
    write_json(
        fusion_diagnostics_path,
        {
            "intrinsics_source": "colmap",
            "depth_source": "vggt",
            "registered_images": len(registered_paths),
            "scale_fallback": fallback_scale,
            "camera_models": sorted({record["colmap_model"] for record in fusion_camera_records}),
            "cross_view_filter": {
                "confidence_threshold": confidence_threshold,
                "neighbors": args.consistency_neighbors,
                "min_shared_points": args.consistency_min_shared_points,
                "relative_threshold": consistency_relative_threshold,
                "relative_threshold_cap": args.consistency_relative_threshold,
                "min_relative_threshold": args.consistency_min_relative_threshold,
                "stride": args.consistency_stride,
            },
            "images": fusion_camera_records,
        },
    )

    elapsed_seconds = time.perf_counter() - started_at
    log_lines = [
        "backend=colmap_vggt",
        f"num_images={len(image_paths)}",
        f"registered_images={len(registered_paths)}",
        f"num_points={len(flat_points)}",
        f"colmap_points={len(points3d)}",
        f"scaled_images={len(scale_estimates)}",
        f"scale_median={float(np.median(used_scales)):.6f}",
        f"scale_min={float(np.min(used_scales)):.6f}",
        f"scale_max={float(np.max(used_scales)):.6f}",
        f"scale_observations_median={float(np.median([estimate.observation_count for estimate in scale_estimates.values()])):.3f}",
        f"scale_log_mad_median={float(np.median([estimate.log_mad for estimate in scale_estimates.values()])):.6f}",
        "fusion_intrinsics=colmap",
        f"fusion_diagnostics={fusion_diagnostics_path.relative_to(output_dir).as_posix()}",
        f"visibility_graph={visibility_graph_path.relative_to(output_dir).as_posix()}",
        f"consistency_diagnostics={consistency_path.relative_to(output_dir).as_posix()}",
        f"consistency_confidence_threshold={confidence_threshold:.6f}",
        f"consistency_candidates={filtered.candidate_points}",
        f"consistency_accepted={filtered.accepted_points}",
        f"consistency_rejected={filtered.rejected_points}",
        f"consistency_unverified={filtered.unverified_points}",
        f"consistency_supported={filtered.supported_points}",
        f"consistency_acceptance_rate={filtered.accepted_points / max(filtered.candidate_points, 1):.6f}",
        f"consistency_relative_threshold={consistency_relative_threshold:.6f}",
        f"consistency_residual_p50={percentile_or_zero(filtered.residual_samples, 50):.6f}",
        f"consistency_residual_p90={percentile_or_zero(filtered.residual_samples, 90):.6f}",
        f"consistency_stride={args.consistency_stride}",
        f"matcher={args.matcher}",
        f"vggt_batch_size={args.vggt_batch_size}",
        f"conf_percentile={args.conf_percentile}",
        f"max_points={args.max_points}",
        f"colmap_seconds={colmap_seconds:.3f}",
        f"vggt_seconds={vggt_seconds:.3f}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        *colmap_logs,
    ]
    (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {geometry_dir / 'points.ply'}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
    print(f"wrote {fusion_diagnostics_path}")
    print(f"wrote {visibility_graph_path}")
    print(f"wrote {consistency_path}")
    print(f"registered_images={len(registered_paths)}")
    print(f"scaled_images={len(scale_estimates)}")
    print(f"num_points={len(flat_points)}")


def run_colmap_pipeline(
    *,
    colmap: str,
    image_dir: Path,
    database_path: Path,
    sparse_dir: Path,
    text_dir: Path,
    matcher: str,
) -> list[str]:
    commands = [
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            "1",
        ],
        [
            colmap,
            "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "1",
        ],
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
        ],
    ]
    command_logs = [run_command(command) for command in commands]
    model_dir, selection_logs = convert_best_sparse_model(colmap, sparse_dir, text_dir)
    command_logs.extend(selection_logs)
    command_logs.append(f"selected_model={model_dir}")
    return command_logs


def convert_best_sparse_model(colmap: str, sparse_dir: Path, text_dir: Path) -> tuple[Path, list[str]]:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse models")

    logs: list[str] = []
    best_model: Path | None = None
    best_count = -1
    for candidate in candidates:
        candidate_text_dir = sparse_dir.parent / f"sparse_txt_{candidate.name}"
        candidate_text_dir.mkdir(parents=True, exist_ok=True)
        logs.append(
            run_command(
                [
                    colmap,
                    "model_converter",
                    "--input_path",
                    str(candidate),
                    "--output_path",
                    str(candidate_text_dir),
                    "--output_type",
                    "TXT",
                ]
            )
        )
        registered_images = count_colmap_text_images(candidate_text_dir / "images.txt")
        logs.append(f"candidate_model={candidate} registered_images={registered_images}")
        if registered_images > best_count:
            best_model = candidate
            best_count = registered_images

    if best_model is None:
        raise RuntimeError("Could not select a COLMAP sparse model")
    text_dir.mkdir(parents=True, exist_ok=True)
    logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(best_model),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )
    return best_model, logs


def count_colmap_text_images(path: Path) -> int:
    data_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return len(data_lines) // 2


def run_vggt_depth_batches(
    *,
    model: torch.nn.Module,
    image_paths: list[Path],
    load_and_preprocess_images: Any,
    pose_encoding_to_extri_intri: Any,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[Path, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("--vggt-batch-size must be positive")
    results: dict[Path, dict[str, Any]] = {}
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        prediction_np, _seconds = infer_image_group(
            model=model,
            image_paths=batch_paths,
            load_and_preprocess_images=load_and_preprocess_images,
            pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
            device=device,
            dtype=dtype,
        )
        colors = load_padded_rgb_images(batch_paths, prediction_np["images"].shape[-2:])
        for index, image_path in enumerate(batch_paths):
            results[image_path] = {
                "depth": np.squeeze(prediction_np["depth"][index]).astype(np.float32),
                "confidence": np.squeeze(prediction_np["depth_conf"][index]).astype(np.float32),
                "intrinsic": prediction_np["intrinsic"][index].astype(np.float32),
                "colors": colors[index],
                "image_shape": prediction_np["images"].shape[-2:],
                "original_size": read_image_size(image_path),
            }
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def estimate_depth_scale(
    *,
    colmap_image: ColmapImage,
    points3d: dict[int, np.ndarray],
    depth: np.ndarray,
    image_shape: tuple[int, int],
    original_size: tuple[int, int],
    min_observations: int,
) -> DepthScaleEstimate | None:
    ratios: list[float] = []
    rotation = qvec_to_rotmat(colmap_image.qvec)
    for x, y, point3d_id in colmap_image.observations:
        point = points3d.get(point3d_id)
        if point is None:
            continue
        colmap_depth = float((rotation @ point + colmap_image.tvec)[2])
        if colmap_depth <= 0:
            continue
        u, v = map_original_pixel_to_vggt(float(x), float(y), original_size, image_shape)
        ui = int(round(u))
        vi = int(round(v))
        if vi < 0 or ui < 0 or vi >= depth.shape[0] or ui >= depth.shape[1]:
            continue
        vggt_depth = float(depth[vi, ui])
        if np.isfinite(vggt_depth) and vggt_depth > 1e-6:
            ratios.append(colmap_depth / vggt_depth)
    if len(ratios) < min_observations:
        return None
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    median_log_ratio = float(np.median(log_ratios))
    return DepthScaleEstimate(
        scale=float(np.exp(median_log_ratio)),
        observation_count=len(ratios),
        log_mad=float(np.median(np.abs(log_ratios - median_log_ratio))),
    )


def build_fusion_camera(
    *,
    colmap_camera: dict[str, Any],
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> FusionCamera:
    source_width = int(colmap_camera["width"])
    source_height = int(colmap_camera["height"])
    if (source_width, source_height) != original_size:
        raise RuntimeError(
            "COLMAP camera size does not match the input image for dense fusion: "
            f"camera={source_width}x{source_height}, image={original_size[0]}x{original_size[1]}"
        )

    model = str(colmap_camera["model"])
    params = [float(value) for value in colmap_camera["params"]]
    if model == "SIMPLE_PINHOLE" and len(params) == 3:
        fx = fy = params[0]
        cx, cy = params[1:]
        radial_distortion: tuple[float, ...] = ()
    elif model == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        radial_distortion = ()
    elif model == "SIMPLE_RADIAL" and len(params) == 4:
        fx = fy = params[0]
        cx, cy = params[1:3]
        radial_distortion = (params[3],)
    elif model == "RADIAL" and len(params) == 5:
        fx = fy = params[0]
        cx, cy = params[1:3]
        radial_distortion = (params[3], params[4])
    else:
        raise RuntimeError(
            f"Unsupported COLMAP camera model '{model}' for dense fusion. "
            "Supported models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL."
        )

    transform = build_vggt_image_transform(original_size, image_shape)
    intrinsic = np.array(
        [
            [fx * transform.scale_x, 0.0, cx * transform.scale_x + transform.pad_left],
            [0.0, fy * transform.scale_y, cy * transform.scale_y + transform.pad_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return FusionCamera(model=model, intrinsic=intrinsic, radial_distortion=radial_distortion)


def build_fusion_camera_record(
    *,
    image_name: str,
    camera_id: int,
    camera: FusionCamera,
    vggt_intrinsic: np.ndarray,
    scale_estimate: DepthScaleEstimate | None,
    fallback_scale: float,
) -> dict[str, Any]:
    vggt_fx = float(vggt_intrinsic[0, 0])
    vggt_fy = float(vggt_intrinsic[1, 1])
    return {
        "image": image_name,
        "camera_id": camera_id,
        "colmap_model": camera.model,
        "fusion_intrinsic": camera.intrinsic.tolist(),
        "radial_distortion": list(camera.radial_distortion),
        "vggt_intrinsic": vggt_intrinsic.tolist(),
        "vggt_to_colmap_focal_ratio": {
            "fx": vggt_fx / float(camera.intrinsic[0, 0]),
            "fy": vggt_fy / float(camera.intrinsic[1, 1]),
        },
        "depth_scale": scale_estimate.scale if scale_estimate is not None else fallback_scale,
        "scale_observations": scale_estimate.observation_count if scale_estimate is not None else 0,
        "scale_log_mad": scale_estimate.log_mad if scale_estimate is not None else None,
        "scale_source": "sparse_colmap" if scale_estimate is not None else "median_fallback",
    }


def unproject_depth_with_colmap_pose(
    *,
    depth: np.ndarray,
    camera: FusionCamera,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    world_points = unproject_depth_pixels_with_colmap_pose(
        depth=depth.reshape(-1),
        u=xx.reshape(-1),
        v=yy.reshape(-1),
        camera=camera,
        qvec=qvec,
        tvec=tvec,
    )
    return world_points.reshape(height, width, 3).astype(np.float32)


def unproject_depth_pixels_with_colmap_pose(
    *,
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    camera: FusionCamera,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    fx = camera.intrinsic[0, 0]
    fy = camera.intrinsic[1, 1]
    cx = camera.intrinsic[0, 2]
    cy = camera.intrinsic[1, 2]
    z = depth.astype(np.float32)
    normalized_x, normalized_y = undistort_radial_coordinates(
        (u.astype(np.float32) - cx) / fx,
        (v.astype(np.float32) - cy) / fy,
        camera.radial_distortion,
    )
    camera_points = np.stack([normalized_x * z, normalized_y * z, z], axis=-1)
    rotation = qvec_to_rotmat(qvec).astype(np.float32)
    world_points = (camera_points.reshape(-1, 3) - tvec.astype(np.float32)) @ rotation
    return world_points.astype(np.float32)


def map_original_pixel_to_vggt(
    x: float,
    y: float,
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    transform = build_vggt_image_transform(original_size, image_shape)
    return x * transform.scale_x + transform.pad_left, y * transform.scale_y + transform.pad_top


def build_vggt_image_transform(
    original_size: tuple[int, int], image_shape: tuple[int, int]
) -> VggtImageTransform:
    original_width, original_height = original_size
    target_height, target_width = image_shape
    target_size = max(target_height, target_width)
    if original_width >= original_height:
        new_width = target_size
        new_height = round(original_height * (new_width / original_width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(original_width * (new_height / original_height) / 14) * 14
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    return VggtImageTransform(
        scale_x=new_width / original_width,
        scale_y=new_height / original_height,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=new_width,
        resized_height=new_height,
    )


def undistort_radial_coordinates(
    distorted_x: np.ndarray,
    distorted_y: np.ndarray,
    radial_distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not radial_distortion:
        return distorted_x, distorted_y

    undistorted_x = distorted_x.copy()
    undistorted_y = distorted_y.copy()
    for _ in range(8):
        radius_squared = undistorted_x * undistorted_x + undistorted_y * undistorted_y
        radial = np.ones_like(radius_squared)
        radius_power = radius_squared
        for coefficient in radial_distortion:
            radial += coefficient * radius_power
            radius_power *= radius_squared
        undistorted_x = distorted_x / radial
        undistorted_y = distorted_y / radial
    return undistorted_x, undistorted_y


def compute_global_confidence_threshold(frames: list[FusionFrame], percentile: float) -> float:
    confidence_parts: list[np.ndarray] = []
    for frame in frames:
        valid_canvas = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
        valid = valid_canvas & np.isfinite(frame.confidence) & (frame.confidence > 0)
        if valid.any():
            confidence_parts.append(frame.confidence[valid])
    if not confidence_parts:
        raise RuntimeError("No positive VGGT depth confidence values were available")
    return float(np.percentile(np.concatenate(confidence_parts), percentile))


def derive_consistency_relative_threshold(
    scale_estimates: Any,
    *,
    min_threshold: float,
    max_threshold: float,
) -> float:
    log_mads = np.asarray([estimate.log_mad for estimate in scale_estimates], dtype=np.float64)
    robust_threshold = float(np.expm1(2 * np.median(log_mads)))
    return float(np.clip(robust_threshold, min_threshold, max_threshold))


def build_covisibility_graph(
    frames: list[FusionFrame],
    *,
    max_neighbors: int,
    min_shared_points: int,
) -> dict[int, list[CovisibilityEdge]]:
    track_images: dict[int, set[int]] = {}
    for frame in frames:
        for _x, _y, point3d_id in frame.colmap_image.observations:
            track_images.setdefault(point3d_id, set()).add(frame.colmap_image.image_id)

    shared_counts: dict[tuple[int, int], int] = {}
    for image_ids in track_images.values():
        for first_id, second_id in combinations(sorted(image_ids), 2):
            key = (first_id, second_id)
            shared_counts[key] = shared_counts.get(key, 0) + 1

    frame_by_id = {frame.colmap_image.image_id: frame for frame in frames}
    centers = {
        image_id: colmap_camera_center(frame.colmap_image)
        for image_id, frame in frame_by_id.items()
    }
    candidates: dict[int, list[CovisibilityEdge]] = {image_id: [] for image_id in frame_by_id}
    for (first_id, second_id), shared_points in shared_counts.items():
        if shared_points < min_shared_points:
            continue
        baseline = float(np.linalg.norm(centers[first_id] - centers[second_id]))
        candidates[first_id].append(CovisibilityEdge(second_id, shared_points, baseline))
        candidates[second_id].append(CovisibilityEdge(first_id, shared_points, baseline))

    return {
        image_id: sorted(
            edges,
            key=lambda edge: (-edge.shared_points, -edge.baseline, edge.target_image_id),
        )[:max_neighbors]
        for image_id, edges in candidates.items()
    }


def build_visibility_graph_payload(
    frames: list[FusionFrame],
    graph: dict[int, list[CovisibilityEdge]],
) -> dict[str, Any]:
    frame_by_id = {frame.colmap_image.image_id: frame for frame in frames}
    return {
        "image_count": len(frames),
        "directed_edge_count": sum(len(edges) for edges in graph.values()),
        "images": [
            {
                "image": frame.image_path.name,
                "image_id": frame.colmap_image.image_id,
                "neighbors": [
                    {
                        "image": frame_by_id[edge.target_image_id].image_path.name,
                        "image_id": edge.target_image_id,
                        "shared_points": edge.shared_points,
                        "baseline": edge.baseline,
                    }
                    for edge in graph.get(frame.colmap_image.image_id, [])
                ],
            }
            for frame in frames
        ],
    }


def filter_points_by_cross_view_consistency(
    frames: list[FusionFrame],
    *,
    covisibility_graph: dict[int, list[CovisibilityEdge]],
    confidence_threshold: float,
    relative_threshold: float,
    stride: int,
) -> ConsistencyFilterResult:
    frame_by_id = {frame.colmap_image.image_id: frame for frame in frames}
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    residual_samples: list[np.ndarray] = []
    image_records: list[dict[str, Any]] = []
    candidate_points = 0
    accepted_points = 0
    rejected_points = 0
    unverified_points = 0
    supported_points = 0

    for frame in frames:
        valid = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
        valid &= np.isfinite(frame.depth) & (frame.depth > 0)
        valid &= np.isfinite(frame.confidence) & (frame.confidence >= confidence_threshold)
        if stride > 1:
            sampled = np.zeros_like(valid)
            sampled[::stride, ::stride] = True
            valid &= sampled

        v, u = np.nonzero(valid)
        source_depth = frame.depth[v, u] * frame.scale
        source_points = unproject_depth_pixels_with_colmap_pose(
            depth=source_depth,
            u=u,
            v=v,
            camera=frame.camera,
            qvec=frame.colmap_image.qvec,
            tvec=frame.colmap_image.tvec,
        )
        neighbors = [
            frame_by_id[edge.target_image_id]
            for edge in covisibility_graph.get(frame.colmap_image.image_id, [])
        ]
        validation = validate_cross_view_consistency(
            source_points,
            neighbors=neighbors,
            confidence_threshold=confidence_threshold,
            relative_threshold=relative_threshold,
        )
        accepted = validation.accepted
        supported = accepted & (validation.support_counts > 0)
        unverified = accepted & (validation.visible_counts == 0)
        rejected = ~accepted
        accepted_residuals = validation.mean_relative_error[supported]

        point_parts.append(source_points[accepted])
        color_parts.append(frame.colors[v[accepted], u[accepted]])
        residual_samples.append(diagnostic_sample(accepted_residuals, 1_000))
        candidate_points += len(source_points)
        accepted_points += int(accepted.sum())
        rejected_points += int(rejected.sum())
        unverified_points += int(unverified.sum())
        supported_points += int(supported.sum())
        image_records.append(
            {
                "image": frame.image_path.name,
                "candidate_points": int(len(source_points)),
                "accepted_points": int(accepted.sum()),
                "rejected_points": int(rejected.sum()),
                "unverified_points": int(unverified.sum()),
                "supported_points": int(supported.sum()),
                "median_relative_error": percentile_or_zero(accepted_residuals, 50),
                "p90_relative_error": percentile_or_zero(accepted_residuals, 90),
            }
        )

    if not point_parts:
        raise RuntimeError("No dense points were available after confidence filtering")
    return ConsistencyFilterResult(
        points=np.concatenate(point_parts, axis=0).astype(np.float32),
        colors=np.concatenate(color_parts, axis=0).astype(np.uint8),
        candidate_points=candidate_points,
        accepted_points=accepted_points,
        rejected_points=rejected_points,
        unverified_points=unverified_points,
        supported_points=supported_points,
        residual_samples=np.concatenate(residual_samples) if residual_samples else np.empty(0, dtype=np.float32),
        image_records=image_records,
    )


def validate_cross_view_consistency(
    source_points: np.ndarray,
    *,
    neighbors: list[FusionFrame],
    confidence_threshold: float,
    relative_threshold: float,
) -> CrossViewValidation:
    point_count = len(source_points)
    support_counts = np.zeros(point_count, dtype=np.int16)
    visible_counts = np.zeros(point_count, dtype=np.int16)
    occluded_counts = np.zeros(point_count, dtype=np.int16)
    residual_sums = np.zeros(point_count, dtype=np.float32)

    for neighbor in neighbors:
        u, v, projected_depth = project_world_points_to_depth_canvas(
            source_points,
            camera=neighbor.camera,
            qvec=neighbor.colmap_image.qvec,
            tvec=neighbor.colmap_image.tvec,
        )
        observed_depth, in_bounds = sample_bilinear(neighbor.depth, u, v)
        observed_confidence, _ = sample_bilinear(neighbor.confidence, u, v)
        valid = in_bounds & valid_depth_canvas_coordinates(u, v, neighbor.original_size, neighbor.image_shape)
        valid &= np.isfinite(projected_depth) & (projected_depth > 0)
        valid &= np.isfinite(observed_depth) & (observed_depth > 0)
        valid &= np.isfinite(observed_confidence) & (observed_confidence >= confidence_threshold)
        observed_depth *= neighbor.scale
        relative_error = np.abs(observed_depth - projected_depth) / np.maximum(
            np.maximum(observed_depth, projected_depth), 1e-6
        )
        occluded = valid & (observed_depth < projected_depth * (1 - relative_threshold))
        visible = valid & ~occluded
        supported = visible & (relative_error <= relative_threshold)
        support_counts += supported
        visible_counts += visible
        occluded_counts += occluded
        residual_sums += np.where(visible, relative_error, 0.0)

    accepted = (support_counts > 0) | (visible_counts == 0)
    mean_relative_error = np.divide(
        residual_sums,
        visible_counts,
        out=np.full(point_count, np.nan, dtype=np.float32),
        where=visible_counts > 0,
    )
    return CrossViewValidation(
        accepted=accepted,
        support_counts=support_counts,
        visible_counts=visible_counts,
        occluded_counts=occluded_counts,
        mean_relative_error=mean_relative_error,
    )


def project_world_points_to_depth_canvas(
    world_points: np.ndarray,
    *,
    camera: FusionCamera,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = qvec_to_rotmat(qvec).astype(np.float32)
    camera_points = world_points @ rotation.T + tvec.astype(np.float32)
    depth = camera_points[:, 2]
    safe_depth = np.where(depth > 1e-6, depth, 1.0)
    distorted_x, distorted_y = distort_radial_coordinates(
        camera_points[:, 0] / safe_depth,
        camera_points[:, 1] / safe_depth,
        camera.radial_distortion,
    )
    u = camera.intrinsic[0, 0] * distorted_x + camera.intrinsic[0, 2]
    v = camera.intrinsic[1, 1] * distorted_y + camera.intrinsic[1, 2]
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32)


def distort_radial_coordinates(
    undistorted_x: np.ndarray,
    undistorted_y: np.ndarray,
    radial_distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not radial_distortion:
        return undistorted_x, undistorted_y
    radius_squared = undistorted_x * undistorted_x + undistorted_y * undistorted_y
    radial = np.ones_like(radius_squared)
    radius_power = radius_squared
    for coefficient in radial_distortion:
        radial += coefficient * radius_power
        radius_power *= radius_squared
    return undistorted_x * radial, undistorted_y * radial


def sample_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    in_bounds = (u >= 0) & (v >= 0) & (u <= width - 1) & (v <= height - 1)
    clipped_u = np.clip(u, 0, width - 1)
    clipped_v = np.clip(v, 0, height - 1)
    u0 = np.floor(clipped_u).astype(np.int32)
    v0 = np.floor(clipped_v).astype(np.int32)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    du = clipped_u - u0
    dv = clipped_v - v0
    top = image[v0, u0] * (1 - du) + image[v0, u1] * du
    bottom = image[v1, u0] * (1 - du) + image[v1, u1] * du
    return (top * (1 - dv) + bottom * dv).astype(np.float32), in_bounds


def valid_depth_canvas_mask(
    original_size: tuple[int, int], image_shape: tuple[int, int]
) -> np.ndarray:
    transform = build_vggt_image_transform(original_size, image_shape)
    valid = np.zeros(image_shape, dtype=bool)
    valid[
        transform.pad_top : transform.pad_top + transform.resized_height,
        transform.pad_left : transform.pad_left + transform.resized_width,
    ] = True
    return valid


def valid_depth_canvas_coordinates(
    u: np.ndarray,
    v: np.ndarray,
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    transform = build_vggt_image_transform(original_size, image_shape)
    return (
        (u >= transform.pad_left)
        & (u <= transform.pad_left + transform.resized_width - 1)
        & (v >= transform.pad_top)
        & (v <= transform.pad_top + transform.resized_height - 1)
    )


def colmap_camera_center(image: ColmapImage) -> np.ndarray:
    return -(qvec_to_rotmat(image.qvec).T @ image.tvec)


def cap_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(points), size=max_points, replace=False)
    selected.sort()
    return points[selected], colors[selected]


def diagnostic_sample(values: np.ndarray, max_samples: int) -> np.ndarray:
    if len(values) <= max_samples:
        return values.astype(np.float32)
    stride = max(1, len(values) // max_samples)
    return values[::stride][:max_samples].astype(np.float32)


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def build_consistency_payload(
    filtered: ConsistencyFilterResult,
    *,
    confidence_threshold: float,
    relative_threshold: float,
    stride: int,
) -> dict[str, Any]:
    return {
        "confidence_threshold": confidence_threshold,
        "relative_threshold": relative_threshold,
        "stride": stride,
        "candidate_points": filtered.candidate_points,
        "accepted_points": filtered.accepted_points,
        "rejected_points": filtered.rejected_points,
        "unverified_points": filtered.unverified_points,
        "supported_points": filtered.supported_points,
        "acceptance_rate": filtered.accepted_points / max(filtered.candidate_points, 1),
        "residual_p50": percentile_or_zero(filtered.residual_samples, 50),
        "residual_p90": percentile_or_zero(filtered.residual_samples, 90),
        "images": filtered.image_records,
    }


def parse_colmap_images_with_points(path: Path) -> list[ColmapImage]:
    images: list[ColmapImage] = []
    data_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for index in range(0, len(data_lines), 2):
        image_parts = data_lines[index].split(maxsplit=9)
        point_parts = data_lines[index + 1].split()
        observations: list[tuple[float, float, int]] = []
        for point_index in range(0, len(point_parts), 3):
            point3d_id = int(point_parts[point_index + 2])
            if point3d_id == -1:
                continue
            observations.append(
                (
                    float(point_parts[point_index]),
                    float(point_parts[point_index + 1]),
                    point3d_id,
                )
            )
        images.append(
            ColmapImage(
                image_id=int(image_parts[0]),
                qvec=np.array([float(value) for value in image_parts[1:5]], dtype=np.float64),
                tvec=np.array([float(value) for value in image_parts[5:8]], dtype=np.float64),
                camera_id=int(image_parts[8]),
                name=image_parts[9],
                observations=observations,
            )
        )
    return images


def parse_colmap_points3d(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        points[int(parts[0])] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    return points


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


if __name__ == "__main__":
    main()
