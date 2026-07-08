from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_colmap_sparse import build_camera_payload, discover_images, run_command
from run_vggt_pointcloud import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_VGGT_REPO_DIR,
    flatten_and_filter_points,
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    scales: dict[str, float] = {}
    for image_path, item in depth_items.items():
        colmap_image = registered_by_name[image_path.name]
        scale = estimate_depth_scale(
            colmap_image=colmap_image,
            points3d=points3d,
            depth=item["depth"],
            image_shape=item["image_shape"],
            original_size=item["original_size"],
            min_observations=args.min_scale_observations,
        )
        if scale is not None:
            scales[image_path.name] = scale
    if not scales:
        raise RuntimeError("Could not estimate VGGT-to-COLMAP depth scale from sparse observations")
    fallback_scale = float(np.median(list(scales.values())))

    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    used_scales: list[float] = []
    for image_path, item in depth_items.items():
        colmap_image = registered_by_name[image_path.name]
        scale = scales.get(image_path.name, fallback_scale)
        used_scales.append(scale)
        points = unproject_depth_with_colmap_pose(
            depth=item["depth"] * scale,
            intrinsic=item["intrinsic"],
            qvec=colmap_image.qvec,
            tvec=colmap_image.tvec,
        )
        point_parts.append(points[None, ...])
        color_parts.append(item["colors"][None, ...])
        confidence_parts.append(item["confidence"][None, ...])

    flat_points, flat_colors = flatten_and_filter_points(
        points=np.concatenate(point_parts, axis=0),
        colors=np.concatenate(color_parts, axis=0),
        confidence=np.concatenate(confidence_parts, axis=0),
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
        seed=args.seed,
    )
    write_ply(geometry_dir / "points.ply", flat_points, flat_colors)
    write_json(geometry_dir / "cameras.json", build_camera_payload(text_dir))

    elapsed_seconds = time.perf_counter() - started_at
    log_lines = [
        "backend=colmap_vggt",
        f"num_images={len(image_paths)}",
        f"registered_images={len(registered_paths)}",
        f"num_points={len(flat_points)}",
        f"colmap_points={len(points3d)}",
        f"scaled_images={len(scales)}",
        f"scale_median={float(np.median(used_scales)):.6f}",
        f"scale_min={float(np.min(used_scales)):.6f}",
        f"scale_max={float(np.max(used_scales)):.6f}",
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
    print(f"registered_images={len(registered_paths)}")
    print(f"scaled_images={len(scales)}")
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
) -> float | None:
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
    return float(np.median(ratios))


def unproject_depth_with_colmap_pose(
    *,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    z = depth.astype(np.float32)
    camera_points = np.stack([(xx - cx) / fx * z, (yy - cy) / fy * z, z], axis=-1)
    rotation = qvec_to_rotmat(qvec).astype(np.float32)
    world_points = (camera_points.reshape(-1, 3) - tvec.astype(np.float32)) @ rotation
    return world_points.reshape(height, width, 3).astype(np.float32)


def map_original_pixel_to_vggt(
    x: float,
    y: float,
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    original_width, original_height = original_size
    target_height, target_width = image_shape
    target_size = max(target_height, target_width)
    if original_width >= original_height:
        new_width = target_size
        new_height = round(original_height * (new_width / original_width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(original_width * (new_height / original_height) / 14) * 14
    pad_left = (target_width - new_width) / 2
    pad_top = (target_height - new_height) / 2
    return x * new_width / original_width + pad_left, y * new_height / original_height + pad_top


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
