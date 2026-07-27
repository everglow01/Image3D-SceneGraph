from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from geometry_utils import estimate_similarity_transform, transform_points


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/vggt/facebook--VGGT-1B")
DEFAULT_VGGT_REPO_DIR = Path("external/vggt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT and export a point cloud plus camera metadata.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing input images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for geometry and log outputs.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_VGGT_REPO_DIR)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--conf-percentile", type=float, default=50.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Run overlapping image groups to reduce VRAM use. 0 means one full VGGT pass.",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=1,
        help="Number of images shared by adjacent groups when --batch-size is active.",
    )
    parser.add_argument("--use-point-map", action="store_true", help="Use VGGT point-map branch instead of depth unprojection.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started_at = time.perf_counter()
    image_paths = discover_images(args.image_dir, args.max_images)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    validate_local_vggt(args.repo_dir, args.checkpoint_dir)
    sys.path.insert(0, str(args.repo_dir.resolve()))

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = select_device(args.device)
    dtype = select_dtype(device, args.precision)
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"num_images={len(image_paths)}")

    load_started_at = time.perf_counter()
    model = load_vggt_model(
        model_cls=VGGT,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        dtype=dtype,
        enable_point=args.use_point_map,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started_at

    reconstruction = run_reconstruction(
        model=model,
        image_paths=image_paths,
        load_and_preprocess_images=load_and_preprocess_images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
        device=device,
        dtype=dtype,
        use_point_map=args.use_point_map,
        batch_size=args.batch_size,
        overlap_size=args.overlap_size,
    )

    point_source = "point_map" if args.use_point_map else "depth_unprojection"
    flat_points, flat_colors = flatten_and_filter_points(
        points=reconstruction["points"],
        colors=reconstruction["colors"],
        confidence=reconstruction["confidence"],
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
        seed=args.seed,
    )

    output_dir = args.output_dir
    geometry_dir = output_dir / "geometry"
    log_dir = output_dir / "logs"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    write_ply(geometry_dir / "points.ply", flat_points, flat_colors)
    cameras = build_camera_payload(
        image_paths=image_paths,
        extrinsic=reconstruction["extrinsic"],
        intrinsic=reconstruction["intrinsic"],
        cam_to_world=closed_form_inverse_se3(reconstruction["extrinsic"])[:, :3, :],
        image_shape=reconstruction["image_shape"],
    )
    write_json(geometry_dir / "cameras.json", cameras)

    elapsed_seconds = time.perf_counter() - started_at
    log_lines = [
        "backend=vggt",
        f"checkpoint_dir={args.checkpoint_dir}",
        f"point_source={point_source}",
        f"num_images={len(image_paths)}",
        f"num_groups={reconstruction['num_groups']}",
        f"batch_size={args.batch_size}",
        f"overlap_size={reconstruction['overlap_size']}",
        f"num_points={len(flat_points)}",
        f"conf_percentile={args.conf_percentile}",
        f"max_points={args.max_points}",
        f"model_load_seconds={load_seconds:.3f}",
        f"inference_seconds={reconstruction['inference_seconds']:.3f}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
    ]
    (log_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {geometry_dir / 'points.ply'}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
    print(f"num_points={len(flat_points)}")
    print(f"num_groups={reconstruction['num_groups']}")
    print(f"inference_seconds={reconstruction['inference_seconds']:.3f}")


def discover_images(image_dir: Path, max_images: int) -> list[Path]:
    if max_images <= 0:
        raise ValueError("--max-images must be positive")
    paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return paths[:max_images]


def validate_local_vggt(repo_dir: Path, checkpoint_dir: Path) -> None:
    if not (repo_dir / "vggt").exists():
        raise SystemExit(f"VGGT repo missing: {repo_dir}")
    if not (checkpoint_dir / "model.safetensors").exists():
        raise SystemExit(f"VGGT checkpoint missing: {checkpoint_dir / 'model.safetensors'}")
    if not (checkpoint_dir / "config.json").exists():
        raise SystemExit(f"VGGT config missing: {checkpoint_dir / 'config.json'}")


def load_vggt_model(
    *,
    model_cls: type[torch.nn.Module],
    checkpoint_dir: Path,
    device: str,
    dtype: torch.dtype,
    enable_point: bool,
) -> torch.nn.Module:
    model = model_cls(enable_point=enable_point, enable_track=False)
    state_dict = load_file(checkpoint_dir / "model.safetensors", device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    del state_dict
    if device == "cuda":
        torch.cuda.empty_cache()

    relevant_missing = [
        key for key in missing_keys if not key.startswith("point_head.") and not key.startswith("track_head.")
    ]
    if relevant_missing:
        raise RuntimeError(f"Missing VGGT weights for enabled modules: {relevant_missing[:10]}")
    if unexpected_keys:
        print(f"ignored_weights={len(unexpected_keys)}")
    if device == "cuda":
        model = model.to(device=device, dtype=dtype)
        if dtype != torch.float32:
            model.camera_head.float()
            model.depth_head.float()
        return model
    return model.to(device)


def run_reconstruction(
    *,
    model: torch.nn.Module,
    image_paths: list[Path],
    load_and_preprocess_images: Any,
    pose_encoding_to_extri_intri: Any,
    unproject_depth_map_to_point_map: Any,
    device: str,
    dtype: torch.dtype,
    use_point_map: bool,
    batch_size: int,
    overlap_size: int,
) -> dict[str, Any]:
    chunks = build_image_chunks(len(image_paths), batch_size, overlap_size)
    global_extrinsics: list[np.ndarray | None] = [None] * len(image_paths)
    global_intrinsics: list[np.ndarray | None] = [None] * len(image_paths)
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    image_shape: tuple[int, int] | None = None
    inference_seconds = 0.0

    for chunk_index, image_indices in enumerate(chunks):
        group_paths = [image_paths[index] for index in image_indices]
        prediction_np, group_seconds = infer_image_group(
            model=model,
            image_paths=group_paths,
            load_and_preprocess_images=load_and_preprocess_images,
            pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
            device=device,
            dtype=dtype,
        )
        inference_seconds += group_seconds
        image_shape = prediction_np["images"].shape[-2:]

        local_extrinsic = prediction_np["extrinsic"]
        local_intrinsic = prediction_np["intrinsic"]
        if use_point_map:
            local_points = prediction_np["world_points"]
            local_confidence = prediction_np["world_points_conf"]
        else:
            local_points = unproject_depth_map_to_point_map(
                prediction_np["depth"], local_extrinsic, local_intrinsic
            )
            local_confidence = prediction_np["depth_conf"]
        local_colors = load_padded_rgb_images(group_paths, prediction_np["images"].shape[-2:])

        global_from_local = np.eye(4, dtype=np.float32)
        overlap_count = count_existing_prefix(image_indices, global_extrinsics)
        if chunk_index > 0:
            if overlap_count == 0:
                raise RuntimeError(f"Image group {chunk_index} has no overlap with previous groups")
            global_from_local = estimate_similarity_world_alignment(
                local_extrinsic[:overlap_count],
                np.stack(
                    [
                        global_extrinsics[image_index]
                        for image_index in image_indices[:overlap_count]
                        if global_extrinsics[image_index] is not None
                    ],
                    axis=0,
                ),
            )
            local_points = transform_points(local_points, global_from_local)

        local_from_global = np.linalg.inv(global_from_local)
        for local_index, image_index in enumerate(image_indices):
            global_extrinsics[image_index] = transform_extrinsic_by_similarity(
                local_extrinsic[local_index],
                global_from_local,
            )
            global_intrinsics[image_index] = local_intrinsic[local_index]

        keep_slice = slice(overlap_count, None) if chunk_index > 0 else slice(None)
        point_parts.append(local_points[keep_slice])
        color_parts.append(local_colors[keep_slice])
        confidence_parts.append(local_confidence[keep_slice])

        del prediction_np
        if device == "cuda":
            torch.cuda.empty_cache()

    if image_shape is None:
        raise RuntimeError("No VGGT image groups were processed")
    if any(extrinsic is None for extrinsic in global_extrinsics) or any(intrinsic is None for intrinsic in global_intrinsics):
        raise RuntimeError("VGGT did not produce camera metadata for every input image")

    return {
        "points": np.concatenate(point_parts, axis=0),
        "colors": np.concatenate(color_parts, axis=0),
        "confidence": np.concatenate(confidence_parts, axis=0),
        "extrinsic": np.stack([extrinsic for extrinsic in global_extrinsics if extrinsic is not None], axis=0),
        "intrinsic": np.stack([intrinsic for intrinsic in global_intrinsics if intrinsic is not None], axis=0),
        "image_shape": image_shape,
        "num_groups": len(chunks),
        "overlap_size": overlap_size if batch_size > 0 else 0,
        "inference_seconds": inference_seconds,
    }


def infer_image_group(
    *,
    model: torch.nn.Module,
    image_paths: list[Path],
    load_and_preprocess_images: Any,
    pose_encoding_to_extri_intri: Any,
    device: str,
    dtype: torch.dtype,
    frames_chunk_size: int | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    images = load_and_preprocess_images([str(path) for path in image_paths], mode="pad").to(device)
    infer_started_at = time.perf_counter()
    autocast_context = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()
    depth_head = getattr(model, "depth_head", None)
    original_forward = depth_head.forward.__func__ if depth_head is not None else None
    patched = False
    if depth_head is not None and frames_chunk_size is not None and original_forward is not None:
        # VGGT's DPT depth head supports frames_chunk_size to process frames in
        # chunks and lower peak activation memory. The default (8) never chunks a
        # 4-view group, so we force it before model(images) runs.
        import functools

        depth_head.forward = functools.partial(original_forward, depth_head, frames_chunk_size=frames_chunk_size)
        patched = True
    try:
        with torch.no_grad():
            with autocast_context:
                predictions = model(images)
    finally:
        if patched:
            depth_head.forward = original_forward.__get__(depth_head)
    infer_seconds = time.perf_counter() - infer_started_at

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    prediction_np = {
        key: value.detach().cpu().numpy().squeeze(0)
        for key, value in predictions.items()
        if isinstance(value, torch.Tensor)
    }
    del predictions, images
    return prediction_np, infer_seconds


def build_image_chunks(num_images: int, batch_size: int, overlap_size: int) -> list[list[int]]:
    if num_images <= 0:
        return []
    if batch_size <= 0 or batch_size >= num_images:
        return [list(range(num_images))]
    if batch_size == 1 and num_images > 1:
        raise ValueError("--batch-size must be at least 2 when more than one image is used")
    if overlap_size <= 0:
        raise ValueError("--overlap-size must be positive when --batch-size is active")
    if overlap_size >= batch_size:
        raise ValueError("--overlap-size must be smaller than --batch-size")

    chunks: list[list[int]] = []
    start = 0
    while start < num_images:
        end = min(start + batch_size, num_images)
        chunks.append(list(range(start, end)))
        if end == num_images:
            break
        start = end - overlap_size
    return chunks


def count_existing_prefix(image_indices: list[int], global_extrinsics: list[np.ndarray | None]) -> int:
    count = 0
    for image_index in image_indices:
        if global_extrinsics[image_index] is None:
            break
        count += 1
    return count


def estimate_similarity_world_alignment(local_extrinsic: np.ndarray, global_extrinsic: np.ndarray) -> np.ndarray:
    if len(local_extrinsic) >= 2:
        local_centers = camera_centers(local_extrinsic)
        global_centers = camera_centers(global_extrinsic)
        return estimate_similarity_transform(local_centers, global_centers)
    return np.linalg.inv(to_homogeneous_extrinsic(global_extrinsic[0])) @ to_homogeneous_extrinsic(local_extrinsic[0])


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)


def transform_extrinsic_by_similarity(
    local_extrinsic: np.ndarray,
    global_from_local: np.ndarray,
) -> np.ndarray:
    linear = global_from_local[:3, :3]
    scale = float(np.cbrt(max(np.linalg.det(linear), 1e-12)))
    world_rotation = linear / scale
    local_rotation = local_extrinsic[:3, :3]
    local_center = camera_centers(local_extrinsic[None, ...])[0]
    global_center = (global_from_local @ np.array([*local_center, 1.0], dtype=np.float32))[:3]

    global_rotation = local_rotation @ world_rotation.T
    u, _s, vt = np.linalg.svd(global_rotation)
    global_rotation = u @ vt
    if np.linalg.det(global_rotation) < 0:
        u[:, -1] *= -1
        global_rotation = u @ vt
    global_translation = -global_rotation @ global_center

    extrinsic = np.empty((3, 4), dtype=np.float32)
    extrinsic[:3, :3] = global_rotation.astype(np.float32)
    extrinsic[:3, 3] = global_translation.astype(np.float32)
    return extrinsic


def to_homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :4] = extrinsic.astype(np.float32)
    return matrix


def load_padded_rgb_images(image_paths: list[Path], image_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = image_shape
    target_size = max(target_height, target_width)
    images: list[np.ndarray] = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)

        pad_height = target_height - array.shape[0]
        pad_width = target_width - array.shape[1]
        if pad_height > 0 or pad_width > 0:
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            array = np.pad(
                array,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        images.append(array)
    return np.stack(images, axis=0)


def select_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    return requested_device


def select_dtype(device: str, precision: str) -> torch.dtype:
    if device == "cpu" or precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def flatten_and_filter_points(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    conf_percentile: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_conf = confidence.reshape(-1)
    valid = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_conf) & (flat_conf > 0)
    if valid.any():
        threshold = np.percentile(flat_conf[valid], conf_percentile)
        valid &= flat_conf >= threshold

    selected_indices = np.flatnonzero(valid)
    if max_points > 0 and len(selected_indices) > max_points:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(selected_indices, size=max_points, replace=False)
        selected_indices.sort()

    return flat_points[selected_indices].astype(np.float32), flat_colors[selected_indices].astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertex_data = np.empty(len(points), dtype=vertex_dtype)
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    with path.open("wb") as file:
        file.write(header)
        vertex_data.tofile(file)


def build_camera_payload(
    *,
    image_paths: list[Path],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    cam_to_world: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    return {
        "coordinate_system": "opencv_camera_from_world",
        "image_shape": {"height": int(image_shape[0]), "width": int(image_shape[1])},
        "cameras": [
            {
                "index": index,
                "image": image_path.name,
                "extrinsic_camera_from_world": extrinsic[index].tolist(),
                "intrinsic": intrinsic[index].tolist(),
                "camera_to_world": cam_to_world[index].tolist(),
            }
            for index, image_path in enumerate(image_paths)
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
