from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image3d_scenegraph.geometry.colmap import resolve_colmap_executable
from image3d_scenegraph.geometry.vggt_ba import (
    VggtBaError,
    bridge_windows,
    estimate_window_edge,
    merge_window_cameras,
    optimize_window_graph,
    sequential_windows,
    supported_image_ids,
    write_initial_colmap_model,
    write_json,
)
from scripts.run_colmap_sparse import (
    build_camera_payload,
    colmap_version,
    read_ply_vertex_count,
)
from scripts.run_vggt_pointcloud import load_vggt_model, select_dtype


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROFILE_ID = "vggt_ba_standard_v1"
WINDOW_SIZE = 8
WINDOW_OVERLAP = 4
QUERY_FRAME_COUNT = 5
MAX_QUERY_POINTS = 2048
MAX_BRIDGES = 16
MIN_SUPPORTED_OBSERVATIONS = 32


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a global COLMAP reconstruction from batched VGGT cameras and BA."
    )
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", type=Path, default=Path("external/vggt"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/vggt/facebook--VGGT-1B"),
    )
    parser.add_argument("--dinov2-repo", type=Path, default=Path("external/dinov2"))
    parser.add_argument("--lightglue-repo", type=Path, default=Path("external/lightglue"))
    parser.add_argument(
        "--dinov2-checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/dinov2_vitb14_reg4_pretrain.pth"),
    )
    parser.add_argument(
        "--aliked-checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/torch-hub/checkpoints/aliked-n16.pth"),
    )
    parser.add_argument(
        "--tracker-checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/vggsfm_v2_tracker.pt"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max-image-size", type=int, default=1280)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.max_image_size < 1 or args.num_threads < 1:
        parser.error("--max-image-size and --num-threads must be positive")

    started_at = time.perf_counter()
    image_paths = discover_images(args.image_dir)
    if len(image_paths) < 12:
        raise SystemExit("VGGT-BA Gaussian geometry requires at least 12 images")
    validate_runtime(args)
    colmap_path = resolve_colmap_executable()
    if colmap_path is None:
        raise SystemExit("COLMAP executable not found")
    colmap = str(colmap_path)
    output_dir = args.output_dir
    work_dir = output_dir / "vggt_ba"
    windows_dir = work_dir / "windows"
    diagnostics_dir = output_dir / "diagnostics"
    geometry_dir = output_dir / "geometry"
    colmap_dir = output_dir / "colmap"
    for path in (windows_dir, diagnostics_dir, geometry_dir, colmap_dir):
        path.mkdir(parents=True, exist_ok=True)

    write_progress(args.progress_file, "vggt_ba_descriptors")
    runtime = load_runtime(args)
    descriptors = compute_dino_descriptors(
        image_paths,
        runtime["dino_model"],
        runtime["device"],
        runtime["torch"],
    )
    bases = sequential_windows(
        len(image_paths), window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP
    )
    bridges = bridge_windows(
        descriptors,
        bases,
        window_size=WINDOW_SIZE,
        minimum_index_gap=WINDOW_SIZE * 2,
        maximum_bridges=MAX_BRIDGES,
    )
    window_specs = [*bases, *bridges]
    write_json(
        work_dir / "profile.json",
        {
            "profile": PROFILE_ID,
            "window_size": WINDOW_SIZE,
            "overlap": WINDOW_OVERLAP,
            "query_frame_num": QUERY_FRAME_COUNT,
            "max_query_points": MAX_QUERY_POINTS,
            "keypoint_extractor": "aliked",
            "seed": args.seed,
            "base_window_count": len(bases),
            "bridge_candidate_count": len(bridges),
        },
    )

    write_progress(args.progress_file, "vggt_ba_windows")
    windows: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    window_records = []
    rejected_bridges = []
    for ordinal, spec in enumerate(window_specs):
        try:
            cameras, record = process_window(
                spec,
                image_paths,
                descriptors,
                runtime,
                windows_dir / spec.window_id,
                args,
            )
        except Exception as exc:
            if spec.kind == "bridge":
                rejected_bridges.append(
                    {"window_id": spec.window_id, "reason": str(exc)}
                )
                continue
            raise
        windows[spec.window_id] = cameras
        window_records.append(record)
        print(
            f"window={spec.window_id} kind={spec.kind} images={len(spec.image_indices)} "
            f"progress={ordinal + 1}/{len(window_specs)}",
            flush=True,
        )

    write_progress(args.progress_file, "vggt_ba_pose_graph")
    edges = []
    edge_rejections = []
    window_ids = list(windows)
    for left_index, source in enumerate(window_ids):
        for target in window_ids[left_index + 1 :]:
            if len(set(windows[source]) & set(windows[target])) < 3:
                continue
            try:
                edges.append(
                    estimate_window_edge(
                        source,
                        target,
                        {
                            index: camera["extrinsic"]
                            for index, camera in windows[source].items()
                        },
                        {
                            index: camera["extrinsic"]
                            for index, camera in windows[target].items()
                        },
                    )
                )
            except VggtBaError as exc:
                edge_rejections.append(
                    {"source": source, "target": target, "reason": str(exc)}
                )
    transforms, graph_metrics = optimize_window_graph(window_ids, edges)
    merged, merge_metrics = merge_window_cameras(windows, transforms)
    kinds = {spec.window_id: spec.kind for spec in window_specs}
    loop_edges = [
        edge
        for edge in edges
        if kinds.get(edge.source) == "bridge" or kinds.get(edge.target) == "bridge"
    ]
    graph_payload = {
        "schema_version": 1,
        "profile": PROFILE_ID,
        "windows": window_records,
        "rejected_bridges": rejected_bridges,
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "shared_indices": list(edge.shared_indices),
                "target_from_source": edge.target_from_source.tolist(),
                "center_residual_p90": edge.center_residual_p90,
                "rotation_residual_p90_degrees": edge.rotation_residual_p90_degrees,
            }
            for edge in edges
        ],
        "rejected_edges": edge_rejections,
        "graph": graph_metrics,
        "merge": merge_metrics,
        "verified_nonlocal_edge_count": len(loop_edges),
        "trajectory_status": (
            "closed_graph_verified" if loop_edges else "open_trajectory_unverified"
        ),
    }
    write_json(work_dir / "window_graph.json", graph_payload)

    image_names = [path.name for path in image_paths]
    image_sizes = {
        index: Image.open(path).size for index, path in enumerate(image_paths)
    }
    initial_record = write_initial_colmap_model(
        work_dir / "initial_model", image_names, merged, image_sizes
    )

    write_progress(args.progress_file, "vggt_ba_feature_extraction")
    command_logs: list[str] = []
    database_path = colmap_dir / "database.db"
    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(args.image_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "OPENCV",
        "--FeatureExtraction.use_gpu",
        "1",
        "--FeatureExtraction.num_threads",
        str(args.num_threads),
    ]
    command_logs.append(run_command(feature_command))
    write_progress(args.progress_file, "vggt_ba_feature_matching")
    command_logs.append(
        run_command(
            [
                colmap,
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                "--FeatureMatching.use_gpu",
                "1",
                "--FeatureMatching.num_threads",
                str(args.num_threads),
            ]
        )
    )
    write_progress(args.progress_file, "vggt_ba_global_triangulation")
    triangulated_dir = work_dir / "triangulated"
    triangulated_dir.mkdir()
    command_logs.append(
        run_command(
            [
                colmap,
                "point_triangulator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(args.image_dir),
                "--input_path",
                str(work_dir / "initial_model"),
                "--output_path",
                str(triangulated_dir),
                "--clear_points",
                "1",
                "--refine_intrinsics",
                "1",
                "--Mapper.num_threads",
                str(args.num_threads),
                "--Mapper.ba_global_function_tolerance",
                "0.000001",
            ]
        )
    )
    write_progress(args.progress_file, "vggt_ba_global_bundle_adjustment")
    bundled_dir = work_dir / "global_model"
    bundled_dir.mkdir()
    command_logs.append(
        run_command(
            [
                colmap,
                "bundle_adjuster",
                "--input_path",
                str(triangulated_dir),
                "--output_path",
                str(bundled_dir),
                "--BundleAdjustment.refine_focal_length",
                "1",
                "--BundleAdjustment.refine_principal_point",
                "0",
                "--BundleAdjustment.refine_extra_params",
                "1",
                "--BundleAdjustmentCeres.function_tolerance",
                "0.000001",
            ]
        )
    )

    write_progress(args.progress_file, "colmap_undistortion")
    undistorted_dir = colmap_dir / "undistorted"
    command_logs.append(
        run_command(
            [
                colmap,
                "image_undistorter",
                "--image_path",
                str(args.image_dir),
                "--input_path",
                str(bundled_dir),
                "--output_path",
                str(undistorted_dir),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(args.max_image_size),
            ]
        )
    )
    sparse_source = undistorted_dir / "sparse"
    sparse_text = undistorted_dir / "sparse_txt"
    sparse_text.mkdir()
    points_ply = geometry_dir / "points.ply"
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(sparse_source),
                "--output_path",
                str(sparse_text),
                "--output_type",
                "TXT",
            ]
        )
    )
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(sparse_source),
                "--output_path",
                str(points_ply),
                "--output_type",
                "PLY",
            ]
        )
    )
    camera_payload = build_camera_payload(sparse_text)
    supported = supported_image_ids(
        sparse_text / "images.txt",
        minimum_observations=MIN_SUPPORTED_OBSERVATIONS,
    )
    camera_payload["images"] = [
        image for image in camera_payload["images"] if int(image["image_id"]) in supported
    ]
    if len(camera_payload["images"]) < 12:
        raise RuntimeError(
            f"VGGT-BA global model has only {len(camera_payload['images'])} geometrically supported cameras"
        )
    write_json(geometry_dir / "cameras.json", camera_payload)

    elapsed = time.perf_counter() - started_at
    diagnostics = {
        "schema_version": 1,
        "profile": PROFILE_ID,
        "geometry_source": "vggt_ba",
        "input_count": len(image_paths),
        "supported_camera_count": len(camera_payload["images"]),
        "supported_camera_rate": len(camera_payload["images"]) / len(image_paths),
        "point_count": read_ply_vertex_count(points_ply),
        "initial_camera": initial_record,
        "window_graph": graph_payload,
        "colmap_executable": colmap,
        "colmap_build": colmap_version(colmap),
        "elapsed_seconds": elapsed,
        "dependencies": dependency_record(args),
    }
    write_json(diagnostics_dir / "vggt_ba.json", diagnostics)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs" / "vggt_ba.log").write_text(
        "\n".join(
            [
                "geometry_source=vggt_ba",
                f"profile={PROFILE_ID}",
                f"input_count={len(image_paths)}",
                f"supported_camera_count={len(camera_payload['images'])}",
                f"point_count={diagnostics['point_count']}",
                f"trajectory_status={graph_payload['trajectory_status']}",
                f"elapsed_seconds={elapsed:.3f}",
                *command_logs,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"supported_cameras={len(camera_payload['images'])}")
    print(f"num_points={diagnostics['point_count']}")
    print(f"trajectory_status={graph_payload['trajectory_status']}")


def discover_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def validate_runtime(args: argparse.Namespace) -> None:
    required = [
        args.repo_dir / "vggt",
        args.checkpoint_dir / "model.safetensors",
        args.dinov2_repo / "hubconf.py",
        args.lightglue_repo / ".git",
        args.dinov2_checkpoint,
        args.aliked_checkpoint,
        args.tracker_checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("VGGT-BA runtime is incomplete: " + "; ".join(missing))
    if (
        args.aliked_checkpoint.name != "aliked-n16.pth"
        or args.aliked_checkpoint.parent.name != "checkpoints"
    ):
        raise SystemExit(
            "ALIKED checkpoint must use the local torch-hub/checkpoints/aliked-n16.pth layout"
        )
    try:
        import lightglue  # noqa: F401
        import pycolmap  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"VGGT-BA dependency is missing: {exc}") from exc


def load_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import random

    import torch
    import torch.nn.functional as functional

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = select_dtype(args.device, args.precision)
    torch.hub.set_dir(str(args.aliked_checkpoint.parent.parent.resolve()))
    sys.path.insert(0, str(args.repo_dir.resolve()))
    from vggt.dependency.vggsfm_utils import (
        build_vggsfm_tracker,
        initialize_feature_extractors,
    )
    from vggt.models.vggt import VGGT

    vggt_model = load_vggt_model(
        model_cls=VGGT,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dtype=dtype,
        enable_point=False,
    ).eval()
    tracker = build_vggsfm_tracker(str(args.tracker_checkpoint)).to(device, dtype).eval()
    extractors = initialize_feature_extractors(
        MAX_QUERY_POINTS, extractor_method="aliked", device=device
    )
    dino_model = torch.hub.load(
        str(args.dinov2_repo.resolve()),
        "dinov2_vitb14_reg",
        source="local",
        pretrained=False,
    )
    payload = torch.load(args.dinov2_checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }
    missing, unexpected = dino_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv2 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    dino_model = dino_model.to(device).eval()
    return {
        "torch": torch,
        "functional": functional,
        "device": device,
        "dtype": dtype,
        "vggt_model": vggt_model,
        "tracker": tracker,
        "extractors": extractors,
        "dino_model": dino_model,
    }


def compute_dino_descriptors(
    image_paths: list[Path], model: Any, device: Any, torch: Any
) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    descriptors = []
    for start in range(0, len(image_paths), 16):
        arrays = []
        for path in image_paths[start : start + 16]:
            image = Image.open(path).convert("RGB").resize((336, 336), Image.Resampling.BICUBIC)
            arrays.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        batch = torch.from_numpy(np.stack(arrays)).to(device)
        with torch.no_grad():
            features = model.forward_features((batch - mean) / std)
            cls = features["x_norm_clstoken"] if isinstance(features, dict) else features
        descriptors.append(cls.detach().float().cpu().numpy())
    return np.concatenate(descriptors, axis=0)


def process_window(
    spec: Any,
    image_paths: list[Path],
    descriptors: np.ndarray,
    runtime: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    import pycolmap
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
    from vggt.dependency.track_predict import _forward_on_query
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    torch = runtime["torch"]
    functional = runtime["functional"]
    paths = [image_paths[index] for index in spec.image_indices]
    images, original_coords = load_and_preprocess_images_square(
        [str(path) for path in paths], 1024
    )
    images = images.to(runtime["device"])
    images_518 = functional.interpolate(
        images, size=(518, 518), mode="bilinear", align_corners=False
    )
    autocast = (
        torch.cuda.amp.autocast(dtype=runtime["dtype"])
        if runtime["device"].type == "cuda"
        else nullcontext()
    )
    started = time.perf_counter()
    with torch.no_grad(), autocast:
        predictions = runtime["vggt_model"](images_518[None])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images_518.shape[-2:]
    )
    points_3d = unproject_depth_map_to_point_map(
        predictions["depth"].squeeze(0).detach().float().cpu().numpy(),
        extrinsic.squeeze(0).detach().float().cpu().numpy(),
        intrinsic.squeeze(0).detach().float().cpu().numpy(),
    )
    confidence = predictions["depth_conf"].squeeze(0).detach().float().cpu().numpy()
    extrinsic_np = extrinsic.squeeze(0).detach().float().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().float().cpu().numpy()
    del predictions, images_518

    fmaps = runtime["tracker"].process_images_to_fmaps(images)
    query_indices = choose_queries(descriptors[list(spec.image_indices)], QUERY_FRAME_COUNT)
    tracks = []
    visibility = []
    point_parts = []
    color_parts = []
    for query_index in query_indices:
        track, visible, _conf, point, color = _forward_on_query(
            query_index,
            images,
            confidence,
            points_3d,
            fmaps,
            runtime["extractors"],
            runtime["tracker"],
            163840,
            True,
            runtime["device"],
        )
        tracks.append(track)
        visibility.append(visible)
        point_parts.append(point)
        color_parts.append(color)
    tracks_np = np.concatenate(tracks, axis=1)
    visibility_np = np.concatenate(visibility, axis=1)
    points_np = np.concatenate(point_parts, axis=0)
    colors_np = np.concatenate(color_parts, axis=0)
    intrinsic_np[:, :2, :] *= 1024 / 518
    mask = visibility_np > 0.2
    reconstruction, _valid = batch_np_matrix_to_pycolmap(
        points_np,
        extrinsic_np,
        intrinsic_np,
        tracks_np,
        np.array([1024, 1024]),
        masks=mask,
        max_reproj_error=8.0,
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
        points_rgb=colors_np,
    )
    if reconstruction is None:
        raise RuntimeError(f"{spec.window_id} did not produce enough track inliers")
    before = reconstruction.compute_mean_reprojection_error()
    options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, options)
    after = reconstruction.compute_mean_reprojection_error()
    if not np.isfinite(after) or after > before * 1.05 + 1e-6:
        raise RuntimeError(
            f"{spec.window_id} local BA reprojection increased: {before} -> {after}"
        )
    reconstruction = restore_original_coordinates(
        reconstruction,
        [path.name for path in paths],
        original_coords.detach().cpu().numpy(),
        1024,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    reconstruction.write(output_dir)
    cameras: dict[int, dict[str, np.ndarray]] = {}
    for local_index, global_index in enumerate(spec.image_indices):
        image = reconstruction.images[local_index + 1]
        camera = reconstruction.cameras[image.camera_id]
        cameras[global_index] = {
            "extrinsic": np.asarray(image.cam_from_world.matrix(), dtype=np.float64)[:3, :4],
            "intrinsic": np.asarray(camera.calibration_matrix(), dtype=np.float64),
        }
    elapsed = time.perf_counter() - started
    record = {
        "window_id": spec.window_id,
        "kind": spec.kind,
        "image_indices": list(spec.image_indices),
        "image_names": [path.name for path in paths],
        "query_indices": query_indices,
        "point_count": len(reconstruction.points3D),
        "track_count": int(tracks_np.shape[1]),
        "mean_reprojection_before": float(before),
        "mean_reprojection_after": float(after),
        "elapsed_seconds": elapsed,
    }
    write_json(output_dir / "window.json", record)
    del images, fmaps
    if runtime["device"].type == "cuda":
        torch.cuda.empty_cache()
    return cameras, record


def choose_queries(descriptors: np.ndarray, count: int) -> list[int]:
    values = np.asarray(descriptors, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    chosen = [0]
    while len(chosen) < min(count, len(values)):
        similarity = values @ values[chosen].T
        minimum_distance = (1.0 - similarity).min(axis=1)
        minimum_distance[chosen] = -1
        chosen.append(int(np.argmax(minimum_distance)))
    return chosen


def restore_original_coordinates(
    reconstruction: Any,
    image_names: list[str],
    original_coords: np.ndarray,
    image_size: int,
) -> Any:
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        image.name = image_names[image_id - 1]
        real_size = original_coords[image_id - 1, -2:]
        ratio = max(real_size) / image_size
        params = camera.params.copy() * ratio
        params[-2:] = real_size / 2
        camera.params = params
        camera.width = int(real_size[0])
        camera.height = int(real_size[1])
        top_left = original_coords[image_id - 1, :2]
        for point in image.points2D:
            point.xy = (point.xy - top_left) * ratio
    return reconstruction


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return (
        "command="
        + " ".join(command)
        + "\nstdout="
        + completed.stdout.strip()
        + "\nstderr="
        + completed.stderr.strip()
    )


def write_progress(path: Path | None, stage: str) -> None:
    if path is not None:
        write_json(path, {"stage": stage})


def dependency_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vggt_revision": git_revision(args.repo_dir),
        "vggt_checkpoint_sha256": sha256_file(args.checkpoint_dir / "model.safetensors"),
        "dinov2_revision": git_revision(args.dinov2_repo),
        "dinov2_checkpoint_sha256": sha256_file(args.dinov2_checkpoint),
        "lightglue_revision": git_revision(args.lightglue_repo),
        "aliked_checkpoint_sha256": sha256_file(args.aliked_checkpoint),
        "vggsfm_tracker_sha256": sha256_file(args.tracker_checkpoint),
        "runtime_network_access": False,
        "research_only": True,
    }


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except (VggtBaError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
