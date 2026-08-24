from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

from image3d_scenegraph.geometry.colmap import resolve_colmap_executable
from image3d_scenegraph.geometry.video_recovery import (
    recover_video_registration,
    sequential_overlap,
)
from image3d_scenegraph.video.keyframes import V2_PROFILE_ID


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP sparse SfM and export a point cloud.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-index", type=parse_gpu_indices)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--max-image-size", type=int)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--gaussian-baseline", action="store_true")
    parser.add_argument("--vocab-tree-path", type=Path)
    parser.add_argument("--video-source", type=Path)
    parser.add_argument("--video-selection", type=Path)
    args = parser.parse_args()
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be at least 1")
    if args.max_image_size is not None and args.max_image_size < 1:
        parser.error("--max-image-size must be at least 1")
    if args.gaussian_baseline and args.matcher == "sequential" and args.vocab_tree_path is None:
        raise SystemExit(
            "Gaussian baseline sequential matching requires --vocab-tree-path for loop closure"
        )
    if (args.video_source is None) != (args.video_selection is None):
        parser.error("--video-source and --video-selection must be provided together")
    video_selection: dict[str, Any] | None = None
    if args.video_source is not None and args.video_selection is not None:
        if not args.video_source.is_file():
            raise SystemExit(f"Video source is missing: {args.video_source}")
        try:
            video_selection = json.loads(
                args.video_selection.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit("Cannot read video selection metadata") from exc

    started_at = time.perf_counter()
    colmap_path = resolve_colmap_executable()
    if colmap_path is None:
        raise SystemExit(
            "COLMAP executable not found. Run `uv run python scripts/setup_colmap_cuda.py --install` "
            "or install COLMAP on PATH."
        )
    colmap = str(colmap_path)
    colmap_build = colmap_version(colmap)

    image_paths = discover_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    output_dir = args.output_dir
    geometry_dir = output_dir / "geometry"
    logs_dir = output_dir / "logs"
    work_dir = output_dir / "colmap"
    sparse_dir = work_dir / "sparse"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(work_dir / "database.db"),
        "--image_path",
        str(args.image_dir),
        "--ImageReader.single_camera",
        "1" if args.single_camera else "0",
        "--FeatureExtraction.use_gpu",
        "1" if args.use_gpu else "0",
    ]
    if args.gpu_index is not None:
        feature_command.extend(("--FeatureExtraction.gpu_index", args.gpu_index))
    if args.gaussian_baseline:
        feature_command.extend(("--ImageReader.camera_model", "OPENCV"))
    if args.num_threads is not None:
        feature_command.extend(("--FeatureExtraction.num_threads", str(args.num_threads)))
    mapper_command = [
        colmap,
        "mapper",
        "--database_path",
        str(work_dir / "database.db"),
        "--image_path",
        str(args.image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    if args.gaussian_baseline:
        mapper_command.extend(("--Mapper.ba_global_function_tolerance", "0.000001"))
    if args.num_threads is not None:
        mapper_command.extend(("--Mapper.num_threads", str(args.num_threads)))
    matcher_command = [
        colmap,
        "sequential_matcher" if args.matcher == "sequential" else "exhaustive_matcher",
        "--database_path",
        str(work_dir / "database.db"),
        "--FeatureMatching.use_gpu",
        "1" if args.use_gpu else "0",
    ]
    if args.matcher == "sequential" and args.vocab_tree_path is not None:
        matcher_command.extend(
            (
                "--SequentialMatching.loop_detection",
                "1",
                "--SequentialMatching.vocab_tree_path",
                str(args.vocab_tree_path),
            )
        )
    sequential_overlap_value: int | None = None
    if args.matcher == "sequential" and video_selection is not None:
        if video_selection.get("profile") == V2_PROFILE_ID:
            sequential_overlap_value = sequential_overlap(video_selection)
            matcher_command.extend(
                ("--SequentialMatching.overlap", str(sequential_overlap_value))
            )
    if args.gpu_index is not None:
        matcher_command.extend(("--FeatureMatching.gpu_index", args.gpu_index))
    if args.num_threads is not None:
        matcher_command.extend(("--FeatureMatching.num_threads", str(args.num_threads)))
    commands = [
        ("colmap_feature_extraction", feature_command),
        ("colmap_feature_matching", matcher_command),
        ("colmap_mapping", mapper_command),
    ]
    command_logs = []
    for stage, command in commands:
        write_progress(args.progress_file, stage)
        command_logs.append(run_command(command))

    model_dir, registered_images, sparse_points = find_largest_sparse_model(sparse_dir)
    initial_registered_images = registered_images
    initial_sparse_points = sparse_points
    recovery_diagnostics: dict[str, Any] | None = None
    if (
        video_selection is not None
        and video_selection.get("profile") == V2_PROFILE_ID
        and args.video_source is not None
        and args.video_selection is not None
    ):
        model_dir, recovery_diagnostics, recovery_logs = recover_video_registration(
            colmap=colmap,
            database_path=work_dir / "database.db",
            image_dir=args.image_dir,
            initial_model=model_dir,
            selection_path=args.video_selection,
            video_source=args.video_source,
            diagnostics_path=output_dir
            / "diagnostics"
            / "video_registration_recovery.json",
            use_gpu=args.use_gpu,
            gpu_index=args.gpu_index,
            num_threads=args.num_threads,
            progress=lambda stage: write_progress(args.progress_file, stage),
        )
        command_logs.extend(recovery_logs)
        registered_images, sparse_points = read_sparse_model_counts(model_dir)
    model_source = model_dir
    text_dir = work_dir / "sparse_txt"
    training_image_dir = args.image_dir
    if args.gaussian_baseline:
        undistorted_dir = work_dir / "undistorted"
        write_progress(args.progress_file, "colmap_undistortion")
        undistort_command = [
            colmap,
            "image_undistorter",
            "--image_path",
            str(args.image_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(undistorted_dir),
            "--output_type",
            "COLMAP",
        ]
        if args.max_image_size is not None:
            undistort_command.extend(("--max_image_size", str(args.max_image_size)))
        command_logs.append(run_command(undistort_command))
        model_source = undistorted_dir / "sparse"
        text_dir = undistorted_dir / "sparse_txt"
        training_image_dir = undistorted_dir / "images"
    text_dir.mkdir(parents=True, exist_ok=True)

    point_cloud_path = geometry_dir / "points.ply"
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_source),
                "--output_path",
                str(point_cloud_path),
                "--output_type",
                "PLY",
            ]
        )
    )
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_source),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )

    camera_payload = build_camera_payload(text_dir)
    if args.gaussian_baseline:
        camera_models = {camera["model"] for camera in camera_payload["cameras"]}
        if not camera_models <= {"SIMPLE_PINHOLE", "PINHOLE"}:
            raise RuntimeError(
                f"COLMAP undistortion produced unsupported camera models: {sorted(camera_models)}"
            )
        if len(camera_payload["images"]) < 12:
            raise RuntimeError("Gaussian baseline requires at least 12 registered images")
    write_json(geometry_dir / "cameras.json", camera_payload)

    num_points = read_ply_vertex_count(point_cloud_path)
    elapsed_seconds = time.perf_counter() - started_at
    log_lines = [
        "backend=colmap",
        f"num_images={len(image_paths)}",
        f"registered_images={len(camera_payload['images'])}",
        f"registration_ratio={len(camera_payload['images']) / len(image_paths):.6f}",
        f"num_points={num_points}",
        f"selected_sparse_model={model_dir.name}",
        f"selected_sparse_registered_images={registered_images}",
        f"selected_sparse_points={sparse_points}",
        f"initial_sparse_registered_images={initial_registered_images}",
        f"initial_sparse_points={initial_sparse_points}",
        f"video_registration_recovery_status={recovery_diagnostics['status'] if recovery_diagnostics is not None else 'not_requested'}",
        f"video_registration_recovery_rounds={len(recovery_diagnostics['rounds']) if recovery_diagnostics is not None else 0}",
        f"training_image_dir={training_image_dir}",
        f"camera_models={','.join(sorted(camera['model'] for camera in camera_payload['cameras']))}",
        f"matcher={args.matcher}",
        f"sequential_overlap={sequential_overlap_value if sequential_overlap_value is not None else 'default'}",
        f"vocab_tree={args.vocab_tree_path if args.vocab_tree_path is not None else 'none'}",
        f"single_camera={args.single_camera}",
        f"colmap_executable={colmap}",
        f"colmap_build={colmap_build}",
        f"use_gpu={args.use_gpu}",
        f"gpu_index={args.gpu_index if args.gpu_index is not None else 'all_visible'}",
        f"num_threads={args.num_threads if args.num_threads is not None else 'auto'}",
        f"max_image_size={args.max_image_size if args.max_image_size is not None else 'original'}",
        f"gaussian_baseline={args.gaussian_baseline}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        *command_logs,
    ]
    (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {point_cloud_path}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
    print(f"registered_images={len(camera_payload['images'])}")
    print(f"num_points={num_points}")


def parse_gpu_indices(value: str) -> str:
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated non-negative integers")
    return ",".join(str(int(part)) for part in parts)


def discover_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def write_progress(path: Path | None, stage: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"stage": stage}) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def colmap_version(colmap: str) -> str:
    completed = subprocess.run(
        [colmap, "-h"], check=True, capture_output=True, text=True
    )
    output = completed.stdout + completed.stderr
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " ".join(lines[:2]) or "unknown"


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return "command=" + " ".join(command) + "\nstdout=" + completed.stdout.strip() + "\nstderr=" + completed.stderr.strip()


def find_largest_sparse_model(sparse_dir: Path) -> tuple[Path, int, int]:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse models")
    ranked = [(*read_sparse_model_counts(path), path) for path in candidates]
    registered_images, points, selected = max(ranked, key=lambda item: (item[0], item[1], item[2].name))
    return selected, registered_images, points


def read_sparse_model_counts(model_dir: Path) -> tuple[int, int]:
    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"
    if images_bin.is_file() and points_bin.is_file():
        return _binary_count(images_bin), _binary_count(points_bin)
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"
    if images_txt.is_file() and points_txt.is_file():
        image_rows = [
            line
            for line in images_txt.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        point_rows = [
            line
            for line in points_txt.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        return len(image_rows) // 2, len(point_rows)
    raise RuntimeError(f"COLMAP sparse model is incomplete: {model_dir}")


def _binary_count(path: Path) -> int:
    with path.open("rb") as handle:
        payload = handle.read(8)
    if len(payload) != 8:
        raise RuntimeError(f"COLMAP binary model file is truncated: {path}")
    return int(struct.unpack("<Q", payload)[0])


def build_camera_payload(text_dir: Path) -> dict[str, Any]:
    cameras = parse_colmap_cameras(text_dir / "cameras.txt")
    images = parse_colmap_images(text_dir / "images.txt")
    return {
        "coordinate_system": "colmap_world",
        "cameras": cameras,
        "images": images,
    }


def parse_colmap_cameras(path: Path) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras.append(
            {
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(value) for value in parts[4:]],
            }
        )
    return cameras


def parse_colmap_images(path: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    data_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for index in range(0, len(data_lines), 2):
        parts = data_lines[index].split(maxsplit=9)
        images.append(
            {
                "image_id": int(parts[0]),
                "qvec": [float(value) for value in parts[1:5]],
                "tvec": [float(value) for value in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        )
    return images


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as file:
        for raw_line in file:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
