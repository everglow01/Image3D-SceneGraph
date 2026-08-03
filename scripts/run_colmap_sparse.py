from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP sparse SfM and export a point cloud.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaussian-baseline", action="store_true")
    args = parser.parse_args()
    if args.gaussian_baseline and args.matcher != "exhaustive":
        raise SystemExit("Gaussian baseline requires exhaustive COLMAP matching")

    started_at = time.perf_counter()
    colmap = shutil.which("colmap")
    if colmap is None:
        raise SystemExit("COLMAP executable not found. Install COLMAP and ensure `colmap` is on PATH.")

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
        "--SiftExtraction.use_gpu",
        "1" if args.use_gpu else "0",
    ]
    if args.gaussian_baseline:
        feature_command.extend(("--ImageReader.camera_model", "OPENCV"))
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
    commands = [
        feature_command,
        [
            colmap,
            "sequential_matcher" if args.matcher == "sequential" else "exhaustive_matcher",
            "--database_path",
            str(work_dir / "database.db"),
            "--SiftMatching.use_gpu",
            "1" if args.use_gpu else "0",
        ],
        mapper_command,
    ]
    command_logs = [run_command(command) for command in commands]

    model_dir, registered_images, sparse_points = find_largest_sparse_model(sparse_dir)
    model_source = model_dir
    text_dir = work_dir / "sparse_txt"
    training_image_dir = args.image_dir
    if args.gaussian_baseline:
        undistorted_dir = work_dir / "undistorted"
        command_logs.append(
            run_command(
                [
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
            )
        )
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
        f"training_image_dir={training_image_dir}",
        f"camera_models={','.join(sorted(camera['model'] for camera in camera_payload['cameras']))}",
        f"matcher={args.matcher}",
        f"single_camera={args.single_camera}",
        f"use_gpu={args.use_gpu}",
        f"gaussian_baseline={args.gaussian_baseline}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        *command_logs,
    ]
    (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {point_cloud_path}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
    print(f"registered_images={len(camera_payload['images'])}")
    print(f"num_points={num_points}")


def discover_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


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
