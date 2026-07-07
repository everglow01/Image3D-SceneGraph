from __future__ import annotations

import argparse
import json
import shutil
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
    args = parser.parse_args()

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
    text_dir = work_dir / "sparse_txt"
    database_path = work_dir / "database.db"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(args.image_dir),
            "--ImageReader.single_camera",
            "1" if args.single_camera else "0",
            "--SiftExtraction.use_gpu",
            "1" if args.use_gpu else "0",
        ],
        [
            colmap,
            "sequential_matcher" if args.matcher == "sequential" else "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "1" if args.use_gpu else "0",
        ],
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(args.image_dir),
            "--output_path",
            str(sparse_dir),
        ],
    ]
    command_logs = [run_command(command) for command in commands]

    model_dir = find_largest_sparse_model(sparse_dir)
    point_cloud_path = geometry_dir / "points.ply"
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_dir),
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
                str(model_dir),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )

    camera_payload = build_camera_payload(text_dir)
    write_json(geometry_dir / "cameras.json", camera_payload)

    num_points = read_ply_vertex_count(point_cloud_path)
    elapsed_seconds = time.perf_counter() - started_at
    log_lines = [
        "backend=colmap",
        f"num_images={len(image_paths)}",
        f"registered_images={len(camera_payload['images'])}",
        f"num_points={num_points}",
        f"matcher={args.matcher}",
        f"single_camera={args.single_camera}",
        f"use_gpu={args.use_gpu}",
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


def find_largest_sparse_model(sparse_dir: Path) -> Path:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse models")
    return max(candidates, key=lambda path: sum(file.stat().st_size for file in path.iterdir() if file.is_file()))


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
