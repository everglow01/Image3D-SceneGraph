from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pointcloud import analyze_pointcloud, read_ply_points_and_colors, vector_to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a point cloud to a dominant RANSAC plane.")
    parser.add_argument("--input", required=True, type=Path, help="Input PLY point cloud.")
    parser.add_argument("--output", required=True, type=Path, help="Output aligned PLY point cloud.")
    parser.add_argument("--diagnostics-output", type=Path, help="Output alignment diagnostics JSON.")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--ransac-iterations", type=int, default=400)
    parser.add_argument("--plane-distance", type=float, default=0.0)
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=0.08)
    parser.add_argument("--plane-index", type=int, default=0)
    parser.add_argument("--target-axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="z")
    parser.add_argument("--keep-plane-height", action="store_true", help="Do not translate the selected plane to target-axis 0.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = align_pointcloud(
        input_path=args.input,
        output_path=args.output,
        diagnostics_output=args.diagnostics_output,
        sample_size=args.sample_size,
        ransac_iterations=args.ransac_iterations,
        plane_distance=args.plane_distance,
        min_plane_inlier_ratio=args.min_plane_inlier_ratio,
        plane_index=args.plane_index,
        target_axis=args.target_axis,
        translate_plane_to_zero=not args.keep_plane_height,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def align_pointcloud(
    *,
    input_path: Path,
    output_path: Path,
    diagnostics_output: Path | None = None,
    sample_size: int = 50_000,
    ransac_iterations: int = 400,
    plane_distance: float = 0.0,
    min_plane_inlier_ratio: float = 0.08,
    plane_index: int = 0,
    target_axis: str = "z",
    translate_plane_to_zero: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    diagnostics = analyze_pointcloud(
        input_path,
        sample_size=sample_size,
        ransac_iterations=ransac_iterations,
        max_planes=max(plane_index + 1, 1),
        plane_distance=plane_distance,
        min_plane_inlier_ratio=min_plane_inlier_ratio,
        seed=seed,
    )
    planes = diagnostics["dominant_planes"]
    if plane_index < 0 or plane_index >= len(planes):
        raise SystemExit(f"Plane index {plane_index} is not available")

    plane = planes[plane_index]
    if plane["inlier_ratio"] < min_plane_inlier_ratio:
        raise SystemExit(
            f"Dominant plane is too weak: ratio={plane['inlier_ratio']:.3f}, "
            f"required={min_plane_inlier_ratio:.3f}"
        )

    points, colors = read_ply_points_and_colors(input_path)
    target = axis_vector(target_axis)
    source_normal = np.asarray(plane["normal"], dtype=np.float64)
    rotation = rotation_from_vectors(source_normal, target)
    pivot = np.asarray(plane["centroid"], dtype=np.float64)
    aligned = ((points.astype(np.float64) - pivot) @ rotation.T) + pivot

    translation = np.zeros(3, dtype=np.float64)
    if translate_plane_to_zero:
        pivot_aligned = ((pivot - pivot) @ rotation.T) + pivot
        axis_index = int(np.argmax(np.abs(target)))
        translation[axis_index] = -pivot_aligned[axis_index]
        aligned += translation

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(output_path, aligned.astype(np.float32), colors)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot + translation - rotation @ pivot

    result = {
        "status": "aligned",
        "input": str(input_path),
        "output": str(output_path),
        "num_points": int(len(points)),
        "colors_preserved": colors is not None,
        "target_axis": target_axis,
        "translate_plane_to_zero": translate_plane_to_zero,
        "source_plane": plane,
        "rotation": matrix_to_json(rotation),
        "translation": vector_to_json(translation),
        "transform": matrix_to_json(transform),
        "pre_alignment_diagnostics": diagnostics,
    }

    if diagnostics_output:
        diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def axis_vector(axis: str) -> np.ndarray:
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[-1]
    vector = np.zeros(3, dtype=np.float64)
    vector[{"x": 0, "y": 1, "z": 2}[name]] = sign
    return vector


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source)
    target = normalize(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        return rotation_around_axis(orthogonal_axis(source), np.pi)

    axis = normalize(np.cross(source, target))
    angle = float(np.arccos(dot))
    return rotation_around_axis(axis, angle)


def rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def orthogonal_axis(vector: np.ndarray) -> np.ndarray:
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(normalize(vector), reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return normalize(np.cross(vector, reference))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    if colors is None:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)
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


def matrix_to_json(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


if __name__ == "__main__":
    main()
