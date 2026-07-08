from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    data_offset: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze point-cloud geometry and dominant planes.")
    parser.add_argument("--input", required=True, type=Path, help="Input PLY point cloud.")
    parser.add_argument("--output", type=Path, help="Output diagnostics JSON. Prints JSON when omitted.")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--ransac-iterations", type=int, default=400)
    parser.add_argument("--max-planes", type=int, default=3)
    parser.add_argument("--plane-distance", type=float, default=0.0, help="RANSAC inlier distance. 0 uses bbox-based auto threshold.")
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diagnostics = analyze_pointcloud(
        args.input,
        sample_size=args.sample_size,
        ransac_iterations=args.ransac_iterations,
        max_planes=args.max_planes,
        plane_distance=args.plane_distance,
        min_plane_inlier_ratio=args.min_plane_inlier_ratio,
        seed=args.seed,
    )

    text = json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def analyze_pointcloud(
    path: Path,
    *,
    sample_size: int = 50_000,
    ransac_iterations: int = 400,
    max_planes: int = 3,
    plane_distance: float = 0.0,
    min_plane_inlier_ratio: float = 0.08,
    seed: int = 42,
) -> dict[str, Any]:
    points = read_ply_points(path)
    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]

    diagnostics: dict[str, Any] = {
        "input": str(path),
        "num_points": int(len(points)),
        "finite_points": int(len(finite_points)),
        "sampled_points": 0,
        "bbox": None,
        "robust_bbox": None,
        "density_points_per_unit3": None,
        "robust_density_points_per_unit3": None,
        "dominant_planes": [],
        "quality_flags": [],
    }

    if len(points) == 0:
        diagnostics["quality_flags"].append("empty_point_cloud")
        return diagnostics
    if len(finite_points) == 0:
        diagnostics["quality_flags"].append("no_finite_points")
        return diagnostics

    bbox = compute_bbox(finite_points, lower=0, upper=100)
    robust_bbox = compute_bbox(finite_points, lower=1, upper=99)
    diagnostics["bbox"] = bbox
    diagnostics["robust_bbox"] = robust_bbox
    diagnostics["density_points_per_unit3"] = density(len(finite_points), bbox["size"])
    diagnostics["robust_density_points_per_unit3"] = density(len(finite_points), robust_bbox["size"])

    if len(finite_points) < 1_000:
        diagnostics["quality_flags"].append("low_point_count")

    sample = sample_points(finite_points, sample_size, seed)
    diagnostics["sampled_points"] = int(len(sample))
    threshold = plane_distance if plane_distance > 0 else auto_plane_distance(robust_bbox["diagonal"])
    planes = find_dominant_planes(
        sample,
        iterations=ransac_iterations,
        max_planes=max_planes,
        distance_threshold=threshold,
        seed=seed,
    )
    diagnostics["dominant_planes"] = planes

    if not planes or planes[0]["inlier_ratio"] < min_plane_inlier_ratio:
        diagnostics["quality_flags"].append("no_dominant_plane")

    return diagnostics


def read_ply_points(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = parse_ply_header(file)
        if header.vertex_count == 0:
            return np.empty((0, 3), dtype=np.float32)
        if header.fmt == "binary_little_endian":
            return read_binary_little_endian_vertices(file, header)
        if header.fmt == "ascii":
            return read_ascii_vertices(file, header)
    raise ValueError(f"Unsupported PLY format: {header.fmt}")


def parse_ply_header(file: Any) -> PlyHeader:
    first = file.readline()
    if first != b"ply\n":
        raise ValueError("Input is not a PLY file")

    fmt = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""

    while True:
        raw = file.readline()
        if not raw:
            raise ValueError("PLY header ended before end_header")
        line = raw.decode("ascii").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[:1] == ["format"]:
            fmt = parts[1]
        elif parts[:1] == ["element"]:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[:1] == ["property"] and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List vertex properties are not supported")
            vertex_properties.append((parts[2], parts[1]))

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format: {fmt}")
    if vertex_count < 0:
        raise ValueError("PLY vertex count cannot be negative")
    names = {name for name, _ in vertex_properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex properties must include x, y, z")
    return PlyHeader(fmt=fmt, vertex_count=vertex_count, vertex_properties=vertex_properties, data_offset=file.tell())


def read_binary_little_endian_vertices(file: Any, header: PlyHeader) -> np.ndarray:
    dtype = np.dtype([(name, PLY_SCALAR_TYPES[scalar_type]) for name, scalar_type in header.vertex_properties])
    vertices = np.fromfile(file, dtype=dtype, count=header.vertex_count)
    if len(vertices) != header.vertex_count:
        raise ValueError(f"Expected {header.vertex_count} vertices, found {len(vertices)}")
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32, copy=False)


def read_ascii_vertices(file: Any, header: PlyHeader) -> np.ndarray:
    if header.vertex_count == 0:
        return np.empty((0, 3), dtype=np.float32)
    raw = np.loadtxt(file, max_rows=header.vertex_count, dtype=np.float32, ndmin=2)
    property_names = [name for name, _ in header.vertex_properties]
    indices = [property_names.index(axis) for axis in ("x", "y", "z")]
    return raw[:, indices].astype(np.float32, copy=False)


def compute_bbox(points: np.ndarray, *, lower: float, upper: float) -> dict[str, Any]:
    if lower == 0 and upper == 100:
        min_corner = points.min(axis=0)
        max_corner = points.max(axis=0)
    else:
        min_corner = np.percentile(points, lower, axis=0)
        max_corner = np.percentile(points, upper, axis=0)
    size = np.maximum(max_corner - min_corner, 0)
    return {
        "min": vector_to_json(min_corner),
        "max": vector_to_json(max_corner),
        "size": vector_to_json(size),
        "diagonal": float(np.linalg.norm(size)),
    }


def density(num_points: int, bbox_size: list[float]) -> float | None:
    volume = float(np.prod(np.asarray(bbox_size, dtype=np.float64)))
    if volume <= 1e-12:
        return None
    return float(num_points / volume)


def sample_points(points: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(points) <= sample_size:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=sample_size, replace=False)
    return points[np.sort(indices)]


def auto_plane_distance(robust_diagonal: float) -> float:
    return max(float(robust_diagonal) * 0.006, 1e-4)


def find_dominant_planes(
    points: np.ndarray,
    *,
    iterations: int,
    max_planes: int,
    distance_threshold: float,
    seed: int,
) -> list[dict[str, Any]]:
    if len(points) < 3 or iterations <= 0 or max_planes <= 0:
        return []

    rng = np.random.default_rng(seed)
    remaining = points
    original_count = len(points)
    planes: list[dict[str, Any]] = []

    for plane_index in range(max_planes):
        if len(remaining) < 3:
            break
        plane = fit_best_plane_ransac(remaining, iterations, distance_threshold, rng)
        if plane is None:
            break
        normal, offset, inlier_mask = plane
        inliers = remaining[inlier_mask]
        if len(inliers) < 3:
            break
        planes.append(describe_plane(plane_index, normal, offset, inliers, len(inliers) / original_count, distance_threshold))
        remaining = remaining[~inlier_mask]

    return planes


def fit_best_plane_ransac(
    points: np.ndarray,
    iterations: int,
    distance_threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    best_count = 0
    best_plane: tuple[np.ndarray, float, np.ndarray] | None = None

    for _ in range(iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[indices]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))
        distances = np.abs(points @ normal + offset)
        inlier_mask = distances <= distance_threshold
        count = int(inlier_mask.sum())
        if count > best_count:
            best_count = count
            best_plane = (normal.astype(np.float64), offset, inlier_mask)

    if best_plane is None:
        return None

    inliers = points[best_plane[2]]
    refined_normal, refined_offset = fit_plane_svd(inliers)
    distances = np.abs(points @ refined_normal + refined_offset)
    refined_mask = distances <= distance_threshold
    return refined_normal, refined_offset, refined_mask


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0:
        raise ValueError("Cannot fit plane with zero normal")
    normal = normal / normal_norm
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def describe_plane(
    plane_index: int,
    normal: np.ndarray,
    offset: float,
    inliers: np.ndarray,
    inlier_ratio: float,
    distance_threshold: float,
) -> dict[str, Any]:
    centroid = inliers.mean(axis=0)
    basis_u, basis_v = plane_basis(normal)
    projected = np.column_stack([(inliers - centroid) @ basis_u, (inliers - centroid) @ basis_v])
    extent = np.percentile(projected, 95, axis=0) - np.percentile(projected, 5, axis=0)
    distances = np.abs(inliers @ normal + offset)
    return {
        "index": plane_index,
        "normal": vector_to_json(normal),
        "offset": float(offset),
        "inlier_count": int(len(inliers)),
        "inlier_ratio": float(inlier_ratio),
        "centroid": vector_to_json(centroid),
        "extent": vector_to_json(extent),
        "area_estimate": float(max(extent[0], 0) * max(extent[1], 0)),
        "rms_distance": float(math.sqrt(np.mean(distances**2))),
        "distance_threshold": float(distance_threshold),
    }


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(normal, reference)
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)
    return basis_u, basis_v


def vector_to_json(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


if __name__ == "__main__":
    main()
