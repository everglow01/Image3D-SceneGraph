from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshOptions:
    method: str = "poisson"
    voxel_size: float = 0.05
    normal_radius: float = 0.2
    normal_max_nn: int = 30
    statistical_neighbors: int = 24
    statistical_std_ratio: float = 2.0
    radius_outlier_neighbors: int = 0
    radius_outlier_radius: float = 0.0
    poisson_depth: int = 8
    density_trim_quantile: float = 0.1
    component_min_ratio: float = 0.03
    edge_trim_quantile: float = 0.98
    edge_trim_factor: float = 2.5
    max_triangles: int = 120_000
    alpha: float = 0.0


def build_mesh_from_pointcloud(
    *,
    input_path: Path,
    output_path: Path,
    diagnostics_output: Path,
    options: MeshOptions,
) -> dict[str, Any]:
    o3d = _require_open3d()

    timings: dict[str, float] = {}
    started = time.perf_counter()
    point_cloud = o3d.io.read_point_cloud(str(input_path))
    input_points = len(point_cloud.points)
    timings["read_seconds"] = time.perf_counter() - started
    if input_points < 4:
        raise ValueError(f"point cloud has too few points for mesh reconstruction: {input_points}")

    started = time.perf_counter()
    point_cloud, preprocessing = _preprocess_point_cloud(point_cloud, options, o3d)
    processed_points = len(point_cloud.points)
    timings["preprocess_seconds"] = time.perf_counter() - started
    if processed_points < 4:
        raise ValueError(f"point cloud has too few points after preprocessing: {processed_points}")

    point_summary = _summarize_point_cloud(point_cloud, o3d)
    method_stats: dict[str, Any] = {}
    started = time.perf_counter()
    if options.method == "poisson":
        mesh, densities, crop_box = _poisson_mesh(point_cloud, options, o3d)
        density_threshold = _trim_low_density_vertices(mesh, densities, options)
        mesh = mesh.crop(crop_box)
        method_stats["density_threshold"] = density_threshold
    elif options.method == "ball_pivoting":
        mesh = _ball_pivoting_mesh(point_cloud, options, o3d)
        density_threshold = None
    elif options.method == "alpha_shape":
        alpha = _alpha_value(point_cloud, options)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
        density_threshold = None
        method_stats["alpha"] = alpha
    else:
        raise ValueError(f"unsupported mesh method: {options.method}")
    timings["reconstruct_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    mesh, cleanup = _cleanup_mesh(mesh, options)
    _transfer_vertex_colors(mesh, point_cloud, o3d)
    mesh.compute_vertex_normals()
    timings["cleanup_seconds"] = time.perf_counter() - started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise RuntimeError(f"failed to write mesh: {output_path}")
    timings["write_seconds"] = time.perf_counter() - started

    diagnostics = {
        "input": str(input_path),
        "output": str(output_path),
        "method": options.method,
        "options": asdict(options),
        "input_points": input_points,
        "processed_points": processed_points,
        "preprocessing": preprocessing,
        "point_summary": point_summary,
        "method_stats": method_stats,
        "cleanup": cleanup,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "density_threshold": density_threshold,
        "bbox_min": np.asarray(mesh.get_axis_aligned_bounding_box().min_bound).tolist(),
        "bbox_max": np.asarray(mesh.get_axis_aligned_bounding_box().max_bound).tolist(),
        "timings": timings,
    }
    diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return diagnostics


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for mesh output; install it with `uv add open3d`") from exc
    return o3d


def _preprocess_point_cloud(point_cloud: Any, options: MeshOptions, o3d: Any) -> tuple[Any, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"raw_points": len(point_cloud.points)}
    point_cloud.remove_non_finite_points()
    diagnostics["finite_points"] = len(point_cloud.points)
    if options.voxel_size > 0:
        before = len(point_cloud.points)
        point_cloud = point_cloud.voxel_down_sample(options.voxel_size)
        diagnostics["voxel_downsampled_points"] = len(point_cloud.points)
        diagnostics["voxel_removed_points"] = before - len(point_cloud.points)

    if options.statistical_neighbors > 0 and len(point_cloud.points) >= max(options.statistical_neighbors, 32):
        before = len(point_cloud.points)
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=options.statistical_neighbors,
            std_ratio=options.statistical_std_ratio,
        )
        diagnostics["statistical_outlier_removed_points"] = before - len(point_cloud.points)

    if (
        options.radius_outlier_neighbors > 0
        and options.radius_outlier_radius > 0
        and len(point_cloud.points) >= options.radius_outlier_neighbors
    ):
        before = len(point_cloud.points)
        point_cloud, _ = point_cloud.remove_radius_outlier(
            nb_points=options.radius_outlier_neighbors,
            radius=options.radius_outlier_radius,
        )
        diagnostics["radius_outlier_removed_points"] = before - len(point_cloud.points)

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=options.normal_radius,
            max_nn=options.normal_max_nn,
        )
    )
    if len(point_cloud.points) >= 30:
        try:
            point_cloud.orient_normals_consistent_tangent_plane(30)
            diagnostics["normal_orientation"] = "consistent_tangent_plane"
        except RuntimeError:
            diagnostics["normal_orientation"] = "failed"
    else:
        diagnostics["normal_orientation"] = "skipped_low_point_count"
    return point_cloud, diagnostics


def _poisson_mesh(point_cloud: Any, options: MeshOptions, o3d: Any) -> tuple[Any, np.ndarray, Any]:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=options.poisson_depth,
    )
    bbox = point_cloud.get_axis_aligned_bounding_box()
    return mesh, np.asarray(densities), bbox


def _ball_pivoting_mesh(point_cloud: Any, options: MeshOptions, o3d: Any) -> Any:
    radii = [
        max(options.voxel_size * 1.5, 0.01),
        max(options.voxel_size * 3.0, 0.02),
        max(options.voxel_size * 6.0, 0.04),
    ]
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(radii),
    )


def _alpha_value(point_cloud: Any, options: MeshOptions) -> float:
    if options.alpha > 0:
        return options.alpha
    spacing = _nearest_neighbor_spacing(np.asarray(point_cloud.points), sample_size=2_000)
    median = spacing.get("median")
    if isinstance(median, float) and median > 0:
        return median * 4.0
    return max(options.voxel_size * 4.0, 0.05)


def _trim_low_density_vertices(mesh: Any, densities: np.ndarray, options: MeshOptions) -> float | None:
    if len(densities) != len(mesh.vertices) or len(densities) == 0:
        return None
    threshold = float(np.quantile(densities, options.density_trim_quantile))
    mesh.remove_vertices_by_mask(densities < threshold)
    return threshold


def _cleanup_mesh(mesh: Any, options: MeshOptions) -> tuple[Any, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "initial_vertices": len(mesh.vertices),
        "initial_triangles": len(mesh.triangles),
    }
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    diagnostics.update(_remove_long_edge_triangles(mesh, options))
    diagnostics.update(_remove_small_components(mesh, options))
    if options.max_triangles > 0 and len(mesh.triangles) > options.max_triangles:
        diagnostics["triangles_before_decimation"] = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(options.max_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        diagnostics["triangles_after_decimation"] = len(mesh.triangles)
    diagnostics["final_vertices"] = len(mesh.vertices)
    diagnostics["final_triangles"] = len(mesh.triangles)
    return mesh, diagnostics


def _remove_long_edge_triangles(mesh: Any, options: MeshOptions) -> dict[str, Any]:
    if options.edge_trim_quantile <= 0 or options.edge_trim_factor <= 0 or len(mesh.triangles) == 0:
        return {"long_edge_removed_triangles": 0}
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        return {"long_edge_removed_triangles": 0}

    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    edge_lengths = np.concatenate(
        [
            np.linalg.norm(a - b, axis=1),
            np.linalg.norm(b - c, axis=1),
            np.linalg.norm(c - a, axis=1),
        ]
    )
    finite_lengths = edge_lengths[np.isfinite(edge_lengths)]
    if len(finite_lengths) == 0:
        return {"long_edge_removed_triangles": 0}

    quantile = min(max(options.edge_trim_quantile, 0.0), 1.0)
    threshold = float(np.quantile(finite_lengths, quantile) * options.edge_trim_factor)
    triangle_max_edges = np.maximum.reduce(
        [
            np.linalg.norm(a - b, axis=1),
            np.linalg.norm(b - c, axis=1),
            np.linalg.norm(c - a, axis=1),
        ]
    )
    remove_mask = triangle_max_edges > threshold
    removed = int(np.count_nonzero(remove_mask))
    if removed:
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    return {
        "long_edge_threshold": threshold,
        "long_edge_removed_triangles": removed,
    }


def _remove_small_components(mesh: Any, options: MeshOptions) -> dict[str, Any]:
    if len(mesh.triangles) == 0:
        return {
            "component_count": 0,
            "largest_component_triangles": 0,
            "small_component_removed_triangles": 0,
        }
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(triangle_clusters)
    counts = np.asarray(cluster_n_triangles)
    if len(counts) <= 1:
        return {
            "component_count": int(len(counts)),
            "largest_component_triangles": int(counts.max()) if len(counts) else 0,
            "small_component_removed_triangles": 0,
        }
    min_triangles = max(20, int(counts.max() * options.component_min_ratio))
    remove_mask = np.array([counts[cluster_id] < min_triangles for cluster_id in cluster_ids])
    removed = int(np.count_nonzero(remove_mask))
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return {
        "component_count": int(len(counts)),
        "largest_component_triangles": int(counts.max()),
        "component_min_triangles": int(min_triangles),
        "small_component_removed_triangles": removed,
    }


def _transfer_vertex_colors(mesh: Any, point_cloud: Any, o3d: Any) -> None:
    if not point_cloud.has_colors() or len(mesh.vertices) == 0:
        return
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    source_colors = np.asarray(point_cloud.colors)
    colors = []
    for vertex in mesh.vertices:
        _, indices, _ = tree.search_knn_vector_3d(vertex, 1)
        colors.append(source_colors[indices[0]] if indices else [0.8, 0.8, 0.8])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))


def _summarize_point_cloud(point_cloud: Any, o3d: Any) -> dict[str, Any]:
    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        return {"bbox": None, "robust_bbox": None, "nearest_neighbor_spacing": None}
    return {
        "bbox": _bbox(points, lower=0, upper=100),
        "robust_bbox": _bbox(points, lower=1, upper=99),
        "nearest_neighbor_spacing": _nearest_neighbor_spacing(points, sample_size=5_000, o3d=o3d),
    }


def _bbox(points: np.ndarray, *, lower: float, upper: float) -> dict[str, Any]:
    if lower == 0 and upper == 100:
        min_corner = points.min(axis=0)
        max_corner = points.max(axis=0)
    else:
        min_corner = np.percentile(points, lower, axis=0)
        max_corner = np.percentile(points, upper, axis=0)
    size = np.maximum(max_corner - min_corner, 0)
    return {
        "min": min_corner.astype(float).tolist(),
        "max": max_corner.astype(float).tolist(),
        "size": size.astype(float).tolist(),
        "diagonal": float(np.linalg.norm(size)),
    }


def _nearest_neighbor_spacing(points: np.ndarray, *, sample_size: int, o3d: Any | None = None) -> dict[str, float | int | None]:
    if len(points) < 2:
        return {"sampled_points": int(len(points)), "p10": None, "median": None, "p90": None}
    if o3d is None:
        o3d = _require_open3d()

    if len(points) > sample_size:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(len(points), size=sample_size, replace=False))
        sample = points[indices]
    else:
        sample = points

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    distances: list[float] = []
    for point in sample:
        _, _, squared = tree.search_knn_vector_3d(point, 2)
        if len(squared) >= 2 and squared[1] > 0:
            distances.append(float(np.sqrt(squared[1])))
    if not distances:
        return {"sampled_points": int(len(sample)), "p10": None, "median": None, "p90": None}
    values = np.asarray(distances, dtype=np.float64)
    return {
        "sampled_points": int(len(sample)),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a triangle mesh from a point cloud.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--method", choices=["poisson", "ball_pivoting", "alpha_shape"], default="poisson")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--normal-radius", type=float, default=0.2)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--statistical-neighbors", type=int, default=24)
    parser.add_argument("--statistical-std-ratio", type=float, default=2.0)
    parser.add_argument("--radius-outlier-neighbors", type=int, default=0)
    parser.add_argument("--radius-outlier-radius", type=float, default=0.0)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--density-trim-quantile", type=float, default=0.1)
    parser.add_argument("--component-min-ratio", type=float, default=0.03)
    parser.add_argument("--edge-trim-quantile", type=float, default=0.98)
    parser.add_argument("--edge-trim-factor", type=float, default=2.5)
    parser.add_argument("--max-triangles", type=int, default=120_000)
    parser.add_argument("--alpha", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = build_mesh_from_pointcloud(
        input_path=args.input,
        output_path=args.output,
        diagnostics_output=args.diagnostics_output,
        options=MeshOptions(
            method=args.method,
            voxel_size=args.voxel_size,
            normal_radius=args.normal_radius,
            normal_max_nn=args.normal_max_nn,
            statistical_neighbors=args.statistical_neighbors,
            statistical_std_ratio=args.statistical_std_ratio,
            radius_outlier_neighbors=args.radius_outlier_neighbors,
            radius_outlier_radius=args.radius_outlier_radius,
            poisson_depth=args.poisson_depth,
            density_trim_quantile=args.density_trim_quantile,
            component_min_ratio=args.component_min_ratio,
            edge_trim_quantile=args.edge_trim_quantile,
            edge_trim_factor=args.edge_trim_factor,
            max_triangles=args.max_triangles,
            alpha=args.alpha,
        ),
    )
    print(f"vertices={diagnostics['vertices']}")
    print(f"triangles={diagnostics['triangles']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
