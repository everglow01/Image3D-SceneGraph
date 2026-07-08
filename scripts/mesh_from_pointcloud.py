from __future__ import annotations

import argparse
import json
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
    poisson_depth: int = 8
    density_trim_quantile: float = 0.1
    component_min_ratio: float = 0.03
    max_triangles: int = 120_000


def build_mesh_from_pointcloud(
    *,
    input_path: Path,
    output_path: Path,
    diagnostics_output: Path,
    options: MeshOptions,
) -> dict[str, Any]:
    o3d = _require_open3d()

    point_cloud = o3d.io.read_point_cloud(str(input_path))
    input_points = len(point_cloud.points)
    if input_points < 4:
        raise ValueError(f"point cloud has too few points for mesh reconstruction: {input_points}")

    point_cloud = _preprocess_point_cloud(point_cloud, options, o3d)
    processed_points = len(point_cloud.points)
    if processed_points < 4:
        raise ValueError(f"point cloud has too few points after preprocessing: {processed_points}")

    if options.method == "poisson":
        mesh, densities, crop_box = _poisson_mesh(point_cloud, options, o3d)
        density_threshold = _trim_low_density_vertices(mesh, densities, options)
        mesh = mesh.crop(crop_box)
    elif options.method == "ball_pivoting":
        mesh = _ball_pivoting_mesh(point_cloud, options, o3d)
        density_threshold = None
    else:
        raise ValueError(f"unsupported mesh method: {options.method}")

    mesh = _cleanup_mesh(mesh, options)
    _transfer_vertex_colors(mesh, point_cloud, o3d)
    mesh.compute_vertex_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise RuntimeError(f"failed to write mesh: {output_path}")

    diagnostics = {
        "input": str(input_path),
        "output": str(output_path),
        "method": options.method,
        "options": asdict(options),
        "input_points": input_points,
        "processed_points": processed_points,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "density_threshold": density_threshold,
        "bbox_min": np.asarray(mesh.get_axis_aligned_bounding_box().min_bound).tolist(),
        "bbox_max": np.asarray(mesh.get_axis_aligned_bounding_box().max_bound).tolist(),
    }
    diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return diagnostics


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for mesh output; install it with `uv add open3d`") from exc
    return o3d


def _preprocess_point_cloud(point_cloud: Any, options: MeshOptions, o3d: Any) -> Any:
    point_cloud.remove_non_finite_points()
    if options.voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(options.voxel_size)

    if len(point_cloud.points) >= max(options.normal_max_nn, 32):
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=options.normal_radius,
            max_nn=options.normal_max_nn,
        )
    )
    if len(point_cloud.points) >= 30:
        try:
            point_cloud.orient_normals_consistent_tangent_plane(30)
        except RuntimeError:
            pass
    return point_cloud


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


def _trim_low_density_vertices(mesh: Any, densities: np.ndarray, options: MeshOptions) -> float | None:
    if len(densities) != len(mesh.vertices) or len(densities) == 0:
        return None
    threshold = float(np.quantile(densities, options.density_trim_quantile))
    mesh.remove_vertices_by_mask(densities < threshold)
    return threshold


def _cleanup_mesh(mesh: Any, options: MeshOptions) -> Any:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    _remove_small_components(mesh, options)
    if options.max_triangles > 0 and len(mesh.triangles) > options.max_triangles:
        mesh = mesh.simplify_quadric_decimation(options.max_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
    return mesh


def _remove_small_components(mesh: Any, options: MeshOptions) -> None:
    if len(mesh.triangles) == 0:
        return
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(triangle_clusters)
    counts = np.asarray(cluster_n_triangles)
    if len(counts) <= 1:
        return
    min_triangles = max(20, int(counts.max() * options.component_min_ratio))
    remove_mask = np.array([counts[cluster_id] < min_triangles for cluster_id in cluster_ids])
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a triangle mesh from a point cloud.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--method", choices=["poisson", "ball_pivoting"], default="poisson")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--normal-radius", type=float, default=0.2)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--density-trim-quantile", type=float, default=0.1)
    parser.add_argument("--component-min-ratio", type=float, default=0.03)
    parser.add_argument("--max-triangles", type=int, default=120_000)
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
            poisson_depth=args.poisson_depth,
            density_trim_quantile=args.density_trim_quantile,
            component_min_ratio=args.component_min_ratio,
            max_triangles=args.max_triangles,
        ),
    )
    print(f"vertices={diagnostics['vertices']}")
    print(f"triangles={diagnostics['triangles']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
