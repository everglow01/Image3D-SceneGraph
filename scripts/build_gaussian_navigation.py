from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from image3d_scenegraph.gaussian.dataset import (
    camera_from_normalized_transform,
    sha256_file,
    validate_contract,
)
from image3d_scenegraph.gaussian.evaluation import load_model_snapshot
from image3d_scenegraph.gaussian.render import RenderCamera, render_gaussians


SCHEMA_VERSION = 1
MIB = 1024 * 1024


class NavigationBuildError(RuntimeError):
    """Raised when conservative navigation assets cannot pass quality gates."""


@dataclass(frozen=True)
class BuildOptions:
    render_longest_edge: int = 512
    alpha_threshold: float = 0.35
    consistency_neighbors: int = 6
    consistency_min_support: int = 1
    consistency_relative_tolerance: float = 0.04
    voxel_length: float = 0.006
    sdf_trunc: float = 0.024
    depth_trunc: float = 4.0
    max_triangles: int = 50_000
    max_glb_bytes: int = 10 * MIB
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class DepthFrame:
    camera: RenderCamera
    depth: np.ndarray
    alpha: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NavigationBuildError(f"cannot read JSON asset {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NavigationBuildError(f"JSON asset must contain an object: {path}")
    return value


def _config_hash(record: dict[str, Any]) -> str:
    config = record.get("effective_config")
    if not isinstance(config, dict):
        raise NavigationBuildError("effective config is missing")
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_inputs(
    contract: dict[str, Any],
    config: dict[str, Any],
    export: dict[str, Any],
    model_path: Path,
) -> list[str]:
    validate_contract(contract)
    train_ids = [str(value) for value in contract["splits"]["train"]]
    held_out = {
        str(value)
        for split in ("validation", "test")
        for value in contract["splits"][split]
    }
    if not train_ids or set(train_ids) & held_out:
        raise NavigationBuildError("Train render selection overlaps Validation/Test")
    if contract["coordinate_system"]["world_units"] != "arbitrary":
        raise NavigationBuildError("navigation validation requires arbitrary world units")
    if export.get("coordinate_frame") != "normalized" or export.get("world_units") != "arbitrary":
        raise NavigationBuildError("Gaussian export must use normalized arbitrary coordinates")
    calculated_config_hash = _config_hash(config)
    expected = {
        "dataset_hash": contract["dataset_hash"],
        "effective_config_hash": calculated_config_hash,
        "model_sha256": sha256_file(model_path),
    }
    for key, value in expected.items():
        if export.get(key) != value:
            raise NavigationBuildError(f"provenance mismatch: {key}")
    if config.get("effective_config_hash") != calculated_config_hash:
        raise NavigationBuildError("effective config hash mismatch")
    return train_ids


def validate_train_images(
    contract: dict[str, Any],
    dataset_root: Path,
    train_ids: list[str],
    contract_path: Path | None = None,
) -> str:
    selected = set(train_ids)
    layout: str | None = None
    for entry in contract["images"]:
        if str(entry["image_id"]) not in selected:
            continue
        filename = Path(str(entry["path"])).name
        candidates = [
            (dataset_root / str(entry["path"]), "contract_paths"),
            (dataset_root / "images" / filename, "trainer_flattened_images"),
        ]
        if contract_path is not None:
            candidates.append(
                (
                    contract_path.parent / "graphdeco-dataset" / "images" / filename,
                    "graphdeco_frozen_train_images",
                )
            )
        matches = [
            (path, candidate_layout)
            for path, candidate_layout in candidates
            if path.is_file() and sha256_file(path) == entry["sha256"]
        ]
        if not matches:
            raise NavigationBuildError(f"Train image hash mismatch: {entry['path']}")
        _, current_layout = matches[0]
        if layout is not None and layout != current_layout:
            raise NavigationBuildError("Train images use mixed dataset layouts")
        layout = current_layout
    return layout or "none"


def load_sparse_initialization(
    contract: dict[str, Any], contract_path: Path
) -> tuple[np.ndarray, str]:
    initialization = contract["initialization"]
    if initialization.get("coordinate_frame") != "normalized":
        raise NavigationBuildError("hybrid floor support requires normalized sparse initialization")
    asset = initialization.get("asset")
    expected_hash = initialization.get("sha256")
    if not isinstance(asset, str) or not isinstance(expected_hash, str):
        raise NavigationBuildError("hybrid floor support requires a contracted sparse asset")
    path = (contract_path.parent / asset).resolve()
    try:
        path.relative_to(contract_path.parent.resolve())
    except ValueError as exc:
        raise NavigationBuildError("sparse initialization path escapes preparation root") from exc
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise NavigationBuildError("sparse initialization hash mismatch")
    try:
        with np.load(path) as archive:
            points = np.asarray(archive["points"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise NavigationBuildError(f"cannot load sparse initialization: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise NavigationBuildError("sparse initialization points must be finite N x 3")
    return points, expected_hash


def _camera_sequence_key(entry: dict[str, Any]) -> tuple[str, int] | None:
    name = Path(str(entry["path"])).stem
    match = re.fullmatch(r"(.*?)(\d+)", name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def load_train_cameras(
    contract: dict[str, Any], train_ids: list[str], longest_edge: int
) -> list[RenderCamera]:
    import torch

    selected = set(train_ids)
    normalization = contract["normalization"]
    cameras: list[RenderCamera] = []
    for entry in contract["images"]:
        image_id = str(entry["image_id"])
        if image_id not in selected:
            continue
        width, height = int(entry["width"]), int(entry["height"])
        scale = min(1.0, longest_edge / max(width, height))
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        intrinsic = np.asarray(entry["intrinsic"], dtype=np.float64).copy()
        intrinsic[0] *= target_width / width
        intrinsic[1] *= target_height / height
        camera_from_normalized = camera_from_normalized_transform(
            entry["camera_from_world"], normalization
        )
        cameras.append(
            RenderCamera(
                image_id=image_id,
                camera_from_normalized=torch.tensor(camera_from_normalized, dtype=torch.float32),
                intrinsic=torch.tensor(intrinsic, dtype=torch.float32),
                width=target_width,
                height=target_height,
            )
        )
    if {camera.image_id for camera in cameras} != selected:
        raise NavigationBuildError("Train split does not resolve to every contracted camera")
    return cameras


def train_trajectory_pairs(
    contract: dict[str, Any], train_ids: list[str], cameras: list[RenderCamera]
) -> list[tuple[int, int]]:
    selected = set(train_ids)
    camera_index = {camera.image_id: index for index, camera in enumerate(cameras)}
    ordered = []
    for entry in contract["images"]:
        image_id = str(entry["image_id"])
        sequence = _camera_sequence_key(entry)
        if image_id in selected and sequence is not None:
            ordered.append((sequence, image_id))
    ordered.sort(key=lambda item: item[0])
    pairs = []
    for (previous, previous_id), (current, current_id) in zip(ordered, ordered[1:]):
        if previous[0] == current[0] and current[1] - previous[1] <= 2:
            pairs.append((camera_index[previous_id], camera_index[current_id]))
    return pairs


def _camera_center(camera: RenderCamera) -> np.ndarray:
    return np.linalg.inv(camera.camera_from_normalized.cpu().numpy())[:3, 3]


def _camera_up(cameras: list[RenderCamera]) -> np.ndarray:
    vectors = []
    for camera in cameras:
        world_from_camera = np.linalg.inv(camera.camera_from_normalized.cpu().numpy())
        vector = -world_from_camera[:3, 1]
        vectors.append(vector / np.linalg.norm(vector))
    up = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 0.5:
        raise NavigationBuildError("Train camera image-up vectors do not define stable scene up")
    return up / norm


def render_train_depth(
    model: Any,
    cameras: list[RenderCamera],
    *,
    sh_degree: int,
    alpha_threshold: float,
    preview_dir: Path,
    deadline: float,
) -> list[DepthFrame]:
    import torch

    frames: list[DepthFrame] = []
    preview_indices = set(np.linspace(0, len(cameras) - 1, min(6, len(cameras)), dtype=int))
    with torch.no_grad():
        for index, camera_cpu in enumerate(cameras):
            _check_deadline(deadline)
            camera = RenderCamera(
                image_id=camera_cpu.image_id,
                camera_from_normalized=camera_cpu.camera_from_normalized.to(model.means.device),
                intrinsic=camera_cpu.intrinsic.to(model.means.device),
                width=camera_cpu.width,
                height=camera_cpu.height,
            )
            rendered = render_gaussians(
                model, camera, sh_degree=sh_degree, background=None, render_mode="RGB+ED"
            )
            if rendered.depth is None:
                raise NavigationBuildError("Gaussian renderer did not return expected depth")
            depth = rendered.depth.detach().float().cpu().numpy()
            alpha = rendered.alpha.detach().float().cpu().numpy()
            depth = np.squeeze(depth).astype(np.float32)
            alpha = np.squeeze(alpha).astype(np.float32)
            valid = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha >= alpha_threshold)
            depth = np.where(valid, depth, 0.0).astype(np.float32)
            alpha = np.where(np.isfinite(alpha), np.clip(alpha, 0, 1), 0).astype(np.float32)
            frames.append(DepthFrame(camera_cpu, depth, alpha))
            if index in preview_indices:
                _write_scalar_preview(preview_dir / f"{index:03d}-{camera.image_id}-depth.png", depth)
                _write_scalar_preview(preview_dir / f"{index:03d}-{camera.image_id}-alpha.png", alpha, unit=True)
    return frames


def camera_neighbors(frames: list[DepthFrame], count: int) -> list[list[int]]:
    centers = np.stack([_camera_center(frame.camera) for frame in frames])
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return [np.argsort(row)[: min(count, len(frames) - 1)].tolist() for row in distances]


def filter_multiview_depth(
    frames: list[DepthFrame],
    *,
    neighbor_count: int,
    min_support: int,
    relative_tolerance: float,
    alpha_threshold: float,
    preview_dir: Path,
    deadline: float,
) -> tuple[list[DepthFrame], dict[str, Any]]:
    neighbors = camera_neighbors(frames, neighbor_count)
    filtered: list[DepthFrame] = []
    total_valid = total_kept = total_supported = total_contradicted = 0
    preview_indices = set(np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int))
    for index, frame in enumerate(frames):
        _check_deadline(deadline)
        valid_y, valid_x = np.nonzero(frame.depth > 0)
        total_valid += len(valid_x)
        support = np.zeros(len(valid_x), dtype=np.uint8)
        contradicted = np.zeros(len(valid_x), dtype=bool)
        if len(valid_x):
            z = frame.depth[valid_y, valid_x].astype(np.float64)
            intrinsic = frame.camera.intrinsic.cpu().numpy().astype(np.float64)
            points_camera = np.stack(
                ((valid_x - intrinsic[0, 2]) * z / intrinsic[0, 0],
                 (valid_y - intrinsic[1, 2]) * z / intrinsic[1, 1], z),
                axis=0,
            )
            world_from_camera = np.linalg.inv(frame.camera.camera_from_normalized.cpu().numpy())
            points_world = world_from_camera[:3, :3] @ points_camera + world_from_camera[:3, 3:4]
            for neighbor_index in neighbors[index]:
                neighbor = frames[neighbor_index]
                transform = neighbor.camera.camera_from_normalized.cpu().numpy().astype(np.float64)
                projected = transform[:3, :3] @ points_world + transform[:3, 3:4]
                projected_z = projected[2]
                neighbor_k = neighbor.camera.intrinsic.cpu().numpy().astype(np.float64)
                u = np.rint(projected[0] * neighbor_k[0, 0] / projected_z + neighbor_k[0, 2]).astype(np.int64)
                v = np.rint(projected[1] * neighbor_k[1, 1] / projected_z + neighbor_k[1, 2]).astype(np.int64)
                inside = (
                    (projected_z > 0)
                    & (u >= 0) & (u < neighbor.camera.width)
                    & (v >= 0) & (v < neighbor.camera.height)
                )
                indices = np.flatnonzero(inside)
                if not len(indices):
                    continue
                observed_depth = neighbor.depth[v[indices], u[indices]].astype(np.float64)
                observed = (
                    (observed_depth > 0)
                    & (neighbor.alpha[v[indices], u[indices]] >= alpha_threshold)
                )
                indices = indices[observed]
                if not len(indices):
                    continue
                observed_depth = neighbor.depth[v[indices], u[indices]].astype(np.float64)
                tolerance = relative_tolerance * np.maximum(observed_depth, projected_z[indices])
                difference = observed_depth - projected_z[indices]
                support[indices] += np.abs(difference) <= tolerance
                contradicted[indices] |= difference > tolerance
        keep = support >= min_support
        output = np.zeros_like(frame.depth)
        output[valid_y[keep], valid_x[keep]] = frame.depth[valid_y[keep], valid_x[keep]]
        filtered.append(DepthFrame(frame.camera, output, frame.alpha))
        total_kept += int(keep.sum())
        total_supported += int((support > 0).sum())
        total_contradicted += int(contradicted.sum())
        if index in preview_indices:
            preview = np.zeros(frame.depth.shape + (3,), dtype=np.uint8)
            preview[valid_y, valid_x] = (235, 104, 52)
            preview[valid_y[support > 0], valid_x[support > 0]] = (27, 175, 122)
            preview[valid_y[keep], valid_x[keep]] = (42, 120, 214)
            Image.fromarray(preview).save(preview_dir / f"{index:03d}-{frame.camera.image_id}-support.png")
    if total_kept == 0:
        raise NavigationBuildError("multi-view filtering rejected every depth pixel")
    return filtered, {
        "input_valid_pixels": total_valid,
        "supported_pixels": total_supported,
        "contradicted_pixels": total_contradicted,
        "kept_pixels": total_kept,
        "kept_ratio": total_kept / max(total_valid, 1),
    }


def mesh_vertex_support(
    mesh: Any,
    frames: list[DepthFrame],
    *,
    relative_tolerance: float,
    voxel: float,
    deadline: float,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    support = np.zeros(len(vertices), dtype=np.uint16)
    for frame in frames:
        _check_deadline(deadline)
        transform = frame.camera.camera_from_normalized.cpu().numpy().astype(np.float64)
        projected = vertices @ transform[:3, :3].T + transform[:3, 3]
        z = projected[:, 2]
        intrinsic = frame.camera.intrinsic.cpu().numpy().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(projected[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2]).astype(np.int64)
            v = np.rint(projected[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2]).astype(np.int64)
        inside = (
            (z > 0)
            & (u >= 0)
            & (u < frame.camera.width)
            & (v >= 0)
            & (v < frame.camera.height)
        )
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        observed = frame.depth[v[indices], u[indices]].astype(np.float64)
        visible = observed > 0
        indices = indices[visible]
        observed = observed[visible]
        tolerance = np.maximum(
            voxel * 2,
            relative_tolerance * np.maximum(observed, z[indices]),
        )
        support[indices] += np.abs(observed - z[indices]) <= tolerance
    return support


def fuse_tsdf(frames: list[DepthFrame], options: BuildOptions, deadline: float) -> tuple[Any, dict[str, Any]]:
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=options.voxel_length,
        sdf_trunc=options.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    integrated = valid_pixels = 0
    for frame in frames:
        _check_deadline(deadline)
        valid = frame.depth > 0
        if not valid.any():
            continue
        depth_image = o3d.geometry.Image(np.ascontiguousarray(frame.depth, dtype=np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((*frame.depth.shape, 3), dtype=np.uint8)),
            depth_image,
            depth_scale=1.0,
            depth_trunc=options.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        k = frame.camera.intrinsic.cpu().numpy()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            frame.camera.width,
            frame.camera.height,
            float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
        )
        volume.integrate(rgbd, intrinsic, frame.camera.camera_from_normalized.cpu().numpy())
        integrated += 1
        valid_pixels += int(valid.sum())
    if integrated < max(3, math.ceil(len(frames) * 0.8)):
        raise NavigationBuildError(
            f"TSDF integrated too few Train views: {integrated}/{len(frames)}"
        )
    mesh = volume.extract_triangle_mesh()
    return mesh, {"input_frames": len(frames), "integrated_frames": integrated, "valid_pixels": valid_pixels}


def clean_mesh(mesh: Any, max_triangles: int) -> dict[str, int]:
    before = len(mesh.triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > max_triangles:
        simplified = mesh.simplify_quadric_decimation(max_triangles)
        mesh.vertices = simplified.vertices
        mesh.triangles = simplified.triangles
        mesh.vertex_normals = simplified.vertex_normals
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return {
        "triangles_before_cleanup": before,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
    }


def deterministic_plane(
    points: np.ndarray, *, distance_threshold: float, iterations: int = 2_000
) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(points), size=(iterations, 3))
    best_count = -1
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for start in range(0, iterations, 64):
        triples = points[samples[start : start + 64]]
        normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-10
        normals[valid] /= lengths[valid, None]
        offsets = np.einsum("ij,ij->i", normals, triples[:, 0])
        distances = np.abs(points @ normals.T - offsets)
        counts = (distances <= distance_threshold).sum(axis=0)
        counts[~valid] = -1
        local = int(np.argmax(counts))
        if int(counts[local]) > best_count:
            best_count = int(counts[local])
            best_normal = normals[local].copy()
            best_offset = float(offsets[local])
    if best_normal is None:
        raise NavigationBuildError("sparse initialization cannot define a plane")
    inliers = np.abs(points @ best_normal - best_offset) <= distance_threshold
    for _ in range(3):
        selected = points[inliers]
        center = selected.mean(axis=0)
        _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
        best_normal = axes[-1]
        best_offset = float(center @ best_normal)
        inliers = np.abs(points @ best_normal - best_offset) <= distance_threshold
    return best_normal, best_offset, np.flatnonzero(inliers)


def infer_floor(
    mesh: Any,
    sparse_points: np.ndarray,
    cameras: list[RenderCamera],
    camera_up: np.ndarray,
    voxel: float,
) -> tuple[np.ndarray, float, float, np.ndarray, dict[str, Any]]:
    points = np.asarray(mesh.vertices)
    camera_centers = np.stack([_camera_center(camera) for camera in cameras])
    normal, offset, sparse_inliers = deterministic_plane(
        sparse_points, distance_threshold=max(voxel * 2.0, 0.012)
    )
    signs = []
    for sign in (1.0, -1.0):
        up = normal * sign
        floor_offset = offset * sign
        heights = camera_centers @ up - floor_offset
        positive = heights > max(voxel * 2, 0.012)
        selected = heights[positive]
        if len(selected) < max(10, math.ceil(len(cameras) * 0.5)):
            continue
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        signs.append(
            {
                "up": up,
                "floor_offset": floor_offset,
                "positive_camera_count": int(positive.sum()),
                "height_median": median,
                "height_mad": mad,
                "height_mad_ratio": mad / max(median, 1e-12),
                "camera_up_alignment": float(up @ camera_up),
            }
        )
    if not signs:
        raise NavigationBuildError("sparse floor plane does not lie below Train cameras")
    winner = max(
        signs,
        key=lambda candidate: (
            candidate["positive_camera_count"],
            candidate["camera_up_alignment"],
            -candidate["height_mad_ratio"],
        ),
    )
    if winner["height_mad_ratio"] > 0.35 or winner["camera_up_alignment"] < 0.8:
        raise NavigationBuildError("sparse floor orientation/height is unstable")
    sparse_floor_points = sparse_points[np.asarray(sparse_inliers, dtype=np.int64)]
    return (
        winner["up"],
        float(winner["floor_offset"]),
        float(winner["height_median"]),
        sparse_floor_points,
        {
            key: value for key, value in winner.items() if key != "up"
        }
        | {
            "source": "contracted_colmap_sparse_dominant_plane",
            "sparse_points": len(sparse_points),
            "sparse_plane_inliers": len(sparse_inliers),
            "sparse_plane_inlier_ratio": len(sparse_inliers) / len(sparse_points),
            "tsdf_vertices": len(points),
        },
    )


def floor_basis(up: np.ndarray, floor_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    basis_u = np.cross(up, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(up, basis_u)
    return up * floor_offset, basis_u, basis_v


def _uv(points: np.ndarray, origin: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray) -> np.ndarray:
    relative = points - origin
    return np.stack((relative @ basis_u, relative @ basis_v), axis=-1)


def _grid_index(points_uv: np.ndarray, minimum: np.ndarray, cell: float) -> np.ndarray:
    return np.floor((points_uv - minimum) / cell).astype(int)


def _draw_grid_line(
    image: Image.Image, start: np.ndarray, end: np.ndarray, *, fill: int, width: int
) -> None:
    ImageDraw.Draw(image).line(
        [tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))],
        fill=fill,
        width=width,
    )


def remove_diagonal_pinches(mask: np.ndarray) -> tuple[np.ndarray, int]:
    output = mask.copy()
    removed = 0
    while True:
        neighborhood = ndimage.convolve(
            output.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
        )
        selected: tuple[int, int] | None = None
        for y in range(output.shape[0] - 1):
            for x in range(output.shape[1] - 1):
                block = output[y : y + 2, x : x + 2]
                if block[0, 0] and block[1, 1] and not block[0, 1] and not block[1, 0]:
                    choices = ((y, x), (y + 1, x + 1))
                elif block[0, 1] and block[1, 0] and not block[0, 0] and not block[1, 1]:
                    choices = ((y, x + 1), (y + 1, x))
                else:
                    continue
                selected = min(choices, key=lambda point: (neighborhood[point], point))
                break
            if selected is not None:
                break
        if selected is None:
            return output, removed
        output[selected] = False
        removed += 1


def protect_train_passages(
    floor_mask: np.ndarray,
    support_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    camera_pixels: np.ndarray,
    camera_heights: np.ndarray,
    trajectory_pairs: list[tuple[int, int]],
    *,
    cell: float,
    height: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    radius_cells = max(1, int(math.ceil(0.18 * height / cell)))
    max_step = 0.5 * height / cell
    candidate = Image.new("1", (floor_mask.shape[1], floor_mask.shape[0]))
    accepted_pairs = 0
    rejected_distance = rejected_level = 0
    for first, second in trajectory_pairs:
        start = camera_pixels[first]
        end = camera_pixels[second]
        if float(np.linalg.norm(end - start)) > max_step:
            rejected_distance += 1
            continue
        if abs(float(camera_heights[first] - camera_heights[second])) > 0.12 * height:
            rejected_level += 1
            continue
        _draw_grid_line(candidate, start, end, fill=1, width=radius_cells * 2 + 1)
        accepted_pairs += 1
    corridor = np.asarray(candidate, dtype=bool).copy()
    corridor &= floor_mask & support_mask
    # Camera trajectories provide free-space evidence only where the Capsule centerline
    # remains clear after the same radius expansion used for normal navigation.
    corridor &= ~ndimage.binary_dilation(obstacle_mask, iterations=radius_cells)
    return corridor, {
        "candidate_pairs": len(trajectory_pairs),
        "accepted_pairs": accepted_pairs,
        "rejected_distance_pairs": rejected_distance,
        "rejected_level_change_pairs": rejected_level,
        "protected_cells": int(corridor.sum()),
    }


def build_walkable_mask(
    mesh: Any,
    sparse_floor_points: np.ndarray,
    cameras: list[RenderCamera],
    trajectory_pairs: list[tuple[int, int]],
    up: np.ndarray,
    floor_offset: float,
    height: float,
    voxel: float,
    debug_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float, tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    cross = np.cross(b - a, c - a)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    centers = (a + b + c) / 3
    levels = centers @ up - floor_offset
    floor_triangles = (np.abs(normals @ up) >= math.cos(math.radians(35))) & (np.abs(levels) <= max(voxel * 3, height * 0.08))
    if floor_triangles.sum() < 20:
        raise NavigationBuildError("collision mesh has no usable walkable floor triangles")

    origin, basis_u, basis_v = floor_basis(up, floor_offset)
    floor_uv = _uv(vertices[triangles[floor_triangles]].reshape(-1, 3), origin, basis_u, basis_v)
    sparse_floor_uv = _uv(sparse_floor_points, origin, basis_u, basis_v)
    camera_centers = np.stack([_camera_center(camera) for camera in cameras])
    camera_uv = _uv(camera_centers, origin, basis_u, basis_v)
    margin = height
    minimum = np.minimum.reduce(
        (floor_uv.min(axis=0), sparse_floor_uv.min(axis=0), camera_uv.min(axis=0))
    ) - margin
    maximum = np.maximum.reduce(
        (floor_uv.max(axis=0), sparse_floor_uv.max(axis=0), camera_uv.max(axis=0))
    ) + margin
    cell = max(voxel * 1.5, height * 0.035)
    shape_xy = np.ceil((maximum - minimum) / cell).astype(int) + 1
    largest = int(shape_xy.max())
    if largest > 1024:
        cell *= largest / 1024
        shape_xy = np.ceil((maximum - minimum) / cell).astype(int) + 1
    width, rows = int(shape_xy[0]), int(shape_xy[1])
    floor_image = Image.new("1", (width, rows))
    draw_floor = ImageDraw.Draw(floor_image)
    for triangle in vertices[triangles[floor_triangles]]:
        polygon = _grid_index(_uv(triangle, origin, basis_u, basis_v), minimum, cell)
        draw_floor.polygon([tuple(point) for point in polygon], fill=1)
    floor_mask = np.asarray(floor_image, dtype=bool).copy()
    sparse_mask = np.zeros_like(floor_mask)
    sparse_pixels = _grid_index(sparse_floor_uv, minimum, cell)
    valid_sparse = (
        (sparse_pixels[:, 0] >= 0)
        & (sparse_pixels[:, 0] < width)
        & (sparse_pixels[:, 1] >= 0)
        & (sparse_pixels[:, 1] < rows)
    )
    sparse_mask[sparse_pixels[valid_sparse, 1], sparse_pixels[valid_sparse, 0]] = True
    sparse_density = ndimage.convolve(
        sparse_mask.astype(np.uint16), np.ones((3, 3), dtype=np.uint16), mode="constant"
    )
    sparse_mask = ndimage.binary_closing(sparse_density >= 2, iterations=2)
    sparse_mask = ndimage.binary_opening(sparse_mask, iterations=1)
    floor_mask |= sparse_mask

    support_image = Image.new("1", (width, rows))
    draw_support = ImageDraw.Draw(support_image)
    support_radius = max(1, int(math.ceil(height * 0.5 / cell)))
    for camera, center_uv in zip(cameras, camera_uv):
        pixel = _grid_index(center_uv[None], minimum, cell)[0]
        draw_support.ellipse(
            (pixel[0] - support_radius, pixel[1] - support_radius, pixel[0] + support_radius, pixel[1] + support_radius),
            fill=1,
        )
        world_from_camera = np.linalg.inv(camera.camera_from_normalized.cpu().numpy())
        k = camera.intrinsic.cpu().numpy()
        intersections = [center_uv]
        for x, y in ((0, camera.height - 1), (camera.width - 1, camera.height - 1), (camera.width / 2, camera.height - 1)):
            direction_camera = np.array([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], 1.0])
            direction = world_from_camera[:3, :3] @ direction_camera
            denominator = float(direction @ up)
            if denominator >= -1e-6:
                continue
            distance = (floor_offset - float(_camera_center(camera) @ up)) / denominator
            if 0 < distance <= height * 4:
                point = _camera_center(camera) + distance * direction
                intersections.append(_uv(point[None], origin, basis_u, basis_v)[0])
        if len(intersections) >= 3:
            polygon = _grid_index(np.asarray(intersections), minimum, cell)
            draw_support.polygon([tuple(point) for point in polygon], fill=1)
    support_mask = np.asarray(support_image, dtype=bool)

    obstacle_image = Image.new("1", (width, rows))
    draw_obstacle = ImageDraw.Draw(obstacle_image)
    vertical = np.abs(normals @ up) < 0.7
    triangle_min = np.minimum.reduce((a @ up, b @ up, c @ up)) - floor_offset
    triangle_max = np.maximum.reduce((a @ up, b @ up, c @ up)) - floor_offset
    obstacles = (
        vertical
        & (triangle_max >= height * 0.08)
        & (triangle_min <= height * 1.12)
    )
    for triangle in vertices[triangles[obstacles]]:
        polygon = _grid_index(_uv(triangle, origin, basis_u, basis_v), minimum, cell)
        points = [tuple(point) for point in polygon] + [tuple(polygon[0])]
        draw_obstacle.line(points, fill=1, width=2)
    obstacle_mask = np.asarray(obstacle_image, dtype=bool).copy()
    labels, obstacle_components = ndimage.label(obstacle_mask)
    component_sizes = np.bincount(labels.ravel())
    minimum_obstacle_cells = max(2, int(math.ceil(0.05 * height / cell)))
    retained_labels = np.flatnonzero(component_sizes >= minimum_obstacle_cells)
    retained_labels = retained_labels[retained_labels != 0]
    obstacle_mask &= np.isin(labels, retained_labels)
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((floor_mask * 255).astype(np.uint8)).save(debug_dir / "floor-mask.png")
    Image.fromarray((sparse_mask * 255).astype(np.uint8)).save(debug_dir / "sparse-floor-mask.png")
    Image.fromarray((support_mask * 255).astype(np.uint8)).save(debug_dir / "support-mask.png")
    Image.fromarray((obstacle_mask * 255).astype(np.uint8)).save(debug_dir / "obstacle-mask-raw.png")

    camera_pixels = _grid_index(camera_uv, minimum, cell)
    camera_heights = camera_centers @ up - floor_offset
    passage_mask, passage_stats = protect_train_passages(
        floor_mask,
        support_mask,
        obstacle_mask,
        camera_pixels,
        camera_heights,
        trajectory_pairs,
        cell=cell,
        height=height,
    )
    radius = 0.18 * height
    iterations = max(1, int(math.ceil(radius / cell)))
    obstacle_mask = ndimage.binary_dilation(obstacle_mask, iterations=iterations)
    walkable = floor_mask & support_mask & ~obstacle_mask
    walkable = ndimage.binary_opening(walkable, iterations=1)
    walkable |= passage_mask
    Image.fromarray((passage_mask * 255).astype(np.uint8)).save(debug_dir / "protected-passages-mask.png")
    Image.fromarray((obstacle_mask * 255).astype(np.uint8)).save(debug_dir / "obstacle-mask-expanded.png")
    Image.fromarray((walkable * 255).astype(np.uint8)).save(debug_dir / "walkable-mask.png")
    labels, count = ndimage.label(walkable)
    if count == 0:
        raise NavigationBuildError("floor/support/collision intersection has no walkable component")
    camera_pixels = _grid_index(camera_uv, minimum, cell)
    scores = []
    for label in range(1, count + 1):
        area = int((labels == label).sum())
        camera_count = sum(
            0 <= x < width and 0 <= y < rows and labels[y, x] == label
            for x, y in camera_pixels
        )
        scores.append((camera_count, area, label))
    camera_count, area, selected_label = max(scores)
    component = labels == selected_label
    component, diagonal_pinch_cells_removed = remove_diagonal_pinches(component)
    labels, count = ndimage.label(component)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        component = labels == int(np.argmax(sizes))
        area = int(component.sum())
        camera_count = sum(
            0 <= x < width and 0 <= y < rows and component[y, x]
            for x, y in camera_pixels
        )
    if camera_count == 0 or area < 25:
        raise NavigationBuildError(
            "no Train camera belongs to a usable walkable component: "
            f"floor_cells={int(floor_mask.sum())}, "
            f"support_cells={int(support_mask.sum())}, "
            f"obstacle_cells={int(obstacle_mask.sum())}, "
            f"walkable_cells={int(walkable.sum())}, "
            f"floor_triangles={int(floor_triangles.sum())}, "
            f"height={height:.6f}, floor_offset={floor_offset:.6f}, "
            f"up={up.tolist()}, components={scores}"
        )
    return component, minimum, cell, (origin, basis_u, basis_v), {
        "floor_triangle_count": int(floor_triangles.sum()),
        "floor_area": float(areas[floor_triangles].sum()),
        "sparse_floor_point_count": len(sparse_floor_points),
        "sparse_floor_cells": int(sparse_mask.sum()),
        "obstacle_triangle_count": int(obstacles.sum()),
        "obstacle_components_before_filter": int(obstacle_components),
        "obstacle_components_retained": int(len(retained_labels)),
        "passages": passage_stats,
        "diagonal_pinch_cells_removed": diagonal_pinch_cells_removed,
        "grid_shape": [rows, width],
        "grid_cell_size": cell,
        "reachable_cells": int(area),
        "reachable_train_cameras": int(camera_count),
    }


def mask_to_polygons(mask: np.ndarray, minimum: np.ndarray, cell: float) -> list[list[list[float]]]:
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    rows, columns = mask.shape
    for y, x in np.argwhere(mask):
        candidates = (
            ((x, y), (x + 1, y), y == 0 or not mask[y - 1, x]),
            ((x + 1, y), (x + 1, y + 1), x == columns - 1 or not mask[y, x + 1]),
            ((x + 1, y + 1), (x, y + 1), y == rows - 1 or not mask[y + 1, x]),
            ((x, y + 1), (x, y), x == 0 or not mask[y, x - 1]),
        )
        for start, end, exposed in candidates:
            if exposed:
                edges.setdefault(start, []).append(end)
    polygons: list[list[list[float]]] = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        current = start
        while True:
            choices = edges.get(current)
            if not choices:
                raise NavigationBuildError("walkable boundary edge graph is open")
            following = choices.pop()
            if not choices:
                del edges[current]
            if following == start:
                break
            loop.append(following)
            current = following
            if len(loop) > mask.size * 4:
                raise NavigationBuildError("walkable boundary edge graph did not close")
        points = np.asarray(loop, dtype=np.float64) * cell + minimum
        points = simplify_polygon(points, cell * 0.25)
        if len(points) >= 3:
            polygons.append(points.tolist())
    polygons.sort(key=lambda polygon: abs(polygon_area(np.asarray(polygon))), reverse=True)
    return polygons


def simplify_polygon(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 4:
        return points

    def rdp(line: np.ndarray) -> np.ndarray:
        if len(line) <= 2:
            return line
        delta = line[-1] - line[0]
        length = float(np.linalg.norm(delta))
        distances = np.abs(delta[0] * (line[:, 1] - line[0, 1]) - delta[1] * (line[:, 0] - line[0, 0])) / max(length, 1e-12)
        index = int(np.argmax(distances))
        if distances[index] <= epsilon:
            return line[[0, -1]]
        return np.concatenate((rdp(line[: index + 1])[:-1], rdp(line[index:])))

    anchor = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    first_arc = rdp(points[: anchor + 1])
    second_arc = rdp(np.concatenate((points[anchor:], points[:1])))
    return np.concatenate((first_arc[:-1], second_arc[:-1]))


def polygon_area(points: np.ndarray) -> float:
    return float(0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > y) != (previous[1] > y):
            crossing = (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1]) + current[0]
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def choose_spawn(
    mesh: Any,
    cameras: list[RenderCamera],
    component: np.ndarray,
    minimum: np.ndarray,
    cell: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    height: float,
) -> dict[str, Any]:
    import open3d as o3d

    origin, basis_u, basis_v = basis
    center_pixel = np.asarray(ndimage.center_of_mass(component))[::-1]
    candidates = []
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    radius = 0.18 * height
    total_height = 1.12 * height
    for camera in cameras:
        camera_center = _camera_center(camera)
        uv = _uv(camera_center[None], origin, basis_u, basis_v)[0]
        pixel = _grid_index(uv[None], minimum, cell)[0]
        x, y = pixel
        if not (0 <= y < component.shape[0] and 0 <= x < component.shape[1] and component[y, x]):
            continue
        floor_point = origin + uv[0] * basis_u + uv[1] * basis_v
        up = np.cross(basis_u, basis_v)
        if up @ (camera_center - floor_point) < 0:
            up = -up
        samples = np.stack(
            [
                floor_point + up * (radius + fraction * (total_height - 2 * radius))
                for fraction in (0.0, 0.5, 1.0)
            ]
        )
        distances = scene.compute_distance(o3d.core.Tensor(samples.astype(np.float32))).numpy()
        if not np.isfinite(distances).all() or float(distances.min()) < radius * 0.85:
            continue
        score = float(np.linalg.norm(pixel - center_pixel))
        candidates.append((score, camera, floor_point, up, uv))
    if not candidates:
        raise NavigationBuildError("no Train camera provides a collision-free spawn")
    _, camera, floor_point, up, uv = min(candidates, key=lambda value: value[0])
    world_from_camera = np.linalg.inv(camera.camera_from_normalized.cpu().numpy())
    forward = world_from_camera[:3, 2]
    forward = forward - up * float(forward @ up)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        forward = basis_v
    else:
        forward /= norm
    return {
        "source_train_image_id": camera.image_id,
        "floor_position": floor_point.tolist(),
        "eye_position": (floor_point + up * height).tolist(),
        "look_direction": forward.tolist(),
        "floor_uv": uv.tolist(),
    }


def write_glb(path: Path, mesh: Any) -> None:
    vertices = np.asarray(mesh.vertices, dtype="<f4")
    triangles = np.asarray(mesh.triangles)
    if len(vertices) >= 65536:
        indices = triangles.astype("<u4", copy=False)
        component_type = 5125
    else:
        indices = triangles.astype("<u2", copy=False)
        component_type = 5123
    index_bytes = indices.tobytes(order="C")
    index_padding = (-len(index_bytes)) % 4
    position_offset = len(index_bytes) + index_padding
    binary = index_bytes + b"\0" * index_padding + vertices.tobytes(order="C")
    document = {
        "asset": {"version": "2.0", "generator": "Image3D-SceneGraph"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(index_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": position_offset, "byteLength": vertices.nbytes, "target": 34962},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": component_type, "count": indices.size, "type": "SCALAR", "min": [int(indices.min())], "max": [int(indices.max())]},
            {"bufferView": 1, "componentType": 5126, "count": len(vertices), "type": "VEC3", "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        ],
    }
    json_bytes = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\0" * ((-len(binary)) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    with path.open("xb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        handle.write(json_bytes)
        handle.write(struct.pack("<I4s", len(binary), b"BIN\0"))
        handle.write(binary)


def filter_collision_mesh(
    mesh: Any,
    vertex_support: np.ndarray,
    *,
    up: np.ndarray,
    floor_offset: float,
    height: float,
    voxel: float,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    triangle_support = np.max(vertex_support[triangles], axis=1)
    supported = triangle_support >= 2
    retained_components = np.zeros(len(counts), dtype=bool)
    minimum_area = (0.05 * height) ** 2
    component_records = []
    for label, count in enumerate(counts):
        selected = clusters == label
        component_triangles = triangles[selected]
        component_vertices = np.unique(component_triangles)
        points = vertices[component_vertices]
        spans = np.ptp(points, axis=0)
        vertical_span = float(np.ptp(points @ up))
        a, b, c = (
            vertices[component_triangles[:, 0]],
            vertices[component_triangles[:, 1]],
            vertices[component_triangles[:, 2]],
        )
        area = float((np.linalg.norm(np.cross(b - a, c - a), axis=1) * 0.5).sum())
        support_ratio = float(supported[selected].mean())
        keep = support_ratio >= 0.2 and (
            area >= minimum_area
            or vertical_span >= 0.12 * height
            or float(spans.max()) >= 0.18 * height
        )
        retained_components[label] = keep
        component_records.append((int(count), area, support_ratio, keep))
    remove = ~retained_components[clusters]
    removed_triangles = int(remove.sum())
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return {
        "components_before_filter": len(counts),
        "components_retained": int(retained_components.sum()),
        "component_removed_triangles": removed_triangles,
        "supported_vertices": int((vertex_support >= 2).sum()),
        "minimum_component_area": minimum_area,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
    }


def remove_floor_overlap(
    mesh: Any,
    component: np.ndarray,
    minimum: np.ndarray,
    cell: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    height: float,
    voxel: float,
) -> int:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    origin, basis_u, basis_v = basis
    up = np.cross(basis_u, basis_v)
    centers = vertices[triangles].mean(axis=1)
    height_from_floor = np.abs((centers - origin) @ up)
    pixels = _grid_index(_uv(centers, origin, basis_u, basis_v), minimum, cell)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < component.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < component.shape[0])
    )
    covered = np.zeros(len(triangles), dtype=bool)
    covered[inside] = component[pixels[inside, 1], pixels[inside, 0]]
    remove = covered & (height_from_floor <= max(voxel * 3, height * 0.06))
    removed = int(remove.sum())
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    non_manifold_vertices = list(mesh.get_non_manifold_vertices())
    if non_manifold_vertices:
        mesh.remove_vertices_by_index(non_manifold_vertices)
        mesh.remove_unreferenced_vertices()
    return removed


def add_navigation_floor(
    mesh: Any,
    component: np.ndarray,
    minimum: np.ndarray,
    cell: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> int:
    import open3d as o3d

    origin, basis_u, basis_v = basis
    vertex_ids: dict[tuple[int, int], int] = {}
    vertices: list[np.ndarray] = []
    triangles: list[list[int]] = []

    def vertex(x: int, y: int) -> int:
        key = (x, y)
        if key not in vertex_ids:
            uv = minimum + np.array([x, y], dtype=np.float64) * cell
            vertex_ids[key] = len(vertices)
            vertices.append(origin + uv[0] * basis_u + uv[1] * basis_v)
        return vertex_ids[key]

    for y, x in np.argwhere(component):
        lower_left = vertex(int(x), int(y))
        lower_right = vertex(int(x + 1), int(y))
        upper_right = vertex(int(x + 1), int(y + 1))
        upper_left = vertex(int(x), int(y + 1))
        triangles.extend(
            ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
        )
    floor = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh += floor
    return len(triangles)


def write_debug_overview(
    path: Path,
    mesh: Any,
    cameras: list[RenderCamera],
    polygons: list[list[list[float]]],
    spawn: dict[str, Any],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    origin, basis_u, basis_v = basis
    vertices_uv = _uv(np.asarray(mesh.vertices), origin, basis_u, basis_v)
    cameras_uv = _uv(np.stack([_camera_center(camera) for camera in cameras]), origin, basis_u, basis_v)
    spawn_uv = np.asarray(spawn["floor_uv"])
    all_points = np.concatenate((vertices_uv, cameras_uv, spawn_uv[None]), axis=0)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    scale = 900 / max(float((maximum - minimum).max()), 1e-6)

    def pixel(point: np.ndarray) -> tuple[int, int]:
        value = (point - minimum) * scale + 50
        return int(value[0]), int(1000 - value[1])

    image = Image.new("RGB", (1000, 1060), "#fcfcfb")
    draw = ImageDraw.Draw(image)
    sampled = vertices_uv[:: max(1, len(vertices_uv) // 30_000)]
    for point in sampled:
        x, y = pixel(point)
        draw.point((x, y), fill="#c3c2b7")
    for polygon in polygons:
        points = [pixel(np.asarray(point)) for point in polygon]
        draw.line(points + [points[0]], fill="#1baf7a", width=3)
    for point in cameras_uv:
        x, y = pixel(point)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#2a78d6")
    x, y = pixel(spawn_uv)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#eb6834", outline="#0b0b0b")
    draw.text((30, 1015), "Train cameras (blue)   Boundary (aqua)   Spawn (orange)", fill="#0b0b0b")
    image.save(path)


def build_navigation_assets(
    *,
    model_path: Path,
    contract_path: Path,
    dataset_root: Path,
    config_path: Path,
    export_path: Path,
    output_dir: Path,
    options: BuildOptions,
) -> dict[str, Any]:
    import open3d as o3d
    import torch

    started = time.perf_counter()
    deadline = started + options.timeout_seconds
    if output_dir.exists():
        raise NavigationBuildError(f"output directory already exists: {output_dir}")
    if not dataset_root.is_dir():
        raise NavigationBuildError(f"dataset root is missing: {dataset_root}")
    contract = _read_json(contract_path)
    config = _read_json(config_path)
    export = _read_json(export_path)
    train_ids = validate_inputs(contract, config, export, model_path)
    train_image_layout = validate_train_images(
        contract, dataset_root, train_ids, contract_path
    )
    sparse_points, sparse_sha256 = load_sparse_initialization(contract, contract_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise NavigationBuildError("Gaussian navigation depth rendering requires CUDA")
    model = load_model_snapshot(model_path, device)
    cameras = load_train_cameras(contract, train_ids, options.render_longest_edge)
    trajectory_pairs = train_trajectory_pairs(contract, train_ids, cameras)
    sh_degree = int(config["effective_config"]["sh_schedule"]["max_degree"])
    frames = render_train_depth(
        model, cameras, sh_degree=sh_degree, alpha_threshold=options.alpha_threshold,
        preview_dir=preview_dir, deadline=deadline,
    )
    filtered, consistency = filter_multiview_depth(
        frames,
        neighbor_count=options.consistency_neighbors,
        min_support=options.consistency_min_support,
        relative_tolerance=options.consistency_relative_tolerance,
        alpha_threshold=options.alpha_threshold,
        preview_dir=preview_dir,
        deadline=deadline,
    )
    del frames, model
    torch.cuda.empty_cache()
    mesh, tsdf = fuse_tsdf(filtered, options, deadline)
    cleanup = clean_mesh(mesh, options.max_triangles)
    vertex_support = mesh_vertex_support(
        mesh,
        filtered,
        relative_tolerance=options.consistency_relative_tolerance,
        voxel=options.voxel_length,
        deadline=deadline,
    )
    camera_up = _camera_up(cameras)
    up, floor_offset, height, sparse_floor_points, floor = infer_floor(
        mesh, sparse_points, cameras, camera_up, options.voxel_length
    )
    cleanup.update(
        filter_collision_mesh(
            mesh,
            vertex_support,
            up=up,
            floor_offset=floor_offset,
            height=height,
            voxel=options.voxel_length,
        )
    )
    cleanup["vertices"] = len(mesh.vertices)
    cleanup["triangles"] = len(mesh.triangles)
    (output_dir / "fusion-diagnostics.json").write_text(
        json.dumps(
            {"consistency": consistency, "tsdf": tsdf, "cleanup": cleanup},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    o3d.io.write_triangle_mesh(
        str(output_dir / "collision-debug.ply"), mesh, write_ascii=False
    )
    if cleanup["triangles"] < 1_000 or cleanup["triangles"] > options.max_triangles:
        raise NavigationBuildError(f"collision triangle quality gate failed: {cleanup['triangles']}")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not np.isfinite(vertices).all() or not np.isfinite(triangles).all():
        raise NavigationBuildError("collision mesh contains non-finite geometry")

    component, minimum, cell, basis, boundary_stats = build_walkable_mask(
        mesh,
        sparse_floor_points,
        cameras,
        trajectory_pairs,
        up,
        floor_offset,
        height,
        options.voxel_length,
        output_dir / "masks",
    )
    polygons = mask_to_polygons(component, minimum, cell)
    if not polygons:
        raise NavigationBuildError("reachable floor has no closed boundary polygon")
    spawn = choose_spawn(mesh, cameras, component, minimum, cell, basis, height)
    if not point_in_polygon(np.asarray(spawn["floor_uv"]), np.asarray(polygons[0])):
        raise NavigationBuildError("spawn lies outside the outer navigation boundary")
    reachable_area = boundary_stats["reachable_cells"] * boundary_stats["grid_cell_size"] ** 2
    reachable_camera_ratio = boundary_stats["reachable_train_cameras"] / len(cameras)
    reachable_area_h2 = reachable_area / (height * height)
    if reachable_camera_ratio < 0.2 or reachable_area_h2 < 0.5:
        raise NavigationBuildError(
            "reachable navigation coverage is insufficient: "
            f"train_camera_ratio={reachable_camera_ratio:.6f}, "
            f"area_h2={reachable_area_h2:.6f}"
        )
    navigation_floor_overlap_removed = remove_floor_overlap(
        mesh,
        component,
        minimum,
        cell,
        basis,
        height=height,
        voxel=options.voxel_length,
    )
    navigation_floor_triangles = add_navigation_floor(
        mesh, component, minimum, cell, basis
    )
    cleanup["navigation_floor_overlap_removed_triangles"] = navigation_floor_overlap_removed
    cleanup["navigation_floor_triangles"] = navigation_floor_triangles
    cleanup["vertices"] = len(mesh.vertices)
    cleanup["triangles"] = len(mesh.triangles)
    if cleanup["triangles"] > options.max_triangles:
        raise NavigationBuildError(
            f"hybrid collision triangle budget exceeded: {cleanup['triangles']}"
        )

    mesh.compute_vertex_normals()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    collision_path = output_dir / "collision.glb"
    write_glb(collision_path, mesh)
    verified_mesh = o3d.io.read_triangle_mesh(str(collision_path))
    if (
        len(verified_mesh.vertices) != len(mesh.vertices)
        or len(verified_mesh.triangles) != len(mesh.triangles)
        or not np.isfinite(np.asarray(verified_mesh.vertices)).all()
    ):
        raise NavigationBuildError("collision GLB failed write/read integrity check")
    collision_bytes = collision_path.stat().st_size
    topology = {
        "self_intersecting": bool(verified_mesh.is_self_intersecting()),
        "vertex_manifold": bool(verified_mesh.is_vertex_manifold()),
        "edge_manifold_allow_boundary": bool(verified_mesh.is_edge_manifold(allow_boundary_edges=True)),
        "orientable": bool(verified_mesh.is_orientable()),
    }
    if (
        topology["self_intersecting"]
        or not topology["vertex_manifold"]
        or not topology["edge_manifold_allow_boundary"]
        or not topology["orientable"]
    ):
        raise NavigationBuildError(f"collision topology quality gate failed: {topology}")
    if collision_bytes > options.max_glb_bytes:
        raise NavigationBuildError(
            f"collision GLB exceeds budget: {collision_bytes} > {options.max_glb_bytes}"
        )
    elapsed = time.perf_counter() - started
    if elapsed > options.timeout_seconds:
        raise NavigationBuildError(
            f"navigation generation exceeded time budget: {elapsed:.2f}s"
        )

    origin, basis_u, basis_v = basis
    navigation = {
        "schema_version": SCHEMA_VERSION,
        "status": "available",
        "coordinate_frame": "normalized",
        "world_units": "arbitrary",
        "up": up.tolist(),
        "floor": {
            "origin": origin.tolist(),
            "basis_u": basis_u.tolist(),
            "basis_v": basis_v.tolist(),
            "plane_offset": floor_offset,
        },
        "boundary": {
            "coordinate_frame": "floor_uv",
            "outer": polygons[0],
            "holes": polygons[1:],
            "derivation": "train_frustum_support_intersect_spawn_reachable_collision_floor",
        },
        "spawn": spawn,
        "estimated_eye_height": height,
        "player": {
            "capsule_total_height": 1.12 * height,
            "capsule_radius": 0.18 * height,
            "max_step": 0.12 * height,
            "max_slope_degrees": 35.0,
        },
        "controls": {
            "speed": {"default": 0.8 * height, "minimum": 0.4 * height, "maximum": 1.2 * height},
            "vertical_fov_degrees": {"default": 70.0, "minimum": 50.0, "maximum": 90.0},
            "mouse_sensitivity_radians_per_pixel": {"default": 0.002, "minimum": 0.0005, "maximum": 0.005},
            "pitch_degrees": {"minimum": -85.0, "maximum": 85.0},
        },
        "provenance": {
            "dataset_hash": contract["dataset_hash"],
            "effective_config_hash": config["effective_config_hash"],
            "model_sha256": export["model_sha256"],
            "export_sha256": sha256_file(export_path),
            "train_image_ids": train_ids,
            "selected_render_image_ids": [frame.camera.image_id for frame in filtered],
            "validation_image_ids_used": [],
            "test_image_ids_used": [],
            "image_rgb_access": "hash_verification_only",
            "train_image_layout": train_image_layout,
            "sparse_initialization_sha256": sparse_sha256,
        },
        "collision": {
            "asset": "collision.glb",
            "sha256": sha256_file(collision_path),
            "bytes": collision_bytes,
            "vertices": cleanup["vertices"],
            "triangles": cleanup["triangles"],
        },
        "generation": {
            "options": asdict(options),
            "elapsed_seconds": elapsed,
            "small_hole_policy": "implicit_tsdf_fusion_only",
            "floor_support": "contracted_colmap_sparse_dominant_plane_intersect_train_support",
            "collision_floor": "triangulated_reachable_train_supported_floor_cells",
            "door_protection": "train_sequence_short_level_consistent_obstacle_clear_corridors",
        },
        "quality": {"consistency": consistency, "tsdf": tsdf, "cleanup": cleanup, "floor": floor, "boundary": boundary_stats, "topology": topology},
    }
    (output_dir / "navigation.json").write_text(
        json.dumps(navigation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "train_only": True,
        "render_resolution_longest_edge": options.render_longest_edge,
        "alpha_threshold": options.alpha_threshold,
        "consistency": consistency,
        "tsdf": tsdf,
        "cleanup": cleanup,
        "floor": floor,
        "boundary": boundary_stats,
        "topology": topology,
        "collision_bytes": collision_bytes,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_debug_overview(output_dir / "overview.png", mesh, cameras, polygons, spawn, basis)
    return navigation


def _write_scalar_preview(path: Path, values: np.ndarray, *, unit: bool = False) -> None:
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        low, high = (0.0, 1.0) if unit else tuple(np.quantile(values[valid], (0.02, 0.98)))
        normalized[valid] = np.clip((values[valid] - low) / max(high - low, 1e-8) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(normalized).save(path)


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > deadline:
        raise NavigationBuildError("navigation generation exceeded five-minute budget")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Train-only collision/navigation evidence from a Gaussian model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--effective-config", type=Path, required=True)
    parser.add_argument("--export-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-longest-edge", type=int, default=512)
    parser.add_argument("--alpha-threshold", type=float, default=0.35)
    parser.add_argument("--voxel-length", type=float, default=0.006)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = BuildOptions(
        render_longest_edge=args.render_longest_edge,
        alpha_threshold=args.alpha_threshold,
        voxel_length=args.voxel_length,
        sdf_trunc=args.voxel_length * 4,
        timeout_seconds=args.timeout_seconds,
    )
    result = build_navigation_assets(
        model_path=args.model,
        contract_path=args.dataset_contract,
        dataset_root=args.dataset_root,
        config_path=args.effective_config,
        export_path=args.export_metadata,
        output_dir=args.output_dir,
        options=options,
    )
    print(f"triangles={result['collision']['triangles']}")
    print(f"collision_bytes={result['collision']['bytes']}")
    print(f"elapsed_seconds={result['generation']['elapsed_seconds']:.3f}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
