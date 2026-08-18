"""Deterministic window and Sim(3) graph utilities for experimental VGGT-BA."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


MIN_SUPPORTED_OBSERVATIONS = 32
MIN_RELIABLE_CAMERAS = 12
MIN_RELIABLE_CAMERA_RATE = 0.70
MIN_TEMPORAL_COVERAGE = 0.80
VGGT_BA_FALLBACK_REASONS = frozenset(
    {
        "vggt_graph_unusable_after_recovery",
        "vggt_seed_geometry_insufficient",
        "vggt_registration_gate_failed",
    }
)


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    image_indices: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class WindowEdge:
    source: str
    target: str
    target_from_source: np.ndarray
    shared_indices: tuple[int, ...]
    center_residual_p90: float
    rotation_residual_p90_degrees: float


class VggtBaError(ValueError):
    """Raised when batched VGGT cameras cannot form a valid global model."""


def sequential_windows(
    image_count: int,
    *,
    window_size: int = 8,
    overlap: int = 4,
) -> list[WindowSpec]:
    if image_count <= 0:
        raise VggtBaError("VGGT-BA requires at least one image")
    if window_size < 3:
        raise VggtBaError("VGGT-BA window size must be at least 3")
    if overlap < 3 or overlap >= window_size:
        raise VggtBaError("VGGT-BA overlap must be at least 3 and smaller than the window")
    if image_count <= window_size:
        return [WindowSpec("base-0000", tuple(range(image_count)), "base")]
    windows: list[WindowSpec] = []
    start = 0
    while start < image_count:
        end = min(start + window_size, image_count)
        indices = tuple(range(start, end))
        if len(indices) < 3:
            previous = windows[-1].image_indices
            indices = tuple(range(max(0, image_count - window_size), image_count))
            if indices == previous:
                break
        windows.append(WindowSpec(f"base-{len(windows):04d}", indices, "base"))
        if end == image_count:
            break
        start = end - overlap
    return windows


def bridge_windows(
    descriptors: np.ndarray,
    base_windows: list[WindowSpec],
    *,
    window_size: int = 8,
    minimum_index_gap: int = 16,
    maximum_bridges: int = 16,
) -> list[WindowSpec]:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise VggtBaError("DINO descriptors must be a finite N x D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise VggtBaError("DINO descriptors cannot contain zero rows")
    normalized = values / norms
    similarity = normalized @ normalized.T
    candidates: list[tuple[float, int, int]] = []
    for left in range(len(values)):
        for right in range(left + minimum_index_gap, len(values)):
            candidates.append((float(similarity[left, right]), left, right))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    owner: dict[int, list[WindowSpec]] = {}
    for window in base_windows:
        for index in window.image_indices:
            owner.setdefault(index, []).append(window)
    bridges: list[WindowSpec] = []
    used_pairs: set[tuple[str, str]] = set()
    side_count = max(3, window_size // 2)
    for _score, left, right in candidates:
        if len(bridges) >= maximum_bridges:
            break
        left_windows = owner.get(left, [])
        right_windows = owner.get(right, [])
        if not left_windows or not right_windows:
            continue
        first = min(left_windows, key=lambda window: (abs(_window_center(window) - left), window.window_id))
        second = min(right_windows, key=lambda window: (abs(_window_center(window) - right), window.window_id))
        if first.window_id == second.window_id:
            continue
        pair = tuple(sorted((first.window_id, second.window_id)))
        if pair in used_pairs:
            continue
        first_members = _nearest_members(first.image_indices, left, side_count)
        second_members = _nearest_members(second.image_indices, right, side_count)
        members = tuple(sorted(set(first_members + second_members)))
        if len(set(members) & set(first.image_indices)) < 3:
            continue
        if len(set(members) & set(second.image_indices)) < 3:
            continue
        if len(members) > window_size:
            members = tuple(sorted(_nearest_members(members, left, window_size // 2) + _nearest_members(members, right, window_size // 2)))
        bridges.append(WindowSpec(f"bridge-{len(bridges):04d}", members, "bridge"))
        used_pairs.add(pair)
    return bridges


def classify_frame_support(
    inlier_counts: list[int],
    *,
    minimum_observations: int = MIN_SUPPORTED_OBSERVATIONS,
) -> tuple[list[int], list[int]]:
    if minimum_observations < 1 or any(count < 0 for count in inlier_counts):
        raise VggtBaError(
            "frame support counts must be non-negative and threshold must be positive"
        )
    strong = [index for index, count in enumerate(inlier_counts) if count >= minimum_observations]
    weak = [index for index, count in enumerate(inlier_counts) if count < minimum_observations]
    return strong, weak


def recovery_windows(
    base_windows: list[WindowSpec],
    reliable_indices: dict[str, set[int]],
    *,
    window_size: int = 8,
    minimum_side_frames: int = 3,
    forced_pairs: set[tuple[str, str]] | None = None,
) -> list[WindowSpec]:
    if window_size < minimum_side_frames * 2:
        raise VggtBaError("recovery window cannot hold both reliable sides")
    usable = [
        window
        for window in base_windows
        if len(reliable_indices.get(window.window_id, set())) >= minimum_side_frames
    ]
    recoveries = []
    forced = forced_pairs or set()
    side_count = window_size // 2
    for left, right in zip(usable, usable[1:], strict=False):
        pair = (left.window_id, right.window_id)
        left_reliable = set(reliable_indices[left.window_id])
        right_reliable = set(reliable_indices[right.window_id])
        shared = left_reliable & right_reliable
        if len(shared) >= minimum_side_frames and pair not in forced:
            continue
        left_candidates = [
            *sorted(left_reliable - right_reliable, reverse=True),
            *sorted(shared, reverse=True),
        ]
        right_candidates = [
            *sorted(right_reliable - left_reliable),
            *sorted(shared),
        ]
        left_members = tuple(left_candidates[:side_count])
        right_members = tuple(right_candidates[:side_count])
        members = tuple(sorted(set(left_members + right_members)))
        if len(set(members) & left_reliable) < minimum_side_frames:
            continue
        if len(set(members) & right_reliable) < minimum_side_frames:
            continue
        if len(members) < minimum_side_frames * 2:
            continue
        recoveries.append(
            WindowSpec(f"recovery-{len(recoveries):04d}", members, "recovery")
        )
    return recoveries


def select_reliable_component(
    windows: dict[str, dict[int, dict[str, np.ndarray]]],
    edges: list[WindowEdge],
    image_count: int,
    *,
    minimum_camera_count: int = MIN_RELIABLE_CAMERAS,
    minimum_camera_rate: float = MIN_RELIABLE_CAMERA_RATE,
    minimum_temporal_coverage: float = MIN_TEMPORAL_COVERAGE,
) -> tuple[list[str], dict[str, Any]]:
    if image_count < 1:
        raise VggtBaError("component selection requires at least one image")
    adjacency = {window_id: set() for window_id in windows}
    for edge in edges:
        if edge.source not in adjacency or edge.target not in adjacency:
            raise VggtBaError("window edge references an unknown window")
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    remaining = set(windows)
    records = []
    while remaining:
        first = min(remaining)
        component = set()
        queue = deque([first])
        while queue:
            window_id = queue.popleft()
            if window_id in component:
                continue
            component.add(window_id)
            remaining.discard(window_id)
            queue.extend(sorted(adjacency[window_id] - component))
        image_indices = sorted(
            {index for window_id in component for index in windows[window_id]}
        )
        camera_rate = len(image_indices) / image_count
        temporal_coverage = (
            (image_indices[-1] - image_indices[0] + 1) / image_count
            if image_indices
            else 0.0
        )
        records.append(
            {
                "window_ids": sorted(component),
                "reliable_camera_count": len(image_indices),
                "reliable_camera_rate": camera_rate,
                "temporal_coverage": temporal_coverage,
                "first_image_index": image_indices[0] if image_indices else None,
                "last_image_index": image_indices[-1] if image_indices else None,
                "usable": (
                    len(image_indices) >= minimum_camera_count
                    and camera_rate >= minimum_camera_rate
                    and temporal_coverage >= minimum_temporal_coverage
                ),
            }
        )
    records.sort(
        key=lambda record: (
            -record["reliable_camera_count"],
            -record["temporal_coverage"],
            record["window_ids"],
        )
    )
    selected = next((record for record in records if record["usable"]), None)
    return (list(selected["window_ids"]) if selected else []), {
        "minimum_camera_count": minimum_camera_count,
        "minimum_camera_rate": minimum_camera_rate,
        "minimum_temporal_coverage": minimum_temporal_coverage,
        "components": records,
        "selected": selected,
    }


def estimate_window_edge(
    source_id: str,
    target_id: str,
    source_extrinsics: dict[int, np.ndarray],
    target_extrinsics: dict[int, np.ndarray],
) -> WindowEdge:
    shared = tuple(sorted(set(source_extrinsics) & set(target_extrinsics)))
    if len(shared) < 3:
        raise VggtBaError("window edge requires at least three shared cameras")
    source_c2w = [_camera_to_world(source_extrinsics[index]) for index in shared]
    target_c2w = [_camera_to_world(target_extrinsics[index]) for index in shared]
    source_centers = np.stack([matrix[:3, 3] for matrix in source_c2w])
    target_centers = np.stack([matrix[:3, 3] for matrix in target_c2w])

    rotations = Rotation.from_matrix(
        np.stack(
            [target[:3, :3] @ source[:3, :3].T for source, target in zip(source_c2w, target_c2w, strict=True)]
        )
    )
    rotation = rotations.mean().as_matrix()
    scales = []
    for left in range(len(shared)):
        for right in range(left + 1, len(shared)):
            source_distance = float(np.linalg.norm(source_centers[left] - source_centers[right]))
            target_distance = float(np.linalg.norm(target_centers[left] - target_centers[right]))
            if source_distance > 1e-8 and target_distance > 1e-8:
                scales.append(target_distance / source_distance)
    if not scales:
        raise VggtBaError("shared camera centers do not define a Sim(3) scale")
    scale = float(np.median(scales))
    if not math.isfinite(scale) or scale <= 0:
        raise VggtBaError("window edge produced an invalid scale")
    translation = np.median(target_centers - scale * (source_centers @ rotation.T), axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation

    transformed = source_centers @ transform[:3, :3].T + transform[:3, 3]
    extent = max(float(np.linalg.norm(target_centers - np.median(target_centers, axis=0), axis=1).max()), 1e-6)
    center_residuals = np.linalg.norm(transformed - target_centers, axis=1) / extent
    rotation_residuals = []
    for source, target in zip(source_c2w, target_c2w, strict=True):
        predicted = rotation @ source[:3, :3]
        difference = Rotation.from_matrix(target[:3, :3] @ predicted.T)
        rotation_residuals.append(float(np.degrees(difference.magnitude())))
    center_p90 = float(np.quantile(center_residuals, 0.9))
    rotation_p90 = float(np.quantile(rotation_residuals, 0.9))
    if center_p90 > 0.25 or rotation_p90 > 15.0:
        raise VggtBaError(
            f"window edge residual is too large: center_p90={center_p90:.4f}, rotation_p90={rotation_p90:.3f}deg"
        )
    return WindowEdge(
        source=source_id,
        target=target_id,
        target_from_source=transform,
        shared_indices=shared,
        center_residual_p90=center_p90,
        rotation_residual_p90_degrees=rotation_p90,
    )


def optimize_window_graph(
    window_ids: list[str],
    edges: list[WindowEdge],
    *,
    anchor_id: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ids = list(dict.fromkeys(window_ids))
    if not ids:
        raise VggtBaError("window graph is empty")
    if len(ids) == 1:
        return {ids[0]: np.eye(4, dtype=np.float64)}, {
            "connected": True,
            "anchor": ids[0],
            "edge_count": 0,
            "initial_cost": 0.0,
            "final_cost": 0.0,
        }
    anchor = anchor_id or ids[0]
    if anchor not in ids:
        raise VggtBaError("window graph anchor is unknown")
    initial = _initialize_graph(ids, edges, anchor)
    variable_ids = [window_id for window_id in ids if window_id != anchor]
    slices = {window_id: slice(index * 7, (index + 1) * 7) for index, window_id in enumerate(variable_ids)}
    x0 = np.concatenate([_encode_similarity(initial[window_id]) for window_id in variable_ids])

    def transforms(parameters: np.ndarray) -> dict[str, np.ndarray]:
        result = {anchor: np.eye(4, dtype=np.float64)}
        for window_id in variable_ids:
            result[window_id] = _decode_similarity(parameters[slices[window_id]])
        return result

    translation_scale = max(
        np.median([np.linalg.norm(edge.target_from_source[:3, 3]) for edge in edges]),
        1e-3,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        current = transforms(parameters)
        values = []
        for edge in edges:
            predicted = np.linalg.inv(current[edge.target]) @ current[edge.source]
            measured = edge.target_from_source
            predicted_scale, predicted_rotation, predicted_translation = decompose_similarity(predicted)
            measured_scale, measured_rotation, measured_translation = decompose_similarity(measured)
            rotation_error = Rotation.from_matrix(measured_rotation.T @ predicted_rotation).as_rotvec()
            values.extend(
                [
                    math.log(predicted_scale / measured_scale),
                    *rotation_error.tolist(),
                    *((predicted_translation - measured_translation) / translation_scale).tolist(),
                ]
            )
        return np.asarray(values, dtype=np.float64)

    initial_residual = residual(x0)
    optimized = least_squares(residual, x0, loss="soft_l1", f_scale=0.05, max_nfev=300)
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise VggtBaError(f"window pose graph optimization failed: {optimized.message}")
    final_residual = residual(optimized.x)
    initial_cost = float(np.mean(initial_residual * initial_residual)) if len(initial_residual) else 0.0
    final_cost = float(np.mean(final_residual * final_residual)) if len(final_residual) else 0.0
    if final_cost > initial_cost * 1.05 + 1e-12:
        raise VggtBaError("window pose graph residual increased")
    return transforms(optimized.x), {
        "connected": True,
        "anchor": anchor,
        "edge_count": len(edges),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "optimizer_evaluations": int(optimized.nfev),
    }


def merge_window_cameras(
    windows: dict[str, dict[int, dict[str, np.ndarray]]],
    global_from_window: dict[str, np.ndarray],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    predictions: dict[int, list[dict[str, np.ndarray]]] = {}
    for window_id, cameras in windows.items():
        transform = global_from_window[window_id]
        for image_index, camera in cameras.items():
            predictions.setdefault(image_index, []).append(
                {
                    "extrinsic": transform_extrinsic(camera["extrinsic"], transform),
                    "intrinsic": np.asarray(camera["intrinsic"], dtype=np.float64),
                }
            )
    merged: dict[int, dict[str, np.ndarray]] = {}
    center_spreads = []
    rotation_spreads = []
    for image_index, entries in sorted(predictions.items()):
        c2w = [_camera_to_world(entry["extrinsic"]) for entry in entries]
        centers = np.stack([matrix[:3, 3] for matrix in c2w])
        center = np.median(centers, axis=0)
        rotation = Rotation.from_matrix(np.stack([matrix[:3, :3] for matrix in c2w])).mean().as_matrix()
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = rotation
        camera_to_world[:3, 3] = center
        merged[image_index] = {
            "extrinsic": np.linalg.inv(camera_to_world)[:3, :4],
            "intrinsic": np.median(np.stack([entry["intrinsic"] for entry in entries]), axis=0),
        }
        center_spreads.extend(np.linalg.norm(centers - center, axis=1).tolist())
        rotation_spreads.extend(
            np.degrees(
                Rotation.from_matrix(np.stack([matrix[:3, :3] @ rotation.T for matrix in c2w])).magnitude()
            ).tolist()
        )
    return merged, {
        "prediction_count": sum(len(entries) for entries in predictions.values()),
        "camera_count": len(merged),
        "center_disagreement_p90": _quantile(center_spreads, 0.9),
        "center_disagreement_max": max(center_spreads, default=0.0),
        "rotation_disagreement_p90_degrees": _quantile(rotation_spreads, 0.9),
        "rotation_disagreement_max_degrees": max(rotation_spreads, default=0.0),
    }


def transform_extrinsic(extrinsic: np.ndarray, global_from_local: np.ndarray) -> np.ndarray:
    local_camera_to_world = _camera_to_world(extrinsic)
    scale, rotation, translation = decompose_similarity(global_from_local)
    global_camera_to_world = np.eye(4, dtype=np.float64)
    global_camera_to_world[:3, :3] = rotation @ local_camera_to_world[:3, :3]
    global_camera_to_world[:3, 3] = (
        scale * rotation @ local_camera_to_world[:3, 3] + translation
    )
    return np.linalg.inv(global_camera_to_world)[:3, :4]


def write_initial_colmap_model(
    output_dir: Path,
    image_names: list[str],
    cameras: dict[int, dict[str, np.ndarray]],
    image_sizes: dict[int, tuple[int, int]],
    *,
    image_ids_by_name: dict[str, int] | None = None,
) -> dict[str, Any]:
    camera_indices = sorted(cameras)
    if not camera_indices:
        raise VggtBaError("global camera model is empty")
    if any(index < 0 or index >= len(image_names) for index in camera_indices):
        raise VggtBaError("global camera model references an unknown image")
    sizes = {image_sizes[index] for index in camera_indices}
    if len(sizes) != 1:
        raise VggtBaError("video VGGT-BA requires one shared image size")
    width, height = next(iter(sizes))
    intrinsics = np.stack([cameras[index]["intrinsic"] for index in camera_indices])
    fx = float(np.median(intrinsics[:, 0, 0]))
    fy = float(np.median(intrinsics[:, 1, 1]))
    cx = float(np.median(intrinsics[:, 0, 2]))
    cy = float(np.median(intrinsics[:, 1, 2]))
    if min(fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
        raise VggtBaError("VGGT intrinsics cannot initialize a shared camera")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cameras.txt").write_text(
        f"# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n1 OPENCV {width} {height} "
        f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g} 0 0 0 0\n",
        encoding="utf-8",
    )
    image_rows = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    image_ids = image_ids_by_name or {
        name: index + 1 for index, name in enumerate(image_names)
    }
    selected_ids = [image_ids.get(image_names[index]) for index in camera_indices]
    if any(image_id is None or image_id < 1 for image_id in selected_ids):
        raise VggtBaError("COLMAP database does not contain every reliable image")
    if len(set(selected_ids)) != len(selected_ids):
        raise VggtBaError("COLMAP database image IDs must be unique")
    for image_index, image_id in zip(camera_indices, selected_ids, strict=True):
        name = image_names[image_index]
        extrinsic = np.asarray(cameras[image_index]["extrinsic"], dtype=np.float64)
        qvec = rotmat_to_qvec(extrinsic[:3, :3])
        tvec = extrinsic[:3, 3]
        image_rows.extend(
            (
                f"{image_id} {' '.join(f'{value:.17g}' for value in qvec)} "
                f"{' '.join(f'{value:.17g}' for value in tvec)} 1 {name}",
                "",
            )
        )
    (output_dir / "images.txt").write_text("\n".join(image_rows) + "\n", encoding="utf-8")
    (output_dir / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n", encoding="utf-8"
    )
    focal_dispersion = float(
        np.median(
            np.abs(intrinsics[:, 0, 0] - fx) / max(fx, 1e-12)
            + np.abs(intrinsics[:, 1, 1] - fy) / max(fy, 1e-12)
        )
        / 2.0
    )
    return {
        "camera_model": "OPENCV",
        "shared_camera": True,
        "camera_count": len(camera_indices),
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal_relative_median_dispersion": focal_dispersion,
    }


def filter_train_supported_points(
    points_path: Path,
    images_path: Path,
    image_root: Path,
    train_image_ids: set[int],
    output_path: Path,
    *,
    minimum_train_observations: int = 2,
) -> dict[str, Any]:
    if minimum_train_observations < 1:
        raise VggtBaError("minimum Train observations must be positive")
    image_records = _parse_colmap_image_observations(images_path)
    unknown = train_image_ids - set(image_records)
    if unknown:
        raise VggtBaError(f"Train split references unknown COLMAP images: {sorted(unknown)}")
    image_cache: dict[int, np.ndarray] = {}
    counts = {
        "input": 0,
        "accepted": 0,
        "heldout_only_rejected": 0,
        "insufficient_train_support_rejected": 0,
        "train_recolor_failures": 0,
        "mixed_track_points": 0,
    }
    output_rows = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    for line in points_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        counts["input"] += 1
        parts = line.split()
        if len(parts) < 8 or (len(parts) - 8) % 2:
            raise VggtBaError("COLMAP points3D.txt has an invalid row")
        track = [
            (int(parts[index]), int(parts[index + 1]))
            for index in range(8, len(parts), 2)
        ]
        train_track = [entry for entry in track if entry[0] in train_image_ids]
        if not train_track:
            counts["heldout_only_rejected"] += 1
            continue
        if len(train_track) < minimum_train_observations:
            counts["insufficient_train_support_rejected"] += 1
            continue
        if len(train_track) != len(track):
            counts["mixed_track_points"] += 1
        colors = []
        for image_id, point2d_index in train_track:
            record = image_records[image_id]
            observations = record["observations"]
            if point2d_index < 0 or point2d_index >= len(observations):
                continue
            x, y, point3d_id = observations[point2d_index]
            if point3d_id != int(parts[0]):
                continue
            if image_id not in image_cache:
                from PIL import Image

                path = image_root / str(record["name"])
                if not path.is_file():
                    raise VggtBaError(f"missing Train image for recoloring: {path}")
                image_cache[image_id] = np.asarray(Image.open(path).convert("RGB"))
            image = image_cache[image_id]
            pixel_x = int(np.clip(round(x), 0, image.shape[1] - 1))
            pixel_y = int(np.clip(round(y), 0, image.shape[0] - 1))
            colors.append(image[pixel_y, pixel_x])
        if not colors:
            counts["train_recolor_failures"] += 1
            continue
        color = np.rint(np.median(np.stack(colors).astype(np.float64), axis=0)).astype(np.uint8)
        output_rows.append(
            " ".join(
                [
                    *parts[:4],
                    str(int(color[0])),
                    str(int(color[1])),
                    str(int(color[2])),
                    *parts[7:],
                ]
            )
        )
        counts["accepted"] += 1
    if counts["accepted"] == 0:
        raise VggtBaError("Train-supported filtering rejected every sparse point")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_rows) + "\n", encoding="utf-8")
    return {
        "minimum_train_observations": minimum_train_observations,
        "train_image_ids": sorted(train_image_ids),
        "counts": counts,
    }


def count_frame_inliers(mask: np.ndarray) -> list[int]:
    return np.asarray(mask, dtype=bool).sum(axis=1).astype(int).tolist()


def read_colmap_database_image_ids(database_path: Path) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT name, image_id FROM images").fetchall()
    result = {str(name): int(image_id) for name, image_id in rows}
    if len(result) != len(rows):
        raise VggtBaError("COLMAP database contains duplicate image names")
    return result


def supported_image_ids(
    images_text: Path,
    *,
    minimum_observations: int = MIN_SUPPORTED_OBSERVATIONS,
) -> set[int]:
    lines = [
        line.strip()
        for line in images_text.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    supported = set()
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header:
            raise VggtBaError("COLMAP images.txt has a missing image row")
        observations = lines[index + 1].split() if index + 1 < len(lines) else []
        image_id = int(header.split(maxsplit=1)[0])
        if len(observations) % 3:
            raise VggtBaError("COLMAP images.txt has invalid observations")
        linked = sum(
            1 for offset in range(2, len(observations), 3) if int(observations[offset]) >= 0
        )
        if linked >= minimum_observations:
            supported.add(image_id)
        index += 2
    return supported


def decompose_similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise VggtBaError("Sim(3) must be a finite 4 x 4 matrix")
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if not math.isfinite(scale) or scale <= 0:
        raise VggtBaError("Sim(3) scale must be positive")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise VggtBaError("Sim(3) rotation must be proper")
    return scale, rotation, matrix[:3, 3].copy()


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    qvec = quaternion_xyzw[[3, 0, 1, 2]]
    return -qvec if qvec[0] < 0 else qvec


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parse_colmap_image_observations(path: Path) -> dict[int, dict[str, Any]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    records: dict[int, dict[str, Any]] = {}
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header:
            raise VggtBaError("COLMAP images.txt has a missing image row")
        parts = header.split(maxsplit=9)
        if len(parts) != 10:
            raise VggtBaError("COLMAP images.txt has an invalid image row")
        image_id = int(parts[0])
        observation_parts = lines[index + 1].split() if index + 1 < len(lines) else []
        if len(observation_parts) % 3:
            raise VggtBaError("COLMAP images.txt has invalid observations")
        observations = [
            (
                float(observation_parts[offset]),
                float(observation_parts[offset + 1]),
                int(observation_parts[offset + 2]),
            )
            for offset in range(0, len(observation_parts), 3)
        ]
        records[image_id] = {"name": parts[9], "observations": observations}
        index += 2
    return records


def _window_center(window: WindowSpec) -> float:
    return float(np.mean(window.image_indices))


def _nearest_members(indices: tuple[int, ...], anchor: int, count: int) -> tuple[int, ...]:
    return tuple(sorted(sorted(indices, key=lambda index: (abs(index - anchor), index))[:count]))


def _camera_to_world(extrinsic: np.ndarray) -> np.ndarray:
    value = np.asarray(extrinsic, dtype=np.float64)
    if value.shape != (3, 4) or not np.isfinite(value).all():
        raise VggtBaError("camera extrinsic must be finite 3 x 4")
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :4] = value
    return np.linalg.inv(homogeneous)


def _initialize_graph(
    window_ids: list[str], edges: list[WindowEdge], anchor: str
) -> dict[str, np.ndarray]:
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {window_id: [] for window_id in window_ids}
    for edge in edges:
        if edge.source not in adjacency or edge.target not in adjacency:
            raise VggtBaError("window edge references an unknown window")
        adjacency[edge.source].append((edge.target, edge.target_from_source))
        adjacency[edge.target].append((edge.source, np.linalg.inv(edge.target_from_source)))
    transforms = {anchor: np.eye(4, dtype=np.float64)}
    queue = deque([anchor])
    while queue:
        source = queue.popleft()
        for target, target_from_source in adjacency[source]:
            if target in transforms:
                continue
            transforms[target] = transforms[source] @ np.linalg.inv(target_from_source)
            queue.append(target)
    missing = sorted(set(window_ids) - set(transforms))
    if missing:
        raise VggtBaError(f"window graph is disconnected: {missing}")
    return transforms


def _encode_similarity(transform: np.ndarray) -> np.ndarray:
    scale, rotation, translation = decompose_similarity(transform)
    return np.asarray([math.log(scale), *Rotation.from_matrix(rotation).as_rotvec(), *translation])


def _decode_similarity(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise VggtBaError("Sim(3) parameters must contain seven finite values")
    scale = math.exp(float(values[0]))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * Rotation.from_rotvec(values[1:4]).as_matrix()
    transform[:3, 3] = values[4:7]
    return transform


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability)) if values else 0.0
