"""Pure COLMAP sparse-model and VGGT grouping computations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np


GROUPING_MIN_SHARED_POINTS = 8
GROUPING_MAX_NEIGHBORS = 16


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    observations: list[tuple[float, float, int]]


@dataclass(frozen=True)
class CovisibilityEdge:
    target_image_id: int
    shared_points: int
    baseline: float


@dataclass(frozen=True)
class VggtGroupSelection:
    groups: list[list[Path]]
    records: list[dict[str, Any]]


def parse_colmap_images_with_points(path: Path) -> list[ColmapImage]:
    images: list[ColmapImage] = []
    data_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for index in range(0, len(data_lines), 2):
        image_parts = data_lines[index].split(maxsplit=9)
        point_parts = data_lines[index + 1].split()
        observations = [
            (float(point_parts[point_index]), float(point_parts[point_index + 1]), point3d_id)
            for point_index in range(0, len(point_parts), 3)
            if (point3d_id := int(point_parts[point_index + 2])) != -1
        ]
        images.append(
            ColmapImage(
                image_id=int(image_parts[0]),
                qvec=np.array([float(value) for value in image_parts[1:5]], dtype=np.float64),
                tvec=np.array([float(value) for value in image_parts[5:8]], dtype=np.float64),
                camera_id=int(image_parts[8]),
                name=image_parts[9],
                observations=observations,
            )
        )
    return images


def parse_colmap_points3d(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        points[int(parts[0])] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    return points


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def colmap_camera_center(image: ColmapImage) -> np.ndarray:
    return -(qvec_to_rotmat(image.qvec).T @ image.tvec)


def colmap_camera_view_axis(image: ColmapImage) -> np.ndarray:
    return qvec_to_rotmat(image.qvec).T[:, 2]


def build_covisibility_graph(
    colmap_images: list[ColmapImage],
    *,
    max_neighbors: int,
    min_shared_points: int,
) -> dict[int, list[CovisibilityEdge]]:
    track_images: dict[int, set[int]] = {}
    for image in colmap_images:
        for _x, _y, point3d_id in image.observations:
            track_images.setdefault(point3d_id, set()).add(image.image_id)

    shared_counts: dict[tuple[int, int], int] = {}
    for image_ids in track_images.values():
        for first_id, second_id in combinations(sorted(image_ids), 2):
            key = (first_id, second_id)
            shared_counts[key] = shared_counts.get(key, 0) + 1

    image_by_id = {image.image_id: image for image in colmap_images}
    centers = {image_id: colmap_camera_center(image) for image_id, image in image_by_id.items()}
    candidates: dict[int, list[CovisibilityEdge]] = {image_id: [] for image_id in image_by_id}
    for (first_id, second_id), shared_points in shared_counts.items():
        if shared_points < min_shared_points:
            continue
        baseline = float(np.linalg.norm(centers[first_id] - centers[second_id]))
        candidates[first_id].append(CovisibilityEdge(second_id, shared_points, baseline))
        candidates[second_id].append(CovisibilityEdge(first_id, shared_points, baseline))

    return {
        image_id: sorted(edges, key=lambda edge: (-edge.shared_points, -edge.baseline, edge.target_image_id))[:max_neighbors]
        for image_id, edges in candidates.items()
    }


def order_images_by_covisibility(
    colmap_images: list[ColmapImage],
    graph: dict[int, list[CovisibilityEdge]],
) -> list[ColmapImage]:
    """Greedily follow strongest unvisited neighbors, restarting at the strongest hub."""
    image_by_id = {image.image_id: image for image in colmap_images}
    strength = {image_id: sum(edge.shared_points for edge in graph.get(image_id, [])) for image_id in image_by_id}
    remaining = set(image_by_id)
    order: list[ColmapImage] = []
    current: int | None = None
    while remaining:
        if current is None or current not in remaining:
            current = max(remaining, key=lambda image_id: (strength[image_id], -image_id))
        order.append(image_by_id[current])
        remaining.discard(current)
        current = next((edge.target_image_id for edge in graph.get(order[-1].image_id, []) if edge.target_image_id in remaining), None)
    return order


def build_image_chunks(num_images: int, batch_size: int, overlap_size: int) -> list[list[int]]:
    if num_images <= 0:
        return []
    if batch_size <= 0 or batch_size >= num_images:
        return [list(range(num_images))]
    if batch_size == 1:
        raise ValueError("--batch-size must be at least 2 when more than one image is used")
    if overlap_size <= 0:
        raise ValueError("--overlap-size must be positive when --batch-size is active")
    if overlap_size >= batch_size:
        raise ValueError("--overlap-size must be smaller than --batch-size")

    chunks: list[list[int]] = []
    start = 0
    while start < num_images:
        end = min(start + batch_size, num_images)
        chunks.append(list(range(start, end)))
        if end == num_images:
            break
        start = end - overlap_size
    return chunks


def build_vggt_groups(
    *,
    registered_paths: list[Path],
    registered_by_name: Mapping[str, ColmapImage],
    grouping: str,
    batch_size: int,
    overlap_size: int,
) -> list[list[Path]]:
    return build_vggt_group_selection(
        registered_paths=registered_paths,
        registered_by_name=registered_by_name,
        grouping=grouping,
        batch_size=batch_size,
        overlap_size=overlap_size,
    ).groups


def build_vggt_group_selection(
    *,
    registered_paths: list[Path],
    registered_by_name: Mapping[str, ColmapImage],
    grouping: str,
    batch_size: int,
    overlap_size: int,
) -> VggtGroupSelection:
    """Build deterministic local groups, preferring direct sparse-track links."""
    if not registered_paths:
        return VggtGroupSelection(groups=[], records=[])
    if batch_size <= 0 or batch_size >= len(registered_paths):
        return VggtGroupSelection(groups=[list(registered_paths)], records=[])
    if grouping == "sequential":
        return VggtGroupSelection(
            groups=[list(registered_paths[start : start + batch_size]) for start in range(0, len(registered_paths), batch_size)],
            records=[],
        )
    if grouping != "covisibility":
        raise ValueError(f"unknown VGGT grouping: {grouping}")

    registered_images = [registered_by_name[path.name] for path in registered_paths]
    image_by_id = {image.image_id: image for image in registered_images}
    graph = build_covisibility_graph(
        registered_images,
        max_neighbors=GROUPING_MAX_NEIGHBORS,
        min_shared_points=GROUPING_MIN_SHARED_POINTS,
    )
    if batch_size == 1:
        raise ValueError("--batch-size must be at least 2 when more than one image is used")
    if overlap_size <= 0:
        raise ValueError("--overlap-size must be positive when --batch-size is active")
    if overlap_size >= batch_size:
        raise ValueError("--overlap-size must be smaller than --batch-size")

    edge_by_target = {
        image_id: {edge.target_image_id: edge for edge in edges}
        for image_id, edges in graph.items()
    }
    path_by_name = {path.name: path for path in registered_paths}
    remaining = set(image_by_id)
    previous_group: list[int] = []
    groups: list[list[Path]] = []
    records: list[dict[str, Any]] = []

    def edge_priority(reference_id: int, target_id: int) -> tuple[float, float, float, int]:
        edge = edge_by_target[reference_id].get(target_id)
        if edge is None:
            edge = edge_by_target[target_id][reference_id]
        reference_axis = colmap_camera_view_axis(image_by_id[reference_id])
        target_axis = colmap_camera_view_axis(image_by_id[target_id])
        view_angle = float(np.degrees(np.arccos(np.clip(reference_axis @ target_axis, -1.0, 1.0))))
        return edge.shared_points, edge.baseline, -view_angle, -target_id

    while remaining:
        bridge_candidates = [
            (edge_priority(reference_id, previous_id), reference_id)
            for previous_id in previous_group
            for reference_id in remaining
            if reference_id in edge_by_target[previous_id]
        ]
        if bridge_candidates:
            reference_id = max(bridge_candidates)[1]
            reference_selection = "strongest_previous_group_bridge"
        else:
            reference_id = max(
                remaining,
                key=lambda image_id: (
                    sum(target_id in remaining for target_id in edge_by_target[image_id]),
                    sum(edge.shared_points for edge in graph[image_id]),
                    -image_id,
                ),
            )
            reference_selection = "uncovered_covisibility_seed"

        overlap_ids = sorted(
            (image_id for image_id in previous_group if image_id in edge_by_target[reference_id]),
            key=lambda image_id: edge_priority(reference_id, image_id),
            reverse=True,
        )[:overlap_size]
        fresh_ids = sorted(
            (image_id for image_id in remaining - {reference_id} if image_id in edge_by_target[reference_id]),
            key=lambda image_id: edge_priority(reference_id, image_id),
            reverse=True,
        )[: batch_size - 1 - len(overlap_ids)]
        fallback_ids = sorted(
            remaining - {reference_id, *fresh_ids},
            key=lambda image_id: (
                sum(target_id in remaining for target_id in edge_by_target[image_id]),
                sum(edge.shared_points for edge in graph[image_id]),
                -image_id,
            ),
            reverse=True,
        )[: batch_size - 1 - len(overlap_ids) - len(fresh_ids)]
        group_ids = [reference_id, *overlap_ids, *fresh_ids, *fallback_ids]
        remaining.difference_update(group_ids)
        groups.append([path_by_name[image_by_id[image_id].name] for image_id in group_ids])
        records.append(
            {
                "group_index": len(groups) - 1,
                "reference": image_by_id[reference_id].name,
                "reference_selection": reference_selection,
                "requested_overlap_size": overlap_size,
                "selected_overlap_size": len(overlap_ids),
                "selected_fresh_size": len(fresh_ids),
                "selected_fallback_size": len(fallback_ids),
                "fallback": (
                    "direct_covisibility_unavailable"
                    if fallback_ids
                    else "none"
                ),
            }
        )
        previous_group = group_ids

    return VggtGroupSelection(groups=groups, records=records)


def build_vggt_group_diagnostics(
    *,
    groups: list[list[Path]],
    registered_by_name: Mapping[str, ColmapImage],
    grouping: str,
    batch_size: int,
    requested_overlap_size: int,
    selection_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    order_source = "input_discovery_order" if grouping == "sequential" or len(groups) <= 1 else "greedy_covisibility_order"
    ordered_names = list(dict.fromkeys(path.name for group in groups for path in group))
    order_index = {name: index for index, name in enumerate(ordered_names)}
    actual_overlaps = [len({path.name for path in groups[index - 1]} & {path.name for path in groups[index]}) for index in range(1, len(groups))]
    if not groups:
        effective_overlap_size, overlap_status = 0, "not_applicable_no_groups"
    elif len(groups) == 1:
        effective_overlap_size, overlap_status = 0, "not_applicable_single_group"
    elif grouping == "sequential":
        effective_overlap_size = 0
        overlap_status = "ignored_by_sequential_grouping" if requested_overlap_size > 0 else "disabled"
    else:
        effective_overlap_size = requested_overlap_size
        overlap_status = "applied" if requested_overlap_size > 0 else "disabled"

    group_records: list[dict[str, Any]] = []
    selection_by_index = {
        record["group_index"]: record for record in selection_records or []
    }
    previous_names: set[str] = set()
    for group_index, group in enumerate(groups):
        reference_path = group[0]
        reference = registered_by_name[reference_path.name]
        reference_tracks = {point_id for _x, _y, point_id in reference.observations}
        reference_center = colmap_camera_center(reference)
        reference_axis = colmap_camera_view_axis(reference)
        members: list[dict[str, Any]] = []
        for position, path in enumerate(group):
            image = registered_by_name[path.name]
            if position == 0:
                shared_tracks = center_distance = view_angle = None
                connection_status = "reference"
            else:
                shared_tracks = len(reference_tracks & {point_id for _x, _y, point_id in image.observations})
                center_distance = float(np.linalg.norm(reference_center - colmap_camera_center(image)))
                cosine = float(np.clip(reference_axis @ colmap_camera_view_axis(image), -1.0, 1.0))
                view_angle = float(np.degrees(np.arccos(cosine)))
                connection_status = "disconnected" if shared_tracks == 0 else "weak" if shared_tracks < GROUPING_MIN_SHARED_POINTS else "connected"
            members.append({
                "image": path.name, "image_id": image.image_id, "group_position": position,
                "source_order_index": order_index[path.name], "is_reference": position == 0,
                "shared_tracks_with_reference": shared_tracks, "camera_center_distance": center_distance,
                "view_angle_degrees": view_angle, "connection_status": connection_status,
            })
        group_names = {path.name for path in group}
        group_record = {
            "group_index": group_index, "order_source": order_source, "size": len(group),
            "incomplete": batch_size > 0 and len(group) < batch_size,
            "effective_overlap_with_previous": len(group_names & previous_names) if group_index else 0,
            "reference": {"image": reference_path.name, "image_id": reference.image_id},
            "reference_selection": selection_by_index.get(group_index, {}).get("reference_selection", "first_group_member"),
            "members": members,
        }
        if group_index in selection_by_index:
            group_record["selection"] = selection_by_index[group_index]
        group_records.append(group_record)
        previous_names = group_names

    return {
        "schema_version": 1, "grouping": grouping, "order_source": order_source,
        "image_count": len(ordered_names), "group_count": len(groups), "batch_size": batch_size,
        "strong_connection_min_shared_tracks": GROUPING_MIN_SHARED_POINTS,
        "overlap": {"requested_size": requested_overlap_size, "effective_size": effective_overlap_size, "applied": overlap_status == "applied", "status": overlap_status},
        "actual_consecutive_overlap_sizes": actual_overlaps, "groups": group_records,
    }


def build_scale_disagreement_diagnostics(
    *,
    colmap_images: list[ColmapImage],
    groups: list[list[Path]],
    scales_by_name: Mapping[str, float],
    min_shared_tracks: int = GROUPING_MIN_SHARED_POINTS,
) -> dict[str, Any]:
    image_by_id = {image.image_id: image for image in colmap_images}
    graph = build_covisibility_graph(colmap_images, max_neighbors=max(len(colmap_images) - 1, 0), min_shared_points=min_shared_tracks)
    group_sets = [{path.name for path in group} for group in groups]
    edges: list[dict[str, Any]] = []
    for first_id in sorted(graph):
        for edge in graph[first_id]:
            second_id = edge.target_image_id
            if first_id >= second_id:
                continue
            first, second = image_by_id[first_id], image_by_id[second_id]
            if first.name not in scales_by_name or second.name not in scales_by_name:
                continue
            first_scale, second_scale = float(scales_by_name[first.name]), float(scales_by_name[second.name])
            if not (np.isfinite(first_scale) and first_scale > 0):
                raise ValueError(f"invalid depth scale for {first.name}: {first_scale}")
            if not (np.isfinite(second_scale) and second_scale > 0):
                raise ValueError(f"invalid depth scale for {second.name}: {second_scale}")
            edges.append({
                "first_image": first.name, "first_image_id": first_id, "second_image": second.name,
                "second_image_id": second_id, "shared_tracks": edge.shared_points,
                "classification": "within_group" if any({first.name, second.name} <= group_names for group_names in group_sets) else "group_boundary",
                "log_scale_difference": abs(float(np.log(first_scale) - np.log(second_scale))),
            })

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        differences = np.asarray([edge["log_scale_difference"] for edge in selected], dtype=np.float64)
        return {"edge_count": len(selected), **{f"log_scale_difference_p{percentile}": float(np.percentile(differences, percentile)) if len(differences) else None for percentile in (50, 90, 95)}}

    return {
        "schema_version": 1, "metric": "absolute_log_depth_scale_difference", "scale_source": "sparse_colmap", "world_scale": "arbitrary",
        "edge_definition": {"undirected": True, "min_shared_tracks": min_shared_tracks, "requires_scale_at_both_ends": True},
        "classification": {"within_group": "both images occur together in at least one VGGT group", "group_boundary": "images never occur together in a VGGT group"},
        "all": summarize(edges), "within_group": summarize([edge for edge in edges if edge["classification"] == "within_group"]),
        "group_boundary": summarize([edge for edge in edges if edge["classification"] == "group_boundary"]), "edges": edges,
    }
