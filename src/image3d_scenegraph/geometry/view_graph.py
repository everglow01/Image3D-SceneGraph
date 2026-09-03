from __future__ import annotations

import math
from typing import Any

from image3d_scenegraph.video.registration import MAX_REGISTERED_GAP_SECONDS


class ViewGraphError(ValueError):
    """Raised when SfM pair evidence cannot form a valid view graph."""


def summarize_view_graph(
    images: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    image_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        image_id = _positive_int(image.get("colmap_image_id"), "image ID")
        if image_id in image_by_id:
            raise ViewGraphError(f"duplicate image ID: {image_id}")
        image_by_id[image_id] = image
    if not image_by_id:
        raise ViewGraphError("view graph has no images")

    adjacency = {image_id: set() for image_id in image_by_id}
    verified_edges: list[tuple[int, int]] = []
    candidate_counts: list[int] = []
    verified_counts: list[int] = []
    survival_ratios: list[float] = []
    geometric_configs: dict[str, int] = {}
    seen_pairs: set[tuple[int, int]] = set()
    candidate_pair_count = 0
    candidate_match_count = 0
    candidate_inlier_count = 0
    guided_inlier_count = 0
    verified_inlier_count = 0
    outlier_count = 0

    for pair in pairs:
        image_ids = pair.get("image_ids")
        if not isinstance(image_ids, list) or len(image_ids) != 2:
            raise ViewGraphError("pair image IDs must contain two entries")
        left = _positive_int(image_ids[0], "left image ID")
        right = _positive_int(image_ids[1], "right image ID")
        if left >= right or left not in image_by_id or right not in image_by_id:
            raise ViewGraphError(f"invalid view graph pair: {left}-{right}")
        edge = (left, right)
        if edge in seen_pairs:
            raise ViewGraphError(f"duplicate view graph pair: {left}-{right}")
        seen_pairs.add(edge)

        candidate = _non_negative_int(
            pair.get("candidate_match_count"), "candidate match count"
        )
        verified = _non_negative_int(pair.get("inlier_count"), "inlier count")
        has_extended_counts = any(
            key in pair
            for key in (
                "candidate_inlier_count",
                "guided_inlier_count",
                "outlier_count",
            )
        )
        if has_extended_counts and not all(
            key in pair
            for key in (
                "candidate_inlier_count",
                "guided_inlier_count",
                "outlier_count",
            )
        ):
            raise ViewGraphError("pair has incomplete guided-matching counts")
        if has_extended_counts:
            candidate_inliers = _non_negative_int(
                pair["candidate_inlier_count"], "candidate inlier count"
            )
            guided_inliers = _non_negative_int(
                pair["guided_inlier_count"], "guided inlier count"
            )
            rejected = _non_negative_int(pair["outlier_count"], "outlier count")
        else:
            candidate_inliers = verified
            guided_inliers = 0
            rejected = candidate - candidate_inliers
        if (
            candidate_inliers > candidate
            or candidate_inliers + guided_inliers != verified
            or candidate_inliers + rejected != candidate
        ):
            raise ViewGraphError(f"inconsistent match counts for pair {left}-{right}")

        config = _non_negative_int(
            pair.get("geometric_config"), "geometric config"
        )
        geometric_configs[str(config)] = geometric_configs.get(str(config), 0) + 1
        candidate_counts.append(candidate)
        candidate_match_count += candidate
        candidate_inlier_count += candidate_inliers
        guided_inlier_count += guided_inliers
        verified_inlier_count += verified
        outlier_count += rejected
        if candidate > 0:
            candidate_pair_count += 1
            survival_ratios.append(candidate_inliers / candidate)
        if verified > 0:
            adjacency[left].add(right)
            adjacency[right].add(left)
            verified_edges.append(edge)
            verified_counts.append(verified)

    components = _connected_components(adjacency)
    largest = components[0]
    degrees = [len(adjacency[image_id]) for image_id in sorted(adjacency)]
    registered_ids = {
        image_id
        for image_id, image in image_by_id.items()
        if image.get("registered") is True
    }
    timestamps = {
        image_id: float(image["source_time_seconds"])
        for image_id, image in image_by_id.items()
        if image.get("source_time_seconds") is not None
    }
    if any(not math.isfinite(value) or value < 0 for value in timestamps.values()):
        raise ViewGraphError("image timestamps must be finite and non-negative")

    tested_pair_count = len(pairs)
    return {
        "schema_version": 1,
        "profile": "sfm_verified_view_graph_v1",
        "edge_definition": "nonempty_two_view_geometry",
        "node_count": len(image_by_id),
        "registered_node_count": len(registered_ids),
        "tested_pair_count": tested_pair_count,
        "candidate_pair_count": candidate_pair_count,
        "verified_edge_count": len(verified_edges),
        "verified_edge_ratio": (
            len(verified_edges) / tested_pair_count if tested_pair_count else 0.0
        ),
        "match_totals": {
            "candidate": candidate_match_count,
            "candidate_inliers": candidate_inlier_count,
            "guided_inliers": guided_inlier_count,
            "verified": verified_inlier_count,
            "outliers": outlier_count,
        },
        "geometric_config_counts": dict(
            sorted(geometric_configs.items(), key=lambda item: int(item[0]))
        ),
        "connected_component_count": len(components),
        "largest_component_node_count": len(largest),
        "largest_component_ratio": len(largest) / len(image_by_id),
        "largest_component_registered_node_count": len(largest & registered_ids),
        "largest_component_unregistered_node_count": len(largest - registered_ids),
        "isolated_node_count": sum(degree == 0 for degree in degrees),
        "degree_one_node_count": sum(degree == 1 for degree in degrees),
        "degree_distribution": _distribution(degrees),
        "component_size_distribution": _distribution(
            [len(component) for component in components]
        ),
        "candidate_match_distribution": _distribution(candidate_counts),
        "verified_inlier_distribution": _distribution(verified_counts),
        "candidate_survival_ratio_distribution": _distribution(survival_ratios),
        "video": _video_summary(
            registered_ids,
            timestamps,
            verified_edges,
        ),
    }


def _connected_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    remaining = set(adjacency)
    components: list[set[int]] = []
    while remaining:
        start = min(remaining)
        component: set[int] = set()
        stack = [start]
        while stack:
            image_id = stack.pop()
            if image_id in component:
                continue
            component.add(image_id)
            stack.extend(adjacency[image_id] - component)
        remaining -= component
        components.append(component)
    return sorted(components, key=lambda component: (-len(component), min(component)))


def _video_summary(
    registered_ids: set[int],
    timestamps: dict[int, float],
    verified_edges: list[tuple[int, int]],
) -> dict[str, Any] | None:
    if not timestamps:
        return None
    edge_spans = [
        abs(timestamps[left] - timestamps[right])
        for left, right in verified_edges
        if left in timestamps and right in timestamps
    ]
    registered_timeline = sorted(
        (timestamps[image_id], image_id)
        for image_id in registered_ids
        if image_id in timestamps
    )
    gaps = [
        (left_time, right_time)
        for (left_time, _left_id), (right_time, _right_id) in zip(
            registered_timeline, registered_timeline[1:]
        )
        if right_time - left_time > MAX_REGISTERED_GAP_SECONDS
    ]
    timed_edges = [
        tuple(sorted((timestamps[left], timestamps[right])))
        for left, right in verified_edges
        if left in timestamps and right in timestamps
    ]
    directly_bridged = sum(
        any(edge_start <= gap_start and edge_end >= gap_end for edge_start, edge_end in timed_edges)
        for gap_start, gap_end in gaps
    )
    return {
        "timestamped_node_count": len(timestamps),
        "verified_edge_time_span_seconds": _distribution(edge_spans),
        "registered_gap_threshold_seconds": MAX_REGISTERED_GAP_SECONDS,
        "registered_gap_count": len(gaps),
        "directly_bridged_registered_gap_count": directly_bridged,
        "unbridged_registered_gap_count": len(gaps) - directly_bridged,
    }


def _distribution(values: list[int] | list[float]) -> dict[str, int | float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _quantile(ordered, 0.5),
        "p90": _quantile(ordered, 0.9),
        "max": ordered[-1],
    }


def _quantile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result == 0:
        raise ViewGraphError(f"{label} must be positive")
    return result


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ViewGraphError(f"{label} must be a non-negative integer")
    return value
