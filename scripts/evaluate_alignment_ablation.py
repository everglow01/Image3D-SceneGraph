#!/usr/bin/env python3
"""Compare frozen single-plane and Manhattan point-cloud alignments."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from align_pointcloud import matrix_to_json, write_binary_ply
from analyze_pointcloud import parse_ply_header, read_ply_points_and_colors, vector_to_json


SCHEMA_VERSION = 1


class AlignmentAblationError(RuntimeError):
    """Raised when frozen alignment evidence cannot be applied safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlignmentAblationError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise AlignmentAblationError(f"{label} must contain an object")
    return value


def finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AlignmentAblationError(f"{label} must be numeric") from exc
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise AlignmentAblationError(f"{label} must be a finite {length}-vector")
    return vector


def finite_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AlignmentAblationError(f"{label} must be numeric") from exc
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise AlignmentAblationError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def validate_rotation(rotation: np.ndarray, label: str) -> None:
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise AlignmentAblationError(f"{label} is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise AlignmentAblationError(f"{label} is not a proper rotation")


def validate_affine_transform(value: Any, label: str) -> np.ndarray:
    transform = finite_matrix(value, (4, 4), label)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise AlignmentAblationError(f"{label} must be affine")
    validate_rotation(transform[:3, :3], f"{label} rotation")
    return transform


def source_hash_from_manhattan(report: dict[str, Any]) -> str:
    value = report.get("input")
    if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
        raise AlignmentAblationError("Manhattan report is missing its point-cloud hash")
    return value["sha256"]


def source_hash_from_gravity(report: dict[str, Any]) -> str:
    inputs = report.get("inputs")
    point_cloud = inputs.get("point_cloud") if isinstance(inputs, dict) else None
    if not isinstance(point_cloud, dict) or not isinstance(point_cloud.get("sha256"), str):
        raise AlignmentAblationError("gravity report is missing its point-cloud hash")
    return point_cloud["sha256"]


def fallback_reasons(
    manhattan_report: dict[str, Any], gravity_report: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if manhattan_report.get("status") != "candidate" or not isinstance(
        manhattan_report.get("best_candidate"), dict
    ):
        reasons.append("manhattan_frame_not_unambiguous")
    selection = gravity_report.get("selection")
    if gravity_report.get("status") != "selected" or not isinstance(selection, dict):
        reasons.append("gravity_axis_not_selected")
    elif selection.get("up_sign_status") != "selected" or selection.get("up_sign") not in (-1, 1):
        reasons.append("gravity_sign_not_selected")
    return reasons


def candidate_axes(
    manhattan_report: dict[str, Any], gravity_report: dict[str, Any]
) -> tuple[np.ndarray, int, list[int], np.ndarray]:
    candidate = manhattan_report["best_candidate"]
    selection = gravity_report["selection"]
    axes = finite_matrix(candidate.get("orthonormal_axes"), (3, 3), "Manhattan axes")
    validate_rotation(axes.T, "Manhattan frame")
    gravity_axes = finite_matrix(
        gravity_report.get("manhattan_candidate", {}).get("axes"),
        (3, 3),
        "gravity-report Manhattan axes",
    )
    if not np.allclose(axes, gravity_axes, atol=1e-9):
        raise AlignmentAblationError("Manhattan and gravity reports contain different axes")

    cluster_ids = candidate.get("cluster_ids")
    gravity_cluster_ids = gravity_report.get("manhattan_candidate", {}).get("cluster_ids")
    if (
        not isinstance(cluster_ids, list)
        or len(cluster_ids) != 3
        or cluster_ids != gravity_cluster_ids
        or not all(isinstance(value, int) for value in cluster_ids)
    ):
        raise AlignmentAblationError("Manhattan and gravity reports contain different cluster IDs")

    winner = selection.get("winner_axis_index")
    if not isinstance(winner, int) or winner not in range(3):
        raise AlignmentAblationError("gravity winner axis index is invalid")
    if selection.get("winner_cluster_id") != cluster_ids[winner]:
        raise AlignmentAblationError("gravity winner cluster does not match the Manhattan frame")
    selected_axis = finite_vector(selection.get("axis"), 3, "selected gravity axis")
    if not np.allclose(selected_axis, axes[winner], atol=1e-9):
        raise AlignmentAblationError("selected gravity axis does not match the Manhattan frame")
    up = finite_vector(selection.get("up_vector"), 3, "selected up vector")
    expected_up = axes[winner] * int(selection["up_sign"])
    if not np.allclose(up, expected_up, atol=1e-9):
        raise AlignmentAblationError("selected up vector does not match axis and sign")
    return axes, winner, cluster_ids, up


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def manhattan_rotation(axes: np.ndarray, winner: int, up: np.ndarray) -> dict[str, Any]:
    horizontal = [index for index in range(3) if index != winner]
    choices: list[tuple[tuple[Any, ...], np.ndarray, list[int], list[int]]] = []
    for order in itertools.permutations(horizontal):
        for signs in itertools.product((-1, 1), repeat=2):
            rotation = np.vstack([signs[0] * axes[order[0]], signs[1] * axes[order[1]], up])
            determinant = float(np.linalg.det(rotation))
            if determinant < 0:
                continue
            validate_rotation(rotation, "Manhattan alignment rotation")
            key = (rotation_angle_degrees(rotation), order, signs)
            choices.append((key, rotation, list(order), list(signs)))
    if not choices:
        raise AlignmentAblationError("no right-handed Manhattan orientation is available")
    key, rotation, order, signs = min(choices, key=lambda item: item[0])
    return {
        "rotation": rotation,
        "rotation_angle_degrees": key[0],
        "horizontal_axis_indices": order,
        "horizontal_signs": signs,
        "yaw_tie_breaker": "minimum_so3_angle_from_identity",
        "horizontal_axis_semantics_assigned": False,
    }


def reliable_planes(manhattan_report: dict[str, Any]) -> list[dict[str, Any]]:
    indices = manhattan_report.get("reliable_plane_indices")
    diagnostics = manhattan_report.get("plane_diagnostics")
    planes = diagnostics.get("dominant_planes") if isinstance(diagnostics, dict) else None
    if not isinstance(indices, list) or not all(isinstance(value, int) for value in indices):
        raise AlignmentAblationError("Manhattan reliable-plane indices are invalid")
    if not isinstance(planes, list):
        raise AlignmentAblationError("Manhattan plane diagnostics are missing")
    by_index = {
        int(plane.get("index", position)): plane
        for position, plane in enumerate(planes)
        if isinstance(plane, dict)
    }
    try:
        selected = [by_index[index] for index in indices]
    except KeyError as exc:
        raise AlignmentAblationError(f"reliable plane {exc.args[0]} is missing") from exc
    for plane in selected:
        finite_vector(plane.get("normal"), 3, "plane normal")
        finite_vector(plane.get("centroid"), 3, "plane centroid")
        support = plane.get("inlier_ratio")
        if not isinstance(support, (int, float)) or not math.isfinite(float(support)) or support < 0:
            raise AlignmentAblationError("plane inlier ratio must be finite and nonnegative")
    return selected


def pivot_plane(
    manhattan_report: dict[str, Any], planes: list[dict[str, Any]], selected_cluster_id: int
) -> dict[str, Any]:
    clusters = manhattan_report.get("normal_clusters")
    if not isinstance(clusters, list):
        raise AlignmentAblationError("Manhattan normal clusters are missing")
    cluster = next(
        (
            value
            for value in clusters
            if isinstance(value, dict) and value.get("id") == selected_cluster_id
        ),
        None,
    )
    indices = cluster.get("plane_indices") if isinstance(cluster, dict) else None
    if not isinstance(indices, list):
        raise AlignmentAblationError("selected Manhattan cluster is missing its planes")
    candidates = [plane for plane in planes if plane.get("index") in indices]
    if not candidates:
        raise AlignmentAblationError("selected Manhattan cluster has no reliable plane")
    return min(
        candidates,
        key=lambda plane: (
            -float(plane["inlier_ratio"]),
            -float(plane.get("area_estimate", 0.0)),
            int(plane["index"]),
        ),
    )


def rigid_transform(rotation: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    translation = np.zeros(3, dtype=np.float64)
    translation[2] = -pivot[2]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot + translation - rotation @ pivot
    return transform


def transformed_up_metrics(rotation: np.ndarray, up: np.ndarray) -> dict[str, Any]:
    transformed = rotation @ up
    dot = float(np.clip(transformed[2], -1.0, 1.0))
    return {
        "vector": vector_to_json(transformed),
        "dot_positive_z": dot,
        "angular_residual_to_positive_z_degrees": math.degrees(math.acos(dot)),
        "positive_z": dot > 0,
    }


def cardinal_axis(vector: np.ndarray) -> tuple[str, float]:
    index = int(np.argmax(np.abs(vector)))
    sign = "-" if vector[index] < 0 else ""
    residual = math.degrees(math.acos(float(np.clip(abs(vector[index]), 0.0, 1.0))))
    return f"{sign}{'xyz'[index]}", residual


def alignment_metrics(
    name: str, rotation: np.ndarray, planes: list[dict[str, Any]], up: np.ndarray
) -> dict[str, Any]:
    records = []
    weighted_sum = 0.0
    total_support = 0.0
    for plane in planes:
        normal = rotation @ finite_vector(plane["normal"], 3, "plane normal")
        axis, residual = cardinal_axis(normal)
        support = float(plane["inlier_ratio"])
        weighted_sum += support * residual
        total_support += support
        records.append(
            {
                "plane_index": int(plane["index"]),
                "inlier_ratio": support,
                "transformed_normal": vector_to_json(normal),
                "nearest_cardinal_axis": axis,
                "angular_residual_degrees": residual,
            }
        )
    return {
        "name": name,
        "rotation": matrix_to_json(rotation),
        "reliable_plane_count": len(records),
        "total_reliable_plane_support": total_support,
        "support_weighted_mean_angular_residual_degrees": (
            weighted_sum / total_support if total_support else None
        ),
        "max_angular_residual_degrees": max(
            (record["angular_residual_degrees"] for record in records), default=None
        ),
        "selected_up": transformed_up_metrics(rotation, up),
        "planes": records,
    }


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]


def evaluate_alignment_ablation(
    point_cloud_path: Path,
    single_alignment_path: Path,
    manhattan_report_path: Path,
    gravity_report_path: Path,
    *,
    output_point_cloud_path: Path | None = None,
    write_candidate: bool = True,
) -> dict[str, Any]:
    source_hash = sha256_file(point_cloud_path)
    single = load_json(single_alignment_path, "single-plane alignment")
    manhattan = load_json(manhattan_report_path, "Manhattan report")
    gravity = load_json(gravity_report_path, "gravity report")
    if source_hash != source_hash_from_manhattan(manhattan):
        raise AlignmentAblationError("point cloud does not match the Manhattan report")
    if source_hash != source_hash_from_gravity(gravity):
        raise AlignmentAblationError("point cloud does not match the gravity report")

    single_transform = validate_affine_transform(single.get("transform"), "single-plane transform")
    planes = reliable_planes(manhattan)
    reasons = fallback_reasons(manhattan, gravity)
    selection: dict[str, Any]
    candidate_output: dict[str, Any] | None = None
    manhattan_metrics: dict[str, Any] | None = None
    manhattan_transform: np.ndarray | None = None
    up: np.ndarray | None = None

    if reasons:
        selection = {
            "requested_strategy": "auto",
            "selected_strategy": "single_plane",
            "fallback": True,
            "fallback_reasons": reasons,
        }
    else:
        axes, winner, cluster_ids, up = candidate_axes(manhattan, gravity)
        orientation = manhattan_rotation(axes, winner, up)
        pivot = pivot_plane(manhattan, planes, cluster_ids[winner])
        pivot_value = finite_vector(pivot["centroid"], 3, "pivot plane centroid")
        manhattan_transform = rigid_transform(orientation["rotation"], pivot_value)
        if output_point_cloud_path is None:
            raise AlignmentAblationError("Manhattan selection requires an output point cloud")
        if write_candidate:
            points, colors = read_ply_points_and_colors(point_cloud_path)
            aligned = apply_transform(points, manhattan_transform)
            output_point_cloud_path.parent.mkdir(parents=True, exist_ok=True)
            write_binary_ply(output_point_cloud_path, aligned.astype(np.float32), colors)
        if not output_point_cloud_path.is_file():
            raise AlignmentAblationError(f"candidate point cloud is missing: {output_point_cloud_path}")
        try:
            with output_point_cloud_path.open("rb") as handle:
                output_header = parse_ply_header(handle)
        except (OSError, ValueError) as exc:
            raise AlignmentAblationError(f"candidate point cloud is invalid: {exc}") from exc
        expected_points = int(single.get("num_points", output_header.vertex_count))
        if output_header.vertex_count != expected_points:
            raise AlignmentAblationError(
                "candidate point count does not match the single-plane alignment"
            )
        output_properties = {name for name, _ in output_header.vertex_properties}
        output_has_colors = {"red", "green", "blue"}.issubset(output_properties)
        expected_colors = bool(single.get("colors_preserved", output_has_colors))
        if output_has_colors != expected_colors:
            raise AlignmentAblationError(
                "candidate color properties do not match the single-plane alignment"
            )
        candidate_output = {
            "path": output_point_cloud_path.as_posix(),
            "sha256": sha256_file(output_point_cloud_path),
            "num_points": output_header.vertex_count,
            "colors_preserved": output_has_colors,
        }
        selection = {
            "requested_strategy": "auto",
            "selected_strategy": "manhattan",
            "fallback": False,
            "fallback_reasons": [],
            "winner_axis_index": winner,
            "winner_cluster_id": cluster_ids[winner],
            "up_sign": gravity["selection"]["up_sign"],
            "up_vector": vector_to_json(up),
            "pivot_plane_index": int(pivot["index"]),
            "pivot": vector_to_json(pivot_value),
            "rotation_angle_degrees": orientation["rotation_angle_degrees"],
            "horizontal_axis_indices": orientation["horizontal_axis_indices"],
            "horizontal_signs": orientation["horizontal_signs"],
            "yaw_tie_breaker": orientation["yaw_tie_breaker"],
            "horizontal_axis_semantics_assigned": False,
            "transform": matrix_to_json(manhattan_transform),
        }
        manhattan_metrics = alignment_metrics(
            "manhattan", manhattan_transform[:3, :3], planes, up
        )

    if up is None:
        selection_value = gravity.get("selection")
        candidate_up = selection_value.get("up_vector") if isinstance(selection_value, dict) else None
        try:
            up = finite_vector(candidate_up, 3, "selected up vector")
        except AlignmentAblationError:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            single_up_metrics = None
        else:
            single_up_metrics = alignment_metrics(
                "single_plane", single_transform[:3, :3], planes, up
            )
    else:
        single_up_metrics = alignment_metrics(
            "single_plane", single_transform[:3, :3], planes, up
        )

    best_candidate = manhattan.get("best_candidate")
    pairwise = best_candidate.get("pairwise") if isinstance(best_candidate, dict) else []
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "g1_24_single_plane_vs_manhattan_alignment",
        "inputs": {
            "point_cloud": {"path": point_cloud_path.as_posix(), "sha256": source_hash},
            "single_alignment": {
                "path": single_alignment_path.as_posix(),
                "sha256": sha256_file(single_alignment_path),
            },
            "manhattan_report": {
                "path": manhattan_report_path.as_posix(),
                "sha256": sha256_file(manhattan_report_path),
            },
            "gravity_report": {
                "path": gravity_report_path.as_posix(),
                "sha256": sha256_file(gravity_report_path),
            },
        },
        "protocol": {
            "ground_truth_used": False,
            "semantic_model_used": False,
            "metric_scale_recovered": False,
            "reconstruction_rerun": False,
            "retained_assets_changed": False,
            "production_alignment_changed": False,
            "local_geometry_changed": False,
            "plane_semantics_assigned": False,
            "experimental_point_cloud_written": candidate_output is not None,
        },
        "selection": selection,
        "output_point_cloud": candidate_output,
        "manhattan_evidence": {
            "status": manhattan.get("status"),
            "ambiguity_reasons": manhattan.get("ambiguity_reasons", []),
            "pairwise": pairwise,
            "mean_orthogonality_residual_degrees": (
                best_candidate.get("mean_orthogonality_residual_degrees")
                if isinstance(best_candidate, dict)
                else None
            ),
            "max_orthogonality_residual_degrees": (
                best_candidate.get("max_orthogonality_residual_degrees")
                if isinstance(best_candidate, dict)
                else None
            ),
            "support_score": (
                best_candidate.get("support_score") if isinstance(best_candidate, dict) else None
            ),
        },
        "metrics": {
            "single_plane": single_up_metrics,
            "manhattan": manhattan_metrics,
            "rigid_transform_invariants": [
                "point_count",
                "pairwise_distances",
                "local_plane_rms",
                "local_thickness",
                "layer_topology",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--single-alignment", required=True, type=Path)
    parser.add_argument("--manhattan-report", required=True, type=Path)
    parser.add_argument("--gravity-report", required=True, type=Path)
    parser.add_argument("--output-point-cloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if not args.check:
            if args.output.exists():
                raise AlignmentAblationError(f"output already exists: {args.output}")
            if args.output_point_cloud.exists():
                raise AlignmentAblationError(
                    f"output point cloud already exists: {args.output_point_cloud}"
                )
        result = evaluate_alignment_ablation(
            args.point_cloud,
            args.single_alignment,
            args.manhattan_report,
            args.gravity_report,
            output_point_cloud_path=args.output_point_cloud,
            write_candidate=not args.check,
        )
        content = json.dumps(result, indent=2) + "\n"
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
                raise AlignmentAblationError(f"analysis output differs: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
    except (AlignmentAblationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{'verified' if args.check else 'wrote'} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
