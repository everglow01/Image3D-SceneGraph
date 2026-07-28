#!/usr/bin/env python3
"""Find diagnostic Manhattan-frame candidates from point-cloud planes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pointcloud import analyze_pointcloud, vector_to_json


SCHEMA_VERSION = 1


class ManhattanAnalysisError(RuntimeError):
    """Raised when Manhattan-frame evidence cannot be evaluated safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_planes(
    planes: list[dict[str, Any]],
    *,
    min_inlier_ratio: float = 0.08,
    cluster_angle_degrees: float = 15.0,
    orthogonality_tolerance_degrees: float = 15.0,
) -> dict[str, Any]:
    validate_thresholds(
        min_inlier_ratio,
        cluster_angle_degrees,
        orthogonality_tolerance_degrees,
    )
    reliable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for position, plane in enumerate(planes):
        index = int(plane.get("index", position))
        reasons: list[str] = []
        try:
            normal = normalized_vector(plane.get("normal"))
        except ManhattanAnalysisError:
            normal = None
            reasons.append("invalid_normal")
        inlier_count = numeric_value(plane.get("inlier_count"), "inlier_count")
        inlier_ratio = numeric_value(plane.get("inlier_ratio"), "inlier_ratio")
        if inlier_count < 3:
            reasons.append("fewer_than_three_inliers")
        if inlier_ratio < min_inlier_ratio:
            reasons.append("inlier_ratio_below_threshold")
        if reasons:
            rejected.append({"plane_index": index, "reasons": reasons})
            continue
        reliable.append(
            {
                "index": index,
                "normal": normal,
                "inlier_count": int(inlier_count),
                "inlier_ratio": inlier_ratio,
                "area_estimate": finite_optional(plane.get("area_estimate"), 0.0),
            }
        )

    reliable.sort(
        key=lambda plane: (
            -plane["inlier_ratio"],
            -plane["area_estimate"],
            plane["index"],
        )
    )
    clusters = cluster_normals(reliable, cluster_angle_degrees)
    partial_pairs = orthogonal_pairs(clusters, orthogonality_tolerance_degrees)
    candidates = candidate_frames(clusters, orthogonality_tolerance_degrees)

    if len(candidates) == 1:
        status = "candidate"
        reasons = []
        best_candidate: dict[str, Any] | None = candidates[0]
    else:
        status = "ambiguous"
        best_candidate = None
        if len(clusters) < 3:
            reasons = ["fewer_than_three_reliable_directions"]
        elif not candidates:
            reasons = ["no_orthogonal_three_direction_candidate"]
        else:
            reasons = ["multiple_orthogonal_three_direction_candidates"]

    evidence = "full" if candidates else "partial" if partial_pairs else "insufficient"
    return {
        "status": status,
        "frame_evidence": evidence,
        "ambiguity_reasons": reasons,
        "reliable_plane_indices": [plane["index"] for plane in reliable],
        "rejected_planes": rejected,
        "normal_clusters": clusters,
        "partial_orthogonal_pairs": partial_pairs,
        "candidates": candidates,
        "best_candidate": best_candidate,
    }


def cluster_normals(
    planes: list[dict[str, Any]],
    angle_tolerance_degrees: float,
) -> list[dict[str, Any]]:
    cosine_threshold = math.cos(math.radians(angle_tolerance_degrees))
    clusters: list[dict[str, Any]] = []
    sums: list[np.ndarray] = []
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        matches = [
            (abs(float(np.dot(normal, np.asarray(cluster["direction"])))), cluster["id"])
            for cluster in clusters
            if abs(float(np.dot(normal, np.asarray(cluster["direction"]))))
            >= cosine_threshold
        ]
        cluster_id = max(matches, key=lambda item: (item[0], -item[1]))[1] if matches else None
        weight = float(plane["inlier_ratio"])
        if cluster_id is None:
            cluster_id = len(clusters)
            sums.append(normal * weight)
            clusters.append(
                {
                    "id": cluster_id,
                    "plane_indices": [plane["index"]],
                    "representative_plane_index": plane["index"],
                    "direction": vector_to_json(normal),
                    "support_inlier_ratio": weight,
                }
            )
            continue

        cluster = clusters[cluster_id]
        direction = np.asarray(cluster["direction"], dtype=np.float64)
        signed_normal = normal if float(np.dot(normal, direction)) >= 0 else -normal
        sums[cluster_id] += signed_normal * weight
        cluster["plane_indices"].append(plane["index"])
        cluster["support_inlier_ratio"] += weight
        cluster["direction"] = vector_to_json(normalize(sums[cluster_id]))
    return clusters


def orthogonal_pairs(
    clusters: list[dict[str, Any]],
    tolerance_degrees: float,
) -> list[dict[str, Any]]:
    pairs = []
    for left, right in itertools.combinations(clusters, 2):
        metrics = pair_metrics(left, right)
        if metrics["residual_degrees"] <= tolerance_degrees:
            pairs.append(metrics)
    return pairs


def candidate_frames(
    clusters: list[dict[str, Any]],
    tolerance_degrees: float,
) -> list[dict[str, Any]]:
    candidates = []
    for selected in itertools.combinations(clusters, 3):
        pairwise = [pair_metrics(left, right) for left, right in itertools.combinations(selected, 2)]
        if any(pair["residual_degrees"] > tolerance_degrees for pair in pairwise):
            continue
        directions = np.column_stack(
            [np.asarray(cluster["direction"], dtype=np.float64) for cluster in selected]
        )
        orthonormal = right_handed_frame(directions)
        support_score = float(sum(cluster["support_inlier_ratio"] for cluster in selected))
        mean_residual = float(np.mean([pair["residual_degrees"] for pair in pairwise]))
        max_residual = float(max(pair["residual_degrees"] for pair in pairwise))
        orthogonality_score = max(0.0, 1.0 - mean_residual / 90.0)
        candidates.append(
            {
                "cluster_ids": [cluster["id"] for cluster in selected],
                "plane_indices": [
                    index for cluster in selected for index in cluster["plane_indices"]
                ],
                "measured_axes": [
                    list(axis) for axis in directions.T.tolist()
                ],
                "orthonormal_axes": [
                    list(axis) for axis in orthonormal.T.tolist()
                ],
                "pairwise": pairwise,
                "mean_orthogonality_residual_degrees": mean_residual,
                "max_orthogonality_residual_degrees": max_residual,
                "support_score": support_score,
                "orthogonality_score": orthogonality_score,
                "score": support_score * orthogonality_score,
            }
        )
    candidates.sort(key=lambda candidate: (-candidate["score"], candidate["cluster_ids"]))
    return candidates


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    dot = abs(
        float(
            np.dot(
                np.asarray(left["direction"], dtype=np.float64),
                np.asarray(right["direction"], dtype=np.float64),
            )
        )
    )
    angle = math.degrees(math.acos(float(np.clip(dot, 0.0, 1.0))))
    return {
        "cluster_ids": [left["id"], right["id"]],
        "absolute_dot": dot,
        "angle_degrees": angle,
        "residual_degrees": abs(90.0 - angle),
    }


def right_handed_frame(directions: np.ndarray) -> np.ndarray:
    unoriented = directions.copy()
    if float(np.linalg.det(unoriented)) < 0:
        unoriented[:, 2] *= -1
    left, _, right = np.linalg.svd(unoriented)
    return left @ right


def normalized_vector(value: Any) -> list[float]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ManhattanAnalysisError("plane normal must be numeric") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ManhattanAnalysisError("plane normal must be a finite 3-vector")
    return vector_to_json(normalize(vector))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ManhattanAnalysisError("plane normal cannot be zero")
    return vector / norm


def numeric_value(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ManhattanAnalysisError(f"{label} must be finite")
    return float(value)


def finite_optional(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default


def validate_thresholds(
    min_inlier_ratio: float,
    cluster_angle_degrees: float,
    orthogonality_tolerance_degrees: float,
) -> None:
    if not 0 <= min_inlier_ratio <= 1:
        raise ManhattanAnalysisError("minimum inlier ratio must be between 0 and 1")
    if not 0 < cluster_angle_degrees < 90:
        raise ManhattanAnalysisError("cluster angle must be between 0 and 90 degrees")
    if not 0 < orthogonality_tolerance_degrees < 90:
        raise ManhattanAnalysisError("orthogonality tolerance must be between 0 and 90 degrees")


def analyze_manhattan_frames(
    input_path: Path,
    *,
    sample_size: int = 50_000,
    ransac_iterations: int = 400,
    max_planes: int = 8,
    plane_distance: float = 0.0,
    min_inlier_ratio: float = 0.08,
    cluster_angle_degrees: float = 15.0,
    orthogonality_tolerance_degrees: float = 15.0,
    seed: int = 42,
) -> dict[str, Any]:
    if max_planes <= 0:
        raise ManhattanAnalysisError("max planes must be positive")
    diagnostics = analyze_pointcloud(
        input_path,
        sample_size=sample_size,
        ransac_iterations=ransac_iterations,
        max_planes=max_planes,
        plane_distance=plane_distance,
        min_plane_inlier_ratio=min_inlier_ratio,
        seed=seed,
    )
    frame_evidence = evaluate_planes(
        diagnostics["dominant_planes"],
        min_inlier_ratio=min_inlier_ratio,
        cluster_angle_degrees=cluster_angle_degrees,
        orthogonality_tolerance_degrees=orthogonality_tolerance_degrees,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "g1_22_manhattan_frame_candidates",
        "input": {
            "point_cloud": input_path.as_posix(),
            "sha256": sha256_file(input_path),
        },
        "protocol": {
            "ground_truth_used": False,
            "point_cloud_written": False,
            "alignment_changed": False,
            "gravity_or_ground_assigned": False,
        },
        "parameters": {
            "sample_size": sample_size,
            "ransac_iterations": ransac_iterations,
            "max_planes": max_planes,
            "plane_distance": plane_distance,
            "min_inlier_ratio": min_inlier_ratio,
            "cluster_angle_degrees": cluster_angle_degrees,
            "orthogonality_tolerance_degrees": orthogonality_tolerance_degrees,
            "seed": seed,
        },
        "plane_diagnostics": diagnostics,
        **frame_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--ransac-iterations", type=int, default=400)
    parser.add_argument("--max-planes", type=int, default=8)
    parser.add_argument("--plane-distance", type=float, default=0.0)
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=0.08)
    parser.add_argument("--cluster-angle-degrees", type=float, default=15.0)
    parser.add_argument("--orthogonality-tolerance-degrees", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        payload = analyze_manhattan_frames(
            args.input,
            sample_size=args.sample_size,
            ransac_iterations=args.ransac_iterations,
            max_planes=args.max_planes,
            plane_distance=args.plane_distance,
            min_inlier_ratio=args.min_plane_inlier_ratio,
            cluster_angle_degrees=args.cluster_angle_degrees,
            orthogonality_tolerance_degrees=args.orthogonality_tolerance_degrees,
            seed=args.seed,
        )
        content = json.dumps(payload, indent=2) + "\n"
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
                raise ManhattanAnalysisError(f"analysis output differs: {args.output}")
        else:
            if args.output.exists():
                raise ManhattanAnalysisError(f"output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
    except (ManhattanAnalysisError, OSError) as exc:
        parser.error(str(exc))
    print(f"{'verified' if args.check else 'wrote'} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
