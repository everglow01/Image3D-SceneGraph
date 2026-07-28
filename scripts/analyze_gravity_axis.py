#!/usr/bin/env python3
"""Select a diagnostic gravity axis from Manhattan-frame evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pointcloud import read_ply_points, sample_points, vector_to_json


SCHEMA_VERSION = 1
SOURCE_WEIGHTS = {
    "imu_orientation": 3.0,
    "camera_orientation": 2.0,
    "camera_center_span": 1.0,
    "point_span": 1.0,
    "plane_ordering": 1.0,
}


class GravityAnalysisError(RuntimeError):
    """Raised when required gravity-analysis evidence is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def qvec_to_rotmat(value: Any) -> np.ndarray:
    vector = finite_vector(value, 4, "COLMAP qvec")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("COLMAP qvec cannot be zero")
    qw, qx, qy, qz = vector / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} cannot be zero")
    return vector / norm


def load_colmap_cameras(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable" if path is None else "invalid",
        "valid_count": 0,
        "rejected_count": 0,
        "rejected": [],
        "records": {},
        "centers": np.empty((0, 3), dtype=np.float64),
        "image_ups": np.empty((0, 3), dtype=np.float64),
    }
    if path is None:
        result["reason"] = "camera_file_not_supplied"
        return result
    if not path.is_file():
        result["reason"] = "camera_file_missing"
        return result
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        result["reason"] = f"camera_file_unreadable: {exc}"
        return result
    if not isinstance(payload, dict) or payload.get("coordinate_system") != "colmap_world":
        result["reason"] = "unsupported_camera_coordinate_system"
        return result
    images = payload.get("images")
    if not isinstance(images, list):
        result["reason"] = "camera_images_not_array"
        return result

    centers: list[np.ndarray] = []
    image_ups: list[np.ndarray] = []
    records: dict[str, dict[str, Any]] = {}
    for position, image in enumerate(images):
        try:
            if not isinstance(image, dict):
                raise ValueError("camera record must be an object")
            name = Path(str(image["name"])).name
            if not name or name in records:
                raise ValueError("camera image name must be nonempty and unique")
            rotation = qvec_to_rotmat(image.get("qvec"))
            translation = finite_vector(image.get("tvec"), 3, "COLMAP tvec")
            center = -(rotation.T @ translation)
            image_up = rotation.T @ np.array([0.0, -1.0, 0.0])
            records[name] = {
                "rotation": rotation,
                "center": center,
                "image_up": image_up,
            }
            centers.append(center)
            image_ups.append(image_up)
        except (KeyError, ValueError) as exc:
            result["rejected"].append({"position": position, "reason": str(exc)})

    result["valid_count"] = len(records)
    result["rejected_count"] = len(result["rejected"])
    result["records"] = records
    result["centers"] = np.stack(centers) if centers else np.empty((0, 3), dtype=np.float64)
    result["image_ups"] = np.stack(image_ups) if image_ups else np.empty((0, 3), dtype=np.float64)
    if records:
        result["status"] = "available"
        result.pop("reason", None)
    else:
        result["reason"] = "no_valid_camera_records"
    return result


def load_imu_up_vectors(path: Path | None, camera_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable" if path is None else "invalid",
        "valid_count": 0,
        "missing_count": len(camera_records),
        "rejected_count": 0,
        "rejected": [],
        "world_up_vectors": np.empty((0, 3), dtype=np.float64),
    }
    if path is None:
        result["reason"] = "imu_file_not_supplied"
        return result
    if not path.is_file():
        result["reason"] = "imu_file_missing"
        return result
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        result["reason"] = f"imu_file_unreadable: {exc}"
        return result
    if not isinstance(payload, dict) or payload.get("coordinate_system") != "opencv_camera":
        result["reason"] = "unsupported_imu_coordinate_system"
        return result
    records = payload.get("records")
    if not isinstance(records, dict):
        result["reason"] = "imu_records_not_object"
        return result

    vectors: list[np.ndarray] = []
    matched_names: set[str] = set()
    for supplied_name in sorted(records):
        name = Path(str(supplied_name)).name
        record = records[supplied_name]
        try:
            if name not in camera_records:
                raise ValueError("no matching camera record")
            if not isinstance(record, dict) or ("up" in record) == ("gravity" in record):
                raise ValueError("record must contain exactly one of up or gravity")
            if "up" in record:
                camera_up = normalize(finite_vector(record["up"], 3, "IMU up"), "IMU up")
            else:
                gravity = normalize(finite_vector(record["gravity"], 3, "IMU gravity"), "IMU gravity")
                camera_up = -gravity
            world_up = camera_records[name]["rotation"].T @ camera_up
            vectors.append(normalize(world_up, "world IMU up"))
            matched_names.add(name)
        except ValueError as exc:
            result["rejected"].append({"image_name": str(supplied_name), "reason": str(exc)})

    result["valid_count"] = len(vectors)
    result["missing_count"] = len(set(camera_records) - matched_names)
    result["rejected_count"] = len(result["rejected"])
    result["world_up_vectors"] = np.stack(vectors) if vectors else np.empty((0, 3), dtype=np.float64)
    if vectors:
        result["status"] = "available"
        result.pop("reason", None)
    else:
        result["reason"] = "no_valid_matched_imu_records"
    return result


def orientation_evidence(
    vectors: np.ndarray,
    axes: np.ndarray,
    *,
    min_records: int,
    min_winner_score: float,
) -> dict[str, Any]:
    if len(vectors) < min_records:
        return {
            "status": "unavailable",
            "reason": "insufficient_records",
            "record_count": int(len(vectors)),
            "scores": None,
        }
    dots = vectors @ axes.T
    scores = np.mean(dots**2, axis=0)
    scores /= float(np.sum(scores))
    ranking = np.argsort(-scores, kind="stable")
    mean_absolute = np.mean(np.abs(dots), axis=0)
    mean_signed = np.mean(dots, axis=0)
    coherence = np.divide(
        np.abs(mean_signed),
        mean_absolute,
        out=np.zeros_like(mean_signed),
        where=mean_absolute > 1e-12,
    )
    status = "available" if float(scores[ranking[0]]) >= min_winner_score else "uninformative"
    result = {
        "status": status,
        "record_count": int(len(vectors)),
        "mean_absolute_dot": vector_to_json(mean_absolute),
        "mean_signed_dot": vector_to_json(mean_signed),
        "sign_coherence": vector_to_json(coherence),
        "scores": vector_to_json(scores),
        "winner_axis_index": int(ranking[0]),
        "winner_score": float(scores[ranking[0]]),
        "winner_margin": float(scores[ranking[0]] - scores[ranking[1]]),
    }
    if status != "available":
        result["reason"] = "no_axis_alignment_consensus"
    return result


def span_evidence(values: np.ndarray, axes: np.ndarray, label: str) -> dict[str, Any]:
    if len(values) < 3:
        return {"status": "unavailable", "reason": f"insufficient_{label}_records", "scores": None}
    projected = values @ axes.T
    lower = np.percentile(projected, 5, axis=0)
    upper = np.percentile(projected, 95, axis=0)
    spans = upper - lower
    if not np.isfinite(spans).all() or np.any(spans <= 1e-9):
        return {
            "status": "uninformative",
            "reason": f"degenerate_{label}_span",
            "robust_span": vector_to_json(spans),
            "scores": None,
        }
    inverse_squared = 1.0 / spans**2
    scores = inverse_squared / float(np.sum(inverse_squared))
    return {
        "status": "available",
        "record_count": int(len(values)),
        "percentile_range": [5.0, 95.0],
        "lower": vector_to_json(lower),
        "upper": vector_to_json(upper),
        "robust_span": vector_to_json(spans),
        "scores": vector_to_json(scores),
    }


def plane_ordering_evidence(
    planes: list[dict[str, Any]],
    reliable_indices: set[int],
    axes: np.ndarray,
    points: np.ndarray,
    camera_centers: np.ndarray,
    *,
    match_tolerance_degrees: float,
) -> dict[str, Any]:
    point_coordinates = points @ axes.T
    camera_coordinates = camera_centers @ axes.T if len(camera_centers) else None
    raw_scores = np.zeros(3, dtype=np.float64)
    records = []
    for position, plane in enumerate(planes):
        index = int(plane.get("index", position))
        try:
            normal = normalize(finite_vector(plane.get("normal"), 3, "plane normal"), "plane normal")
            centroid = finite_vector(plane.get("centroid"), 3, "plane centroid")
            support_value = plane.get("inlier_ratio")
            if not isinstance(support_value, (int, float)):
                raise ValueError("plane inlier ratio must be numeric")
            support = float(support_value)
            if not math.isfinite(support) or support < 0:
                raise ValueError("plane inlier ratio must be finite and nonnegative")
        except (TypeError, ValueError) as exc:
            records.append({"plane_index": index, "status": "invalid", "reason": str(exc)})
            continue
        dots = np.abs(axes @ normal)
        axis_index = int(np.argmax(dots))
        residual = math.degrees(math.acos(float(np.clip(dots[axis_index], 0.0, 1.0))))
        coordinate = float(centroid @ axes[axis_index])
        point_lower = float(np.mean(point_coordinates[:, axis_index] <= coordinate))
        point_upper = 1.0 - point_lower
        camera_lower = None
        camera_upper = None
        side_scores = [max(point_lower, point_upper)]
        if camera_coordinates is not None:
            camera_lower = float(np.mean(camera_coordinates[:, axis_index] <= coordinate))
            camera_upper = 1.0 - camera_lower
            side_scores.append(max(camera_lower, camera_upper))
        one_side_score = float(np.prod(side_scores) ** (1.0 / len(side_scores)))
        reliable = index in reliable_indices
        matched = residual <= match_tolerance_degrees
        contribution = support * one_side_score if reliable and matched else 0.0
        raw_scores[axis_index] += contribution
        records.append(
            {
                "plane_index": index,
                "status": "matched" if matched else "unmatched",
                "axis_index": axis_index,
                "angular_residual_degrees": residual,
                "coordinate": coordinate,
                "inlier_ratio": support,
                "reliable": reliable,
                "point_fraction_below": point_lower,
                "point_fraction_above": point_upper,
                "camera_fraction_below": camera_lower,
                "camera_fraction_above": camera_upper,
                "one_side_score": one_side_score,
                "score_contribution": contribution,
            }
        )
    total = float(np.sum(raw_scores))
    if total <= 1e-12:
        return {
            "status": "unavailable",
            "reason": "no_reliable_matched_boundary_planes",
            "match_tolerance_degrees": match_tolerance_degrees,
            "raw_scores": vector_to_json(raw_scores),
            "scores": None,
            "planes": records,
        }
    return {
        "status": "available",
        "match_tolerance_degrees": match_tolerance_degrees,
        "raw_scores": vector_to_json(raw_scores),
        "scores": vector_to_json(raw_scores / total),
        "planes": records,
    }


def combine_evidence(
    sources: dict[str, dict[str, Any]],
    axes: np.ndarray,
    cluster_ids: list[int],
    *,
    min_sources: int,
    min_winner_score: float,
    min_winner_margin: float,
    sign_min_mean_absolute_dot: float,
    sign_min_coherence: float,
) -> dict[str, Any]:
    available = [name for name, value in sources.items() if value.get("status") == "available"]
    weighted = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    for name in available:
        score = np.asarray(sources[name]["scores"], dtype=np.float64)
        weight = SOURCE_WEIGHTS[name]
        weighted += weight * score
        total_weight += weight
    scores = weighted / total_weight if total_weight else np.zeros(3, dtype=np.float64)
    ranking = np.argsort(-scores, kind="stable")
    winner = int(ranking[0])
    winner_score = float(scores[winner])
    margin = float(scores[winner] - scores[ranking[1]])
    directional = [name for name in ("imu_orientation", "camera_orientation") if name in available]
    reasons = []
    directional_winners = {int(sources[name]["winner_axis_index"]) for name in directional}
    if len(available) < min_sources:
        reasons.append("insufficient_available_evidence_sources")
    if not directional:
        reasons.append("no_reliable_directional_evidence")
    if len(directional_winners) > 1:
        reasons.append("directional_sources_disagree")
    if directional_winners and winner not in directional_winners:
        reasons.append("combined_directional_disagreement")
    if winner_score < min_winner_score:
        reasons.append("winner_score_below_threshold")
    if margin < min_winner_margin:
        reasons.append("winner_margin_below_threshold")

    selected = not reasons
    sign_candidates: list[tuple[str, int]] = []
    if selected:
        for name in directional:
            source = sources[name]
            mean_absolute = float(source["mean_absolute_dot"][winner])
            coherence = float(source["sign_coherence"][winner])
            mean_signed = float(source["mean_signed_dot"][winner])
            if mean_absolute >= sign_min_mean_absolute_dot and coherence >= sign_min_coherence:
                sign_candidates.append((name, 1 if mean_signed >= 0 else -1))
    signs = {sign for _, sign in sign_candidates}
    if len(signs) == 1:
        up_sign_status = "selected"
        up_sign = next(iter(signs))
        up_vector = vector_to_json(axes[winner] * up_sign)
        gravity_vector = vector_to_json(-axes[winner] * up_sign)
    elif len(signs) > 1:
        up_sign_status = "ambiguous"
        up_sign = None
        up_vector = None
        gravity_vector = None
    else:
        up_sign_status = "unavailable"
        up_sign = None
        up_vector = None
        gravity_vector = None

    return {
        "status": "selected" if selected else "ambiguous",
        "ambiguity_reasons": reasons,
        "available_sources": available,
        "source_weights": SOURCE_WEIGHTS,
        "scores": vector_to_json(scores),
        "winner_axis_index": winner if selected else None,
        "winner_cluster_id": cluster_ids[winner] if selected else None,
        "winner_score": winner_score,
        "winner_margin": margin,
        "axis": vector_to_json(axes[winner]) if selected else None,
        "up_sign_status": up_sign_status,
        "up_sign": up_sign,
        "up_vector": up_vector,
        "gravity_vector": gravity_vector,
        "sign_sources": [name for name, _ in sign_candidates],
    }


def evaluate_gravity_axes(
    axes: np.ndarray,
    cluster_ids: list[int],
    points: np.ndarray,
    planes: list[dict[str, Any]],
    reliable_plane_indices: set[int],
    camera_centers: np.ndarray,
    camera_ups: np.ndarray,
    imu_ups: np.ndarray,
    *,
    min_camera_records: int = 3,
    min_imu_records: int = 3,
    camera_consensus_score: float = 0.45,
    imu_consensus_score: float = 0.70,
    plane_match_tolerance_degrees: float = 15.0,
    min_evidence_sources: int = 2,
    min_winner_score: float = 0.45,
    min_winner_margin: float = 0.15,
    sign_min_mean_absolute_dot: float = 0.50,
    sign_min_coherence: float = 0.75,
) -> dict[str, Any]:
    axes = np.asarray(axes, dtype=np.float64)
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        raise GravityAnalysisError("Manhattan axes must be a finite 3x3 array")
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5):
        raise GravityAnalysisError("Manhattan axes must be orthonormal")
    if len(cluster_ids) != 3:
        raise GravityAnalysisError("Manhattan candidate must contain three cluster IDs")

    sources = {
        "imu_orientation": orientation_evidence(
            imu_ups, axes, min_records=min_imu_records, min_winner_score=imu_consensus_score
        ),
        "camera_orientation": orientation_evidence(
            camera_ups,
            axes,
            min_records=min_camera_records,
            min_winner_score=camera_consensus_score,
        ),
        "camera_center_span": span_evidence(camera_centers, axes, "camera_center"),
        "point_span": span_evidence(points, axes, "point"),
        "plane_ordering": plane_ordering_evidence(
            planes,
            reliable_plane_indices,
            axes,
            points,
            camera_centers,
            match_tolerance_degrees=plane_match_tolerance_degrees,
        ),
    }
    selection = combine_evidence(
        sources,
        axes,
        cluster_ids,
        min_sources=min_evidence_sources,
        min_winner_score=min_winner_score,
        min_winner_margin=min_winner_margin,
        sign_min_mean_absolute_dot=sign_min_mean_absolute_dot,
        sign_min_coherence=sign_min_coherence,
    )
    return {"evidence": sources, "selection": selection}


def analyze_gravity_axis(
    manhattan_report_path: Path,
    point_cloud_path: Path,
    *,
    cameras_path: Path | None = None,
    imu_path: Path | None = None,
    sample_size: int = 50_000,
    seed: int = 42,
) -> dict[str, Any]:
    report = read_json(manhattan_report_path)
    if not isinstance(report, dict):
        raise GravityAnalysisError("Manhattan report must be a JSON object")
    expected_hash = report.get("input", {}).get("sha256")
    actual_hash = sha256_file(point_cloud_path)
    if expected_hash != actual_hash:
        raise GravityAnalysisError("point-cloud hash does not match Manhattan report")

    camera_data = load_colmap_cameras(cameras_path)
    imu_data = load_imu_up_vectors(imu_path, camera_data["records"])
    inputs = {
        "manhattan_report": {
            "path": manhattan_report_path.as_posix(),
            "sha256": sha256_file(manhattan_report_path),
        },
        "point_cloud": {"path": point_cloud_path.as_posix(), "sha256": actual_hash},
        "cameras": input_descriptor(cameras_path),
        "imu": input_descriptor(imu_path),
    }
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "g1_23_gravity_axis_evidence",
        "inputs": inputs,
        "protocol": {
            "ground_truth_used": False,
            "semantic_model_used": False,
            "point_cloud_written": False,
            "alignment_changed": False,
            "plane_semantics_assigned": False,
            "exif_orientation_used": False,
        },
        "parameters": {
            "sample_size": sample_size,
            "seed": seed,
            "source_weights": SOURCE_WEIGHTS,
            "min_camera_records": 3,
            "min_imu_records": 3,
            "camera_consensus_score": 0.45,
            "imu_consensus_score": 0.70,
            "plane_match_tolerance_degrees": 15.0,
            "min_evidence_sources": 2,
            "min_winner_score": 0.45,
            "min_winner_margin": 0.15,
            "sign_min_mean_absolute_dot": 0.50,
            "sign_min_coherence": 0.75,
        },
        "metadata": {
            "cameras": serializable_metadata(camera_data),
            "imu": serializable_metadata(imu_data),
            "exif_orientation": {
                "status": "not_used",
                "reason": "pixel_storage_orientation_is_not_gravity",
            },
        },
    }

    candidate = report.get("best_candidate")
    if report.get("status") != "candidate" or not isinstance(candidate, dict):
        return {
            **base,
            "status": "ambiguous",
            "ambiguity_reasons": ["manhattan_frame_not_unambiguous"],
            "evidence": {},
            "selection": None,
        }
    try:
        axes = np.asarray(candidate["orthonormal_axes"], dtype=np.float64)
        cluster_ids = [int(value) for value in candidate["cluster_ids"]]
        planes = report["plane_diagnostics"]["dominant_planes"]
        reliable_indices = {int(value) for value in report["reliable_plane_indices"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise GravityAnalysisError("Manhattan candidate is missing required evidence") from exc
    if not isinstance(planes, list):
        raise GravityAnalysisError("Manhattan plane diagnostics must be an array")

    points = read_ply_points(point_cloud_path)
    points = points[np.isfinite(points).all(axis=1)]
    points = sample_points(points, sample_size, seed).astype(np.float64, copy=False)
    if len(points) < 3:
        raise GravityAnalysisError("point cloud contains fewer than three finite points")
    evaluated = evaluate_gravity_axes(
        axes,
        cluster_ids,
        points,
        planes,
        reliable_indices,
        camera_data["centers"],
        camera_data["image_ups"],
        imu_data["world_up_vectors"],
    )
    return {
        **base,
        "status": evaluated["selection"]["status"],
        "ambiguity_reasons": evaluated["selection"]["ambiguity_reasons"],
        "manhattan_candidate": {
            "cluster_ids": cluster_ids,
            "axes": [list(axis) for axis in axes.tolist()],
        },
        "sampled_point_count": int(len(points)),
        **evaluated,
    }


def input_descriptor(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def serializable_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if key not in {"records", "centers", "image_ups", "world_up_vectors"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manhattan-report", required=True, type=Path)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--imu", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        payload = analyze_gravity_axis(
            args.manhattan_report,
            args.point_cloud,
            cameras_path=args.cameras,
            imu_path=args.imu,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        content = json.dumps(payload, indent=2) + "\n"
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
                raise GravityAnalysisError(f"analysis output differs: {args.output}")
        else:
            if args.output.exists():
                raise GravityAnalysisError(f"output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
    except (GravityAnalysisError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"{'verified' if args.check else 'wrote'} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
