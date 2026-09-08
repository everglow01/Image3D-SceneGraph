#!/usr/bin/env python3
"""Compare frozen primary/model-7 overlap without merging or reconstruction."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np

from analyze_aliked_dense_gaps import distribution, read_model
from geometry_utils import estimate_similarity_transform, transform_points
from image3d_scenegraph.geometry.colmap import sha256_file
from image3d_scenegraph.geometry.grouping import (
    colmap_camera_center, parse_colmap_images_with_points, parse_colmap_points3d,
    qvec_to_rotmat,
)


def rotation_angle(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def compare_cameras(reference, candidate):
    names = sorted(set(reference) & set(candidate))
    target = np.array([colmap_camera_center(reference[n]) for n in names])
    source = np.array([colmap_camera_center(candidate[n]) for n in names])
    transform = estimate_similarity_transform(source, target)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    rotation = transform[:3, :3] / scale
    radius = float(np.median(np.linalg.norm(target - np.median(target, axis=0), axis=1)))
    if radius <= 1e-12:
        raise ValueError("degenerate common camera extent")
    residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    orientation = [rotation_angle(
        qvec_to_rotmat(candidate[n].qvec) @ rotation.T @ qvec_to_rotmat(reference[n].qvec).T
    ) for n in names]
    leave_one_out = []
    for i, name in enumerate(names):
        keep = np.arange(len(names)) != i
        try:
            fitted = estimate_similarity_transform(source[keep], target[keep])
            error = np.linalg.norm(transform_points(source[i:i + 1], fitted)[0] - target[i])
            leave_one_out.append({"name": name, "excluded_camera_residual_to_radius": float(error / radius)})
        except ValueError as exc:
            leave_one_out.append({"name": name, "unavailable": str(exc)})
    return transform, radius, {
        "common_names": names,
        "candidate_to_reference_sim3": transform.tolist(),
        "similarity_scale": scale,
        "reference_common_median_radius": radius,
        "reference_centered_singular_values": np.linalg.svd(target - target.mean(axis=0), compute_uv=False).tolist(),
        "candidate_centered_singular_values": np.linalg.svd(source - source.mean(axis=0), compute_uv=False).tolist(),
        "center_residual_to_reference_common_radius": distribution(residuals / radius),
        "orientation_residual_degrees": distribution(orientation),
        "per_camera": [{"name": n, "center_residual_to_radius": float(e / radius),
                        "orientation_residual_degrees": a} for n, e, a in zip(names, residuals, orientation)],
        "leave_one_camera_out": leave_one_out,
    }


def shared_point_groups(reference_observations, candidate_observations):
    shared = defaultdict(set)
    shared_observations = 0
    for image_id in sorted(set(reference_observations) & set(candidate_observations)):
        left, right = reference_observations[image_id], candidate_observations[image_id]
        for index in sorted(set(left) & set(right)):
            shared[(left[index], right[index])].add(image_id)
            shared_observations += 1
    reference_partners, candidate_partners = defaultdict(set), defaultdict(set)
    for left, right in shared:
        reference_partners[left].add(right)
        candidate_partners[right].add(left)
    groups = {
        "all_shared_point_pairs": list(shared),
        "mutually_unique_point_pairs": [p for p in shared if len(reference_partners[p[0]]) == len(candidate_partners[p[1]]) == 1],
        "mutually_unique_seen_in_two_common_images": [p for p, ids in shared.items()
            if len(ids) >= 2 and len(reference_partners[p[0]]) == len(candidate_partners[p[1]]) == 1],
    }
    return shared, groups, {
        "shared_feature_observation_count": shared_observations,
        "reference_points_with_multiple_candidate_partners": sum(len(v) > 1 for v in reference_partners.values()),
        "candidate_points_with_multiple_reference_partners": sum(len(v) > 1 for v in candidate_partners.values()),
    }


def compare_points(reference_observations, candidate_observations, reference_xyz, candidate_xyz, transform, radius):
    shared, groups, summary = shared_point_groups(reference_observations, candidate_observations)
    reports = {}
    for name, pairs in groups.items():
        target = np.array([reference_xyz[a] for a, _ in pairs]).reshape(-1, 3)
        source = np.array([candidate_xyz[b] for _, b in pairs]).reshape(-1, 3)
        residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        reports[name] = {
            "point_pair_count": len(pairs),
            "common_image_support": distribution(len(shared[p]) for p in pairs),
            "camera_fitted_residual_to_reference_common_radius": distribution(residuals / radius),
        }
    return {**summary, "groups": reports}


def orientation_similarity(reference_centers, candidate_centers, world_rotations):
    u, _, vt = np.linalg.svd(np.mean(world_rotations, axis=0))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    source = candidate_centers @ rotation.T
    source_centered = source - source.mean(axis=0)
    target_centered = reference_centers - reference_centers.mean(axis=0)
    variance = float(np.sum(source_centered ** 2))
    if variance <= 1e-12:
        raise ValueError("zero camera variance for orientation-constrained fit")
    scale = float(np.sum(source_centered * target_centered) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"orientation-constrained camera scale is not positive: {scale}")
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = reference_centers.mean(axis=0) - scale * source.mean(axis=0)
    return transform


def cross_check_alignment(cameras, observations, xyz, center_transform, radius):
    names = sorted(set(cameras[0]) & set(cameras[1]))
    target = np.array([colmap_camera_center(cameras[0][n]) for n in names])
    source = np.array([colmap_camera_center(cameras[1][n]) for n in names])
    world_rotations = [qvec_to_rotmat(cameras[0][n].qvec).T @ qvec_to_rotmat(cameras[1][n].qvec) for n in names]
    _, groups, _ = shared_point_groups(*observations)
    pairs = groups["mutually_unique_seen_in_two_common_images"]
    target_points = np.array([xyz[0][a] for a, _ in pairs]).reshape(-1, 3)
    source_points = np.array([xyz[1][b] for _, b in pairs]).reshape(-1, 3)
    report = {}
    fits = {
        "camera_centers": lambda: center_transform,
        "camera_orientations": lambda: orientation_similarity(target, source, world_rotations),
        "shared_points": lambda: estimate_similarity_transform(source_points, target_points),
    }
    for method, fit in fits.items():
        try:
            transform = fit()
        except ValueError as exc:
            report[method] = {"status": "unavailable", "reason": str(exc)}
            continue
        scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
        rotation = transform[:3, :3] / scale
        report[method] = {
            "status": "evaluated", "candidate_to_reference_sim3": transform.tolist(), "scale": scale,
            "camera_center_residual_to_radius": distribution(np.linalg.norm(transform_points(source, transform) - target, axis=1) / radius),
            "camera_orientation_residual_degrees": distribution(rotation_angle(rotation.T @ r) for r in world_rotations),
            "shared_point_pair_count": len(pairs),
            "shared_point_residual_to_radius": distribution(np.linalg.norm(transform_points(source_points, transform) - target_points, axis=1) / radius),
        }
    return report


def audit(root, *, cross_checks=False):
    root = root.resolve()
    prior_path = root / "gap-support-v1.json"
    prior = json.loads(prior_path.read_text())
    if prior.get("profile") != "aliked_dense_gap_support_v1" or not prior.get("source_unchanged"):
        raise ValueError("expected completed dense-gap audit")
    paths = [root / "positive" / name for name in ("model_0_txt", "model_7_txt")]
    source_paths = [root / "selection.json", root / "positive/database.db"]
    source_paths += [p / n for p in paths for n in ("cameras.txt", "images.txt", "points3D.txt")]
    expected = {str(p.relative_to(root)): prior["source_hashes"][str(p.relative_to(root))] for p in source_paths}
    expected[prior_path.name] = sha256_file(prior_path)
    source_paths.append(prior_path)
    for p in source_paths:
        if sha256_file(p) != expected[str(p.relative_to(root))]:
            raise ValueError("source differs from frozen gap audit")
    database = root / "positive/database.db"
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("nonempty WAL in frozen database")
    selection = json.loads((root / "selection.json").read_text())["selected"]
    times = {v["path"]: v["time_seconds"] for v in selection}
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        names = dict(db.execute("SELECT image_id,name FROM images"))
        counts = dict(db.execute("SELECT image_id,rows FROM keypoints"))
    timestamps = {i: times[n] for i, n in names.items()}
    model_data = [read_model(p, names, counts, timestamps) for p in paths]
    cameras = [{im.name: im for im in parse_colmap_images_with_points(p / "images.txt")} for p in paths]
    if len(set(cameras[0]) & set(cameras[1])) != 9:
        raise ValueError("expected nine frozen common cameras")
    transform, radius, camera_report = compare_cameras(*cameras)
    xyz = [parse_colmap_points3d(p / "points3D.txt") for p in paths]
    observations = [m[0] for m in model_data]
    points = compare_points(*observations, *xyz, transform, radius)
    cross_report = cross_check_alignment(cameras, observations, xyz, transform, radius) if cross_checks else None
    for p in source_paths:
        if sha256_file(p) != expected[str(p.relative_to(root))]:
            raise ValueError("source changed during overlap audit")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("database changed during overlap audit")
    return {
        "profile": "aliked_model7_cross_alignment_v1" if cross_checks else "aliked_model7_overlap_v1",
        "cross_alignment": cross_report,
        "source_hashes": expected,
        "source_unchanged": True,
        "camera_comparison": camera_report,
        "point_comparison": points,
        "common_camera_times": {n: times[n] for n in camera_report["common_names"]},
        "camera_parameters_text": {p.name: (p / "cameras.txt").read_text() for p in paths},
        "limitations": [
            "All nine common cameras fit one proper Sim3; no RANSAC or outlier removal.",
            "Point IDs are model-local; correspondence uses the same database image and feature index.",
            "Baseline point_comparison uses the camera-center fit; cross_alignment labels each independent fit.",
            "First-order agreement is not ground truth, metric accuracy or permission to merge.",
            "Excluded-camera checks diagnose fit sensitivity, not Test-split evaluation.",
        ],
        "model_merged": False, "reconstruction_started": False,
        "training_started": False, "test_rgb_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-checks", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite overlap audit")
    record = audit(args.root, cross_checks=args.cross_checks)
    record["code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output),
                      "cameras": record["camera_comparison"], "points": record["point_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
