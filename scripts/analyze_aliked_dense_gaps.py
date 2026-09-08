#!/usr/bin/env python3
"""Read-only gap support audit of a frozen dense ALIKED experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np

from image3d_scenegraph.geometry.colmap import sha256_file
from image3d_scenegraph.geometry.grouping import (
    colmap_camera_center,
    parse_colmap_images_with_points,
    parse_colmap_points3d,
)
from image3d_scenegraph.geometry.view_graph import summarize_view_graph
from image3d_scenegraph.video.registration import MAX_REGISTERED_GAP_SECONDS


PAIR_BASE = 2147483647
MIN_PAIR_MATCHES = 15


def distribution(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite diagnostic values")
    return dict(zip(("count", "min", "p50", "p90", "max"), (
        len(values), float(values.min()), float(np.median(values)),
        float(np.quantile(values, 0.9)), float(values.max()),
    )))


def read_model(path, image_names, feature_counts, timestamps):
    images = parse_colmap_images_with_points(path / "images.txt")
    centers = {image.image_id: colmap_camera_center(image) for image in images}
    for image in images:
        if image_names.get(image.image_id) != image.name:
            raise ValueError("model/database image identity mismatch")
    xyz = parse_colmap_points3d(path / "points3D.txt")
    observations = {image.image_id: {} for image in images}
    tracks = {}
    errors, angles, spans = [], [], []
    for line in (path / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 12 or (len(fields) - 8) % 2:
            raise ValueError("invalid sparse point track")
        point_id = int(fields[0])
        track = [(int(fields[i]), int(fields[i + 1])) for i in range(8, len(fields), 2)]
        ids = [image_id for image_id, _ in track]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate image in sparse track")
        for image_id, index in track:
            if image_id not in observations or not 0 <= index < feature_counts[image_id]:
                raise ValueError("invalid sparse observation index")
            if index in observations[image_id]:
                raise ValueError("duplicate sparse observation")
            observations[image_id][index] = point_id
        tracks[point_id] = set(ids)
        errors.append(float(fields[7]))
        first, last = min(ids, key=timestamps.get), max(ids, key=timestamps.get)
        rays = np.array([xyz[point_id] - centers[first], xyz[point_id] - centers[last]])
        lengths = np.linalg.norm(rays, axis=1)
        if np.any(lengths == 0):
            raise ValueError("zero-length sparse observation ray")
        cosine = np.dot(rays[0], rays[1]) / np.prod(lengths)
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        spans.append(timestamps[last] - timestamps[first])
    for image in images:
        if set(observations[image.image_id].values()) != {p for _, _, p in image.observations}:
            raise ValueError("inconsistent images/points3D track membership")
    ids = set(observations)
    return observations, tracks, {
        "path": path.name,
        "registered_count": len(ids),
        "image_ids": sorted(ids),
        "start_seconds": min(timestamps[i] for i in ids),
        "end_seconds": max(timestamps[i] for i in ids),
        "point_count": len(tracks),
        "observations_per_image": distribution(map(len, observations.values())),
        "track_length": distribution(map(len, tracks.values())),
        "track_time_span_seconds": distribution(spans),
        "point_mean_reprojection_error_pixels": distribution(errors),
        "first_last_ray_angle_degrees": distribution(angles),
        "first_last_ray_angle_below_one_degree_count": sum(a < 1 for a in angles),
    }


def direct_support(connection, observations, feature_counts):
    registered = set(observations)
    options = {i: defaultdict(set) for i in feature_counts if i not in registered}
    neighbors = {i: set() for i in options}
    pair_totals = Counter()
    for pair_id, rows, cols, blob in connection.execute(
        "SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows >= ?",
        (MIN_PAIR_MATCHES,),
    ):
        left, right = divmod(pair_id, PAIR_BASE)
        if left not in feature_counts or right not in feature_counts or left >= right:
            raise ValueError("invalid pair identity")
        if (left in registered) == (right in registered):
            continue
        if cols != 2 or blob is None or len(blob) != rows * cols * 4:
            raise ValueError("invalid match array")
        matches = np.frombuffer(blob, dtype="<u4").reshape(rows, cols)
        if np.any(matches[:, 0] >= feature_counts[left]) or np.any(matches[:, 1] >= feature_counts[right]):
            raise ValueError("match index exceeds feature count")
        if left in registered:
            source, target, matches = left, right, matches[:, ::-1]
        else:
            source, target = right, left
        neighbors[target].add(source)
        pair_totals[target] += rows
        for target_index, source_index in matches:
            point_id = observations[source].get(int(source_index))
            if point_id is not None:
                options[target][int(target_index)].add(point_id)
    records = {}
    for image_id, candidates in options.items():
        points = set().union(*candidates.values()) if candidates else set()
        records[image_id] = {
            "primary_verified_neighbors": len(neighbors[image_id]),
            "primary_verified_correspondences": pair_totals[image_id],
            "candidate_2d_count": len(candidates),
            "candidate_3d_count": len(points),
            "candidate_2d3d_pair_count": sum(map(len, candidates.values())),
            "ambiguous_2d_count": sum(len(p) > 1 for p in candidates.values()),
            "one_to_one_support_upper_bound": min(len(candidates), len(points)),
        }
    return records


def missing_intervals(timestamps, registered):
    ordered = sorted(registered, key=timestamps.get)
    intervals = []
    if timestamps[ordered[0]] > min(timestamps.values()):
        intervals.append(("prefix", min(timestamps.values()), timestamps[ordered[0]]))
    for left, right in zip(ordered, ordered[1:]):
        if timestamps[right] - timestamps[left] > MAX_REGISTERED_GAP_SECONDS:
            intervals.append(("internal", timestamps[left], timestamps[right]))
    if timestamps[ordered[-1]] < max(timestamps.values()):
        intervals.append(("suffix", timestamps[ordered[-1]], max(timestamps.values())))
    return intervals


def audit(root):
    root = root.resolve()
    result_path, selection_path = root / "results.json", root / "selection.json"
    result = json.loads(result_path.read_text())
    if result.get("profile") != "aliked_positive_dense_bruteforce_v1":
        raise ValueError("expected frozen dense ALIKED experiment")
    cell = result["cell"]
    paths = [(root.parent / model["path"]).resolve() for model in cell["models"]]
    for path in paths:
        path.relative_to(root)
    primary_path = (root.parent / cell["primary"]["path"]).resolve()
    if not paths or primary_path != paths[0]:
        raise ValueError("unexpected representative model ordering")
    database = root / "positive" / "database.db"
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("database has pending WAL data; immutable audit requires a settled snapshot")
    sources = [result_path, selection_path, database, root / "positive" / "mapping.log",
               root / "positive" / "mapping.request.json"]
    sources += [path / filename for path in paths for filename in ("cameras.txt", "images.txt", "points3D.txt")]
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sources}
    selection = json.loads(selection_path.read_text())["selected"]
    selected = {item["path"]: item for item in selection}
    if len(selected) != len(selection):
        raise ValueError("duplicate selected filename")
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        names = dict(connection.execute("SELECT image_id, name FROM images"))
        counts = dict(connection.execute("SELECT image_id, rows FROM keypoints"))
        if set(names.values()) != set(selected) or set(names) != set(counts):
            raise ValueError("selection/database mismatch")
        timestamps = {i: selected[name]["time_seconds"] for i, name in names.items()}
        models = [read_model(p, names, counts, timestamps) for p in paths]
        primary, tracks, _ = models[0]
        support = direct_support(connection, primary, counts)
        pairs = []
        for pair_id, candidates, verified, config in connection.execute(
            "SELECT m.pair_id, m.rows, COALESCE(t.rows,0), COALESCE(t.config,0) "
            "FROM matches m LEFT JOIN two_view_geometries t ON m.pair_id=t.pair_id ORDER BY m.pair_id"
        ):
            pairs.append({"image_ids": list(divmod(pair_id, PAIR_BASE)),
                          "candidate_match_count": candidates, "inlier_count": verified,
                          "geometric_config": config})
    graph_images = [{"colmap_image_id": i, "registered": i in primary,
                     "source_time_seconds": timestamps[i]} for i in sorted(names)]
    eligible = [pair for pair in pairs if pair["inlier_count"] >= MIN_PAIR_MATCHES]
    regions = []
    for kind, start, end in missing_intervals(timestamps, primary):
        ids = {i for i, t in timestamps.items() if start <= t <= end and i not in primary}
        left = {i for i in primary if timestamps[i] <= start}
        right = {i for i in primary if timestamps[i] >= end}
        bridge_tracks = [ids for ids in tracks.values() if ids & left and ids & right]
        touching = [p for p in eligible if set(p["image_ids"]) & ids]
        regions.append({
            "kind": kind, "start_seconds": start, "end_seconds": end,
            "duration_seconds": end - start, "unregistered_count": len(ids),
            "feature_count": distribution(counts[i] for i in ids),
            "touching_pair_config_counts": dict(Counter(str(p["geometric_config"]) for p in touching)),
            "direct_support": {key: distribution(support[i][key] for i in ids)
                               for key in next(iter(support.values()), {})},
            "images_with_zero_candidate_2d": sum(support[i]["candidate_2d_count"] == 0 for i in ids),
            "images_with_at_least_30_candidate_2d": sum(support[i]["candidate_2d_count"] >= 30 for i in ids),
            "primary_tracks_spanning_interval": len(bridge_tracks),
            "other_model_registration": [{"path": record["path"], "count": len(ids & set(obs))}
                                         for obs, _, record in models[1:]],
            "image_ids": sorted(ids),
        })
    overlap = [{"left": a[2]["path"], "right": b[2]["path"],
                "shared_registered_images": len(set(a[0]) & set(b[0]))}
               for a, b in combinations(models, 2)]
    for path in sources:
        if sha256_file(path) != hashes[str(path.relative_to(root))]:
            raise ValueError("source changed during audit")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("database changed during audit")
    return {
        "profile": "aliked_dense_gap_support_v1",
        "source_hashes": hashes,
        "source_unchanged": True,
        "mapper_graph_policy": {"min_num_matches": MIN_PAIR_MATCHES, "ignore_watermarks": False},
        "limitations": [
            "Direct support is from final primary tracks, not historical PnP state or validated inliers.",
            "Support excludes registration attempts, PnP, spatial conditioning and track completion.",
            "First/last ray angle is not maximum triangulation angle or ground truth.",
            "Graph connectivity, local-model overlap and registration do not certify physical geometry.",
        ],
        "all_verified_graph": summarize_view_graph(graph_images, pairs),
        "mapper_eligible_graph": summarize_view_graph(graph_images, eligible),
        "models": [record for _, _, record in models],
        "model_overlaps": overlap,
        "missing_intervals": regions,
        "unregistered_images": [{"image_id": i, "name": names[i], "time_seconds": timestamps[i],
                                 "feature_count": counts[i], **support[i]}
                                for i in sorted(support, key=timestamps.get)],
        "reconstruction_started": False,
        "training_started": False,
        "test_rgb_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite audit output")
    record = audit(args.root)
    record["code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output),
                      "connected_components": record["mapper_eligible_graph"]["connected_component_count"],
                      "intervals": record["missing_intervals"]}, indent=2))


if __name__ == "__main__":
    main()
