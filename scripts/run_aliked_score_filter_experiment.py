#!/usr/bin/env python3
"""Frozen small-clip ALIKED score-filter experiment; no recovery, training or downloads."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import time

import numpy as np

from analyze_sfm_small_matches import read_pair
from build_colmap_aliked_score_filter import CANDIDATE_ROOT
from image3d_scenegraph.geometry.colmap import (
    resolve_colmap_camera_calibration,
    resolve_colmap_feature_profile,
    resolve_colmap_geometric_verification,
    resolve_colmap_local_matcher,
    sha256_file,
)
from image3d_scenegraph.geometry.grouping import (
    colmap_camera_center,
    parse_colmap_images_with_points,
)
from image3d_scenegraph.geometry.sfm_pose_health import build_sfm_pose_health_from_text
from image3d_scenegraph.video.keyframes import (
    V2_PROFILE_ID,
    materialize_video_candidates,
)
from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT, MIN_VIDEO_REGISTRATION_RATE, MIN_VIDEO_TEMPORAL_COVERAGE,
    analyze_registration_timeline,
)
from run_sfm_reference_audit import compare_pairs, database_features


SOURCE = "outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2/aliked-lightglue"
SOURCE_SELECTION_SHA = "9c877d158c3b87051f09ca72d04c86a50cf18add59969d53483dba1a10ba8151"
CLIPS = {"weak_696_743": (696, 743), "healthy_448_495": (448, 495)}
EXPANDED_CLIP = {"expanded_672_863": (672, 863)}
DENSE_CANDIDATE_COUNT = 446
SOURCE_VIDEO = "outputs/jobs/20260902_030611_a94e38dd/input/num4_room.mp4"
SMOKE_FRAMES = (468, 469, 710, 713)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sample_gpu_memory_mib():
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        return int(completed.stdout.strip()) if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def run_stage(directory, stage, command, timeout=1200):
    write_json(directory / f"{stage}.request.json", {"command": command, "timeout_seconds": timeout})
    start = time.monotonic()
    peak, telemetry_failures, timed_out = None, 0, False
    with (directory / f"{stage}.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                memory = sample_gpu_memory_mib()
                if memory is None:
                    telemetry_failures += 1
                else:
                    peak = max(peak or 0, memory)
                if time.monotonic() - start > timeout:
                    timed_out = True
                    break
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    result = {
        "exit_code": process.returncode, "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - start,
        "gpu_total_memory_peak_sampled_mib": peak,
        "gpu_sample_failures": telemetry_failures,
        "gpu_sample_interval_seconds": 2,
    }
    write_json(directory / f"{stage}.result.json", result)
    print(stage, json.dumps(result), flush=True)
    if timed_out or process.returncode != 0:
        raise RuntimeError(f"{stage} failed: {result}")
    return result


def database_summary(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        counts = dict(db.execute("SELECT i.name,k.rows FROM images i JOIN keypoints k USING(image_id) ORDER BY i.name"))
        mismatches = db.execute(
            "SELECT count(*) FROM keypoints k LEFT JOIN descriptors d USING(image_id) "
            "WHERE d.image_id IS NULL OR k.rows != d.rows"
        ).fetchone()[0]
        if mismatches:
            raise ValueError("keypoint/descriptor row mismatch")
        return {"feature_counts": counts, **{
            table: {"pairs": db.execute(f"SELECT count(*) FROM {table}").fetchone()[0],
                    "correspondences": db.execute(f"SELECT coalesce(sum(rows),0) FROM {table}").fetchone()[0]}
            for table in ("matches", "two_view_geometries")
        }}


def compare_databases(before, after, names):
    from scipy.spatial import cKDTree

    records = []
    with sqlite3.connect(before.as_uri() + "?mode=ro", uri=True) as old, sqlite3.connect(after.as_uri() + "?mode=ro", uri=True) as new:
        for name in names:
            a, b = database_features(old, name), database_features(new, name)
            xy0, xy1 = a["keypoints"][0], b["keypoints"][0]
            d0, d1 = a["descriptors"][0], b["descriptors"][0]
            if not len(xy0) or not len(xy1):
                raise ValueError("smoke control has no features")
            distance, index = cKDTree(xy0).query(xy1)
            error = np.max(np.abs(d1 - d0[index]), axis=1)
            records.append({
                "name": name, "before_count": len(xy0), "after_count": len(xy1),
                "max_coordinate_distance_px": float(distance.max()),
                "max_descriptor_component_error": float(error.max()),
                "numerical_subset_passed": bool(np.all(distance < 0.01) and np.all(error < 1e-4)),
            })
    return records


def extraction(project, binary, directory, images, image_list, feature_id, use_gpu):
    feature = resolve_colmap_feature_profile(feature_id, project)
    command = [
        str(binary), "feature_extractor", "--database_path", str(directory / "database.db"),
        "--image_path", str(images), "--image_list_path", str(image_list),
        *resolve_colmap_camera_calibration("shared_opencv_v1").image_reader_options,
        *feature.extraction_options,
        "--FeatureExtraction.use_gpu", str(int(use_gpu)), "--FeatureExtraction.gpu_index", "0",
        "--FeatureExtraction.num_threads", "4", "--default_random_seed", "0",
    ]
    return run_stage(directory, "extraction", command)


def smoke(project, root, binaries, selected, images):
    directory = root / "smoke"
    directory.mkdir()
    names = [Path(selected[i]["path"]).name for i in SMOKE_FRAMES]
    image_list = directory / "images.txt"
    image_list.write_text("\n".join(names) + "\n")
    result = {"profile": "aliked_positive_smoke_v1", "records": [], "passed": True}
    for gpu in (False, True):
        outputs = {}
        for arm in ("original", "positive"):
            cell = directory / f"{'cuda' if gpu else 'cpu'}_{arm}"
            cell.mkdir()
            extraction(project, binaries[arm], cell, images, image_list, "aliked_n16rot_v1", gpu)
            outputs[arm] = cell
        records = compare_databases(outputs["original"] / "database.db", outputs["positive"] / "database.db", names)
        counts = database_summary(outputs["positive"] / "database.db")["feature_counts"]
        counters = [list(map(int, match)) for match in re.findall(
            r"positive_scores_v1: candidates=(\d+) nonpositive=(\d+) retained=(\d+)",
            (outputs["positive"] / "extraction.log").read_text(),
        )]
        passed = (len(counters) == len(names)
                  and sorted(x[2] for x in counters) == sorted(counts.values())
                  and any(x[1] > 0 for x in counters)
                  and all(x["numerical_subset_passed"] for x in records)
                  and all(x["after_count"] <= x["before_count"] for x in records))
        result["records"].append({"backend": "cuda" if gpu else "cpu", "comparisons": records,
                                  "score_counters": counters, "passed": passed})
        result["passed"] &= passed
    write_json(directory / "results.json", result)
    if not result["passed"]:
        raise ValueError("extraction contract smoke failed; do not run geometry")


def map_and_evaluate(root, cell, db, binary, images, timestamps, record):
    sparse = cell / "sparse"
    sparse.mkdir()
    run_stage(cell, "mapping", [binary, "mapper", "--database_path", str(db),
              "--image_path", str(images), "--output_path", str(sparse),
              "--Mapper.num_threads", "4", "--Mapper.ba_global_function_tolerance", "0.000001",
              "--default_random_seed", "0"])
    for model in sorted(sparse.iterdir()):
        if not (model / "images.bin").is_file():
            continue
        text = cell / f"model_{model.name}_txt"
        text.mkdir()
        run_stage(cell, f"convert_{model.name}", [binary, "model_converter", "--input_path", str(model),
                  "--output_path", str(text), "--output_type", "TXT"])
        health = build_sfm_pose_health_from_text(
            model_dir=text, selected_timestamps=timestamps, database_path=db
        )
        write_json(text / "pose_health.json", health)
        registered = [
            image.name for image in parse_colmap_images_with_points(text / "images.txt")
        ]
        timeline = analyze_registration_timeline(timestamps, registered)
        write_json(text / "registration.json", timeline)
        point_count = sum(
            1 for line in (text / "points3D.txt").read_text().splitlines()
            if line and not line.startswith("#")
        )
        record["models"].append({
            "path": str(text.relative_to(root)), "registered_count": len(registered),
            "point_count": point_count, "pose_status": health["status"],
            "reason_codes": health["reason_codes"], "timeline": timeline,
        })
    record["models"].sort(
        key=lambda value: (-value["registered_count"], -value["point_count"], value["path"])
    )
    primary = record["models"][0] if record["models"] else None
    record["primary"] = primary
    record["product_gate_passed"] = bool(
        primary and primary["registered_count"] >= MIN_VIDEO_REGISTERED_COUNT
        and primary["timeline"]["registration_rate"] >= MIN_VIDEO_REGISTRATION_RATE
        and primary["timeline"]["temporal_coverage"] >= MIN_VIDEO_TEMPORAL_COVERAGE
    )
    record["acceptance_passed"] = bool(
        record["product_gate_passed"] and primary["pose_status"] == "passed"
    )
    record["status"] = "evaluated" if primary else "no_model"


def geometry(project, root, binaries, selected, images):
    if not json.loads((root / "smoke/results.json").read_text())["passed"]:
        raise ValueError("geometry requires a passing extraction smoke")
    directory = root / "geometry"
    directory.mkdir()
    results = []
    for clip, (first, last) in CLIPS.items():
        clip_dir = directory / clip
        clip_dir.mkdir()
        names = [Path(selected[i]["path"]).name for i in range(first, last + 1)]
        timestamps = {Path(selected[i]["path"]).name: selected[i]["time_seconds"] for i in range(first, last + 1)}
        image_list = clip_dir / "images.txt"
        image_list.write_text("\n".join(names) + "\n")
        write_json(clip_dir / "selection.json", {"scope": "clip_only_not_full_video", "selected": [selected[i] for i in range(first, last + 1)]})
        for arm, feature_id, matcher_id in (
            ("original", "aliked_n16rot_v1", "lightglue"),
            ("positive", "aliked_n16rot_v1", "lightglue"),
            ("sift_control", "sift_v1", "bruteforce"),
        ):
            cell = clip_dir / arm
            cell.mkdir()
            record = {"clip": clip, "arm": arm, "feature_profile": feature_id,
                      "matcher": matcher_id, "status": "failed", "models": []}
            try:
                extraction(project, binaries.get(arm, binaries["original"]), cell, images, image_list, feature_id, True)
                # Only extraction changes; both arms use the original matcher and Mapper binary.
                binary = str(binaries["original"])
                feature = resolve_colmap_feature_profile(feature_id, project)
                matcher = resolve_colmap_local_matcher(feature, matcher_id, project)
                db = cell / "database.db"
                run_stage(cell, "matching", [binary, "exhaustive_matcher", "--database_path", str(db),
                          *matcher.matching_options, *resolve_colmap_geometric_verification("default_v1").matching_options,
                          "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                          "--FeatureMatching.num_threads", "4", "--default_random_seed", "0"])
                record["database"] = database_summary(db)
                if set(record["database"]["feature_counts"]) != set(names):
                    raise ValueError("extracted image set mismatch")
                map_and_evaluate(root, cell, db, binary, images, timestamps, record)
            except (RuntimeError, ValueError) as exc:
                record["error"] = str(exc)
            write_json(cell / "results.json", record)
            results.append(record)
            print("CELL", json.dumps(record), flush=True)
    write_json(directory / "results.json", {"profile": "aliked_positive_geometry_v1", "cells": results,
                                            "recovery_applied": False, "training_started": False, "test_rgb_loaded": False})


def copy_feature_database(source, destination):
    if destination.exists():
        raise ValueError(f"refusing to overwrite database: {destination}")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as old, sqlite3.connect(
        destination
    ) as new:
        old.backup(new)
        new.execute("DELETE FROM matches")
        new.execute("DELETE FROM two_view_geometries")


def brute_force_geometry(project, root, binaries, selected, images):
    clip = "weak_696_743"
    first, last = CLIPS[clip]
    timestamps = {
        Path(selected[index]["path"]).name: selected[index]["time_seconds"]
        for index in range(first, last + 1)
    }
    names = sorted(timestamps)
    directory = root / "bruteforce"
    directory.mkdir()
    binary = str(binaries["original"])
    feature = resolve_colmap_feature_profile("aliked_n16rot_v1", project)
    matcher = resolve_colmap_local_matcher(feature, "bruteforce", project)
    for arm in ("original", "positive"):
        cell = directory / arm
        cell.mkdir()
        source_db = root / "geometry" / clip / arm / "database.db"
        db = cell / "database.db"
        record = {
            "clip": clip, "arm": arm, "feature_profile": "aliked_n16rot_v1",
            "matcher": "bruteforce", "status": "failed", "models": [],
            "feature_database_source": str(source_db.relative_to(root)),
            "feature_database_source_sha256": sha256_file(source_db),
        }
        try:
            copy_feature_database(source_db, db)
            run_stage(cell, "matching", [
                binary, "exhaustive_matcher", "--database_path", str(db),
                *matcher.matching_options,
                *resolve_colmap_geometric_verification("default_v1").matching_options,
                "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                "--FeatureMatching.num_threads", "4", "--default_random_seed", "0",
            ])
            record["database"] = database_summary(db)
            if set(record["database"]["feature_counts"]) != set(names):
                raise ValueError("copied image set mismatch")
            map_and_evaluate(root, cell, db, binary, images, timestamps, record)
        except (RuntimeError, ValueError) as exc:
            record["error"] = str(exc)
        write_json(cell / "results.json", record)
        print("BRUTEFORCE_CELL", json.dumps(record), flush=True)
    write_bruteforce_summary(root)


def write_bruteforce_summary(root):
    directory = root / "bruteforce"
    results = [
        json.loads((directory / arm / "results.json").read_text())
        for arm in ("original", "positive")
    ]
    comparison = None
    if all(record["status"] == "evaluated" for record in results):
        by_arm = {record["arm"]: record for record in results}
        names = sorted(by_arm["original"]["database"]["feature_counts"])
        comparison = {
            "matches": compare_match_databases(
                directory / "original/database.db", directory / "positive/database.db", names
            ),
            "camera_centers": align_camera_centers(
                root / by_arm["original"]["primary"]["path"],
                root / by_arm["positive"]["primary"]["path"],
            ),
        }
    write_json(directory / "results.json", {
        "profile": "aliked_positive_bruteforce_v1", "cells": results,
        "comparison": comparison, "recovery_applied": False,
        "training_started": False, "test_rgb_loaded": False,
    })


def expanded_bruteforce_geometry(project, root, binaries, selected, images):
    clip, (first, last) = next(iter(EXPANDED_CLIP.items()))
    timestamps = {
        Path(selected[index]["path"]).name: selected[index]["time_seconds"]
        for index in range(first, last + 1)
    }
    names = sorted(timestamps)
    directory = root / "expanded-bruteforce"
    directory.mkdir()
    image_list = directory / "images.txt"
    image_list.write_text("\n".join(names) + "\n")
    write_json(directory / "selection.json", {
        "scope": "expanded_clip_only_not_full_video",
        "selected": [selected[index] for index in range(first, last + 1)],
    })
    binary = str(binaries["original"])
    feature = resolve_colmap_feature_profile("aliked_n16rot_v1", project)
    matcher = resolve_colmap_local_matcher(feature, "bruteforce", project)
    results = []
    for arm in ("original", "positive"):
        cell = directory / arm
        cell.mkdir()
        record = {
            "clip": clip, "arm": arm, "feature_profile": "aliked_n16rot_v1",
            "matcher": "bruteforce", "status": "failed", "models": [],
        }
        try:
            extraction(
                project, binaries[arm], cell, images, image_list,
                "aliked_n16rot_v1", True,
            )
            db = cell / "database.db"
            run_stage(cell, "matching", [
                binary, "exhaustive_matcher", "--database_path", str(db),
                *matcher.matching_options,
                *resolve_colmap_geometric_verification("default_v1").matching_options,
                "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                "--FeatureMatching.num_threads", "4", "--default_random_seed", "0",
            ])
            record["database"] = database_summary(db)
            if set(record["database"]["feature_counts"]) != set(names):
                raise ValueError("expanded image set mismatch")
            map_and_evaluate(root, cell, db, binary, images, timestamps, record)
        except (RuntimeError, ValueError) as exc:
            record["error"] = str(exc)
        write_json(cell / "results.json", record)
        results.append(record)
        print("EXPANDED_BRUTEFORCE_CELL", json.dumps(record), flush=True)
    by_arm = {record["arm"]: record for record in results}
    comparison = None
    if all("database" in record for record in results):
        comparison = {
            "matches": compare_match_databases(
                directory / "original/database.db", directory / "positive/database.db", names
            ),
            "camera_centers": None,
        }
        if all(record["status"] == "evaluated" for record in results):
            try:
                comparison["camera_centers"] = align_camera_centers(
                    root / by_arm["original"]["primary"]["path"],
                    root / by_arm["positive"]["primary"]["path"],
                )
            except ValueError as exc:
                comparison["camera_center_error"] = str(exc)
    write_json(directory / "results.json", {
        "profile": "aliked_positive_expanded_bruteforce_v1", "cells": results,
        "comparison": comparison, "stage_timeout_seconds": 1200,
        "recovery_applied": False, "training_started": False, "test_rgb_loaded": False,
    })


def select_dense_candidates(selection, start, end):
    return [
        item for item in selection["candidates"]
        if start <= item["time_seconds"] <= end and not item.get("rejection_reason")
    ]


def dense_candidate_bruteforce(project, root, binaries, selected, source):
    clip, (first, last) = next(iter(EXPANDED_CLIP.items()))
    source_selection = json.loads((source / "selection.json").read_text())
    video = project / SOURCE_VIDEO
    if sha256_file(video) != source_selection["source_sha256"]:
        raise ValueError("source video hash mismatch")
    start = selected[first]["time_seconds"]
    end = selected[last]["time_seconds"]
    candidates = select_dense_candidates(source_selection, start, end)
    if len(candidates) != DENSE_CANDIDATE_COUNT:
        raise ValueError("dense viable candidate count changed")
    directory = root / "dense-bruteforce"
    directory.mkdir()
    images = directory / "images"
    started = time.monotonic()
    paths = materialize_video_candidates(
        video,
        images,
        candidates,
        {
            "profile": V2_PROFILE_ID,
            "selected": source_selection["selected"],
            "rotation": source_selection["rotation"],
        },
    )
    materialization = {
        "elapsed_seconds": time.monotonic() - started,
        "candidate_count": len(paths), "source_video_sha256": source_selection["source_sha256"],
        "source_selection_profile": source_selection["profile"],
        "selection_policy": "all_nonrejected_6fps_candidates_in_expanded_time_range_v1",
    }
    timestamps = {
        path.name: float(candidate["time_seconds"])
        for path, candidate in zip(paths, candidates)
    }
    image_list = directory / "images.txt"
    image_list.write_text("\n".join(timestamps) + "\n")
    write_json(directory / "selection.json", {
        **materialization, "scope": "dense_clip_only_not_full_video",
        "start_time_seconds": start, "end_time_seconds": end,
        "selected": [
            {
                "candidate_index": int(candidate["candidate_index"]),
                "pts": int(candidate["pts"]), "time_seconds": float(candidate["time_seconds"]),
                "path": path.name, "sha256": sha256_file(path),
            }
            for path, candidate in zip(paths, candidates)
        ],
    })
    cell = directory / "positive"
    cell.mkdir()
    record = {
        "clip": clip, "arm": "positive_dense_6fps", "feature_profile": "aliked_n16rot_v1",
        "matcher": "bruteforce", "status": "failed", "models": [],
        "materialization": materialization,
    }
    try:
        extraction(
            project, binaries["positive"], cell, images, image_list,
            "aliked_n16rot_v1", True,
        )
        db = cell / "database.db"
        binary = str(binaries["original"])
        feature = resolve_colmap_feature_profile("aliked_n16rot_v1", project)
        matcher = resolve_colmap_local_matcher(feature, "bruteforce", project)
        run_stage(cell, "matching", [
            binary, "exhaustive_matcher", "--database_path", str(db),
            *matcher.matching_options,
            *resolve_colmap_geometric_verification("default_v1").matching_options,
            "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
            "--FeatureMatching.num_threads", "4", "--default_random_seed", "0",
        ])
        record["database"] = database_summary(db)
        if set(record["database"]["feature_counts"]) != set(timestamps):
            raise ValueError("dense image set mismatch")
        map_and_evaluate(root, cell, db, binary, images, timestamps, record)
    except (RuntimeError, ValueError) as exc:
        record["error"] = str(exc)
    write_json(cell / "results.json", record)
    write_json(directory / "results.json", {
        "profile": "aliked_positive_dense_bruteforce_v1", "cell": record,
        "stage_timeout_seconds": 1200, "recovery_applied": False,
        "training_started": False, "test_rgb_loaded": False,
    })


def map_positive_feature_indices(original, positive):
    old_xy, old_desc = original["keypoints"][0], original["descriptors"][0]
    new_xy, new_desc = positive["keypoints"][0], positive["descriptors"][0]
    mapping = np.empty(len(new_xy), dtype=np.int64)
    cursor = 0
    max_coordinate_error = 0.0
    max_descriptor_error = 0.0
    for index, (xy, descriptor) in enumerate(zip(new_xy, new_desc)):
        while cursor < len(old_xy):
            coordinate_error = float(np.linalg.norm(xy - old_xy[cursor]))
            descriptor_error = float(np.max(np.abs(descriptor - old_desc[cursor])))
            if coordinate_error < 0.01 and descriptor_error < 1e-4:
                break
            cursor += 1
        if cursor == len(old_xy):
            raise ValueError("positive-score features are not an ordered subset")
        mapping[index] = cursor
        max_coordinate_error = max(max_coordinate_error, coordinate_error)
        max_descriptor_error = max(max_descriptor_error, descriptor_error)
        cursor += 1
    return mapping, {
        "original_count": len(old_xy), "positive_count": len(new_xy),
        "removed_count": len(old_xy) - len(new_xy),
        "max_coordinate_error_px": max_coordinate_error,
        "max_descriptor_component_error": max_descriptor_error,
    }


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "min": float(values.min()),
        "p50": float(np.median(values)), "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def compare_match_databases(original_path, positive_path, names):
    feature_records = []
    mappings = {}
    retained = {}
    with sqlite3.connect(original_path.as_uri() + "?mode=ro", uri=True) as old, sqlite3.connect(
        positive_path.as_uri() + "?mode=ro", uri=True
    ) as new:
        for name in names:
            mapping, record = map_positive_feature_indices(
                database_features(old, name), database_features(new, name)
            )
            mappings[name] = mapping
            retained[name] = set(mapping.tolist())
            feature_records.append({"name": name, **record})
        table_reports = {}
        for table in ("matches", "two_view_geometries"):
            totals = {
                "original_count": 0, "original_retained_count": 0,
                "positive_count": 0, "common_count": 0,
                "original_with_removed_endpoint_count": 0,
                "original_retained_only_count": 0, "positive_only_count": 0,
            }
            pair_records = []
            for left, right in combinations(names, 2):
                old_pairs = read_pair(old, left, right, table)
                new_pairs = read_pair(new, left, right, table)
                if old_pairs is None or new_pairs is None:
                    raise ValueError(f"missing exhaustive {table} pair")
                old_set = set(map(tuple, old_pairs.tolist()))
                old_retained = {
                    pair for pair in old_set
                    if pair[0] in retained[left] and pair[1] in retained[right]
                }
                mapped_new = {
                    (int(mappings[left][pair[0]]), int(mappings[right][pair[1]]))
                    for pair in new_pairs
                }
                common = old_retained & mapped_new
                pair_records.append({
                    "names": [left, right],
                    "retained_jaccard": compare_pairs(old_retained, mapped_new)["jaccard"],
                    "original_count": len(old_set),
                    "original_with_removed_endpoint_count": len(old_set) - len(old_retained),
                    "positive_count": len(mapped_new), "common_count": len(common),
                })
                totals["original_count"] += len(old_set)
                totals["original_retained_count"] += len(old_retained)
                totals["positive_count"] += len(mapped_new)
                totals["common_count"] += len(common)
                totals["original_with_removed_endpoint_count"] += len(old_set) - len(old_retained)
                totals["original_retained_only_count"] += len(old_retained - mapped_new)
                totals["positive_only_count"] += len(mapped_new - old_retained)
            union = (
                totals["original_retained_count"] + totals["positive_count"]
                - totals["common_count"]
            )
            totals["retained_jaccard"] = totals["common_count"] / union if union else 1.0
            jaccards = [record["retained_jaccard"] for record in pair_records]
            table_reports[table] = {
                **totals,
                "pair_retained_jaccard": summarize_values(jaccards),
                "lowest_pair_retained_jaccard": sorted(
                    pair_records, key=lambda value: (value["retained_jaccard"], value["names"])
                )[:10],
            }
    return {
        "features": {
            "original_count": sum(record["original_count"] for record in feature_records),
            "positive_count": sum(record["positive_count"] for record in feature_records),
            "removed_count": sum(record["removed_count"] for record in feature_records),
            "records": feature_records,
        },
        "tables": table_reports,
    }


def align_camera_centers(original_path, positive_path):
    originals = {
        image.name: colmap_camera_center(image)
        for image in parse_colmap_images_with_points(original_path / "images.txt")
    }
    positives = {
        image.name: colmap_camera_center(image)
        for image in parse_colmap_images_with_points(positive_path / "images.txt")
    }
    original_names = set(originals)
    positive_names = set(positives)
    names = sorted(original_names & positive_names)
    if len(names) < 3:
        raise ValueError("camera alignment requires at least three common images")
    reference = np.stack([originals[name] for name in names])
    candidate = np.stack([positives[name] for name in names])
    reference_centered = reference - reference.mean(axis=0)
    candidate_centered = candidate - candidate.mean(axis=0)
    left, singular, right = np.linalg.svd(candidate_centered.T @ reference_centered)
    correction = np.ones(3)
    correction[-1] = np.sign(np.linalg.det(left @ right))
    rotation = (left * correction) @ right
    scale = float(np.sum(singular * correction) / np.sum(candidate_centered ** 2))
    aligned = scale * candidate_centered @ rotation + reference.mean(axis=0)
    residuals = np.linalg.norm(aligned - reference, axis=1)
    radius = float(np.median(np.linalg.norm(reference - np.median(reference, axis=0), axis=1)))
    if radius <= 1e-12:
        raise ValueError("original model has degenerate camera extent")
    worst = int(np.argmax(residuals))
    return {
        "original_camera_count": len(original_names),
        "positive_camera_count": len(positive_names),
        "common_camera_count": len(names),
        "original_only_names": sorted(original_names - positive_names),
        "positive_only_names": sorted(positive_names - original_names),
        "similarity_scale": scale,
        "reflection_used": bool(correction[-1] < 0), "world_units": "arbitrary",
        "residual_world": summarize_values(residuals),
        "residual_to_original_median_radius": summarize_values(residuals / radius),
        "maximum_residual_name": names[worst],
    }


def compare_geometry(root):
    output = root / "comparison.json"
    geometry_result = root / "geometry/results.json"
    cells = json.loads(geometry_result.read_text())["cells"]
    report = {
        "profile": "aliked_positive_comparison_v1",
        "geometry_results_sha256": sha256_file(geometry_result),
        "clips": {}, "training_started": False, "test_rgb_loaded": False,
    }
    for clip in CLIPS:
        arms = {cell["arm"]: cell for cell in cells if cell["clip"] == clip}
        original = arms["original"]
        positive = arms["positive"]
        if original["status"] != "evaluated" or positive["status"] != "evaluated":
            raise ValueError("comparison requires evaluated ALIKED arms")
        names = sorted(original["database"]["feature_counts"])
        clip_root = root / "geometry" / clip
        report["clips"][clip] = {
            "matches": compare_match_databases(
                clip_root / "original/database.db", clip_root / "positive/database.db", names
            ),
            "camera_centers": align_camera_centers(
                root / original["primary"]["path"], root / positive["primary"]["path"]
            ),
        }
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke", "geometry", "compare", "bruteforce", "bruteforce-compare",
            "expanded-bruteforce", "dense-bruteforce",
        ),
        required=True,
    )
    args = parser.parse_args()
    project, root = args.project.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = project / SOURCE
    if sha256_file(source / "selection.json") != SOURCE_SELECTION_SHA:
        raise ValueError("frozen source selection changed")
    selected = {int(Path(item["path"]).name.split("_")[1]): item
                for item in json.loads((source / "selection.json").read_text())["selected"]}
    used = (
        set(SMOKE_FRAMES)
        | {i for first, last in CLIPS.values() for i in range(first, last + 1)}
        | {i for first, last in EXPANDED_CLIP.values() for i in range(first, last + 1)}
    )
    for i in used:
        if sha256_file(source / "input" / Path(selected[i]["path"]).name) != selected[i]["sha256"]:
            raise ValueError(f"image hash mismatch at frame {i}")
    binaries = {"original": project / "external/colmap-4-cuda/install/bin/colmap",
                "positive": project / CANDIDATE_ROOT / "install/bin/colmap"}
    build = json.loads((project / CANDIDATE_ROOT / "build-record.json").read_text())
    if (sha256_file(binaries["original"]) != build["baseline_binary_sha256"]
            or sha256_file(binaries["positive"]) != build["binary_sha256"]):
        raise ValueError("binary hash mismatch")
    if subprocess.check_output(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("project tracked files must be clean")
    if args.stage in {
        "geometry", "compare", "bruteforce", "bruteforce-compare",
        "expanded-bruteforce", "dense-bruteforce",
    }:
        previous = json.loads((root / "smoke-request.json").read_text())
        if previous["build"] != build or previous["source_selection_sha256"] != SOURCE_SELECTION_SHA:
            raise ValueError("geometry and extraction smoke provenance mismatch")
    write_json(root / f"{args.stage}-request.json", {
        "project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
        "source_selection_sha256": SOURCE_SELECTION_SHA, "build": build,
        "clips": CLIPS, "expanded_clip": EXPANDED_CLIP,
        "smoke_frames": SMOKE_FRAMES, "gpu": "0", "production_default_changed": False,
    })
    if args.stage == "smoke":
        smoke(project, root, binaries, selected, source / "input")
    elif args.stage == "geometry":
        geometry(project, root, binaries, selected, source / "input")
    elif args.stage == "compare":
        compare_geometry(root)
    elif args.stage == "bruteforce":
        brute_force_geometry(project, root, binaries, selected, source / "input")
    elif args.stage == "bruteforce-compare":
        write_bruteforce_summary(root)
    elif args.stage == "expanded-bruteforce":
        expanded_bruteforce_geometry(
            project, root, binaries, selected, source / "input"
        )
    else:
        dense_candidate_bruteforce(project, root, binaries, selected, source)


if __name__ == "__main__":
    main()
