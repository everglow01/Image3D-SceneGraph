#!/usr/bin/env python3
"""Frozen small-clip ALIKED score-filter experiment; no recovery, training or downloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import time

import numpy as np

from build_colmap_aliked_score_filter import CANDIDATE_ROOT
from image3d_scenegraph.geometry.colmap import (
    resolve_colmap_camera_calibration,
    resolve_colmap_feature_profile,
    resolve_colmap_geometric_verification,
    resolve_colmap_local_matcher,
    sha256_file,
)
from image3d_scenegraph.geometry.grouping import parse_colmap_images_with_points
from image3d_scenegraph.geometry.sfm_pose_health import build_sfm_pose_health_from_text
from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT, MIN_VIDEO_REGISTRATION_RATE, MIN_VIDEO_TEMPORAL_COVERAGE,
    analyze_registration_timeline,
)
from run_sfm_reference_audit import database_features


SOURCE = "outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2/aliked-lightglue"
SOURCE_SELECTION_SHA = "9c877d158c3b87051f09ca72d04c86a50cf18add59969d53483dba1a10ba8151"
CLIPS = {"weak_696_743": (696, 743), "healthy_448_495": (448, 495)}
SMOKE_FRAMES = (468, 469, 710, 713)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_stage(directory, stage, command, timeout=1200):
    write_json(directory / f"{stage}.request.json", {"command": command, "timeout_seconds": timeout})
    start = time.monotonic()
    peak, timed_out = None, False
    with (directory / f"{stage}.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                memory = subprocess.run(
                    ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if memory.returncode == 0 and memory.stdout.strip().isdigit():
                    peak = max(peak or 0, int(memory.stdout.strip()))
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
        "gpu_total_memory_peak_sampled_mib": peak, "gpu_sample_interval_seconds": 2,
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
                    health = build_sfm_pose_health_from_text(model_dir=text, selected_timestamps=timestamps, database_path=db)
                    write_json(text / "pose_health.json", health)
                    registered = [image.name for image in parse_colmap_images_with_points(text / "images.txt")]
                    timeline = analyze_registration_timeline(timestamps, registered)
                    write_json(text / "registration.json", timeline)
                    point_count = sum(1 for line in (text / "points3D.txt").read_text().splitlines() if line and not line.startswith("#"))
                    record["models"].append({"path": str(text.relative_to(root)), "registered_count": len(registered),
                                             "point_count": point_count, "pose_status": health["status"],
                                             "reason_codes": health["reason_codes"], "timeline": timeline})
                record["models"].sort(key=lambda x: (-x["registered_count"], -x["point_count"], x["path"]))
                primary = record["models"][0] if record["models"] else None
                record["primary"] = primary
                record["clip_product_passed"] = bool(primary and primary["pose_status"] == "passed"
                    and primary["registered_count"] >= MIN_VIDEO_REGISTERED_COUNT
                    and primary["timeline"]["registration_rate"] >= MIN_VIDEO_REGISTRATION_RATE
                    and primary["timeline"]["temporal_coverage"] >= MIN_VIDEO_TEMPORAL_COVERAGE)
                record["status"] = "evaluated" if primary else "no_model"
            except (RuntimeError, ValueError) as exc:
                record["error"] = str(exc)
            write_json(cell / "results.json", record)
            results.append(record)
            print("CELL", json.dumps(record), flush=True)
    write_json(directory / "results.json", {"profile": "aliked_positive_geometry_v1", "cells": results,
                                            "recovery_applied": False, "training_started": False, "test_rgb_loaded": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "geometry"), required=True)
    args = parser.parse_args()
    project, root = args.project.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = project / SOURCE
    if sha256_file(source / "selection.json") != SOURCE_SELECTION_SHA:
        raise ValueError("frozen source selection changed")
    selected = {int(Path(item["path"]).name.split("_")[1]): item
                for item in json.loads((source / "selection.json").read_text())["selected"]}
    used = set(SMOKE_FRAMES) | {i for first, last in CLIPS.values() for i in range(first, last + 1)}
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
    if args.stage == "geometry":
        previous = json.loads((root / "smoke-request.json").read_text())
        if previous["build"] != build or previous["source_selection_sha256"] != SOURCE_SELECTION_SHA:
            raise ValueError("geometry and extraction smoke provenance mismatch")
    write_json(root / f"{args.stage}-request.json", {
        "project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
        "source_selection_sha256": SOURCE_SELECTION_SHA, "build": build, "clips": CLIPS,
        "smoke_frames": SMOKE_FRAMES, "gpu": "0", "production_default_changed": False,
    })
    (smoke if args.stage == "smoke" else geometry)(project, root, binaries, selected, source / "input")


if __name__ == "__main__":
    main()
