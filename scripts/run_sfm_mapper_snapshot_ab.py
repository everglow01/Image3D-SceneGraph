#!/usr/bin/env python3
"""Same retained SQLite snapshot, two primary solvers; no RGB or reconstruction recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.geometry.colmap import build_global_mapper_commands
from image3d_scenegraph.geometry.grouping import parse_colmap_images_with_points
from image3d_scenegraph.geometry.reprojection import pixel_reprojection_errors
from image3d_scenegraph.geometry.sfm_pose_health import (
    build_sfm_pose_health_from_text,
    selected_timestamps_from_payload,
)
from image3d_scenegraph.geometry.video_recovery import v2_mapper_options
from image3d_scenegraph.video.keyframes import V2_PROFILE_ID
from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT,
    MIN_VIDEO_REGISTRATION_RATE,
    MIN_VIDEO_TEMPORAL_COVERAGE,
)


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def freeze_databases(source: Path, root: Path) -> dict:
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(source) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("source database must be settled without WAL/journal")
    before = sha256_file(source)
    snapshot = root / "snapshot.db"
    if snapshot.exists():
        raise FileExistsError(snapshot)
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(snapshot) as dst:
            src.backup(dst)
    digest = sha256_file(snapshot)
    inodes = {snapshot.stat().st_ino, source.stat().st_ino}
    for arm in ("incremental", "global"):
        directory = root / arm
        directory.mkdir()
        path = directory / "database.db"
        shutil.copyfile(snapshot, path)
        if path.stat().st_ino in inodes or sha256_file(path) != digest:
            raise ValueError("arm databases must be independent identical copies")
        inodes.add(path.stat().st_ino)
    if sha256_file(source) != before:
        raise ValueError("source database changed during snapshot")
    return {"source_sha256": before, "snapshot_sha256": digest}


def run_stage(root: Path, label: str, command: list[str], stages: list, timeout: int) -> None:
    log = root / f"{label}.log"
    record = {"stage": label, "command": command, "log": log.relative_to(root).as_posix(),
              "status": "running", "timeout_seconds": timeout}
    stages.append(record)
    write_json(root / "stages.json", {"stages": stages})
    start = time.monotonic()
    try:
        with log.open("x") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        record.update(exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed")
        result.check_returncode()
    except subprocess.TimeoutExpired:
        record.update(status="timeout", exit_code=None)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        if record["status"] == "running":
            record["status"] = "failed"
        write_json(root / "stages.json", {"stages": stages})


def arm_commands(colmap: Path, root: Path, arm: str, gpu_index: str) -> list[list[str]]:
    directory = root / arm
    common = ["--database_path", str(directory / "database.db"),
              "--image_path", str(root / "no-rgb"), "--output_path", str(directory / "sparse")]
    if arm == "incremental":
        return [[str(colmap), "mapper", *common, "--default_random_seed", "0",
                 "--Mapper.num_threads", "8", "--Mapper.extract_colors", "0",
                 "--Mapper.ba_use_gpu", "0", "--Mapper.ba_global_function_tolerance", "0.000001",
                 "--Mapper.image_list_path", str(root / "seed.txt"),
                 *v2_mapper_options({"profile": V2_PROFILE_ID})]]
    return list(build_global_mapper_commands(
        colmap, database_path=directory / "database.db", image_dir=root / "no-rgb",
        output_dir=directory / "sparse", use_gpu=True, gpu_index=gpu_index,
        num_threads=8, image_list_path=root / "seed.txt",
    ))


def evaluate_candidate(text: Path, database: Path, timestamps: dict, image_names: dict) -> dict:
    images = parse_colmap_images_with_points(text / "images.txt")
    if any(image.name not in timestamps or image_names.get(image.image_id) != image.name for image in images):
        raise ValueError("model contains non-seed image or database identity mismatch")
    health = build_sfm_pose_health_from_text(
        model_dir=text, selected_timestamps=timestamps, database_path=database,
    )
    timeline = health.get("temporal", {}).get("registration_timeline")
    reasons = list(health["reason_codes"])
    if health["status"] != "passed" and not reasons:
        reasons.append("pose_health_failed")
    if len(images) < MIN_VIDEO_REGISTERED_COUNT:
        reasons.append("registered_count_below_gate")
    if timeline is None:
        reasons.append("video_registration_timeline_missing")
    else:
        if timeline["registration_rate"] < MIN_VIDEO_REGISTRATION_RATE:
            reasons.append("registration_rate_below_gate")
        if timeline["temporal_coverage"] < MIN_VIDEO_TEMPORAL_COVERAGE:
            reasons.append("temporal_coverage_below_gate")
    record = {"accepted": False, "gate_reason_codes": reasons, "health": health,
              "registration_timeline": timeline, "registered_count": len(images)}
    try:
        _, pixels = pixel_reprojection_errors(text)
        record.update(accepted=not reasons, point_count=pixels["point_count"], pixel_reprojection=pixels)
    except ValueError as exc:
        record["evaluation_error"] = f"pixel_reprojection_failed: {exc}"
    return record


def run_arm(colmap: Path, root: Path, arm: str, gpu_index: str, timestamps: dict,
            image_names: dict, timeout: int) -> dict:
    directory = root / arm
    database = directory / "database.db"
    record = {"status": "running", "database_before_sha256": sha256_file(database),
              "candidates": [], "selected": None, "stages": []}
    try:
        sparse = directory / "sparse"
        sparse.mkdir()
        for command in arm_commands(colmap, root, arm, gpu_index):
            run_stage(directory, command[1], command, record["stages"], timeout)
        record["solver_elapsed_seconds"] = sum(stage["elapsed_seconds"] for stage in record["stages"])
        models = [sparse] if (sparse / "images.bin").is_file() else sorted(
            path for path in sparse.iterdir() if path.is_dir() and (path / "images.bin").is_file()
        )
        if not models:
            raise ValueError("solver produced no models")
        for index, model in enumerate(models):
            text = directory / f"text-{index}"
            text.mkdir()
            candidate = {"model_path": model.relative_to(root).as_posix(), "accepted": False,
                         "model_files_sha256": {p.name: sha256_file(p) for p in sorted(model.glob("*.bin"))}}
            record["candidates"].append(candidate)
            try:
                run_stage(directory, f"convert-{index}", [str(colmap), "model_converter", "--input_path",
                          str(model), "--output_path", str(text), "--output_type", "TXT"], record["stages"], timeout)
                candidate.update(evaluate_candidate(text, database, timestamps, image_names))
                candidate["text_files_sha256"] = {p.name: sha256_file(p) for p in sorted(text.glob("*.txt"))}
            except Exception as exc:
                candidate.update(evaluation_error=f"{type(exc).__name__}: {exc}", accepted=False)
        healthy = [item for item in record["candidates"] if item["accepted"]]
        if healthy:
            record["selected"] = max(healthy, key=lambda item: (item["registered_count"], item["point_count"]))
        record["status"] = "evaluation_failed" if any("evaluation_error" in c for c in record["candidates"]) else "completed"
    except Exception as exc:
        record.update(status="execution_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        record["database_after_sha256"] = sha256_file(database)
        write_json(directory / "result.json", record)
    return record


def compare_arms(arms: dict) -> dict:
    baseline, candidate = (arms[arm].get("selected") for arm in ("incremental", "global"))
    if any(arm["status"] != "completed" for arm in arms.values()) or not baseline or not candidate:
        return {"status": "not_comparable", "reason": "both_arms_need_successful_evaluation_and_healthy_selection"}
    checks = {
        "registration_retention_ge_95pct": candidate["registered_count"] >= 0.95 * baseline["registered_count"],
        "coverage_not_lower": candidate["registration_timeline"]["temporal_coverage"] >= baseline["registration_timeline"]["temporal_coverage"],
        "point_retention_ge_90pct": candidate["point_count"] >= 0.9 * baseline["point_count"],
        "median_pixel_error_le_1_2x": candidate["pixel_reprojection"]["median_reprojection_error_pixels"] <= 1.2 * baseline["pixel_reprojection"]["median_reprojection_error_pixels"],
    }
    return {"status": "passed" if all(checks.values()) else "failed_gates", "checks": checks,
            "solver_time_ratio": arms["global"]["solver_elapsed_seconds"] / arms["incremental"]["solver_elapsed_seconds"]}


def run_experiment(workspace: Path, output: Path, colmap: Path, gpu_index: str, timeout: int) -> int:
    workspace, output, colmap = workspace.resolve(), output.absolute(), colmap.resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError("experiment output or temporary directory already exists")
    sources = {"database": workspace / "colmap/database.db",
               "seed": workspace / "colmap/v2-mapper-seed.txt",
               "selection": workspace / "frames/selection.json"}
    source_hashes = {key: sha256_file(path) for key, path in sources.items()}
    seeds = sources["seed"].read_text().splitlines()
    if len(seeds) != 1000 or len(set(seeds)) != 1000:
        raise ValueError("expected the existing unique 1000-image seed list")
    timestamps = selected_timestamps_from_payload(json.loads(sources["selection"].read_text()))
    timestamps = {name: timestamps[name] for name in seeds}
    temporary.mkdir(parents=True)
    record = {"schema_version": 1, "profile": "retained_snapshot_mapper_ab_v2",
              "source_workspace": str(workspace), "source_files_sha256": source_hashes,
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "colmap_binary_sha256": sha256_file(colmap), "colmap_path": str(colmap),
              "seed_count": len(seeds), "arms": {}, "test_rgb": "not_loaded",
              "scope": "retained feature/match snapshot; not full-pipeline or equal-solver-method timing",
              "color_policy": "Incremental extract_colors=0; Global uses empty existing directory, missing RGB warnings expected",
              "incremental_ba": "CPU, unchanged standard_v2 baseline", "global_gpu_index": gpu_index}
    try:
        (temporary / "no-rgb").mkdir()
        shutil.copyfile(sources["seed"], temporary / "seed.txt")
        shutil.copyfile(sources["selection"], temporary / "selection.json")
        write_json(temporary / "seed-timestamps.json", timestamps)
        record["database_snapshot"] = freeze_databases(sources["database"], temporary)
        with sqlite3.connect((temporary / "snapshot.db").as_uri() + "?mode=ro", uri=True) as db:
            image_names = dict(db.execute("SELECT image_id, name FROM images"))
        if not set(seeds).issubset(image_names.values()) or any(
            Path(name).name != name or name in {".", "..", ""} for name in image_names.values()
        ):
            raise ValueError("database image paths must be basenames and contain all seeds")
        write_json(temporary / "comparison.json", record)
        for arm in ("incremental", "global"):
            record["arms"][arm] = run_arm(colmap, temporary, arm, gpu_index, timestamps, image_names, timeout)
            write_json(temporary / "comparison.json", record)
        record["comparison"] = compare_arms(record["arms"])
        record["execution_status"] = "completed" if all(a["status"] == "completed" for a in record["arms"].values()) else "failed"
    except Exception as exc:
        record.update(execution_status="failed", error=f"{type(exc).__name__}: {exc}")
    record["source_files_after_sha256"] = {key: sha256_file(path) for key, path in sources.items()}
    record["source_unchanged"] = record["source_files_after_sha256"] == source_hashes
    snapshot = temporary / "snapshot.db"
    record["snapshot_after_sha256"] = sha256_file(snapshot) if snapshot.exists() else None
    record["snapshot_unchanged"] = record["snapshot_after_sha256"] == record.get("database_snapshot", {}).get("snapshot_sha256") and snapshot.exists()
    if not record["source_unchanged"] or not record["snapshot_unchanged"]:
        record["execution_status"] = "failed"
        record["comparison"] = {"status": "not_comparable", "reason": "source_or_snapshot_integrity_failed"}
    record["publication_status"] = "published_evidence"
    write_json(temporary / "comparison.json", record)
    if output.exists():
        raise FileExistsError(output)
    temporary.rename(output)
    print(json.dumps({"output": str(output), "execution_status": record["execution_status"],
                      "comparison": record.get("comparison")}, indent=2))
    if record["execution_status"] != "completed":
        return 1
    return 0 if record["comparison"]["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not args.gpu_index.isdigit():
        parser.error("timeout must be positive and gpu-index must name one device")
    return run_experiment(args.source_workspace, args.output_dir, args.colmap, args.gpu_index, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
