#!/usr/bin/env python3
"""Run the frozen signed control; the absolute arm remains separately gated."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.config import (
    assert_single_field_ablation,
    resolve_internal_config,
    resolved_config_record,
)
from image3d_scenegraph.gaussian.render import require_distributed_absgrad
from image3d_scenegraph.gaussian.replay import validate_replay_bundle
from run_video_4k_comparison import read_json, require_resources, revision, write_json


SOURCE_PROTOCOL_SHA256 = "c3061cb25911e2faa96e87783ad2b3569a9d5c4a3084830889f05ae6a5e4af69"
SOURCE_CONFIG_HASH = "0792df888f842a0932b5ae45c008916b5ae7ad374582c143779fed0ac599e738"
MINIMUM_RUNNING_FREE_BYTES = 4 * 1024**3
STAGE_TIMEOUT_SECONDS = 6 * 3600


def paired_configs(original: dict) -> dict:
    records = {}
    for arm, absolute in (("signed", False), ("absolute", True)):
        record = resolved_config_record(resolve_internal_config("absgrad_ablation_v1", {
            "resolution": {"longest_edge": 1920},
            "opacity_reset": {"recovery_prune": {"enabled": True}},
            "densification": {"absgrad": absolute},
        }))
        restored = copy.deepcopy(record["effective_config"])
        restored["schema_version"] = 10
        restored["densification"].pop("absgrad")
        if restored != original["effective_config"] or original["effective_config_hash"] != SOURCE_CONFIG_HASH:
            raise ValueError("source Project configuration differs from the frozen baseline")
        records[arm] = record
    if assert_single_field_ablation(
        records["signed"]["effective_config"], records["absolute"]["effective_config"]
    ) != "densification.absgrad":
        raise ValueError("unexpected ablation leaf")
    return records


def stop_reason(free_bytes: int, elapsed: float) -> str | None:
    if free_bytes < MINIMUM_RUNNING_FREE_BYTES:
        return "free_disk_below_4_gib"
    if elapsed > STAGE_TIMEOUT_SECONDS:
        return "stage_exceeded_6_hours"
    return None


def run_stage(root: Path, name: str, arguments: list[str]) -> None:
    require_resources(root, minimum_free_gib=8)
    command = [sys.executable, *arguments]
    write_json(root / f"{name}.command.json", {"argv": command})
    print(f"stage={name} started", flush=True)
    started = time.monotonic()
    failure = None
    with (root / f"{name}.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    failure = stop_reason(shutil.disk_usage(root).free, time.monotonic() - started)
                    if failure:
                        (root / "cancel").touch(exist_ok=False)
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        break
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(root / f"{name}.exit.json", {
                "returncode": process.returncode, "elapsed_seconds": time.monotonic() - started,
                "resource_failure": failure,
            })
    if failure or process.returncode:
        raise RuntimeError(f"{name} failed: {failure or process.returncode}; retained {root}")
    print(f"stage={name} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sh3-smoke", type=Path, required=True)
    args = parser.parse_args()
    root, source = args.output_dir.resolve(), args.source_experiment.resolve()
    if root.exists():
        raise ValueError("output already exists; inspect it instead of retrying")
    code = revision()
    require_distributed_absgrad()
    require_resources(root.parent, minimum_free_gib=20)
    smoke = read_json(args.sh3_smoke)
    if smoke.get("status") != "passed" or smoke.get("sh_degree") != 3 or smoke.get("world_size") != 2:
        raise ValueError("a passing two-GPU SH3 smoke is required")
    if sha256_file(source / "protocol.json") != SOURCE_PROTOCOL_SHA256:
        raise ValueError("source protocol changed")
    protocol = read_json(source / "protocol.json")
    for path, expected in protocol["files"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source identity changed: {path}")
    replay = Path(protocol["replay"])
    validate_replay_bundle(replay)
    records = paired_configs(read_json(source / "project.config.json"))
    root.mkdir()
    for arm, record in records.items():
        write_json(root / f"{arm}.config.json", record)
    from gsplat import rendering

    write_json(root / "protocol.json", {
        "profile": "absgrad_signed_control_v1", "status": "frozen_before_training", "code": code,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256, "source": str(source),
        "dataset_hash": protocol["dataset_hash"], "replay": str(replay),
        "config_hashes": {arm: record["effective_config_hash"] for arm, record in records.items()},
        "overlay_sha256": sha256_file(Path(rendering.__file__)),
        "sh3_smoke_sha256": sha256_file(args.sh3_smoke),
        "seed": 20260729, "world_size": 2, "main_updates": 30000, "main_camera_samples": 60000,
        "train_only_updates": 2000, "train_only_camera_samples": 4000,
        "minimum_running_free_bytes": MINIMUM_RUNNING_FREE_BYTES,
        "stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
        "absolute_arm": "not_started_pending_quality_gate_freeze",
        "quality_decision": "not_run", "test_rgb": "not_loaded", "promotion_eligible": False,
        "test_bytes": "existing replay validation hashes all images; no Test RGB tensor or metric is loaded",
    })
    output = root / "signed"
    output.mkdir()
    common = ["--dataset-contract", str(replay / "dataset.json"), "--dataset-root", str(replay),
              "--resolved-config-json", str(root / "signed.config.json")]
    training = output / "training"
    run_stage(output, "train", [
        "scripts/run_gaussian_training.py", *common, "--run-dir", str(training),
        "--trainer", "project", "--initialization", "frozen", "--distributed",
        "--no-intermediate-previews", "--cancel-file", str(output / "cancel"),
    ])
    result = read_json(training / "attempts/train-001/artifacts/result.json")
    if result["iteration"] != 30000 or result["world_size"] != 2:
        raise ValueError("main training budget mismatch")
    sor = output / "sor"
    run_stage(output, "sor", [
        "scripts/filter_gaussian_sor.py", "--model-snapshot", str(training / result["model_path"]),
        "--output-dir", str(sor), "--nb-neighbors", "30", "--std-ratio", "2.0", "--band-opacity", "0.05",
    ])
    selection = output / "selection"
    run_stage(output, "selection", [
        "scripts/evaluate_gaussian.py", *common, "--model", str(sor / "filtered-model.pt"),
        "--split", "validation", "--output-dir", str(selection),
        "--progress", str(training / result["progress_path"]),
    ])
    final = output / "train-only"
    run_stage(output, "train-only", [
        "scripts/run_gaussian_final_fit.py", *common, "--source-model", str(sor / "filtered-model.pt"),
        "--selection-evaluation", str(selection / "evaluation.json"), "--output-dir", str(final),
        "--train-only-control", "--distributed", "--cancel-file", str(output / "cancel"),
    ])
    record = read_json(final / "record.json")
    if (record["optimizer_updates"], record["camera_samples"], record["input_splits"], record["topology_changed"]) != (2000, 4000, ["train"], False):
        raise ValueError("Train-only budget mismatch")
    write_json(root / "complete.json", {
        "status": "signed_control_complete", "absolute_arm": "not_started_pending_quality_gate_freeze",
        "selection_evaluation_sha256": sha256_file(selection / "evaluation.json"),
        "final_evaluation_sha256": sha256_file(final / "evaluation.json"),
        "final_model_sha256": sha256_file(final / "model.pt"),
        "export": "not_run; retain native model and evaluations without redundant PLY/ZIP copies",
        "test_rgb": "not_loaded", "promotion_eligible": False,
    })
    print("signed_control_complete", flush=True)


if __name__ == "__main__":
    main()
