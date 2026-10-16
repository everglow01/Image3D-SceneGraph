#!/usr/bin/env python3
"""Run the approved MCMC arms serially; retain failures and continue without retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from image3d_scenegraph.file_integrity import sha256_file, verify_file
from image3d_scenegraph.gaussian.checkpoint import _COMPONENT_FILES, _json_bytes
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu

import run_mcmc_stability_ablation as ablation


PROJECT = ablation.PROJECT
SUITE = "mcmc-stability-suite-20261016-v1"


def verify_arm(root: Path, arm: str, baseline_sequence: str | None) -> dict:
    protocol = json.loads((root / "protocol.json").read_text())
    output = root / arm
    complete = json.loads((output / "complete.json").read_text())
    if (complete["status"] != "complete" or complete["arm"] != arm
            or complete["protocol_sha256"] != sha256_file(root / "protocol.json")):
        raise ValueError("arm completion does not match the protocol")
    resources = complete["resources"]
    if (resources["optimizer_updates_observed"], resources["camera_samples_observed"],
            resources["max_observed_global_gaussians"]) != (30000, 60000, 3000000):
        raise ValueError("incomplete training budget")
    sequence = resources["camera_sequence_sha256"]
    if baseline_sequence is not None and sequence != baseline_sequence:
        raise ValueError("camera sequence differs from the suite baseline")
    training = output / "training/attempts/train-001"
    digest = hashlib.sha256()
    updates, validation_iterations = 0, []
    with (training / "artifacts/progress.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "validation":
                if event["status"] != "complete" or event["successful_views"] != 351:
                    raise ValueError("incomplete Validation")
                validation_iterations.append(event["iteration"])
            else:
                updates += 1
                if event["iteration"] != updates or event["world_size"] != 2:
                    raise ValueError("missing or reordered training updates")
                batch = event["batch_view_ids"]
                if len(batch) != 2:
                    raise ValueError("expected two cameras per update")
                digest.update(("\0".join(batch) + "\n").encode())
    if (updates != 30000 or digest.hexdigest() != sequence
            or validation_iterations != [3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000]):
        raise ValueError("training sequence or Validation schedule differs")
    elapsed = {}
    for stage in ("train", "sor", "selection"):
        record = json.loads((output / f"{stage}.exit.json").read_text())
        if record["returncode"] != 0 or record["resource_failure"] is not None:
            raise ValueError(f"{stage} did not complete")
        elapsed[stage] = record["elapsed_seconds"]
    checkpoint = training / "checkpoints/iteration_000030000"
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    config = json.loads((root / f"{arm}.config.json").read_text())
    if (metadata["iteration"] != 30000 or metadata["purpose"] != "final"
            or metadata["provenance"]["code_hash"] != protocol["training_code_hash"]
            or metadata["provenance"]["dataset_hash"] != protocol["dataset_hash"]
            or metadata["provenance"]["effective_config_hash"] != config["effective_config_hash"]
            or set(metadata["files"]) != set(_COMPONENT_FILES)):
        raise ValueError("checkpoint identity differs")
    for name, filename in _COMPONENT_FILES.items():
        entry = metadata["files"][name]
        if entry["path"] != filename or (checkpoint / filename).is_symlink():
            raise ValueError("unexpected checkpoint component path")
        verify_file(checkpoint / filename, expected_size=entry["bytes"],
                    expected_sha256=entry["sha256"], label=name)
    payload = {key: value for key, value in metadata.items() if key != "checkpoint_hash"}
    if hashlib.sha256(_json_bytes(payload)).hexdigest() != metadata["checkpoint_hash"]:
        raise ValueError("checkpoint metadata hash differs")
    if sha256_file(output / "sor/filtered-model.pt") != complete["selection_model_sha256"]:
        raise ValueError("selected model hash differs")
    evaluation_path = output / "selection/evaluation.json"
    if sha256_file(evaluation_path) != complete["selection_evaluation_sha256"]:
        raise ValueError("Selection evaluation hash differs")
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation["status"] != "complete" or evaluation["successful_views"] != 351:
        raise ValueError("incomplete Selection")
    return {"status": "verified", "arm": arm, "complete_sha256": sha256_file(output / "complete.json"),
            "checkpoint_hash": metadata["checkpoint_hash"], "resources": resources,
            "selected_iteration": complete["selected_iteration"], "stage_seconds": elapsed,
            "selection_raw_psnr": evaluation["psnr"]["mean"],
            "selection_display_psnr": evaluation["display_psnr"]["mean"],
            "selection_ssim": evaluation["ssim"]["mean"],
            "selection_gaussians": evaluation["gaussian_count"],
            "final_raw_psnr": complete["final_raw_psnr"]["mean"],
            "final_max_rgb": complete["final_max_rgb"],
            "final_negative_psnr_views": complete["final_negative_psnr_views"],
            "final_dead_fractions": [x["dead_fraction"] for x in complete["final_parameter_health_by_rank"]],
            "test_rgb": "not_loaded", "promotion_eligible": False}


def run_arm(root: Path, arm: str) -> int:
    subprocess.run(["nvidia-smi", "-L"], check=True)
    require_idle_gpu()
    unit = f"image3d-mcmc-{arm}-20261016-v3"
    command = ["systemd-run", "--wait", "--pipe", "--collect", f"--unit={unit}",
        f"--working-directory={PROJECT}", "-p", "MemoryMax=infinity",
        "-p", "MemoryHigh=infinity", "-p", "MemorySwapMax=infinity",
        "-p", "OOMPolicy=kill", "-p", "RuntimeMaxSec=23400",
        "/usr/bin/env", "-u", "PYTHONPATH", "OMP_NUM_THREADS=2", "PYTHONUNBUFFERED=1",
        sys.executable, str(PROJECT / "scripts/run_mcmc_stability_ablation.py"),
        "--output-root", str(root / "experiment"), "--arm", arm]
    with (root / f"{arm}.unit.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    ablation.write_json(root / f"{arm}.unit.exit.json", {"returncode": result.returncode})
    return result.returncode


def run_suite(root: Path) -> int:
    with FileLease(PROJECT / f"outputs/.{SUITE}.lock"):
        if root.exists():
            raise ValueError("suite output already exists; no overwrite, resume or retry")
        with (PROJECT / f"outputs/.{SUITE}.once").open("x"):
            pass
        root.mkdir(parents=True)
        started = time.monotonic()
        records = []
        returncode = 1
        ablation.write_json(root / "suite.run.json", {"pid": os.getpid(),
            "arms": list(ablation.ARMS), "profile": ablation.UNLIMITED_PROFILE,
            "script_sha256": sha256_file(Path(__file__)), "failure_policy": "record_and_continue"})
        try:
            experiment = root / "experiment"
            ablation.prepare(SimpleNamespace(output_root=experiment, unlimited_host_memory=True,
                source_job=PROJECT / "outputs/jobs/20261010_172759_d4dda4b3",
                gates_root=PROJECT / "outputs/analysis/mcmc-stability-gates-20261013-v3",
                floor_overlay=PROJECT / "outputs/analysis/mcmc-stability-gates-20261013-v1/overlay",
                moments_overlay=PROJECT / "outputs/analysis/mcmc-stability-gates-20261013-v2/moments-overlay",
                memory_gate=PROJECT / "outputs/analysis/mcmc-validation-memory-20261013-v1/summary.json"))
            baseline = None
            with (root / "suite.events.jsonl").open("x", buffering=1) as events:
                for arm in ablation.ARMS:
                    if arm != "baseline" and baseline is None:
                        record = {"arm": arm, "status": "skipped", "reason": "matched_baseline_unavailable"}
                    else:
                        events.write(json.dumps({"arm": arm, "status": "started"}) + "\n")
                        print(f"suite arm={arm} started", flush=True)
                        try:
                            code = run_arm(root, arm)
                            if code != 0:
                                raise RuntimeError(f"unit exited with code {code}")
                            record = verify_arm(experiment, arm,
                                None if baseline is None else baseline["resources"]["camera_sequence_sha256"])
                            if arm == "baseline":
                                baseline = record
                            record["raw_psnr_delta"] = record["selection_raw_psnr"] - baseline["selection_raw_psnr"]
                            record["display_psnr_delta"] = record["selection_display_psnr"] - baseline["selection_display_psnr"]
                        except Exception as exc:
                            record = {"arm": arm, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                            record["stage_failure_files"] = {
                                stage: str(path.relative_to(root))
                                for stage in ("train", "sor", "selection")
                                if (path := experiment / arm / f"{stage}.failure.json").is_file()
                            }
                    ablation.write_json(root / f"{arm}.review.json", record)
                    records.append(record)
                    events.write(json.dumps(record) + "\n")
                    print(f"suite arm={arm} status={record['status']}", flush=True)
            passed = all(record["status"] == "verified" for record in records)
            ablation.write_json(root / "suite.summary.json", {
                "status": "complete" if passed else "completed_with_failures", "arms": records,
                "protocol_sha256": sha256_file(experiment / "protocol.json"),
                "test_rgb": "not_loaded", "promotion_eligible": False})
            returncode = 0 if passed else 1
        except BaseException as exc:
            ablation.write_json(root / "suite.failure.json", {"status": "aborted",
                "reason": f"{type(exc).__name__}: {exc}", "reviewed_arms": [r["arm"] for r in records]})
            raise
        finally:
            ablation.write_json(root / "suite.exit.json", {"returncode": returncode,
                "elapsed_seconds": time.monotonic() - started, "reviewed_arms": [r["arm"] for r in records]})
        return returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or PROJECT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
        parser.error("this suite is authorized only in the specified GPU-host repository")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=PROJECT, check=True)
    raise SystemExit(run_suite(args.output_root.resolve()))


if __name__ == "__main__":
    main()
