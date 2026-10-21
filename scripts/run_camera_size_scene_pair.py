#!/usr/bin/env python3
"""Frozen apartment original/target-camera-size comparison, 12k updates on a 30k schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import TrainingMonitor, run_stage, write_json
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from scripts.prepare_gsplat_camera_sizes import file_hashes, verify_overlay
from scripts.run_camera_size_lifecycle import OVERLAY, OVERLAY_SHA, verify_pair
from scripts.run_mcmc_stability_ablation import PROJECT, LIMITS, configs, code_hash, unlimited_host_cgroup

SOURCE = PROJECT / "outputs/jobs/20261010_172759_d4dda4b3"
GATES = PROJECT / "outputs/analysis/camera-size-lifecycle-20261021-v1"
ARMS = ("baseline", "repair")
UPDATES = 12000
PROFILE = "apartment_camera_size_pair_12k_v1"


def verify_validation(evaluation, protocol):
    views = evaluation["per_view"]
    if (evaluation["successful_views"] != 351 or evaluation["failed_views"]
            or sorted(v["image_id"] for v in views) != protocol["validation_ids"]):
        raise ValueError("Validation view identity or completeness differs")
    if any(not math.isfinite(v[k]) for v in views for k in ("psnr", "ssim", "display_psnr", "display_ssim")):
        raise ValueError("nonfinite Validation metric")
    if any(v["rgb_range"]["nonfinite_count"] for v in views):
        raise ValueError("nonfinite Validation RGB")
    health = evaluation["mcmc_health_by_rank"]
    if len(health) != 2 or any(row[k] for row in health for k in (
            "activated_scale_nonfinite_count", "log_scale_nan_count", "log_scale_negative_inf_count",
            "log_scale_positive_inf_count", "opacity_nonfinite_count", "origin_distance_nonfinite_count")):
        raise ValueError("nonfinite or missing rank health")


def verify_terminal(training, result, protocol, arm):
    from image3d_scenegraph.gaussian.checkpoint import _COMPONENT_FILES, _json_bytes, load_attempt

    if (result["iteration"], result["nominal_iterations"], result["requested_stop_iteration"],
            result["world_size"], result["gaussian_cap"], result["termination_reason"]) != (
            UPDATES, 30000, UPDATES, 2, 3000000, "execution_budget"):
        raise ValueError("actual/nominal training budget differs")
    expected_path = "attempts/train-001/checkpoints/iteration_000012000"
    if result["final_checkpoint_path"] != expected_path:
        raise ValueError("wrong terminal checkpoint path")
    path = training / expected_path
    metadata = json.loads((path / "checkpoint.json").read_text())
    expected_provenance = {"dataset_hash": protocol["dataset_hash"], "effective_config_hash": protocol["config_hash"],
                          "code_hash": protocol["core"], "environment_hash": protocol["environments"][arm]["hash"]}
    if metadata["provenance"] != expected_provenance or metadata["iteration"] != UPDATES or metadata["purpose"] != "final":
        raise ValueError("terminal checkpoint identity differs")
    attempt = load_attempt(training, "train-001")
    if attempt.provenance.__dict__ != expected_provenance:
        raise ValueError("attempt identity differs")
    payload = {k: v for k, v in metadata.items() if k != "checkpoint_hash"}
    digest = hashlib.sha256(_json_bytes(payload)).hexdigest()
    if digest != metadata["checkpoint_hash"] or digest != result["final_checkpoint_hash"]:
        raise ValueError("checkpoint metadata hash mismatch")
    if set(metadata["files"]) != set(_COMPONENT_FILES):
        raise ValueError("checkpoint components differ")
    for key, name in _COMPONENT_FILES.items():
        entry, file = metadata["files"][key], path / name
        if (entry["path"] != name or file.is_symlink() or not file.is_file()
                or file.stat().st_size != entry["bytes"] or sha256_file(file) != entry["sha256"]):
            raise ValueError(f"checkpoint component mismatch: {key}")
    scheduler = json.loads((path / "scheduler.bin").read_text())
    if scheduler != {"iteration": UPDATES, "nominal_iterations": 30000, "requested_stop_iteration": UPDATES}:
        raise ValueError("checkpoint scheduler budget differs")
    model = (training / result["model_path"]).resolve()
    if not model.is_relative_to(training.resolve()) or not model.is_file():
        raise ValueError("candidate model missing or outside attempt")
    verify_validation(result["final_validation"], protocol)
    validation_steps = []
    with (training / "attempts/train-001/artifacts/progress.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "validation":
                verify_validation(event, protocol)
                validation_steps.append(event["iteration"])
    if validation_steps != protocol["validation_steps"]:
        raise ValueError("Validation schedule differs")
    if any(event[k] != v for k, v in result["final_validation"].items()):
        raise ValueError("terminal Validation differs from final progress event")
    return {"checkpoint_sha256": digest, "model_sha256": sha256_file(model),
            "result_sha256": sha256_file(training / "attempts/train-001/artifacts/result.json")}


def prepare(args):
    import gsplat
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle

    gate_path = GATES / "summary.json"
    gates = json.loads(gate_path.read_text())
    verify_pair(gates["arms"])
    core = code_hash()
    if gates["status"] != "passed" or not gates["cross_environment_rejected"] or any(
        row["training_core"] != core for row in gates["arms"].values()
    ):
        raise ValueError("same-core native lifecycle gates required")
    if gates["overlay_sha256"] != OVERLAY_SHA or sha256_file(OVERLAY / "overlay.json") != OVERLAY_SHA:
        raise ValueError("lifecycle overlay changed")
    verify_overlay(OVERLAY, OVERLAY / "gsplat")
    original_config = SOURCE / "gaussian/preparation/train-001/effective_config.json"
    config = configs()["baseline"]
    original = json.loads(original_config.read_text())
    if any(original[k] != config[k] for k in ("requested_profile", "effective_config_hash", "effective_config")):
        raise ValueError("source MCMC configuration changed")
    replay = SOURCE / "gaussian/replay"
    replay_record = validate_replay_bundle(replay)
    dataset = json.loads((replay / "dataset.json").read_text())
    if {k: len(v) for k, v in dataset["splits"].items()} != {"train": 2808, "validation": 351, "test": 351}:
        raise ValueError("source split changed")
    if shutil.disk_usage(PROJECT).free < 20*1024**3:
        raise ValueError("requires 20 GiB free disk")
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / "config.json", config)
    bound = [gate_path, GATES / "protocol.json", original_config, replay / "dataset.json", replay / "replay.json",
             replay / replay_record["camera_path"], replay / replay_record["initialization_path"],
             replay / replay_record["initialization_diagnostics_path"], OVERLAY / "overlay.json",
             Path(__file__).resolve(), PROJECT / "scripts/run_gaussian_training.py", args.output_root / "config.json"]
    package = Path(gsplat.__file__).resolve().parent
    write_json(args.output_root / "protocol.json", {"profile": PROFILE, "core": core,
        "replay": str(replay), "dataset_hash": dataset["dataset_hash"], "config_hash": config["effective_config_hash"],
        "updates": UPDATES, "nominal_iterations": 30000, "camera_samples": 24000,
        "validation_steps": [3000, 5000, 7000, 10000, UPDATES], "world_size": 2,
        "validation_ids": sorted(dataset["splits"]["validation"]),
        "environments": {arm: {"name": gates["arms"][arm]["environment"], "hash": gates["arms"][arm]["environment_hash"]}
                         for arm in ARMS}, "overlay": str(OVERLAY), "overlay_sha256": OVERLAY_SHA,
        "files": {str(p): sha256_file(p) for p in bound}, "source_package": str(package), "source_hashes": file_hashes(package),
        "limits": LIMITS, "host_limits": None, "arms": list(ARMS), "source_job": str(SOURCE),
        "evaluation": "351-view endpoint raw/display plus same-step rank health; best candidate is supplementary",
        "test_rgb": "not_decoded_or_evaluated; existing replay integrity checks retained",
        "sor_final_fit_publication": "not_run", "promotion_eligible": False})


def run(args):
    protocol = json.loads((args.output_root / "protocol.json").read_text())
    if (protocol["profile"] != PROFILE or protocol["core"] != code_hash() or protocol["updates"] != UPDATES
            or protocol["limits"] != LIMITS or protocol["host_limits"] is not None):
        raise ValueError("frozen protocol differs")
    for name, digest in protocol["files"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"frozen source changed: {name}")
    if file_hashes(Path(protocol["source_package"])) != protocol["source_hashes"]:
        raise ValueError("installed package changed")
    verify_overlay(OVERLAY, OVERLAY / "gsplat")
    unlimited_host_cgroup(f"image3d-camera-size-apartment-{args.arm}-20261021-v1.service")
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        if shutil.disk_usage(PROJECT).free < 20*1024**3:
            raise ValueError("requires 20 GiB free disk")
        output = args.output_root / args.arm
        output.mkdir(exist_ok=False)
        training = output / "training"
        prefix = ["/usr/bin/env", "-u", "PYTHONPATH"] if args.arm == "baseline" else ["/usr/bin/env", "PYTHONPATH=" + str(OVERLAY)]
        command = prefix + [sys.executable, "scripts/run_gaussian_training.py", "--dataset-contract", protocol["replay"] + "/dataset.json",
            "--dataset-root", protocol["replay"], "--resolved-config-json", str(args.output_root / "config.json"),
            "--run-dir", str(training), "--trainer", "mcmc", "--initialization", "frozen", "--distributed",
            "--stop-after-iteration", str(UPDATES), "--no-intermediate-previews",
            "--resource-telemetry-dir", str(output / "memory"), "--cancel-file", str(output / "cancel")]
        monitor = TrainingMonitor(training / "attempts/train-001/artifacts/progress.jsonl", output / "memory",
                                  updates=UPDATES, main_stage=True, limits=LIMITS)
        run_stage(output, "train", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
        result = json.loads((training / "attempts/train-001/artifacts/result.json").read_text())
        hashes = verify_terminal(training, result, protocol, args.arm)
        if file_hashes(Path(protocol["source_package"])) != protocol["source_hashes"]:
            raise ValueError("installed package changed during training")
        verify_overlay(OVERLAY, OVERLAY / "gsplat")
        final = result["final_validation"]
        write_json(output / "complete.json", {"status": "complete", "arm": args.arm,
            "protocol_sha256": sha256_file(args.output_root / "protocol.json"), **hashes, "resources": monitor.record(),
            "selected_iteration": result["candidate_iteration"], "terminal_iteration": result["iteration"],
            "final_raw_psnr": final["psnr"], "final_raw_ssim": final["ssim"],
            "final_display_psnr": final["display_psnr"], "final_display_ssim": final["display_ssim"],
            "final_negative_psnr_views": sum(v["psnr"] < 0 for v in final["per_view"]),
            "final_parameter_health_by_rank": final["mcmc_health_by_rank"], "per_view": final["per_view"],
            "test_rgb": "not_loaded", "promotion_eligible": False})


def summarize(root):
    records = {arm: json.loads((root / arm / "complete.json").read_text()) for arm in ARMS if (root / arm / "complete.json").exists()}
    paired = len(records) == 2
    if paired:
        paired = (records["baseline"]["protocol_sha256"] == records["repair"]["protocol_sha256"]
                  and records["baseline"]["resources"]["camera_sequence_sha256"] == records["repair"]["resources"]["camera_sequence_sha256"])
    write_json(root / "summary.json", {"status": "complete" if paired else "incomplete_or_unpaired",
        "arms": records, "paired": paired, "test_rgb": "not_loaded", "promotion_eligible": False,
        "raw_psnr_delta": records["repair"]["final_raw_psnr"]["mean"] - records["baseline"]["final_raw_psnr"]["mean"] if paired else None})
    if not paired:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--arm", choices=ARMS)
    mode.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅在已授权远端执行12k对照，不运行本机模型")
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(PROJECT / "outputs/analysis"):
        raise ValueError("independent research output required")
    if args.prepare:
        prepare(args)
    elif args.summarize:
        summarize(args.output_root)
    else:
        run(args)


if __name__ == "__main__":
    main()
