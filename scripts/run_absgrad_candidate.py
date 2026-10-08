#!/usr/bin/env python3
"""Prepare or explicitly execute the frozen absolute arm; never promote defaults."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import (
    LIMITS, POLL_SECONDS, TELEMETRY_START_SECONDS, TELEMETRY_STALE_SECONDS,
    TrainingMonitor, run_stage, write_json,
)
from image3d_scenegraph.gaussian.config import (
    ResolvedGaussianConfig, assert_single_field_ablation, resolved_config_record,
)
from image3d_scenegraph.gpu_lease import FileLease


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNED_REVISION = "9cf78ad84abdc2da61f03512af09a3484b7e75bd"
SIGNED_PROTOCOL_SHA256 = "8e9171e51744171daec3e33059b6b79cbcee20625793ff76fb838b1d672245cb"
SIGNED_CONFIG_SHA256 = "626298d7deef592a38498a8c13cf85d79b8dbb3493c1786ae51e56ebaeaaaa68"
SIGNED_FINAL_RECORD_SHA256 = "b128f96bc972fd634d2e4bca11278b4a321ede02a0e1a4e4b2b4582ccc2c1865"
QUALITY_SHA256 = "315f8f41dce5cde23d01abe74fdbda4dc23e0d7760eec3c24259bab02e8a137e"
ROI_SHA256 = "fcdeedd25c456708825a8ec5b929c46f8f741ccaec44d6ca46c7833ac94b2b13"
MAIN_CAMERA_SHA256 = "8415b1bd17cf77dc0207938f1189b18b79992ff531aafa63d0a71c61d08d1a86"
TRAIN_ONLY_CAMERA_SHA256 = "b9a49e99d9b9ce3a248791b8fae6d20ebf76cd9361ce69765df383df02f0aab8"
ABSOLUTE_CONFIG_HASH = "865aa98a29f2f4271289a6c20ca0771a93e8019d787e58877f7cec75924b01cc"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_json(path: Path, expected: str) -> dict:
    if sha256_file(path) != expected:
        raise ValueError(f"frozen identity mismatch: {path}")
    return read_json(path)


def gate_template() -> dict:
    return {
        "schema_version": 1, "status": "DRAFT_NOT_APPROVED",
        "absolute_training_authorized": False,
        "quality_basis": "single_scene_engineering_effect_not_training_variance",
        "quality_proposal": "outputs/analysis/absgrad-stage2b-gate-proposal-20261008-v1/gate-proposal.json",
        "quality_proposal_sha256": QUALITY_SHA256,
        "quality_proposal_scope": "proposed_quality_gates; report both frozen endpoints",
        "train_roi": "outputs/analysis/absgrad-signed-train-roi-20261008-v1/train-roi.json",
        "train_roi_sha256": ROI_SHA256,
        "resource_limits": copy.deepcopy(LIMITS),
        "runtime_policy": {
            "startup_free_gib": 20, "stage_free_gib": 8, "cancel_below_free_gib": 4,
            "hard_stage_timeout_seconds": 21600, "cancel_grace_seconds": 120,
            "term_grace_seconds": 30, "poll_seconds": POLL_SECONDS,
            "telemetry_start_seconds": TELEMETRY_START_SECONDS,
            "telemetry_stale_seconds": TELEMETRY_STALE_SECONDS,
            "memory_scope": "main and Train-only, including final merge; not SOR/selection",
        },
        "test_authorized": False, "default_changes_authorized": False,
    }


def validate_gate(gate: dict) -> None:
    expected = gate_template()
    expected["status"] = "APPROVED_FOR_CANDIDATE_EXECUTION"
    expected["absolute_training_authorized"] = True
    if json.dumps(gate, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("candidate requires the explicitly approved frozen gate contract")


def absolute_config(signed: dict) -> dict:
    resolved_config_record(ResolvedGaussianConfig(
        requested_profile=signed["requested_profile"],
        effective_config=signed["effective_config"],
        effective_config_hash=signed["effective_config_hash"],
    ))
    config = copy.deepcopy(signed["effective_config"])
    config["densification"]["absgrad"] = True
    if assert_single_field_ablation(signed["effective_config"], config) != "densification.absgrad":
        raise ValueError("unexpected candidate ablation leaf")
    return resolved_config_record(ResolvedGaussianConfig(
        requested_profile="absgrad_ablation_v1", effective_config=config,
        effective_config_hash=ABSOLUTE_CONFIG_HASH,
    ))


def validate_evaluation(path: Path, *, final: bool) -> None:
    record = read_json(path)
    role = "held_out_after_train_only_control" if final else "held_out_model_selection"
    if (record.get("status") != "complete" or record.get("num_views") != 377
            or record.get("successful_views") != 377 or record.get("failed_views") != []
            or record.get("quality_role") != role
            or record.get("selection_eligible") is not (not final)):
        raise ValueError("incomplete Validation or incorrect evaluation role")


def preflight(signed: Path, gate_path: Path, gate_sha256: str, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; do not overwrite or resume")
    if output.absolute() != output.resolve() or not output.resolve().is_relative_to(PROJECT_ROOT / "outputs/experiments"):
        raise ValueError("candidate output must be a new non-symlink experiment directory")
    gate = checked_json(gate_path, gate_sha256)
    validate_gate(gate)
    checked_json(PROJECT_ROOT / gate["quality_proposal"], QUALITY_SHA256)
    checked_json(PROJECT_ROOT / gate["train_roi"], ROI_SHA256)
    protocol = checked_json(signed / "protocol.json", SIGNED_PROTOCOL_SHA256)
    baseline = checked_json(signed / "signed.config.json", SIGNED_CONFIG_SHA256)
    final_record = checked_json(signed / "signed/train-only/record.json", SIGNED_FINAL_RECORD_SHA256)
    complete = read_json(signed / "complete.json")
    if complete.get("status") != "signed_control_complete":
        raise ValueError("signed baseline is not complete")
    for relative, expected in (
        ("signed/selection/evaluation.json", final_record["selection_evaluation_sha256"]),
        ("signed/train-only/evaluation.json", final_record["evaluation_sha256"]),
        ("signed/train-only/model.pt", final_record["final_model_sha256"]),
    ):
        if sha256_file(signed / relative) != expected:
            raise ValueError(f"signed evidence changed: {relative}")
    source = Path(protocol["source"])
    original = checked_json(source / "protocol.json", protocol["source_protocol_sha256"])
    for path, expected in original["files"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source identity changed: {path}")
    candidate = absolute_config(baseline)
    return {"signed_protocol": protocol, "signed_final_record": final_record,
            "absolute_config": candidate, "gate": gate, "gate_sha256": gate_sha256}


def execute(root: Path, prepared: dict, *, lease_fd: int) -> None:
    from gsplat import rendering
    from image3d_scenegraph.gaussian.render import require_distributed_absgrad
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    from image3d_scenegraph.gaussian.trainer import training_provenance
    from run_video_4k_comparison import require_resources, revision

    code = revision()
    subprocess.run(["git", "ls-files", "--error-unmatch", "scripts/run_absgrad_candidate.py",
                    "src/image3d_scenegraph/gaussian/absgrad_resources.py"],
                   cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--exit-code", SIGNED_REVISION, "--", "src",
                    "scripts/filter_gaussian_sor.py", "scripts/evaluate_gaussian.py",
                    ":!src/image3d_scenegraph/gaussian/absgrad_resources.py"],
                   cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
    require_distributed_absgrad()
    require_resources(root.parent, minimum_free_gib=20)
    gpu_inventory = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
    if len(gpu_inventory.strip().splitlines()) != 2 or gpu_inventory.count("NVIDIA L2") != 2:
        raise ValueError("candidate requires the same two NVIDIA L2 GPUs as signed")
    protocol = prepared["signed_protocol"]
    if sha256_file(Path(rendering.__file__)) != protocol["overlay_sha256"]:
        raise ValueError("isolated rendering identity changed")
    provenance = training_provenance(dataset_hash=protocol["dataset_hash"],
        effective_config_hash=ABSOLUTE_CONFIG_HASH, world_size=2)
    for field in ("code_hash", "environment_hash"):
        if getattr(provenance, field) != prepared["signed_final_record"][field]:
            raise ValueError(f"signed/candidate {field} mismatch")
    replay = Path(protocol["replay"])
    validate_replay_bundle(replay)
    root.mkdir()
    write_json(root / "absolute.config.json", prepared["absolute_config"])
    write_json(root / "gate.json", prepared["gate"])
    write_json(root / "protocol.json", {
        "profile": "absgrad_absolute_candidate_v1", "code": code,
        "signed_training_revision": SIGNED_REVISION,
        "signed_protocol_sha256": SIGNED_PROTOCOL_SHA256,
        "gate_sha256": prepared["gate_sha256"], "replay": str(replay),
        "dataset_hash": protocol["dataset_hash"], "overlay_sha256": protocol["overlay_sha256"],
        "effective_config_hash": ABSOLUTE_CONFIG_HASH,
        "main_camera_sequence_sha256": MAIN_CAMERA_SHA256,
        "train_only_camera_sequence_sha256": TRAIN_ONLY_CAMERA_SHA256,
        "code_hash": provenance.code_hash, "environment_hash": provenance.environment_hash,
        "resource_limits": LIMITS, "gpu_inventory": gpu_inventory,
        "quality_decision": "pending_paired_roi_review",
        "test_rgb": "not_loaded", "promotion_eligible": False,
        "telemetry": "opt-in CLI wrapper; 10s cumulative per-rank reserved peak; includes final model merge",
    })
    run_pipeline(root, replay, lease_fd=lease_fd, require_resources=require_resources)


def run_pipeline(root: Path, replay: Path, *, lease_fd: int, require_resources) -> None:
    output = root / "absolute"
    output.mkdir()
    common = ["--dataset-contract", str(replay / "dataset.json"), "--dataset-root", str(replay),
              "--resolved-config-json", str(root / "absolute.config.json")]

    def stage(name, arguments, monitor=None):
        require_resources(output, minimum_free_gib=8)
        run_stage(output, name, [sys.executable, *arguments], cwd=PROJECT_ROOT,
                  monitor=monitor, pass_fds=(lease_fd,))

    training = output / "training"
    progress = training / "attempts/train-001/artifacts/progress.jsonl"
    telemetry = output / "train-memory"
    main_monitor = TrainingMonitor(progress, telemetry, updates=30000, main_stage=True)
    stage("train", ["scripts/run_gaussian_training.py", *common, "--run-dir", str(training),
        "--trainer", "project", "--initialization", "frozen", "--distributed",
        "--no-intermediate-previews", "--cancel-file", str(output / "cancel"),
        "--resource-telemetry-dir", str(telemetry)],
        main_monitor)
    if main_monitor.record()["camera_sequence_sha256"] != MAIN_CAMERA_SHA256:
        raise ValueError("main camera sampling sequence differs from signed")
    result = read_json(training / "attempts/train-001/artifacts/result.json")
    if result["iteration"] != 30000 or result["world_size"] != 2:
        raise ValueError("main training budget mismatch")
    peaks = result["per_rank_peak_reserved_bytes"]
    if len(peaks) != 2 or any(p > cap for p, cap in zip(peaks, LIMITS["max_per_rank_peak_reserved_bytes"])):
        raise ValueError("final main memory record exceeds resource limit")
    sor = output / "sor"
    stage("sor", ["scripts/filter_gaussian_sor.py", "--model-snapshot", str(training / result["model_path"]),
        "--output-dir", str(sor), "--nb-neighbors", "30", "--std-ratio", "2.0", "--band-opacity", "0.05"])
    selection = output / "selection"
    stage("selection", ["scripts/evaluate_gaussian.py", *common, "--model", str(sor / "filtered-model.pt"),
        "--split", "validation", "--output-dir", str(selection), "--progress", str(progress)])
    validate_evaluation(selection / "evaluation.json", final=False)
    final = output / "train-only"
    telemetry = output / "train-only-memory"
    final_monitor = TrainingMonitor(final / "progress.jsonl", telemetry, updates=2000, main_stage=False)
    stage("train-only", ["scripts/run_gaussian_final_fit.py", *common,
        "--source-model", str(sor / "filtered-model.pt"),
        "--selection-evaluation", str(selection / "evaluation.json"), "--output-dir", str(final),
        "--train-only-control", "--distributed", "--cancel-file", str(output / "cancel"),
        "--resource-telemetry-dir", str(telemetry)],
        final_monitor)
    if final_monitor.record()["camera_sequence_sha256"] != TRAIN_ONLY_CAMERA_SHA256:
        raise ValueError("Train-only camera sampling sequence differs from signed")
    record = read_json(final / "record.json")
    if (record["optimizer_updates"], record["camera_samples"], record["world_size"], record["input_splits"],
            record["topology_changed"], record["source_model_unchanged"]) != (2000, 4000, 2, ["train"], False, True):
        raise ValueError("Train-only budget or topology mismatch")
    validate_evaluation(final / "evaluation.json", final=True)
    write_json(root / "complete.json", {
        "status": "absolute_training_complete_quality_pending",
        "selection_evaluation_sha256": sha256_file(selection / "evaluation.json"),
        "final_evaluation_sha256": sha256_file(final / "evaluation.json"),
        "final_model_sha256": sha256_file(final / "model.pt"),
        "quality_decision": "pending_all_frozen_rois_and_visual_review",
        "test_rgb": "not_loaded", "promotion_eligible": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    template = commands.add_parser("gate-template")
    template.add_argument("--output", type=Path, required=True)
    for name in ("preflight", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--signed-experiment", type=Path, required=True)
        command.add_argument("--gate-contract", type=Path, required=True)
        command.add_argument("--gate-sha256", required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "gate-template":
        write_json(args.output, gate_template())
        return
    if args.action == "execute":
        if socket.gethostname() != "i-94B8D131" or PROJECT_ROOT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
            raise ValueError("candidate execution is restricted to the authorized remote workspace")
        with FileLease(PROJECT_ROOT / "outputs/.gpu.lock") as lease:
            prepared = preflight(args.signed_experiment, args.gate_contract, args.gate_sha256, args.output_dir)
            execute(args.output_dir.resolve(), prepared, lease_fd=lease.fileno())
    else:
        preflight(args.signed_experiment, args.gate_contract, args.gate_sha256, args.output_dir)
        print("preflight_passed; no training or output directory created")


if __name__ == "__main__":
    main()
