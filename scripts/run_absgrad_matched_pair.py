#!/usr/bin/env python3
"""One new same-code signed/absolute pair after the checkpoint implementation changed."""
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
    STREAMING_HOST_POLICY, STREAMING_UNIT, QUALITY_LIMITS, available_host_bytes, quality_cgroup, run_stage, write_json,
)
from image3d_scenegraph.gpu_lease import FileLease
from importlib import import_module

candidate = import_module(".run_absgrad_candidate", __package__) if __package__ else import_module("run_absgrad_candidate")

PROFILE = "absgrad_streaming_matched_pair_v1"
SAVE_EVIDENCE = {
    "outputs/analysis/checkpoint-memory-stream-20261009-v1/summary.json": "18bec896cfc0462114c4a954271dd53c73fd55ccf5c235775377df278660e30a",
    "outputs/analysis/checkpoint-streaming-sh3-20261009-v1/candidate/summary.json": "5bae52aea4f5d228d20628d2c6cbb26287edaa74c234998cdce3226544f84652",
    "outputs/analysis/checkpoint-streaming-sh3-20261009-v1/candidate/trainer/attempts/train-001/checkpoints/iteration_000000012/checkpoint.json": "456972ffebbb536d55b193b64d179410ba2a25d09feb8f63fe3ace32a94240cd",
}


def matched_gate_template() -> dict:
    gate = candidate.gate_template(quality_exploration=True)
    gate.update(schema_version=3, profile=PROFILE, comparison_identity="fresh signed then absolute; identical current code/environment",
                old_signed_role="historical reference only; not the new matched control",
                host_policy=copy.deepcopy(STREAMING_HOST_POLICY), systemd_unit=STREAMING_UNIT,
                authorized_fresh_arms=["signed", "absolute"], automatic_retry_or_resume=False,
                save_verification=copy.deepcopy(SAVE_EVIDENCE))
    return gate


def validate_matched_gate(gate: dict) -> None:
    expected = matched_gate_template()
    expected.update(status="APPROVED_FOR_CANDIDATE_EXECUTION", absolute_training_authorized=True)
    if json.dumps(gate, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("new matched pair requires its separately approved exact gate")


def receipt(signed: Path) -> dict:
    paths = ["protocol.json", "signed.config.json", "complete.json", "signed/train-only/record.json",
             "signed/train-only/model.pt", "signed/train-only/evaluation.json", "signed/selection/evaluation.json"]
    return {name: sha256_file(signed / name) for name in paths}


def execute(output: Path, old_signed: Path, gate_path: Path, gate_sha: str, *, expected_revision: str) -> None:
    if output.exists() or output.is_symlink() or output.absolute() != output.resolve() or not output.resolve().is_relative_to(candidate.PROJECT_ROOT / "outputs/experiments"):
        raise ValueError("matched pair requires a fresh non-symlink experiment directory")
    gate = candidate.checked_json(gate_path, gate_sha)
    validate_matched_gate(gate)
    group = quality_cgroup(policy=STREAMING_HOST_POLICY, unit=STREAMING_UNIT)
    if available_host_bytes() < STREAMING_HOST_POLICY["startup_available_bytes"]:
        raise ValueError("matched pair requires 18 GiB available host RAM")
    from gsplat import rendering
    from image3d_scenegraph.gaussian.trainer import training_provenance
    from image3d_scenegraph.gaussian.render import require_distributed_absgrad
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    from run_video_4k_comparison import revision, require_resources

    code = revision()
    if code != expected_revision:
        raise ValueError("execution Git revision differs from explicit launch identity")
    source_protocol = candidate.checked_json(old_signed / "protocol.json", candidate.SIGNED_PROTOCOL_SHA256)
    baseline = candidate.checked_json(old_signed / "signed.config.json", candidate.SIGNED_CONFIG_SHA256)
    old_record = candidate.checked_json(old_signed / "signed/train-only/record.json", candidate.SIGNED_FINAL_RECORD_SHA256)
    candidate.checked_json(candidate.PROJECT_ROOT / gate["quality_proposal"], candidate.QUALITY_SHA256)
    candidate.checked_json(candidate.PROJECT_ROOT / gate["train_roi"], candidate.ROI_SHA256)
    source = Path(source_protocol["source"])
    original = candidate.checked_json(source / "protocol.json", source_protocol["source_protocol_sha256"])
    for path, digest in original["files"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen source identity changed: {path}")
    require_distributed_absgrad()
    if sha256_file(Path(rendering.__file__)) != source_protocol["overlay_sha256"]:
        raise ValueError("isolated rendering changed")
    inventory = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    if len(inventory.strip().splitlines()) != 2 or inventory.count("NVIDIA L2") != 2:
        raise ValueError("matched pair requires the same two L2 GPUs")
    replay = Path(source_protocol["replay"])
    validate_replay_bundle(replay)
    records = {"signed": baseline, "absolute": candidate.absolute_config(baseline)}
    provenance = training_provenance(dataset_hash=source_protocol["dataset_hash"],
        effective_config_hash=baseline["effective_config_hash"], world_size=2)
    if provenance.environment_hash != old_record["environment_hash"]:
        raise ValueError("historical execution environment changed")
    evidence = [candidate.checked_json(candidate.PROJECT_ROOT / path, digest) for path, digest in SAVE_EVIDENCE.items()]
    if any(row["returncode"] != 0 for row in evidence[0]["cases"]) or evidence[1]["status"] != "passed":
        raise ValueError("save/GPU verification is incomplete")
    for key in ("code_hash", "environment_hash"):
        if evidence[2]["provenance"][key] != getattr(provenance, key):
            raise ValueError(f"save/GPU verification {key} differs from execution")
    def admission(root, *, minimum_free_gib):
        if revision() != code:
            raise ValueError("code changed between stages")
        require_resources(root, minimum_free_gib=minimum_free_gib)
    admission(output.parent, minimum_free_gib=20)
    output.mkdir()
    write_json(output / "gate.json", gate)
    write_json(output / "protocol.json", {"profile": PROFILE, "code": code, "gate_sha256": gate_sha,
        "code_hash": provenance.code_hash, "environment_hash": provenance.environment_hash,
        "historical_signed_protocol_sha256": candidate.SIGNED_PROTOCOL_SHA256, "original_gate_sha256": candidate.ORIGINAL_GATE_SHA256,
        "fresh_arms": ["signed", "absolute"], "promotion_eligible": False})
    with FileLease(candidate.PROJECT_ROOT / "outputs/.gpu.lock") as lease:
        for arm, dirname in (("signed", "signed-control"), ("absolute", "absolute-candidate")):
            root = output / dirname
            root.mkdir()
            write_json(root / f"{arm}.config.json", records[arm])
            protocol = {**copy.deepcopy(source_protocol), "profile": PROFILE, "code": code,
                "arm": arm, "gate_sha256": gate_sha, "code_hash": provenance.code_hash,
                "environment_hash": provenance.environment_hash, "original_resource_failure_retained": True,
                "effective_config_hash": records[arm]["effective_config_hash"], "resource_limits": QUALITY_LIMITS,
                "host_policy": STREAMING_HOST_POLICY, "host_cgroup": str(group), "gpu_inventory": inventory,
                "main_camera_sequence_sha256": candidate.MAIN_CAMERA_SHA256, "train_only_camera_sequence_sha256": candidate.TRAIN_ONLY_CAMERA_SHA256,
                "test_rgb": "not_loaded", "promotion_eligible": False}
            protocol.pop("absolute_arm", None)
            if arm == "absolute":
                signed = output / "signed-control"
                protocol["matched_signed"] = receipt(signed)
                current_record = candidate.read_json(signed / "signed/train-only/record.json")
                if current_record["code_hash"] != provenance.code_hash or current_record["environment_hash"] != provenance.environment_hash:
                    raise ValueError("new signed/candidate core or environment mismatch")
            write_json(root / "protocol.json", protocol)
            candidate.run_pipeline(root, replay, lease_fd=lease.fileno(), require_resources=admission,
                         limits=QUALITY_LIMITS, host_group=group, host_policy=STREAMING_HOST_POLICY, arm=arm)
        candidate_root = output / "absolute-candidate"
        admission(output, minimum_free_gib=8)
        run_stage(candidate_root / "absolute", "paired-quality", [sys.executable,
            "scripts/evaluate_absgrad_pair.py", "--signed-experiment", str(output / "signed-control"),
            "--candidate-experiment", str(candidate_root), "--output-dir", str(candidate_root / "paired-quality")],
            cwd=candidate.PROJECT_ROOT, pass_fds=(lease.fileno(),), host_group=group, host_policy=STREAMING_HOST_POLICY)
    write_json(output / "complete.json", {"status": "matched_pair_numerical_report_complete_visual_pending",
        "report_sha256": sha256_file(candidate_root / "paired-quality/report.json"), "promotion_eligible": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-template", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--historical-signed", type=Path)
    parser.add_argument("--gate-contract", type=Path)
    parser.add_argument("--gate-sha256")
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    if args.gate_template:
        write_json(args.gate_template, matched_gate_template())
        return
    if socket.gethostname() != "i-94B8D131" or candidate.PROJECT_ROOT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
        raise ValueError("matched execution is restricted to the authorized remote repository")
    if not all((args.output_dir, args.historical_signed, args.gate_contract, args.gate_sha256, args.expected_revision)):
        parser.error("all execution identity arguments are required")
    execute(args.output_dir, args.historical_signed, args.gate_contract, args.gate_sha256,
            expected_revision=args.expected_revision)


if __name__ == "__main__":
    main()
