#!/usr/bin/env python3
"""Frozen AbsGrad identities and pipelines shared by thin CLI entrypoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import (
    LIMITS, POLL_SECONDS, TELEMETRY_START_SECONDS, TELEMETRY_STALE_SECONDS,
    QUALITY_LIMITS, QUALITY_HOST_POLICY, QUALITY_UNIT, available_host_bytes, quality_cgroup,
    TrainingMonitor, run_stage, write_json, failure_record,
    STREAMING_HOST_POLICY, STREAMING_UNIT,
)
from image3d_scenegraph.gaussian.config import (
    ResolvedGaussianConfig, assert_single_field_ablation, resolved_config_record,
)
from image3d_scenegraph.gpu_lease import FileLease


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIGNED_REVISION = "9cf78ad84abdc2da61f03512af09a3484b7e75bd"
SIGNED_PROTOCOL_SHA256 = "8e9171e51744171daec3e33059b6b79cbcee20625793ff76fb838b1d672245cb"
SIGNED_CONFIG_SHA256 = "626298d7deef592a38498a8c13cf85d79b8dbb3493c1786ae51e56ebaeaaaa68"
SIGNED_FINAL_RECORD_SHA256 = "b128f96bc972fd634d2e4bca11278b4a321ede02a0e1a4e4b2b4582ccc2c1865"
QUALITY_SHA256 = "315f8f41dce5cde23d01abe74fdbda4dc23e0d7760eec3c24259bab02e8a137e"
ROI_SHA256 = "fcdeedd25c456708825a8ec5b929c46f8f741ccaec44d6ca46c7833ac94b2b13"
MAIN_CAMERA_SHA256 = "8415b1bd17cf77dc0207938f1189b18b79992ff531aafa63d0a71c61d08d1a86"
TRAIN_ONLY_CAMERA_SHA256 = "b9a49e99d9b9ce3a248791b8fae6d20ebf76cd9361ce69765df383df02f0aab8"
ABSOLUTE_CONFIG_HASH = "865aa98a29f2f4271289a6c20ca0771a93e8019d787e58877f7cec75924b01cc"
ORIGINAL_GATE_SHA256 = "313dc2cb9f5b901a8c94879141a63d8fe710461f0c002f83877e2e982876f86f"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_json(path: Path, expected: str) -> dict:
    if sha256_file(path) != expected:
        raise ValueError(f"frozen identity mismatch: {path}")
    return read_json(path)


def gate_template(*, quality_exploration=False) -> dict:
    gate = {
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
    if quality_exploration:
        gate.update(schema_version=2, profile="absgrad_quality_exploration_v1",
            resource_limits=copy.deepcopy(QUALITY_LIMITS),
            original_gate_sha256=ORIGINAL_GATE_SHA256,
            original_resource_gate_role="report_only; previous failure retained; no PASS_FOR_REPLICATION",
            host_policy=copy.deepcopy(QUALITY_HOST_POLICY), systemd_unit=QUALITY_UNIT)
        gate["runtime_policy"].update(poll_seconds=2, cancel_grace_seconds=0,
            term_grace_seconds=5, stop_policy="own_process_group_TERM_KILL_no_cancel_checkpoint",
            memory_scope="host cgroup all stages; per-rank reserved main and Train-only")
    return gate


def validate_gate(gate: dict, *, quality_exploration=False) -> None:
    expected = gate_template(quality_exploration=quality_exploration)
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


def preflight(signed: Path, gate_path: Path, gate_sha256: str, output: Path, *, quality_exploration=False) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; do not overwrite or resume")
    if output.absolute() != output.resolve() or not output.resolve().is_relative_to(PROJECT_ROOT / "outputs/experiments"):
        raise ValueError("candidate output must be a new non-symlink experiment directory")
    gate = checked_json(gate_path, gate_sha256)
    validate_gate(gate, quality_exploration=quality_exploration)
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
            "absolute_config": candidate, "gate": gate, "gate_sha256": gate_sha256,
            "quality_exploration": quality_exploration, "signed_path": str(signed)}


def execute(root: Path, prepared: dict, *, lease_fd: int) -> None:
    exploration = prepared["quality_exploration"]
    group = quality_cgroup() if exploration else None
    if exploration and available_host_bytes() < QUALITY_HOST_POLICY["startup_available_bytes"]:
        raise ValueError("quality exploration requires at least 22 GiB available host RAM")
    limits = QUALITY_LIMITS if exploration else LIMITS
    from gsplat import rendering
    from image3d_scenegraph.gaussian.render import require_distributed_absgrad
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    from image3d_scenegraph.gaussian.trainer import training_provenance

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
    with failure_record(root, "candidate", "execution"):
        write_json(root / "absolute.config.json", prepared["absolute_config"])
        write_json(root / "gate.json", prepared["gate"])
        write_json(root / "protocol.json", {
            "profile": "absgrad_quality_exploration_v1" if exploration else "absgrad_absolute_candidate_v1", "code": code,
            "signed_training_revision": SIGNED_REVISION,
            "signed_protocol_sha256": SIGNED_PROTOCOL_SHA256,
            "gate_sha256": prepared["gate_sha256"], "replay": str(replay),
            "dataset_hash": protocol["dataset_hash"], "overlay_sha256": protocol["overlay_sha256"],
            "effective_config_hash": ABSOLUTE_CONFIG_HASH,
            "main_camera_sequence_sha256": MAIN_CAMERA_SHA256,
            "train_only_camera_sequence_sha256": TRAIN_ONLY_CAMERA_SHA256,
            "code_hash": provenance.code_hash, "environment_hash": provenance.environment_hash,
            "resource_limits": limits, "gpu_inventory": gpu_inventory,
            "host_cgroup": None if group is None else str(group),
            "host_policy": QUALITY_HOST_POLICY if exploration else None,
            "original_gate_sha256": ORIGINAL_GATE_SHA256 if exploration else None,
            "original_resource_failure_retained": exploration,
            "quality_decision": "pending_paired_roi_review",
            "test_rgb": "not_loaded", "promotion_eligible": False,
            "telemetry": "opt-in CLI wrapper; 10s cumulative per-rank reserved peak; includes final model merge",
        })
        run_pipeline(root, replay, lease_fd=lease_fd, require_resources=require_resources,
                     limits=limits, host_group=group)
        if exploration:
            run_stage(root / "absolute", "paired-quality", [sys.executable,
                "scripts/evaluate_absgrad_pair.py", "--signed-experiment", prepared["signed_path"],
                "--candidate-experiment", str(root), "--output-dir", str(root / "paired-quality")],
                cwd=PROJECT_ROOT, pass_fds=(lease_fd,), host_group=group,
                admission=lambda: require_resources(root, minimum_free_gib=8))


def run_pipeline(root: Path, replay: Path, *, lease_fd: int, require_resources,
                 limits=None, host_group: Path | None = None, host_policy=None, arm="absolute", recovered_main: dict | None = None) -> None:
    if arm not in {"signed", "absolute"}:
        raise ValueError("unsupported frozen arm")
    if recovered_main is not None and arm != "signed":
        raise ValueError("only the frozen signed main training may be recovered")
    limits = LIMITS if limits is None else limits
    output = root / arm
    output.mkdir()
    common = ["--dataset-contract", str(replay / "dataset.json"), "--dataset-root", str(replay),
              "--resolved-config-json", str(root / f"{arm}.config.json")]

    def stage(name, arguments, monitor=None, verify=None):
        run_stage(output, name, [sys.executable, *arguments], cwd=PROJECT_ROOT,
                  monitor=monitor, pass_fds=(lease_fd,), host_group=host_group, host_policy=host_policy,
                  admission=lambda: require_resources(output, minimum_free_gib=8), verify=verify)

    training = output / "training"
    progress = training / "attempts/train-001/artifacts/progress.jsonl"
    telemetry = output / "train-memory"
    main_monitor = TrainingMonitor(progress, telemetry, updates=30000, main_stage=True, limits=limits)
    result = {}
    def verify_train():
        if main_monitor.record()["camera_sequence_sha256"] != MAIN_CAMERA_SHA256:
            raise ValueError("main camera sampling sequence differs from signed")
        result.update(read_json(training / "attempts/train-001/artifacts/result.json"))
        if result["iteration"] != 30000 or result["world_size"] != 2:
            raise ValueError("main training budget mismatch")
        peaks = result["per_rank_peak_reserved_bytes"]
        if len(peaks) != 2 or any(p > cap for p, cap in zip(peaks, limits["max_per_rank_peak_reserved_bytes"], strict=True)):
            raise ValueError("final main memory record exceeds resource limit")
    if recovered_main is None:
        stage("train", ["scripts/run_gaussian_training.py", *common, "--run-dir", str(training),
            "--trainer", "project", "--initialization", "frozen", "--distributed",
            "--no-intermediate-previews", "--cancel-file", str(output / "cancel"),
            "--resource-telemetry-dir", str(telemetry)], main_monitor, verify_train)
        model = training / result["model_path"]
    else:
        model, progress = Path(recovered_main["model_path"]), Path(recovered_main["progress_path"])
        for path, key in ((model, "model_sha256"), (progress, "progress_sha256")):
            if sha256_file(path) != recovered_main[key]:
                raise ValueError("recovered main input changed")
        write_json(output / "train-recovery.json", recovered_main)
    sor = output / "sor"
    stage("sor", ["scripts/filter_gaussian_sor.py", "--model-snapshot", str(model),
        "--output-dir", str(sor), "--nb-neighbors", "30", "--std-ratio", "2.0", "--band-opacity", "0.05"])
    selection = output / "selection"
    stage("selection", ["scripts/evaluate_gaussian.py", *common, "--model", str(sor / "filtered-model.pt"),
        "--split", "validation", "--output-dir", str(selection), "--progress", str(progress)],
        verify=lambda: validate_evaluation(selection / "evaluation.json", final=False))
    final = output / "train-only"
    telemetry = output / "train-only-memory"
    final_monitor = TrainingMonitor(final / "progress.jsonl", telemetry, updates=2000, main_stage=False, limits=limits)
    def verify_final():
        if final_monitor.record()["camera_sequence_sha256"] != TRAIN_ONLY_CAMERA_SHA256:
            raise ValueError("Train-only camera sampling sequence differs from signed")
        record = read_json(final / "record.json")
        if (record["optimizer_updates"], record["camera_samples"], record["world_size"], record["input_splits"],
                record["topology_changed"], record["source_model_unchanged"]) != (2000, 4000, 2, ["train"], False, True):
            raise ValueError("Train-only budget or topology mismatch")
        validate_evaluation(final / "evaluation.json", final=True)
    stage("train-only", ["scripts/run_gaussian_final_fit.py", *common,
        "--source-model", str(sor / "filtered-model.pt"),
        "--selection-evaluation", str(selection / "evaluation.json"), "--output-dir", str(final),
        "--train-only-control", "--distributed", "--cancel-file", str(output / "cancel"),
        "--resource-telemetry-dir", str(telemetry)], final_monitor, verify_final)
    write_json(root / "complete.json", {
        "status": "signed_control_complete" if arm == "signed" else "absolute_training_complete_quality_pending",
        "selection_evaluation_sha256": sha256_file(selection / "evaluation.json"),
        "final_evaluation_sha256": sha256_file(final / "evaluation.json"),
        "final_model_sha256": sha256_file(final / "model.pt"),
        "quality_decision": "pending_all_frozen_rois_and_visual_review",
        "test_rgb": "not_loaded", "promotion_eligible": False,
    })


def revision() -> str:
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def require_resources(root: Path, *, minimum_free_gib: int) -> None:
    import os
    import shutil
    from image3d_scenegraph.gpu_lease import require_idle_gpu

    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=10)
    indices = set(inventory.split())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = list(indices) if visible is None else [v.strip() for v in visible.split(",")]
    if len(requested) != 2 or len(set(requested)) != 2 or not set(requested).issubset(indices):
        raise ValueError("experiment requires exactly two visible GPU indices")
    require_idle_gpu()
    if shutil.disk_usage(root).free < minimum_free_gib * 1024**3:
        raise ValueError(f"at least {minimum_free_gib} GiB free disk required")
    if available_host_bytes() < 12 * 1024**3:
        raise ValueError("at least 12 GiB available host memory required")


PROFILE = "absgrad_streaming_matched_pair_v1"
SAVE_EVIDENCE = {
    "outputs/analysis/checkpoint-memory-review-20261009-v1/summary.json": "8536b197a3d28e8f38bb44fc812d2d16c79369eeeea3ac1bd638e1c36981a50d",
    "outputs/analysis/absgrad-review-sh3-20261009-v1/candidate/summary.json": "09ea236afbe9c60c5f294204c296bc5c820a02c3f877b476f071339198cf4acd",
    "outputs/analysis/absgrad-review-sh3-20261009-v1/candidate/trainer/attempts/train-001/checkpoints/iteration_000000012/checkpoint.json": "f0d5cedef4d14faee412e72e098ecb49f53b67559a94b1c10939f3bf9cc9353e",
}



RECOVERED_PROFILE = "absgrad_recovered_matched_pair_v1"
CONTINUATION_UNIT = "image3d-absgrad-continued-20261009-v1.service"
CONTINUATION_HOST_POLICY = {**STREAMING_HOST_POLICY, "minimum_available_bytes": 2 * 1024**3,
    "stop_current_bytes": None, "memory_max_bytes": None, "memory_swap_max_bytes": None}
RECOVERY_ROOT = "outputs/experiments/absgrad-save-recovery-20261009-v1"
INTERRUPTED_ROOT = "outputs/experiments/absgrad-review-matched-20261009-v1/experiment"
RECOVERY_SOURCES = {
    f"{RECOVERY_ROOT}/complete.json": "95aa1f65b7a3787c6bcdc477f4bfcbafac65cb895361bf2cdcdd91c3e5b6aefa",
    f"{RECOVERY_ROOT}/recovery-provenance.json": "d9fb52db0319c82a9a26608fcda7580463f67b9ff094a72d1da2410e4c249c61",
    f"{INTERRUPTED_ROOT}/gate.json": "f3d9c0019f2db2fc739f9f234652de4e7f1ba48769b5dfb702b56b43aa3e66e3",
    f"{INTERRUPTED_ROOT}/signed-control/protocol.json": "b90649cf7d534858cb551398dc9d3b76c7a8f6dd28214293d16374413d8be6bb",
    f"{INTERRUPTED_ROOT}/signed-control/signed/train.exit.json": "b0466ce3900c7af95e358073be0fd3c75c3d8c921157e6333c1a5b0d2b9831ae",
    f"{INTERRUPTED_ROOT}/signed-control/signed/training/attempts/train-001/artifacts/progress.jsonl": "3852824a96031cb30e4814ba6849c7f265d142a3f5cf7cda10a5bbcf748b103b",
}


def continuation_cgroup() -> Path:
    expected = f"/system.slice/{CONTINUATION_UNIT}"
    if f"0::{expected}" not in Path("/proc/self/cgroup").read_text().splitlines():
        raise ValueError("continuation requires its isolated systemd cgroup")
    group = Path("/sys/fs/cgroup") / expected.lstrip("/")
    for name in ("memory.max", "memory.high", "memory.swap.max"):
        if (group / name).read_text().strip() != "max":
            raise ValueError(f"continuation requires unlimited {name}")
    if (group / "memory.oom.group").read_text().strip() != "1":
        raise ValueError("continuation requires group OOM isolation")
    return group


def recovered_signed_main(provenance) -> dict:
    from image3d_scenegraph.gaussian.checkpoint import load_checkpoint

    for name, digest in RECOVERY_SOURCES.items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"recovery source changed: {name}")
    source = PROJECT_ROOT / INTERRUPTED_ROOT / "signed-control"
    original = read_json(source / "protocol.json")
    validate_matched_gate(read_json(source.parent / "gate.json"))
    checked_json(source / "signed.config.json", SIGNED_CONFIG_SHA256)
    for key in ("code_hash", "environment_hash", "dataset_hash", "effective_config_hash"):
        if original[key] != getattr(provenance, key):
            raise ValueError(f"recovered signed {key} mismatch")
    recovery = PROJECT_ROOT / RECOVERY_ROOT
    receipt = read_json(recovery / "complete.json")
    if (receipt["status"] != "checkpoint_and_model_recovered_verified" or receipt["iteration"] != 30000
            or receipt["training_rerun"] or not all(receipt[key] for key in
                ("loader_verified", "model_rank_order_equal", "source_hashes_unchanged"))):
        raise ValueError("recovery is incomplete")
    for name, digest in read_json(recovery / "recovery-provenance.json")["source_hashes"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"original recovery input changed: {name}")
    training = recovery / "recovered-training"
    checkpoint = training / "attempts/train-001/checkpoints/iteration_000030000"
    model = training / "attempts/train-001/artifacts/model.pt"
    if str(checkpoint) != receipt["checkpoint_path"] or str(model) != receipt["model_path"]:
        raise ValueError("unexpected recovered output path")
    loaded = load_checkpoint(training, "train-001", 30000, expected_provenance=provenance)
    if loaded.record.checkpoint_hash != receipt["checkpoint_hash"] or sha256_file(model) != receipt["model_sha256"]:
        raise ValueError("recovered checkpoint or model changed")
    failure = read_json(source / "signed/train.exit.json")
    observed = failure["resources"]
    if (failure["resource_failure"] != "task_memory_at_11.5_gib" or
            (observed["optimizer_updates_observed"], observed["camera_samples_observed"],
             observed["camera_sequence_sha256"]) != (30000, 60000, MAIN_CAMERA_SHA256)):
        raise ValueError("interrupted signed main budget mismatch")
    progress = source / "signed/training/attempts/train-001/artifacts/progress.jsonl"
    return {"status": "recovered_main_not_native_training_success", "main_training_revision": original["code"],
        "model_path": str(model), "model_sha256": receipt["model_sha256"],
        "progress_path": str(progress), "progress_sha256": sha256_file(progress),
        "checkpoint_hash": receipt["checkpoint_hash"], "source_train_exit": str(source / "signed/train.exit.json"),
        "source_train_exit_sha256": sha256_file(source / "signed/train.exit.json"),
        "recovery_receipt_sha256": RECOVERY_SOURCES[f"{RECOVERY_ROOT}/complete.json"],
        "resource_comparability": "no native signed lifecycle result; time and peak ratios unavailable"}


def matched_gate_template(*, recovered=False) -> dict:
    gate = gate_template(quality_exploration=True)
    gate.update(schema_version=3, profile=PROFILE, comparison_identity="fresh signed then absolute; identical current code/environment",
                old_signed_role="historical reference only; not the new matched control",
                host_policy=copy.deepcopy(STREAMING_HOST_POLICY), systemd_unit=STREAMING_UNIT,
                authorized_fresh_arms=["signed", "absolute"], automatic_retry_or_resume=False,
                save_verification=copy.deepcopy(SAVE_EVIDENCE))
    if recovered:
        gate.update(schema_version=4, profile=RECOVERED_PROFILE,
            comparison_identity="same training core/environment; signed main recovered without retraining",
            host_policy=copy.deepcopy(CONTINUATION_HOST_POLICY), systemd_unit=CONTINUATION_UNIT,
            authorized_fresh_arms=["absolute"], recovery_sources=copy.deepcopy(RECOVERY_SOURCES))
    return gate


def validate_matched_gate(gate: dict, *, recovered=False) -> None:
    expected = matched_gate_template(recovered=recovered)
    expected.update(status="APPROVED_FOR_CANDIDATE_EXECUTION", absolute_training_authorized=True)
    if json.dumps(gate, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("new matched pair requires its separately approved exact gate")


def receipt(signed: Path) -> dict:
    paths = ["protocol.json", "signed.config.json", "complete.json", "signed/train-only/record.json",
             "signed/train-only/model.pt", "signed/train-only/evaluation.json", "signed/selection/evaluation.json"]
    return {name: sha256_file(signed / name) for name in paths}


def execute_matched(output: Path, old_signed: Path, gate_path: Path, gate_sha: str, *, expected_revision: str, recovered=False) -> None:
    if output.exists() or output.is_symlink() or output.absolute() != output.resolve() or not output.resolve().is_relative_to(PROJECT_ROOT / "outputs/experiments"):
        raise ValueError("matched pair requires a fresh non-symlink experiment directory")
    gate = checked_json(gate_path, gate_sha)
    validate_matched_gate(gate, recovered=recovered)
    policy = CONTINUATION_HOST_POLICY if recovered else STREAMING_HOST_POLICY
    # JSON null means no task cap; the unchanged core monitor compares a numeric threshold.
    runtime_policy = {**policy, "stop_current_bytes": float("inf")} if recovered else policy
    profile = RECOVERED_PROFILE if recovered else PROFILE
    group = continuation_cgroup() if recovered else quality_cgroup(policy=policy, unit=STREAMING_UNIT)
    if available_host_bytes() < policy["startup_available_bytes"]:
        raise ValueError("matched pair requires 18 GiB available host RAM")
    from gsplat import rendering
    from image3d_scenegraph.gaussian.trainer import training_provenance
    from image3d_scenegraph.gaussian.render import require_distributed_absgrad
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle

    code = revision()
    if code != expected_revision:
        raise ValueError("execution Git revision differs from explicit launch identity")
    source_protocol = checked_json(old_signed / "protocol.json", SIGNED_PROTOCOL_SHA256)
    baseline = checked_json(old_signed / "signed.config.json", SIGNED_CONFIG_SHA256)
    old_record = checked_json(old_signed / "signed/train-only/record.json", SIGNED_FINAL_RECORD_SHA256)
    checked_json(PROJECT_ROOT / gate["quality_proposal"], QUALITY_SHA256)
    checked_json(PROJECT_ROOT / gate["train_roi"], ROI_SHA256)
    source = Path(source_protocol["source"])
    original = checked_json(source / "protocol.json", source_protocol["source_protocol_sha256"])
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
    records = {"signed": baseline, "absolute": absolute_config(baseline)}
    provenance = training_provenance(dataset_hash=source_protocol["dataset_hash"],
        effective_config_hash=baseline["effective_config_hash"], world_size=2)
    if provenance.environment_hash != old_record["environment_hash"]:
        raise ValueError("historical execution environment changed")
    evidence = [checked_json(PROJECT_ROOT / path, digest) for path, digest in SAVE_EVIDENCE.items()]
    if any(row["returncode"] != 0 for row in evidence[0]["cases"]) or evidence[1]["status"] != "passed":
        raise ValueError("save/GPU verification is incomplete")
    for key in ("code_hash", "environment_hash"):
        if evidence[2]["provenance"][key] != getattr(provenance, key):
            raise ValueError(f"save/GPU verification {key} differs from execution")
    recovered_main = recovered_signed_main(provenance) if recovered else None
    def admission(root, *, minimum_free_gib):
        if revision() != code:
            raise ValueError("code changed between stages")
        require_resources(root, minimum_free_gib=minimum_free_gib)
    admission(output.parent, minimum_free_gib=20)
    output.mkdir()
    with failure_record(output, "matched_pair", "execution"):
        write_json(output / "gate.json", gate)
        write_json(output / "protocol.json", {"profile": profile, "code": code, "gate_sha256": gate_sha,
            "code_hash": provenance.code_hash, "environment_hash": provenance.environment_hash,
            "historical_signed_protocol_sha256": SIGNED_PROTOCOL_SHA256, "original_gate_sha256": ORIGINAL_GATE_SHA256,
            "fresh_arms": gate["authorized_fresh_arms"], "promotion_eligible": False})
        with FileLease(PROJECT_ROOT / "outputs/.gpu.lock") as lease:
            for arm, dirname in (("signed", "signed-control"), ("absolute", "absolute-candidate")):
                root = output / dirname
                root.mkdir()
                write_json(root / f"{arm}.config.json", records[arm])
                protocol = {**copy.deepcopy(source_protocol), "profile": profile, "code": code,
                    "arm": arm, "gate_sha256": gate_sha, "code_hash": provenance.code_hash,
                    "environment_hash": provenance.environment_hash, "original_resource_failure_retained": True,
                    "effective_config_hash": records[arm]["effective_config_hash"], "resource_limits": QUALITY_LIMITS,
                    "host_policy": policy, "host_cgroup": str(group), "gpu_inventory": inventory,
                    "main_camera_sequence_sha256": MAIN_CAMERA_SHA256, "train_only_camera_sequence_sha256": TRAIN_ONLY_CAMERA_SHA256,
                    "test_rgb": "not_loaded", "promotion_eligible": False}
                protocol.pop("absolute_arm", None)
                if recovered_main is not None:
                    protocol["signed_main_recovery"] = recovered_main
                if arm == "absolute":
                    signed = output / "signed-control"
                    protocol["matched_signed"] = receipt(signed)
                    current_record = read_json(signed / "signed/train-only/record.json")
                    if current_record["code_hash"] != provenance.code_hash or current_record["environment_hash"] != provenance.environment_hash:
                        raise ValueError("new signed/candidate core or environment mismatch")
                write_json(root / "protocol.json", protocol)
                recovery_args = {"recovered_main": recovered_main} if recovered and arm == "signed" else {}
                run_pipeline(root, replay, lease_fd=lease.fileno(), require_resources=admission,
                             limits=QUALITY_LIMITS, host_group=group, host_policy=runtime_policy, arm=arm, **recovery_args)
            candidate_root = output / "absolute-candidate"
            run_stage(candidate_root / "absolute", "paired-quality", [sys.executable,
                "scripts/evaluate_absgrad_pair.py", "--signed-experiment", str(output / "signed-control"),
                "--candidate-experiment", str(candidate_root), "--output-dir", str(candidate_root / "paired-quality")],
                cwd=PROJECT_ROOT, pass_fds=(lease.fileno(),), host_group=group, host_policy=runtime_policy,
                admission=lambda: admission(output, minimum_free_gib=8))
            write_json(output / "complete.json", {"status": "matched_pair_numerical_report_complete_visual_pending",
                "report_sha256": sha256_file(candidate_root / "paired-quality/report.json"), "promotion_eligible": False})
