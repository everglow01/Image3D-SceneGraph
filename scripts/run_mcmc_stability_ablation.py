#!/usr/bin/env python3
"""Fresh, single-factor apartment MCMC runs using the existing trainer and resource gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.config import (
    assert_single_field_ablation, resolve_internal_config, resolved_config_record,
)
from image3d_scenegraph.gaussian.absgrad_resources import (
    STREAMING_HOST_POLICY, TrainingMonitor, quality_cgroup, run_stage, write_json,
)
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_HASH = "02f02cfe692b6e1c266a5d0fc2677ad1c9085110f7f70bdf6349a32d23911c7c"
ARMS = ("baseline", "raw_loss", "relocation_floor", "destination_moments", "position_lr")
PROFILE = "apartment_mcmc_single_factor_memory_v2"
HOST_POLICY = {**STREAMING_HOST_POLICY, "memory_high_bytes": None}
UNLIMITED_PROFILE = "apartment_mcmc_single_factor_unlimited_host_memory_v3"
UNLIMITED_HOST_POLICY = {**HOST_POLICY, **dict.fromkeys((
    "startup_available_bytes", "minimum_available_bytes", "stop_current_bytes",
    "memory_max_bytes", "memory_high_bytes", "memory_swap_max_bytes",
))}
LIMITS = {"max_observed_global_gaussians": 3_000_000,
          "max_main_stage_wall_seconds": 21600,
          "validation_progress_timeout_seconds": 300,
          "max_per_rank_peak_reserved_bytes": [18 * 1024**3] * 2}


def configs() -> dict:
    records = {}
    for arm in ARMS:
        overrides = {"resolution": {"longest_edge": 2048}}
        if arm == "raw_loss":
            overrides["loss"] = {"clamp_render": False}
        if arm == "position_lr":
            overrides["learning_rate"] = {"position": {"delay_multiplier": 1.}}
        records[arm] = resolved_config_record(resolve_internal_config("mcmc_v1", overrides))
    assert records["baseline"]["effective_config_hash"] == SOURCE_CONFIG_HASH
    for arm, field in (("raw_loss", "loss.clamp_render"),
                       ("position_lr", "learning_rate.position.delay_multiplier")):
        assert assert_single_field_ablation(records["baseline"]["effective_config"],
                                            records[arm]["effective_config"]) == field
    assert records["baseline"] == records["relocation_floor"] == records["destination_moments"]
    return records


def code_hash() -> str:
    from image3d_scenegraph.gaussian.trainer import training_provenance
    return training_provenance(dataset_hash="0" * 64, effective_config_hash="0" * 64,
                               world_size=2).code_hash


def prepare(args) -> None:
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle

    if any(value is None for value in (args.source_job, args.gates_root, args.floor_overlay, args.moments_overlay, args.memory_gate)):
        raise ValueError("preparation requires source Job, three-arm gates, overlays and the validation memory gate")
    source = args.source_job.resolve()
    if source.name != "20261010_172759_d4dda4b3":
        raise ValueError("this experiment is bound to the audited apartment MCMC Job")
    original_path = source / "gaussian/preparation/train-001/effective_config.json"
    records = configs()
    original = json.loads(original_path.read_text())
    if any(original[key] != records["baseline"][key]
           for key in ("requested_profile", "effective_config_hash", "effective_config")):
        raise ValueError("source configuration changed")
    replay = source / "gaussian/replay"
    validate_replay_bundle(replay)
    dataset = json.loads((replay / "dataset.json").read_text())
    if {key: len(value) for key, value in dataset["splits"].items()} != {"train": 2808, "validation": 351, "test": 351}:
        raise ValueError("apartment split changed")
    core = code_hash()
    memory_path = args.memory_gate.resolve()
    memory_report = json.loads(memory_path.read_text())
    if (memory_report["status"] != "passed" or memory_report["training_code_hash"] != core
            or memory_report["host_policy"] != HOST_POLICY
            or memory_report["global_gaussians"] != 3000000
            or memory_report["validation_views"] != 351 or memory_report["passes"] != 3):
        raise ValueError("validation memory gate differs from the new baseline")
    gate_hashes, environments = {str(memory_path): sha256_file(memory_path)}, {}
    for arm, folder in (("baseline", "baseline"), ("relocation_floor", "relocation"),
                        ("destination_moments", "destination")):
        path = args.gates_root.resolve() / folder / "summary.json"
        report = json.loads(path.read_text())
        if report["status"] != "passed" or report["world_size"] != 2 or len(report["ranks"]) != 2:
            raise ValueError("incomplete GPU gates")
        for rank, record in enumerate(report["ranks"]):
            if record["rank"] != rank or record["status"] != "passed" or record["training_code_hash"] != core:
                raise ValueError("GPU gate training core differs")
        if report["ranks"][0]["environment"] != report["ranks"][1]["environment"]:
            raise ValueError("GPU gate rank environments differ")
        gate_hashes[str(path)] = sha256_file(path)
        environments[arm] = report["ranks"][0]["environment"]
    overlays = {"relocation_floor": str(args.floor_overlay.resolve()),
                "destination_moments": str(args.moments_overlay.resolve())}
    for arm, folder in overlays.items():
        path = Path(folder) / "overlay.json"
        if json.loads(path.read_text())["profile"] != f"mcmc_{arm}_overlay_v1":
            raise ValueError("wrong single-factor overlay")
        if not environments[arm].endswith("sha256=" + sha256_file(path)):
            raise ValueError("overlay differs from GPU gate")
        gate_hashes[str(path)] = sha256_file(path)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for arm, record in records.items():
        write_json(args.output_root / f"{arm}.config.json", record)
    bound = [original_path, replay / "replay.json", replay / "dataset.json",
             PROJECT / "scripts/run_gaussian_training.py", Path(__file__).resolve(),
             PROJECT / "scripts/run_mcmc_stability_suite.py"]
    write_json(args.output_root / "protocol.json", {
        "profile": UNLIMITED_PROFILE if args.unlimited_host_memory else PROFILE,
        "status": "frozen_before_training",
        "training_code_hash": core, "source_job": str(source), "replay": str(replay),
        "dataset_hash": dataset["dataset_hash"], "world_size": 2, "updates": 30000,
        "camera_samples": 60000, "seed": 20260729, "longest_edge": 2048,
        "gaussian_cap": 3000000, "arms": list(ARMS), "overlays": overlays,
        "gpu_gate_environments": environments,
        "files": {**gate_hashes, **{str(path): sha256_file(path) for path in bound}},
        "configs": {arm: sha256_file(args.output_root / f"{arm}.config.json") for arm in ARMS},
        "host_policy": UNLIMITED_HOST_POLICY if args.unlimited_host_memory else HOST_POLICY,
        "limits": LIMITS,
        "evaluation": "same eight full Validation checkpoints; raw selection unchanged; common SOR and Selection evaluation",
        "comparison": "paired raw/display metrics, negative-PSNR views, RGB excursions, terminal per-rank health and resource deltas; no automatic promotion",
        "final_fit": "not_run_in_this_main_training_instability_ablation",
        "test_rgb": "not_loaded; replay verification hashes file bytes only",
        "existing_jobs_modified": False,
    })


def unlimited_host_cgroup(unit: str) -> Path:
    expected = f"/system.slice/{unit}"
    if f"0::{expected}" not in Path("/proc/self/cgroup").read_text().splitlines():
        raise ValueError("unlimited host memory requires its own systemd cgroup")
    group = Path("/sys/fs/cgroup") / expected.lstrip("/")
    for directory in (group, group.parent, group.parent.parent):
        for name in ("memory.max", "memory.high", "memory.swap.max"):
            path = directory / name
            if directory == group.parent.parent and not path.exists():
                continue
            if path.read_text().strip() != "max":
                raise ValueError(f"host memory is still limited: {path}")
    return group


def run(args) -> None:
    root = args.output_root
    protocol = json.loads((root / "protocol.json").read_text())
    unlimited = protocol["profile"] == UNLIMITED_PROFILE
    policy = UNLIMITED_HOST_POLICY if unlimited else HOST_POLICY
    if protocol["profile"] not in (PROFILE, UNLIMITED_PROFILE) or protocol["host_policy"] != policy or protocol["limits"] != LIMITS:
        raise ValueError("a matching frozen resource protocol is required; old protocols remain immutable")
    if protocol["training_code_hash"] != code_hash():
        raise ValueError("training core changed since preparation")
    for filename, digest in protocol["files"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"frozen input changed: {filename}")
    for arm, digest in protocol["configs"].items():
        if sha256_file(root / f"{arm}.config.json") != digest:
            raise ValueError("frozen arm configuration changed")
    if args.arm != "baseline" and not (root / "baseline/complete.json").exists():
        raise ValueError("a completed matched baseline is required")
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise ValueError("less than 20 GiB free disk at admission")
    suffix = "20261016-v3" if unlimited else "20261013-v2"
    unit = f"image3d-mcmc-{args.arm}-{suffix}.service"
    if unlimited:
        unlimited_host_cgroup(unit)
        group = None
    else:
        group = quality_cgroup(policy=policy, unit=unit)
        if (group / "memory.high").read_text().strip() != "max":
            raise ValueError("memory-v2 requires no cgroup memory.high throttling")
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        output = root / args.arm
        output.mkdir(exist_ok=False)
        command_prefix = [sys.executable]
        if args.arm in protocol["overlays"]:
            command_prefix = ["/usr/bin/env", "PYTHONPATH=" + protocol["overlays"][args.arm], sys.executable]
        elif os.environ.get("PYTHONPATH"):
            raise ValueError("baseline/config-only arms must use the unmodified installed library")
        common = ["--dataset-contract", protocol["replay"] + "/dataset.json",
                  "--dataset-root", protocol["replay"],
                  "--resolved-config-json", str(root / f"{args.arm}.config.json")]
        training = output / "training"
        monitor = TrainingMonitor(training / "attempts/train-001/artifacts/progress.jsonl",
                                  output / "memory", updates=30000, main_stage=True, limits=protocol["limits"])
        def stage(name, arguments, training_monitor=None):
            run_stage(output, name, command_prefix + arguments, cwd=PROJECT,
                      monitor=training_monitor, pass_fds=(lease.fileno(),),
                      host_group=group, host_policy=protocol["host_policy"])
        stage("train", ["scripts/run_gaussian_training.py", *common, "--run-dir", str(training),
                        "--trainer", "mcmc", "--initialization", "frozen", "--distributed",
                        "--no-intermediate-previews", "--resource-telemetry-dir", str(output / "memory"),
                        "--cancel-file", str(output / "cancel")], monitor)
        result = json.loads((training / "attempts/train-001/artifacts/result.json").read_text())
        if (result["iteration"], result["world_size"], result["gaussian_cap"]) != (30000, 2, 3000000):
            raise ValueError("training budget differs")
        if args.arm != "baseline":
            baseline = json.loads((root / "baseline/complete.json").read_text())
            if baseline["resources"]["camera_sequence_sha256"] != monitor.record()["camera_sequence_sha256"]:
                raise ValueError("camera sample sequence differs from baseline")
        stage("sor", ["scripts/filter_gaussian_sor.py", "--model-snapshot", str(training / result["model_path"]),
                      "--output-dir", str(output / "sor"), "--nb-neighbors", "30", "--std-ratio", "2.0", "--band-opacity", "0.05"])
        stage("selection", ["scripts/evaluate_gaussian.py", *common,
                            "--model", str(output / "sor/filtered-model.pt"), "--split", "validation",
                            "--output-dir", str(output / "selection"), "--progress", str(training / result["progress_path"])])
        selected = json.loads((output / "selection/evaluation.json").read_text())
        if selected["status"] != "complete" or selected["successful_views"] != 351:
            raise ValueError("Selection evaluation is incomplete")
        final = result["final_validation"]
        write_json(output / "complete.json", {
            "status": "complete", "arm": args.arm, "protocol_sha256": sha256_file(root / "protocol.json"),
            "selected_iteration": result["candidate_iteration"], "resources": monitor.record(),
            "selection_raw_psnr": selected["psnr"], "selection_display_psnr": selected["display_psnr"],
            "selection_ssim": selected["ssim"], "selection_model_sha256": sha256_file(output / "sor/filtered-model.pt"),
            "selection_evaluation_sha256": sha256_file(output / "selection/evaluation.json"),
            "final_raw_psnr": final["psnr"], "final_display_psnr": final["display_psnr"],
            "final_negative_psnr_views": sum(view["psnr"] < 0 for view in final["per_view"]),
            "final_parameter_health_by_rank": final["mcmc_health_by_rank"],
            "final_max_rgb": max(view["rgb_range"]["max"] for view in final["per_view"]),
            "final_above_one_channels": sum(view["rgb_range"]["above_one_count"] for view in final["per_view"]),
            "test_rgb": "not_loaded", "promotion_eligible": False,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--arm", choices=ARMS)
    parser.add_argument("--source-job", type=Path)
    parser.add_argument("--gates-root", type=Path)
    parser.add_argument("--floor-overlay", type=Path)
    parser.add_argument("--moments-overlay", type=Path)
    parser.add_argument("--memory-gate", type=Path)
    parser.add_argument("--unlimited-host-memory", action="store_true",
                        help="prepare a new protocol without task RAM/swap limits or host-memory stop gates")
    args = parser.parse_args()
    if args.unlimited_host_memory and not args.prepare:
        parser.error("--unlimited-host-memory is only valid with --prepare; runs use their frozen protocol")
    args.output_root = args.output_root.resolve()
    if args.prepare:
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
