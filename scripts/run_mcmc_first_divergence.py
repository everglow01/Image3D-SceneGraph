#!/usr/bin/env python3
"""Independent T1 native-resume A/A one-step probe; execution requires separate authorization."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import socket
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import run_stage
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from scripts import run_mcmc_lifecycle_trace as lifecycle
from scripts.mcmc_first_divergence import (STAGES, PARAMETERS, ATOL, RTOL, ITERATION, GIB, FirstDivergenceProbe,
    ProbeComplete, write_new, safe_file, compare_snapshots, summarize_comparison, verify_snapshot_files, validate_stage_schema)
from scripts.run_mcmc_stability_ablation import PROJECT, LIMITS, unlimited_host_cgroup
from scripts.prepare_gsplat_camera_sizes import file_hashes

PROFILE = "mcmc_first_divergence_t1_v1"
BRANCHES = ("a", "b")
DEPENDENCIES = ("scripts/run_mcmc_first_divergence.py", "scripts/mcmc_first_divergence.py",
                "scripts/run_mcmc_lifecycle_trace.py", "scripts/mcmc_lifecycle_observer.py",
                "scripts/run_camera_size_scene_pair.py", "scripts/run_camera_size_lifecycle.py",
                "scripts/run_mcmc_stability_ablation.py", "scripts/prepare_gsplat_camera_sizes.py",
                "src/image3d_scenegraph/gpu_lease.py", "src/image3d_scenegraph/file_integrity.py")
read = lifecycle.read


def verify_inputs(p):
    if lifecycle.core_hash() != lifecycle.CORE or sha256_file(lifecycle.SOURCE / "protocol.json") != lifecycle.SOURCE_PROTOCOL:
        raise ValueError("frozen source differs")
    if p["source"] != str(lifecycle.SOURCE) or p["pair"] != read(lifecycle.SOURCE / "protocol.json"):
        raise ValueError("source protocol identity differs")
    for name, digest in p["bound_files"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"bound file changed: {name}")
    if file_hashes(Path(p["pair"]["source_package"])) != p["pair"]["source_hashes"]:
        raise ValueError("original package changed")
    meta = lifecycle.verify_checkpoint(Path(p["parent_training"]), "train-001", 12000, lifecycle.parent_provenance(p, "baseline"))
    expected_disk = max(20 * GIB, 12 * sum(f["bytes"] for f in meta["files"].values()) + 8 * GIB)
    if "disk_admission_bytes" in p and p["disk_admission_bytes"] != expected_disk:
        raise ValueError("snapshot disk admission budget differs")
    if meta["checkpoint_hash"] != p["parent_checkpoint_hash"]:
        raise ValueError("parent checkpoint identity differs")
    return meta


def protocol_at(root, digest):
    if not digest or sha256_file(root / "protocol.json") != digest:
        raise ValueError("explicit protocol SHA required")
    p = read(root / "protocol.json")
    expected = {"profile": PROFILE, "core": lifecycle.CORE, "branches": list(BRANCHES), "environment": "baseline",
                "start": 12000, "end": ITERATION, "stages": list(STAGES), "atol": ATOL, "rtol": RTOL,
                "nominal_iterations": 30000, "updates_per_branch": 1, "host_limits": None, "limits": LIMITS}
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported first-divergence protocol")
    required = {str(PROJECT / name) for name in DEPENDENCIES}
    required |= {str(lifecycle.SOURCE / "protocol.json"), str(lifecycle.SOURCE / "config.json"),
                 str(lifecycle.SOURCE / "baseline/complete.json"), str(lifecycle.SOURCE / "baseline/training/attempts/train-001/attempt.json")}
    if not required <= p["bound_files"].keys():
        raise ValueError("missing executable/input bindings")
    if p["parent_training"] != str(lifecycle.SOURCE / "baseline/training"):
        raise ValueError("wrong parent")
    return p


def prepare(root):
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    pair = read(lifecycle.SOURCE / "protocol.json")
    parent = lifecycle.SOURCE / "baseline/training"
    result = read(parent / "attempts/train-001/artifacts/result.json")
    terminal = lifecycle.verify_terminal(parent, result, pair, "baseline")
    complete = read(lifecycle.SOURCE / "baseline/complete.json")
    if complete["status"] != "complete" or any(complete[k] != v for k, v in terminal.items()):
        raise ValueError("source baseline incomplete")
    p = {"profile": PROFILE, "core": lifecycle.CORE, "source": str(lifecycle.SOURCE), "pair": pair,
         "environment": "baseline", "branches": list(BRANCHES), "start": 12000, "end": ITERATION,
         "nominal_iterations": 30000, "updates_per_branch": 1, "stages": list(STAGES), "atol": ATOL, "rtol": RTOL,
         "parent_training": str(parent), "parent_checkpoint_hash": terminal["checkpoint_sha256"],
         "bound_files": {str(PROJECT / name): sha256_file(PROJECT / name) for name in DEPENDENCIES},
         "limits": LIMITS, "host_limits": None, "test_boundary": "replay_hash_only_no_Test_decode",
         "snapshot_format": "typed_json_and_nonpickle_npy", "comparison_reference": "a_actual_b_reference",
         "termination": "P4_probe_signal_not_native_training_completion", "T2_T3": "not_authorized_or_implemented",
         "status": "prepared_not_executed"}
    p["bound_files"].update(pair["files"])
    for path in (lifecycle.SOURCE / "protocol.json", lifecycle.SOURCE / "config.json", lifecycle.SOURCE / "baseline/complete.json",
                 parent / "attempts/train-001/attempt.json", parent / "attempts/train-001/artifacts/result.json"):
        p["bound_files"][str(path)] = sha256_file(path)
    meta = verify_inputs(p)
    p["replay_record"] = validate_replay_bundle(Path(pair["replay"]))
    size = sum(f["bytes"] for f in meta["files"].values())
    p["disk_admission_bytes"] = max(20 * GIB, 12 * size + 8 * GIB)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "protocol.json", p)


def verify_worker(stage, rank):
    r = read(stage / f"worker-rank-{rank}.json")
    if (r["status"], r["rank"], r["iteration"], r["new_updates"], r["optimizer_steps"], r["native_training_complete"], r["validation_evaluated"]) != (
            "probe_complete", rank, ITERATION, 1, list(PARAMETERS), False, False):
        raise ValueError("worker identity differs")
    if r["calls"] != dict.fromkeys(("restore", "batch", "render", "backward", "strategy"), 1) or list(r["manifests"]) != list(STAGES):
        raise ValueError("probe call/stage counts differ")
    phases = [json.loads(line) for line in (stage / f"phases-rank-{rank}.jsonl").read_text().splitlines()]
    if phases != [{"stage": s, "iteration": ITERATION, "manifest_sha256": r["manifests"][s]} for s in STAGES]:
        raise ValueError("missing or reordered phases")
    for s in STAGES:
        if sha256_file(safe_file(stage, f"rank-{rank}/{s}/manifest.json")) != r["manifests"][s]:
            raise ValueError("phase manifest changed")
        validate_stage_schema(verify_snapshot_files(stage / f"rank-{rank}" / s), s)
    return r


class ProbeMonitor:
    def __init__(self, stage):
        self.stage, self.telemetry = stage, stage / "memory"
        self.peaks = [0, 0]
        self.completed = False

    def __call__(self, elapsed, *, final=False):
        if elapsed > LIMITS["max_main_stage_wall_seconds"]:
            return "probe_timeout"
        for rank in (0, 1):
            path = self.telemetry / f"rank-{rank}.json"
            if path.exists():
                value = read(path)
                self.peaks[rank] = max(self.peaks[rank], value["peak_reserved_bytes"])
                if self.peaks[rank] > LIMITS["max_per_rank_peak_reserved_bytes"][rank]:
                    return "gpu_reserved_limit"
            elif final:
                return "missing_memory_telemetry"
        if final:
            for rank in (0, 1):
                verify_worker(self.stage, rank)
            self.completed = True

    def record(self):
        return {"optimizer_updates_observed": int(self.completed), "camera_samples_observed": 2 * int(self.completed),
                "absolute_iteration": ITERATION if self.completed else None, "per_rank_peak_reserved_bytes": self.peaks}


def completed(root, branch, digest):
    stage = root / branch
    c = read(stage / "complete.json")
    if (c["status"], c["branch"], c["protocol_sha256"]) != ("probe_complete", branch, digest):
        raise ValueError("branch completion differs")
    if (c["resources"]["optimizer_updates_observed"], c["resources"]["camera_samples_observed"], c["resources"]["absolute_iteration"]) != (1, 2, ITERATION):
        raise ValueError("probe budget differs")
    required = {f"{name}-rank-{r}.{suffix}" for r in (0, 1) for name, suffix in (("worker", "json"), ("phases", "jsonl"))}
    if not required <= c["files"].keys():
        raise ValueError("missing worker binding")
    for name, h in c["files"].items():
        if sha256_file(safe_file(stage, name)) != h:
            raise ValueError("completed file changed")
    for rank in (0, 1):
        verify_worker(stage, rank)
    return c


def run_branch(args, p):
    unlimited_host_cgroup(f"image3d-first-divergence-t1-{args.branch}-v1.service")
    if args.branch == "b":
        completed(args.output_root, "a", args.protocol_sha256)
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        from image3d_scenegraph.gaussian.replay import validate_replay_bundle
        verify_inputs(p)
        if validate_replay_bundle(Path(p["pair"]["replay"])) != p["replay_record"]:
            raise ValueError("replay differs")
        if shutil.disk_usage(PROJECT).free < p["disk_admission_bytes"]:
            raise ValueError("insufficient disk for two-branch snapshots")
        stage = args.output_root / args.branch
        stage.mkdir(exist_ok=False)
        parent = Path(p["parent_training"]) / "attempts/train-001"
        destination = stage / "training/attempts/train-001"
        destination.mkdir(parents=True)
        shutil.copyfile(parent / "attempt.json", destination / "attempt.json")
        shutil.copytree(parent / "checkpoints/iteration_000012000", destination / "checkpoints/iteration_000012000")
        lifecycle.verify_checkpoint(stage / "training", "train-001", 12000, lifecycle.parent_provenance(p, "baseline"))
        command = ["/usr/bin/env", "-u", "PYTHONPATH", sys.executable, str(Path(__file__).resolve()),
                   "--output-root", str(args.output_root), "--worker", "--branch", args.branch,
                   "--protocol-sha256", args.protocol_sha256, "--authorize-execution", "--lease-fd", str(lease.fileno())]
        monitor = ProbeMonitor(stage)
        run_stage(stage, "probe", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
        verify_inputs(p)
        for rank in (0, 1):
            verify_worker(stage, rank)
        files = [*stage.glob("worker-rank-*.json"), *stage.glob("phases-rank-*.jsonl"), stage / "probe.exit.json"]
        write_new(stage / "complete.json", {"status": "probe_complete", "branch": args.branch, "protocol_sha256": args.protocol_sha256,
                  "resources": monitor.record(), "files": {str(f.relative_to(stage)): sha256_file(f) for f in files}})


def worker(local_rank, rank, world_size, args):
    import torch
    import gsplat
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig
    from image3d_scenegraph.gaussian.initialization import load_frozen_initialization
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    if world_size != 2 or rank != local_rank:
        raise ValueError("exactly two local ranks required")
    torch.cuda.set_device(local_rank)
    p = protocol_at(args.output_root, args.protocol_sha256)
    verify_inputs(p)
    if Path(gsplat.__file__).resolve().parent != Path(p["pair"]["source_package"]).resolve():
        raise ValueError("not original gsplat")
    cfg = read(lifecycle.SOURCE / "config.json")
    resolved = ResolvedGaussianConfig(cfg["requested_profile"], cfg["effective_config"], cfg["effective_config_hash"])
    replay, bundle = Path(p["pair"]["replay"]), p["replay_record"]
    contract = read(replay / "dataset.json")
    initialization = load_frozen_initialization(replay / bundle["initialization_path"], replay / bundle["initialization_diagnostics_path"],
                                               expected_sha256=bundle["initialization_sha256"])
    provenance = trainer.training_provenance(dataset_hash=contract["dataset_hash"], effective_config_hash=resolved.effective_config_hash, world_size=2)
    if asdict(provenance) != lifecycle.parent_provenance(p, "baseline"):
        raise ValueError("runtime provenance differs")
    stage = args.output_root / args.branch
    failure = stage / f"failure-rank-{rank}"
    failure.mkdir(exist_ok=False)
    probe = FirstDivergenceProbe(stage, rank)
    with memory_telemetry(stage / "memory", local_rank, rank), failure_record(failure, "T1", args.branch):
        try:
            with probe.installed():
                trainer.train_gaussians(contract=contract, dataset_root=replay, initialization=initialization,
                    resolved_config=resolved, run_dir=stage / "training", attempt_id="probe-001", attempt_kind="resume",
                    parent_attempt_id="train-001", resume_iteration=12000, local_rank=local_rank, world_rank=rank, world_size=2,
                    stop_after_iteration=ITERATION, save_intermediate_previews=False,
                    cancel_requested=lambda: (stage / "cancel").exists())
        except ProbeComplete:
            probe.finish()
        else:
            raise ValueError("native lifecycle unexpectedly continued beyond probe")


def compare(root, p, digest):
    verify_inputs(p)
    complete = {b: completed(root, b, digest) for b in BRANCHES}
    ranks = {str(r): {s: compare_snapshots(root / "a" / f"rank-{r}" / s, root / "b" / f"rank-{r}" / s) for s in STAGES} for r in (0, 1)}
    report = summarize_comparison(ranks)
    report.update(profile=PROFILE, protocol_sha256=digest, stages=ranks, atol=ATOL, rtol=RTOL,
                  complete_sha256={b: sha256_file(root / b / "complete.json") for b in complete},
                  new_updates=2, new_camera_samples=4, T2_T3="not_run", old_gate1="not_reclassified")
    write_new(root / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "worker", "compare"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--authorize-prepare", action="store_true")
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--authorize-comparison", action="store_true")
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("本机禁止Torch/真实张量快照执行；仅允许指定远端")
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(PROJECT / "outputs/analysis") or args.output_root == PROJECT / "outputs/analysis":
        raise SystemExit("需要独立研究目录")
    if args.prepare:
        if not args.authorize_prepare:
            raise SystemExit("准备未获显式授权")
        prepare(args.output_root)
        return
    if args.compare:
        if not args.authorize_comparison:
            raise SystemExit("比较未获显式授权")
    elif not args.authorize_execution or args.branch is None:
        raise SystemExit("执行未获显式授权或缺分支")
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.run:
        run_branch(args, p)
    elif args.compare:
        compare(args.output_root, p, args.protocol_sha256)
    else:
        unlimited_host_cgroup(f"image3d-first-divergence-t1-{args.branch}-v1.service")
        import fcntl
        if args.lease_fd is None or os.readlink(f"/proc/self/fd/{args.lease_fd}") != str(PROJECT / "outputs/.gpu.lock"):
            raise ValueError("worker requires inherited GPU lease")
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from gsplat.distributed import cli
        if torch.cuda.device_count() != 2:
            raise ValueError("exactly two GPUs required")
        cli(worker, args, verbose=True)


if __name__ == "__main__":
    main()
