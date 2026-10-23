"""Frozen ten-case connected propagation; remote execution is explicitly gated."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from ..e004_fixed_b0 import legacy, run as fixed
from . import analyze
from .probe import ORDER, STAGES, PropagationProbe, require_finite

PROJECT = legacy.PROJECT
MODULE = "experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e006_connected_propagation.run"
REL = "experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e006_connected_propagation"
RUNS = PROJECT / REL / "runs"
PROFILE = "mcmc_connected_propagation_v1"
UNIT = "image3d-connected-propagation-v1.service"
B1_ROOT = fixed.RUNS_ROOT / "workspace-20261023-v1"
B1_SHA = "34126cd50d0c7d02fe508cfec5701cc1c4768cf7e5b0d6efa2f09a5b01025ce0"
B1_COMPARISON_SHA = "cab6a91e20999442fb8d70f0d80285f1b8e4b6a2e55ddae209ed5ec5915037b8"
PARENT_SHA = "080952acf0ef989b160b64d9bb8e17f8acb7adf2bf7e18b19e3ae1d94d3c2105"
CASES = (("native-0", None), ("same-0", 0), ("same-1", 0), ("same-2", 0),
         ("native-1", None), ("varied-1", 1), ("varied-2", 2), ("varied-3", 3), ("varied-4", 4), ("native-2", None))
FILES = tuple(REL + "/" + name for name in ("run.py", "probe.py", "analyze.py",
    "tests/test_connected_propagation.py", "tests/test_connected_propagation_remote.py"))
snapshots = legacy.snapshots
read, write_new, sha256 = legacy.read, snapshots.write_new, snapshots.sha256_file
CONTRACT = {"profile": PROFILE, "source": str(fixed.SOURCE), "source_protocol_sha256": fixed.SOURCE_SHA,
    "source_comparison_sha256": fixed.COMPARISON_SHA, "B1_root": str(B1_ROOT), "B1_protocol_sha256": B1_SHA,
    "B1_comparison_sha256": B1_COMPARISON_SHA, "parent_checkpoint_sha256": PARENT_SHA,
    "core": legacy.CORE, "start": 12000, "end": 12001, "nominal_iterations": 30000,
    "cases": [{"name": name, "trial": trial} for name, trial in CASES], "order": list(ORDER),
    "atol": snapshots.ATOL, "rtol": snapshots.RTOL, "global_updates": 10, "camera_forwards": 20,
    "loss_backwards": 20, "original_rasterizer_backwards": 6, "injected_returns": 14,
    "optimizer_group_steps": 120, "strategy_posts": 0, "limits": legacy.LIMITS, "host_limits": None,
    "suite_timeout_seconds": 21600, "stop": "after_last_Adam_P3_before_strategy",
    "initialization": "pinned_extension_import_before_native_restore_no_warmup",
    "preflight_budget": {"synthetic_scalar_forwards": 2, "loss_backwards": 2, "injected_returns": 4,
        "optimizer_group_steps": 12, "real_rasterizer_forwards": 0, "real_rasterizer_backwards": 0,
        "strategy_posts": 0, "coordinated_rejection_checks": 1},
    "test_boundary": "no_Test_or_Validation_image_decode", "old_gate1": "not_reclassified",
    "interpretation": "instrumented_single_step_not_nonintervention_or_long_term_proof"}


def publish_status(path, value):
    temporary = path.with_suffix(".pending")
    write_new(temporary, value)
    os.link(temporary, path)  # Atomic, exclusive publication; never overwrite a terminal record.
    temporary.unlink()


def output_root(path, preparing=False):
    path = Path(path).absolute()
    try:
        relative = path.relative_to(RUNS)
    except ValueError as exc:
        raise ValueError("output must be inside E006 runs") from exc
    result = snapshots.safe_file(RUNS, relative)
    if result == RUNS or (preparing and result.exists()):
        raise ValueError("new independent output directory required")
    return result


def native_source():
    return fixed.verify_source()


def execution_commit():
    names = list(dict.fromkeys((*fixed.DEPENDENCIES, *FILES)))
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *names], cwd=PROJECT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *names], cwd=PROJECT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()


def parent_metadata(source, training=None):
    from scripts import run_mcmc_first_divergence as native
    training = Path(source["parent_training"]) if training is None else training
    meta = native.lifecycle.verify_checkpoint(training, "train-001", 12000, native.lifecycle.parent_provenance(source, "baseline"))
    if meta["checkpoint_hash"] != PARENT_SHA:
        raise ValueError("parent checkpoint differs")
    return meta


def admission_bytes(checkpoint_bytes, snapshot_bytes):
    return max(20 * snapshots.GIB, checkpoint_bytes + len(CASES) * snapshot_bytes + 8 * snapshots.GIB)


def prepare(root):
    commit = execution_commit()
    source = native_source()
    previous = fixed.protocol_at(B1_ROOT, B1_SHA)
    fixed.verify_inputs(previous)
    if sha256(B1_ROOT / "comparison.json") != B1_COMPARISON_SHA:
        raise ValueError("B1 comparison changed")
    old_report = read(B1_ROOT / "comparison.json")
    if old_report["complete_sha256"] != sha256(B1_ROOT / "replay/complete.json"):
        raise ValueError("B1 completion changed")
    if old_report["observed_pairwise_status"] != "numerically_equivalent" or old_report["protocol_sha256"] != B1_SHA:
        raise ValueError("B1 source not accepted")
    meta = parent_metadata(source)
    bound = dict(previous["bound_files"])
    for name in FILES:
        bound[str(PROJECT / name)] = sha256(PROJECT / name)
    for path in (B1_ROOT / "protocol.json", B1_ROOT / "comparison.json", B1_ROOT / "replay/complete.json"):
        bound[str(path)] = sha256(path)
    for rank in (0, 1):
        fixed.verify_worker(B1_ROOT / "replay", rank)
        for path in (B1_ROOT / f"replay/worker-rank-{rank}.json", B1_ROOT / f"replay/rank-{rank}/initialization.json"):
            bound[str(path)] = sha256(path)
        for trial in range(5):
            folder = B1_ROOT / f"replay/rank-{rank}/trial-{trial}"
            record = snapshots.load_snapshot(folder)
            bound[str(folder / "manifest.json")] = sha256(folder / "manifest.json")
            bound.update({str(folder / name): v["sha256"] for name, v in record["files"].items()})
    parent = Path(source["parent_training"]) / "attempts/train-001"
    for path in [parent / "attempt.json", *(parent / "checkpoints/iteration_000012000").iterdir()]:
        bound[str(path)] = sha256(path)
    size = sum(v["bytes"] for v in meta["files"].values())
    snapshot_bytes = sum(sum(x["bytes"] for x in read(fixed.SOURCE / f"a/rank-{r}/{stage}/manifest.json")["files"].values())
                         for r in (0, 1) for stage in ORDER)
    protocol = {**CONTRACT, "implementation_commit": commit, "bound_files": bound, "parent_training": source["parent_training"],
        "checkpoint_bytes": size, "snapshot_bytes_per_case": snapshot_bytes,
        "disk_admission_bytes": admission_bytes(size, snapshot_bytes), "points": previous["points"],
        "package": previous["package"], "package_hashes": previous["package_hashes"], "status": "prepared_not_executed"}
    verify_inputs(protocol)
    if shutil.disk_usage(PROJECT).free < protocol["disk_admission_bytes"]:
        raise ValueError("insufficient space for all ten cases")
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "protocol.json", protocol)


def protocol_at(root, digest):
    legacy.verify()
    if not digest or sha256(snapshots.safe_file(root, "protocol.json")) != digest:
        raise ValueError("explicit protocol SHA required")
    p = read(root / "protocol.json")
    if any(p.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("connected propagation contract differs")
    if p["disk_admission_bytes"] != admission_bytes(p["checkpoint_bytes"], p["snapshot_bytes_per_case"]):
        raise ValueError("disk budget differs")
    for name in (*fixed.DEPENDENCIES, *FILES):
        path = snapshots.safe_file(PROJECT, name)
        if p["bound_files"].get(str(path)) != sha256(path):
            raise ValueError("execution dependency binding differs")
    return p


def verify_inputs(p):
    if execution_commit() != p["implementation_commit"]:
        raise ValueError("execution commit changed")
    source = native_source()
    previous = fixed.protocol_at(B1_ROOT, B1_SHA)
    fixed.verify_inputs(previous)
    meta = parent_metadata(source)
    size = sum(v["bytes"] for v in meta["files"].values())
    snap_size = sum(sum(x["bytes"] for x in read(fixed.SOURCE / f"a/rank-{r}/{s}/manifest.json")["files"].values())
                    for r in (0, 1) for s in ORDER)
    if (p["parent_training"] != source["parent_training"] or p["points"] != previous["points"]
            or p["checkpoint_bytes"] != size or p["snapshot_bytes_per_case"] != snap_size
            or p["package"] != previous["package"] or p["package_hashes"] != previous["package_hashes"]):
        raise ValueError("source, point selection or storage identity differs")
    required = set(previous["bound_files"]) | {str(PROJECT / n) for n in FILES}
    required |= {str(B1_ROOT / n) for n in ("protocol.json", "comparison.json", "replay/complete.json")}
    for rank in (0, 1):
        required |= {str(B1_ROOT / f"replay/worker-rank-{rank}.json"), str(B1_ROOT / f"replay/rank-{rank}/initialization.json")}
        for trial in range(5):
            folder = B1_ROOT / f"replay/rank-{rank}/trial-{trial}"
            required.add(str(folder / "manifest.json"))
            required.update(str(snapshots.safe_file(folder, name)) for name in read(folder / "manifest.json")["files"])
    parent = Path(p["parent_training"]) / "attempts/train-001"
    required.add(str(parent / "attempt.json"))
    required.update(str(x) for x in (parent / "checkpoints/iteration_000012000").iterdir())
    if not required <= p["bound_files"].keys():
        raise ValueError("missing source bindings")
    for name, digest in p["bound_files"].items():
        path = Path(name)
        if not path.is_relative_to(PROJECT) or sha256(snapshots.safe_file(PROJECT, path.relative_to(PROJECT))) != digest:
            raise ValueError("bound input changed: " + name)
    if sha256(B1_ROOT / "comparison.json") != B1_COMPARISON_SHA or shutil.disk_usage(PROJECT).free < 4 * snapshots.GIB:
        raise ValueError("source comparison or disk reserve differs")


def verify_worker(stage, name, rank):
    trial = dict(CASES)[name]
    r = read(stage / f"worker-rank-{rank}.json")
    expected = {"status": "probe_complete", "rank": rank, "iteration": 12001, "stages": list(STAGES),
        "order": list(ORDER), "new_updates": 1, "forward_calls": 1, "rasterizer_dispatch_calls": 1,
        "original_backward_calls": int(trial is None), "injected_backward_returns": int(trial is not None),
        "optimizer_steps": list(snapshots.PARAMETERS), "native_training_complete": False, "validation_evaluated": False,
        "calls": {"restore": 1, "batch": 1, "render": 1, "backward": 1, "strategy": 0},
        "injection": str(B1_ROOT / f"replay/rank-{rank}/trial-{trial}") if trial is not None else None}
    if any(r.get(k) != v for k, v in expected.items()) or list(r["manifests"]) != list(STAGES) or list(r["boundary_manifests"]) != ["B0", "B1"]:
        raise ValueError("worker identity, stages or budget differs")
    for stage_name, digest in {**r["manifests"], **r["boundary_manifests"]}.items():
        folder = stage / f"rank-{rank}" / stage_name
        if sha256(snapshots.safe_file(folder, "manifest.json")) != digest:
            raise ValueError("worker manifest changed")
        record = require_finite(folder)
        if stage_name in STAGES:
            snapshots.validate_stage_schema(record, stage_name)
        else:
            legacy.boundary.validate_boundary_schema(record, stage_name)
    phases = [json.loads(line) for line in (stage / f"phases-rank-{rank}.jsonl").read_text().splitlines()]
    if phases != [{"stage": s, "iteration": 12001, "manifest_sha256": r["manifests"][s]} for s in STAGES]:
        raise ValueError("phase log differs")
    return r


def completed(root, name, digest):
    stage = root / name
    c = read(stage / "complete.json")
    required = {"case.exit.json"} | {f"{prefix}-rank-{r}.{ext}" for r in (0, 1)
                                   for prefix, ext in (("worker", "json"), ("phases", "jsonl"))}
    if (c["status"], c["case"], c["protocol_sha256"]) != ("probe_complete", name, digest) or set(c["files"]) != required:
        raise ValueError("incomplete case")
    for name, h in c["files"].items():
        if sha256(snapshots.safe_file(stage, name)) != h:
            raise ValueError("completion binding differs")
    exit_record = read(stage / "case.exit.json")
    if exit_record["returncode"] != 0 or exit_record["resource_failure"] is not None:
        raise ValueError("case execution failed")
    for rank in (0, 1):
        verify_worker(stage, c["case"], rank)
    return c


class CaseMonitor:
    def __init__(self, stage, name, deadline):
        self.stage, self.name, self.deadline = stage, name, deadline
        self.telemetry, self.peaks = stage / "memory", [0, 0]

    def __call__(self, elapsed, *, final=False):
        if time.monotonic() > self.deadline:
            return "suite_timeout"
        for rank in (0, 1):
            path = self.telemetry / f"rank-{rank}.json"
            if path.exists():
                self.peaks[rank] = max(self.peaks[rank], read(path)["peak_reserved_bytes"])
                if self.peaks[rank] > legacy.LIMITS["max_per_rank_peak_reserved_bytes"][rank]:
                    return "gpu_reserved_limit"
            elif final:
                return "missing_memory_telemetry"
        if final:
            for rank in (0, 1):
                verify_worker(self.stage, self.name, rank)

    def record(self):
        return {"per_rank_peak_reserved_bytes": self.peaks}


def worker(local_rank, rank, world_size, args):
    import torch
    import gsplat
    from scripts import run_mcmc_first_divergence as native
    from scripts import run_mcmc_backward_boundary  # noqa: F401 -- prime imports before RNG restore
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig
    from image3d_scenegraph.gaussian.initialization import load_frozen_initialization
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    if world_size != 2 or local_rank != rank:
        raise ValueError("exactly two local ranks required")
    torch.cuda.set_device(rank)
    p = protocol_at(args.output_root, args.protocol_sha256)
    verify_inputs(p)
    source = read(fixed.SOURCE / "protocol.json")
    parent_metadata(source, args.output_root / "training")
    if Path(gsplat.__file__).resolve().parent != Path(p["package"]).resolve():
        raise ValueError("original gsplat required")
    spec = find_spec("gsplat.csrc")
    if spec is None or not isinstance(spec.loader, ExtensionFileLoader) or not spec.origin:
        raise ValueError("pinned precompiled extension required; no JIT fallback")
    binary = snapshots.safe_file(Path(p["package"]), Path(spec.origin).relative_to(p["package"]))
    if sha256(binary) != p["package_hashes"].get(str(binary.relative_to(p["package"]))):
        raise ValueError("extension identity differs")
    extension = import_module("gsplat.csrc")
    if Path(extension.__file__).resolve() != binary.resolve() or not callable(getattr(extension, "rasterize_to_pixels_3dgs_bwd", None)):
        raise ValueError("loaded extension differs")
    config = read(native.lifecycle.SOURCE / "config.json")
    resolved = ResolvedGaussianConfig(config["requested_profile"], config["effective_config"], config["effective_config_hash"])
    replay = Path(source["pair"]["replay"])
    bundle = validate_replay_bundle(replay)
    if bundle != source["replay_record"]:
        raise ValueError("replay bundle differs")
    contract = read(replay / "dataset.json")
    initialization = load_frozen_initialization(replay / bundle["initialization_path"], replay / bundle["initialization_diagnostics_path"],
                                               expected_sha256=bundle["initialization_sha256"])
    provenance = trainer.training_provenance(dataset_hash=contract["dataset_hash"], effective_config_hash=resolved.effective_config_hash, world_size=2)
    if asdict(provenance) != native.lifecycle.parent_provenance(source, "baseline"):
        raise ValueError("runtime provenance differs")
    trial = dict(CASES)[args.case]
    stage = args.output_root / args.case
    failure = stage / f"failure-rank-{rank}"
    failure.mkdir(exist_ok=False)
    instance = PropagationProbe(stage, rank, fixed.SOURCE / "a",
        args.output_root / "native-0" if args.case != "native-0" else None,
        B1_ROOT / f"replay/rank-{rank}/trial-{trial}" if trial is not None else None, contract["splits"]["train"])
    with memory_telemetry(stage / "memory", local_rank, rank), failure_record(failure, "connected_P3", args.case):
        try:
            with instance.installed():
                trainer.train_gaussians(contract=contract, dataset_root=replay, initialization=initialization,
                    resolved_config=resolved, run_dir=args.output_root / "training", attempt_id=args.case,
                    attempt_kind="resume", parent_attempt_id="train-001", resume_iteration=12000,
                    local_rank=rank, world_rank=rank, world_size=2, stop_after_iteration=12001,
                    save_intermediate_previews=False, cancel_requested=lambda: (stage / "cancel").exists())
        except snapshots.ProbeComplete:
            instance.finish()
        else:
            raise ValueError("native lifecycle continued beyond P3")


def compare(root, p, digest):
    verify_inputs(p)
    for name, _ in CASES:
        completed(root, name, digest)
    report = analyze.compare(root, p)
    verify_inputs(p)
    write_new(root / "comparison.json", {**report, "profile": PROFILE, "protocol_sha256": digest,
        "implementation_commit": p["implementation_commit"],
        "complete_sha256": {name: sha256(root / name / "complete.json") for name, _ in CASES},
        "old_gate1": "not_reclassified", "independent_nonintervention_passed": False,
        "MCMC_long_term_causality_proven": False})


def run(args, p):
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    legacy.unlimited_host_cgroup(UNIT)
    root = args.output_root
    deadline = time.monotonic() + CONTRACT["suite_timeout_seconds"]
    with (root / "execution.once").open("x"):
        pass
    publish_status(root / "started.json", {"pid": os.getpid(), "proc_start": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
        "started_unix": time.time(), "protocol_sha256": args.protocol_sha256})
    def expired(signum, frame):
        raise TimeoutError("entire suite exceeded six hours")
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(CONTRACT["suite_timeout_seconds"])
    terminal = {"status": "failed", "exit_code": 1, "protocol_sha256": args.protocol_sha256}
    try:
        with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
            require_idle_gpu()
            verify_inputs(p)
            if shutil.disk_usage(PROJECT).free < p["disk_admission_bytes"]:
                raise ValueError("insufficient disk admission for all cases")
            parent = Path(p["parent_training"]) / "attempts/train-001"
            destination = root / "training/attempts/train-001"
            destination.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(parent / "attempt.json", destination / "attempt.json")
            shutil.copytree(parent / "checkpoints/iteration_000012000", destination / "checkpoints/iteration_000012000")
            parent_metadata(read(fixed.SOURCE / "protocol.json"), root / "training")
            for name, _ in CASES:
                if time.monotonic() >= deadline:
                    raise TimeoutError("suite deadline")
                if name == "native-1":
                    gate = analyze.same_gate(root)
                    write_new(root / "same-gate.json", gate)
                    if not gate["passed"]:
                        terminal.update(status="downstream_repeatability_blocked", exit_code=2)
                        raise ValueError("same B1 downstream gate failed; later cases not run")
                subprocess.run(["nvidia-smi", "-L"], check=True)
                require_idle_gpu()
                stage = root / name
                stage.mkdir(exist_ok=False)
                command = ["/usr/bin/env", "-u", "PYTHONPATH", sys.executable, "-m", MODULE,
                    "--worker", "--case", name, "--output-root", str(root), "--protocol-sha256", args.protocol_sha256,
                    "--authorize-execution", "--lease-fd", str(lease.fileno())]
                monitor = CaseMonitor(stage, name, deadline)
                legacy.run_stage(stage, "case", command, cwd=PROJECT, monitor=monitor,
                                 pass_fds=(lease.fileno(),), host_group=None)
                parent_metadata(read(fixed.SOURCE / "protocol.json"), root / "training")
                files = [stage / "case.exit.json", *stage.glob("worker-rank-*.json"), *stage.glob("phases-rank-*.jsonl")]
                write_new(stage / "complete.json", {"status": "probe_complete", "case": name,
                    "protocol_sha256": args.protocol_sha256, "resources": monitor.record(),
                    "files": {f.name: sha256(f) for f in files}})
            verify_inputs(p)
        compare(root, p, args.protocol_sha256)
        terminal.update(status="completed", exit_code=0, comparison_sha256=sha256(root / "comparison.json"))
    except BaseException as exc:
        terminal["error"] = f"{type(exc).__name__}: {exc}"
        if terminal["exit_code"] == 2:
            raise SystemExit(2) from exc
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        publish_status(root / "execution.json", {**terminal, "ended_unix": time.time()})


def monitor(root, digest):
    start = time.monotonic()
    while time.monotonic() - start < CONTRACT["suite_timeout_seconds"] + 300:
        if (root / "execution.json").is_file():
            end = read(root / "execution.json")
            if end["protocol_sha256"] != digest:
                raise ValueError("monitor protocol differs")
            expected_codes = {"completed": 0, "failed": 1, "downstream_repeatability_blocked": 2}
            if end.get("status") not in expected_codes or end["exit_code"] != expected_codes[end["status"]]:
                raise ValueError("invalid terminal state")
            if end["status"] == "completed" and sha256(snapshots.safe_file(root, "comparison.json")) != end["comparison_sha256"]:
                raise ValueError("terminal comparison changed")
            print(json.dumps(end, ensure_ascii=False), flush=True)
            return end["exit_code"]
        if (root / "started.json").is_file():
            s = read(root / "started.json")
            if s["protocol_sha256"] != digest:
                raise ValueError("monitor start protocol differs")
            proc = Path(f"/proc/{s['pid']}/stat")
            try:
                live = proc.read_text().split()[21] == s["proc_start"]
            except FileNotFoundError:
                live = False
            if not live:
                # Terminal write precedes process exit; recheck after the liveness query.
                if (root / "execution.json").is_file():
                    continue
                raise RuntimeError("execution disappeared without terminal record")
        elif time.monotonic() - start > 300:
            raise RuntimeError("execution did not start")
        done = [name for name, _ in CASES if (root / name / "complete.json").is_file()]
        print(json.dumps({"completed_cases": done, "status": "waiting"}), flush=True)
        time.sleep(30)
    raise TimeoutError("monitor deadline; execution was not stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "worker", "compare", "monitor"):
        modes.add_argument("--" + name, action="store_true")
    for name in ("prepare", "execution", "comparison"):
        parser.add_argument("--authorize-" + name, action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--case", choices=[name for name, _ in CASES])
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case is not None and not args.worker:
        parser.error("--case is worker-only; --run executes the frozen suite")
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅指定远端可执行；本机仅允许help与NoTorch测试")
    authorized = (args.authorize_prepare if args.prepare else args.authorize_comparison if args.compare
                  else True if args.monitor else args.authorize_execution and (not args.run or args.authorize_comparison))
    if not authorized:
        raise SystemExit("未获显式阶段授权")
    args.output_root = output_root(args.output_root, args.prepare)
    if args.prepare:
        prepare(args.output_root)
        return
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.monitor:
        raise SystemExit(monitor(args.output_root, args.protocol_sha256))
    if args.run:
        run(args, p)
    elif args.compare:
        compare(args.output_root, p, args.protocol_sha256)
    else:
        legacy.unlimited_host_cgroup(UNIT)
        if args.case is None or args.lease_fd is None or os.readlink(f"/proc/self/fd/{args.lease_fd}") != str(PROJECT / "outputs/.gpu.lock"):
            raise ValueError("worker needs frozen case and inherited GPU lease")
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from gsplat.distributed import cli
        if torch.cuda.device_count() != 2:
            raise ValueError("exactly two GPUs required")
        cli(worker, args, verbose=True)


if __name__ == "__main__":
    main()
