"""E007 phase A only: offline E006 masks, no training and no phase-B entry."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from ..e006_connected_propagation import run as connected
from . import masks

legacy, snapshots = connected.legacy, connected.snapshots
read, write_new, sha256 = connected.read, connected.write_new, connected.sha256
PROJECT = connected.PROJECT
REL = "experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e007_observer_admission"
MODULE = REL.replace("/", ".") + ".run"
RUNS = PROJECT / REL / "runs"
SOURCE = connected.RUNS / "connected-20261023-v1"
SOURCE_SHA = "f09237d0163ea4f0818e3ae952bc08cdf51abddb09ec34c36e43333dc58b3f35"
COMPARISON_SHA = "e34a334677a84c61de8b840fb5bc763b583d70bb25caa2ae01c4e833442c8c3d"
UNIT = "image3d-observer-masks-v1.service"
FILES = tuple(REL + "/" + name for name in (
    "run.py", "masks.py", "tests/test_masks.py", "tests/test_masks_remote.py"))
DEPENDENCIES = tuple(dict.fromkeys((*connected.fixed.DEPENDENCIES, *connected.FILES, *FILES)))
CONTRACT = {"profile": "mcmc_observer_admission_masks_v1", "phase": "A_only",
    "source": str(SOURCE), "source_protocol_sha256": SOURCE_SHA,
    "source_comparison_sha256": COMPARISON_SHA, "cases": [name for name, _ in connected.CASES],
    "ranks": [0, 1], "stages": ["P0", "P3"], "owner_size": 1500000,
    "methods": list(masks.METHODS), "death_threshold": .005, "render_cutoff": 1 / 255,
    "cuda_sigmoid_calls": 40, "cpu_sigmoid_calls": 40,
    "new_training_updates": 0, "rasterizer_forwards": 0, "backwards": 0, "strategy_calls": 0,
    "suite_timeout_seconds": 3600, "max_output_bytes": 1024**3,
    "disk_admission_bytes": 9 * 1024**3, "max_per_rank_reserved_bytes": 18 * 1024**3,
    "host_limits": None, "stage_B": "not_implemented_not_authorized",
    "preflight_budget": {"cuda_sigmoid_calls": 4, "rasterizer_calls": 0, "backwards": 0, "optimizer_steps": 0}}


def output_root(path, preparing=False):
    path = Path(path).absolute()
    try:
        relative = path.relative_to(RUNS)
    except ValueError as exc:
        raise ValueError("output must be inside E007 runs") from exc
    result = snapshots.safe_file(RUNS, relative)
    if result == RUNS or (preparing and result.exists()):
        raise ValueError("new independent output directory required")
    return result


def execution_commit():
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *DEPENDENCIES], cwd=PROJECT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *DEPENDENCIES], cwd=PROJECT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()


def source_inventory():
    bound = {}

    def checked(relative, expected=None):
        path = snapshots.safe_file(SOURCE, relative)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("source identity changed: " + str(relative))
        bound[str(path)] = digest
        return read(path)

    protocol = checked("protocol.json", SOURCE_SHA)
    comparison = checked("comparison.json", COMPARISON_SHA)
    if (any(protocol.get(k) != v for k, v in connected.CONTRACT.items())
            or comparison["protocol_sha256"] != SOURCE_SHA or not comparison["same_input_gate_passed"]
            or comparison["status"] != "analysis_complete"
            or comparison["implementation_commit"] != protocol["implementation_commit"]):
        raise ValueError("E006 source contract not accepted")
    cases, runtime = {}, {}
    for name, trial in connected.CASES:
        complete = checked(f"{name}/complete.json", comparison["complete_sha256"][name])
        if (complete["case"], complete["status"], complete["protocol_sha256"]) != (name, "probe_complete", SOURCE_SHA):
            raise ValueError("source case is incomplete")
        end = checked(f"{name}/case.exit.json", complete["files"]["case.exit.json"])
        if end["returncode"] != 0 or end["resource_failure"] is not None:
            raise ValueError("source case failed")
        cases[name] = {}
        for rank in (0, 1):
            worker_name = f"worker-rank-{rank}.json"
            worker = checked(f"{name}/{worker_name}", complete["files"][worker_name])
            if (worker["rank"] != rank or worker["iteration"] != 12001 or worker["new_updates"] != 1
                    or worker["calls"]["strategy"] != 0 or worker["validation_evaluated"]
                    or worker["original_backward_calls"] != int(trial is None)):
                raise ValueError("source topology, identity or training budget differs")
            cases[name][str(rank)] = {}
            for stage in CONTRACT["stages"]:
                relative = f"{name}/rank-{rank}/{stage}/manifest.json"
                digest = worker["manifests"][stage]
                record = checked(relative, digest)
                if snapshots.digest_state(record["tree"]) != record["tree_sha256"]:
                    raise ValueError("snapshot tree changed")
                leaves = snapshots.validate_stage_schema(record, stage)
                leaf = leaves[("parameters", "opacities")]
                if leaf["dtype"] != "float32" or leaf["shape"] != [CONTRACT["owner_size"]]:
                    raise ValueError("frozen opacity shape/dtype differs")
                file = snapshots.safe_file(SOURCE / name / f"rank-{rank}" / stage, leaf["file"])
                if file.stat().st_size != leaf["bytes"] or sha256(file) != leaf["sha256"]:
                    raise ValueError("opacity input changed")
                bound[str(file)] = leaf["sha256"]
                cases[name][str(rank)][stage] = digest
                if stage == "P0":
                    env = {k: leaves[("runtime", k)]["value"] for k in ("torch", "cuda", "gpu", "rank")}
                    if env["rank"] != rank or (str(rank) in runtime and runtime[str(rank)] != env):
                        raise ValueError("source runtime differs between cases")
                    runtime[str(rank)] = env
    counts = comparison["point_and_opacity_summary"]
    return {"cases": cases, "runtime": runtime, "owner_size": CONTRACT["owner_size"],
            "counts": {name: {str(r): {k: counts[name][str(r)][k] for k in masks.DEATH_KEYS}
                              for r in (0, 1)} for name, _ in connected.CASES},
            "bound_files": bound, "verification_scope": "signed_P0_P3_manifests_and_opacity_only_not_all_E006_arrays"}


def prepare(root):
    commit = execution_commit()
    inventory = source_inventory()
    if shutil.disk_usage(PROJECT).free < CONTRACT["disk_admission_bytes"]:
        raise ValueError("insufficient disk admission")
    p = {**CONTRACT, "implementation_commit": commit, "source_inventory": inventory,
         "code_sha256": {n: sha256(PROJECT / n) for n in DEPENDENCIES}, "status": "prepared_not_executed"}
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "protocol.json", p)


def protocol_at(root, digest):
    if not digest or sha256(snapshots.safe_file(root, "protocol.json")) != digest:
        raise ValueError("explicit protocol SHA required")
    p = read(root / "protocol.json")
    if any(p.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("phase-A contract differs")
    if set(p["code_sha256"]) != set(DEPENDENCIES):
        raise ValueError("missing execution dependency")
    for name, expected in p["code_sha256"].items():
        if sha256(snapshots.safe_file(PROJECT, name)) != expected:
            raise ValueError("execution dependency changed")
    return p


def verify_inputs(p):
    if execution_commit() != p["implementation_commit"] or source_inventory() != p["source_inventory"]:
        raise ValueError("implementation or source identity changed")


def directory_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def reserve(root, extra):
    # Leave space for final JSON and failure evidence; the monitor also checks actual usage.
    if extra < 0 or directory_bytes(root) + extra + 2 * 1024**2 > CONTRACT["max_output_bytes"]:
        raise ValueError("phase-A output budget exceeded")
    if shutil.disk_usage(root).free < extra + 4 * 1024**3:
        raise ValueError("disk reserve would be breached")


def verify_result(root, digest):
    result = read(snapshots.safe_file(root, "comparison.json"))
    if (result["status"] != "analysis_complete" or result["protocol_sha256"] != digest
            or result["stage_B_authorized"] or result["long_term_causality_proven"]):
        raise ValueError("invalid phase-A result")
    rows = [row for owner in result["pairs"].values() for group in owner.values()
            for pair in group.values() for row in pair.values()]
    rows += [row for pair in result["method_comparison"].values() for row in pair.values()]
    if len(rows) != 360 or {r["path"] for r in rows} != set(result["files"]):
        raise ValueError("incomplete or duplicate result references")
    for row in rows:
        if result["files"][row["path"]] != {k: row[k] for k in ("bytes", "sha256")}:
            raise ValueError("bitset index differs")
        masks.verify_bitset(root, row)
    if len(result["files"]) != 360 or directory_bytes(root) > CONTRACT["max_output_bytes"]:
        raise ValueError("incomplete masks or output budget exceeded")
    return result


class AnalysisMonitor:
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline
        self.peaks = [0, 0]

    def __call__(self, elapsed, *, final=False):
        if time.monotonic() >= self.deadline:
            return "phase_A_timeout"
        if directory_bytes(self.root) > CONTRACT["max_output_bytes"]:
            return "phase_A_output_limit"
        for rank in (0, 1):
            file = self.root / "memory" / f"rank-{rank}.json"
            if file.exists():
                row = read(file)
                self.peaks[rank] = max(self.peaks[rank], row["peak_reserved_bytes"])
                if self.peaks[rank] > CONTRACT["max_per_rank_reserved_bytes"]:
                    return "gpu_reserved_limit"
                if final and not row["finished"]:
                    return "memory_telemetry_incomplete"
            elif final:
                return "missing_memory_telemetry"

    def record(self):
        return {"per_rank_peak_reserved_bytes": self.peaks}


def worker(args, p):
    if args.lease_fd is None or os.readlink(f"/proc/self/fd/{args.lease_fd}") != str(PROJECT / "outputs/.gpu.lock"):
        raise ValueError("inherited GPU lease required")
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    legacy.unlimited_host_cgroup(UNIT)
    verify_inputs(p)
    import torch
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry
    if torch.cuda.device_count() != 2:
        raise ValueError("exactly two GPUs required")
    for rank in (0, 1):
        expected = p["source_inventory"]["runtime"][str(rank)]
        if {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(rank), "rank": rank} != expected:
            raise ValueError("Torch/CUDA/GPU differs from frozen E006 runtime")
    root = args.output_root
    with ExitStack() as stack:
        for rank in (0, 1):
            stack.enter_context(memory_telemetry(root / "memory", rank, rank))
        def progress(value):
            reserve(root, 4096)
            with (root / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)
        result = masks.analyze(SOURCE, root, p["source_inventory"], lambda n: reserve(root, n), progress)
        verify_inputs(p)
        files = {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in root.glob("*.bits.gz")}
        reserve(root, len(json.dumps(result).encode()) + 128 * 1024)
        write_new(root / "comparison.json", {**result, "protocol_sha256": args.protocol_sha256,
            "implementation_commit": p["implementation_commit"], "source_protocol_sha256": SOURCE_SHA,
            "source_comparison_sha256": COMPARISON_SHA, "files": files,
            "runtime": p["source_inventory"]["runtime"], "budget": {k: CONTRACT[k] for k in (
                "cuda_sigmoid_calls", "cpu_sigmoid_calls", "new_training_updates", "rasterizer_forwards", "backwards", "strategy_calls")}})
        verify_result(root, args.protocol_sha256)


def run(args, p):
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    legacy.unlimited_host_cgroup(UNIT)
    root = args.output_root
    with (root / "execution.once").open("x"):
        pass
    connected.publish_status(root / "started.json", {"pid": os.getpid(),
        "proc_start": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
        "protocol_sha256": args.protocol_sha256, "started_unix": time.time()})
    terminal = {"status": "failed", "exit_code": 1, "protocol_sha256": args.protocol_sha256}
    def expired(signum, frame):
        raise TimeoutError("phase A exceeded one hour")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(CONTRACT["suite_timeout_seconds"])
    try:
        with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
            require_idle_gpu()
            verify_inputs(p)
            if shutil.disk_usage(PROJECT).free < CONTRACT["disk_admission_bytes"]:
                raise ValueError("insufficient disk admission")
            subprocess.run(["nvidia-smi", "-L"], check=True)
            monitor = AnalysisMonitor(root, time.monotonic() + CONTRACT["suite_timeout_seconds"])
            command = ["/usr/bin/env", "-u", "PYTHONPATH", sys.executable, "-m", MODULE,
                       "--worker", "--output-root", str(root), "--protocol-sha256", args.protocol_sha256,
                       "--authorize-analysis", "--lease-fd", str(lease.fileno())]
            legacy.run_stage(root, "analysis", command, cwd=PROJECT, monitor=monitor,
                             pass_fds=(lease.fileno(),), host_group=None)
            verify_result(root, args.protocol_sha256)
            verify_inputs(p)
        terminal.update(status="completed", exit_code=0, comparison_sha256=sha256(root / "comparison.json"),
                        analysis_exit_sha256=sha256(root / "analysis.exit.json"))
    except BaseException as exc:
        terminal["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        connected.publish_status(root / "execution.json", {**terminal, "ended_unix": time.time()})


def monitor(root, digest):
    started = time.monotonic()
    while time.monotonic() - started < CONTRACT["suite_timeout_seconds"] + 300:
        if (root / "execution.json").exists():
            end = read(root / "execution.json")
            if end["protocol_sha256"] != digest or (end["status"], end["exit_code"]) not in (("completed", 0), ("failed", 1)):
                raise ValueError("monitor terminal identity differs")
            if end["exit_code"] == 0:
                verify_result(root, digest)
                if sha256(root / "comparison.json") != end["comparison_sha256"] or sha256(root / "analysis.exit.json") != end["analysis_exit_sha256"]:
                    raise ValueError("terminal evidence changed")
            print(json.dumps(end), flush=True)
            return end["exit_code"]
        if (root / "started.json").exists():
            record = read(root / "started.json")
            if record["protocol_sha256"] != digest:
                raise ValueError("monitor start identity differs")
            proc = Path(f"/proc/{record['pid']}/stat")
            try:
                live = proc.read_text().split()[21] == record["proc_start"]
            except FileNotFoundError:
                live = False
            if not live and not (root / "execution.json").exists():
                raise RuntimeError("execution disappeared without terminal record")
        elif time.monotonic() - started > 300:
            raise RuntimeError("execution did not start")
        path = root / "progress.jsonl"
        print(path.read_text().splitlines()[-1] if path.exists() and path.stat().st_size else "waiting_for_analysis", flush=True)
        time.sleep(30)
    raise TimeoutError("monitor deadline; analysis was not stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "worker", "monitor"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--authorize-prepare", action="store_true")
    parser.add_argument("--authorize-analysis", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅指定远端可执行；本机仅help和NoTorch检查")
    if not (args.authorize_prepare if args.prepare else True if args.monitor else args.authorize_analysis):
        raise SystemExit("未获显式阶段A授权")
    args.output_root = output_root(args.output_root, args.prepare)
    if args.prepare:
        prepare(args.output_root)
        return
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.monitor:
        raise SystemExit(monitor(args.output_root, args.protocol_sha256))
    if args.worker:
        worker(args, p)
    else:
        run(args, p)


if __name__ == "__main__":
    main()
