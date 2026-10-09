#!/usr/bin/env python3
"""Bounded two-rank checkpoint memory profiling, controlled outside the task cgroup."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time

from image3d_scenegraph.gaussian.absgrad_resources import available_host_bytes, write_json, process_memory
from image3d_scenegraph.gpu_lease import FileLease

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3




def worker(local_rank, rank, world_size, args):
    import torch
    from image3d_scenegraph.gaussian import trainer as t
    from image3d_scenegraph.gaussian.checkpoint import create_attempt
    from image3d_scenegraph.gaussian.config import resolve_internal_config
    from image3d_scenegraph.gaussian.model import GaussianModel

    if world_size != 2 or local_rank != rank:
        raise ValueError("exactly two local ranks required")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    t.seed_training(20260729)
    case = args.output_dir
    write_json(case / f"rank-{rank}.pid.json", {"pid": os.getpid()})
    events = (case / f"rank-{rank}.events.jsonl").open("x", buffering=1)
    stack = []

    def emit(event):
        torch.cuda.synchronize(device)
        row = {"time": time.monotonic(), "rank": rank, "event": event, "stack": list(stack),
            **process_memory(os.getpid()), "cuda_allocated": torch.cuda.memory_allocated(device),
            "cuda_reserved": torch.cuda.memory_reserved(device),
            "cuda_peak_reserved": torch.cuda.max_memory_reserved(device)}
        events.write(json.dumps(row) + "\n")
        temp = case / f"rank-{rank}.phase.tmp"
        temp.write_text(json.dumps(row))
        os.replace(temp, case / f"rank-{rank}.phase.json")
        if row["cuda_peak_reserved"] > 18 * GIB:
            raise RuntimeError("profiling exceeds 18 GiB per-rank reserved")

    @contextmanager
    def phase(name):
        stack.append(name)
        emit("enter")
        try:
            yield
        finally:
            emit("exit")
            stack.pop()

    def instrument(module, name, label=None):
        original = getattr(module, name)
        @functools.wraps(original)
        def wrapped(*a, **kw):
            with phase(label or name):
                return original(*a, **kw)
        setattr(module, name, wrapped)

    for name in ("_model_bytes", "_checkpoint_state", "_pack_checkpoint_shards",
                 "_write_latest_distributed_checkpoint", "_write_latest_checkpoint",
                 "_merge_model_shards", "load_checkpoint", "write_checkpoint",
                 "_write_streaming_checkpoint", "_pack_checkpoint_files", "_save_torch_file", "save_model_snapshot"):
        if hasattr(t, name):
            instrument(t, name)
    instrument(torch.distributed, "gather_object", "gather_object")
    original_bytes = t._torch_bytes
    def measured_bytes(value):
        keys = set(value) if isinstance(value, dict) else set()
        label = "serialize_optimizer" if "means" in keys else "serialize_" + ("model" if "state_dict" in keys else "other")
        with phase(label):
            return original_bytes(value)
    t._torch_bytes = measured_bytes
    count = len(range(rank, args.count, 2))
    with phase("construct_gpu_state"):
        model = GaussianModel.from_points(torch.full((count, 3), 0.1, device=device),
            torch.full((count, 3), 0.5, device=device), torch.full((count,), 0.01, device=device),
            initial_opacity=0.1, max_sh_degree=3)
        config = resolve_internal_config("absgrad_ablation_v1").effective_config
        optimizers = model.optimizers(config["learning_rate"])
        for name, optimizer in optimizers.items():
            param = model.params[name]
            optimizer.state[param] = {"step": torch.tensor(1.0),
                "exp_avg": torch.full_like(param, 0.001), "exp_avg_sq": torch.full_like(param, 0.002)}
        strategy = {"grad2d": torch.zeros(count, device=device), "count": torch.ones(count, device=device),
                    "scene_scale": 1.0}
        # Representative bounded cache, not a claim about the failed job's live image set.
        cached_views = [bytearray(512 * 1024**2)]
        history = [{"iteration": 5418, "loss": 0.1}]
        provenance = t.training_provenance(dataset_hash="a" * 64, effective_config_hash="b" * 64, world_size=2)
        if rank == 0:
            create_attempt(case / "training", attempt_id="train-001", kind="fresh", provenance=provenance)
    torch.distributed.barrier()
    emit("steady_state")
    with phase("release_views"):
        cached_views.clear()
        import gc
        gc.collect()
    with phase(args.path + "_save"):
        if args.path == "cancel" and hasattr(t, "_write_streaming_checkpoint"):
            emit("cancel_no_checkpoint")
        elif hasattr(t, "_write_streaming_checkpoint"):
            final_record = t._write_streaming_checkpoint(case / "training", attempt_id="train-001",
                iteration=5418, provenance=provenance, model=model, optimizers=optimizers,
                strategy_state=strategy, camera_order=list(range(3016)), camera_cursor=7, history=history,
                world_rank=rank, world_size=2)
        else:
            t._write_latest_distributed_checkpoint(case / "training", attempt_id="train-001",
                iteration=5418, purpose="periodic" if args.path == "cancel" else "final",
                validation_score=None, provenance=provenance,
                state=t._checkpoint_state(model, optimizers, strategy, list(range(3016)), 7, history, 5418),
                world_rank=rank, world_size=2)
            final_record = None
    if args.count == 13 and args.path == "final" and hasattr(t, "_write_streaming_checkpoint"):
        import numpy as np
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a.cpu(), b.cpu())
            elif isinstance(a, np.ndarray):
                assert np.array_equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                assert a == b
        expected = t._checkpoint_state(model, optimizers, strategy, list(range(3016)), 7, history, 5418)
        actual = t.load_checkpoint(case / "training", "train-001", 5418).state
        for name in ("model", "optimizer", "densification", "rng"):
            shard = t._checkpoint_rank_bytes(getattr(actual, name), rank, 2)
            equal(t._torch_load(shard, torch.device("cpu")), t._torch_load(getattr(expected, name), torch.device("cpu")))
        assert actual.scheduler == expected.scheduler and actual.metric_history == expected.metric_history
        write_json(case / f"rank-{rank}.equivalence.json", {"all_components_equal": True, "rng_equal": True})
    if args.path == "final":
        if rank == 0 and final_record is None:
            with phase("terminal_reload"):
                retained_checkpoint = t.load_checkpoint(case / "training", "train-001", 5418,
                                                        expected_provenance=provenance)
                assert retained_checkpoint.record.iteration == 5418
        with phase("best_snapshot"):
            t.save_model_snapshot(model, case / f"model-rank-{rank}.pt")
        torch.distributed.barrier()
        if rank == 0:
            with phase("terminal_merge"):
                merged = t._merge_model_shards([case / f"model-rank-{i}.pt" for i in range(2)], case / "model.pt")
                assert merged.count == args.count
    emit("complete")
    events.close()
    torch.distributed.barrier()


def monitor_case(case, unit, command):
    group = Path("/sys/fs/cgroup/system.slice") / (unit + ".service")
    started = time.monotonic()
    with (case / "service.log").open("x") as log, (case / "memory.jsonl").open("x", buffering=1) as samples:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        peak = {"current": 0, "anon": 0, "file": 0, "kernel": 0}
        reason = None
        try:
            while process.poll() is None:
                row = {"time": time.monotonic(), "available": available_host_bytes(), "ranks": []}
                try:
                    row["current"] = int((group / "memory.current").read_text())
                    row["cgroup_peak"] = int((group / "memory.peak").read_text()) if (group / "memory.peak").exists() else None
                    row["stat"] = {k: int(v) for k, v in (line.split() for line in (group / "memory.stat").read_text().splitlines())}
                    row["events"] = (group / "memory.events").read_text()
                    for key in peak:
                        peak[key] = max(peak[key], row["current"] if key == "current" else row["stat"].get(key, 0))
                except FileNotFoundError:
                    pass
                for rank in range(2):
                    path = case / f"rank-{rank}.phase.json"
                    if path.exists():
                        phase = json.loads(path.read_text())
                        row["ranks"].append({"rank": rank, "phase": phase["stack"],
                            "cuda_peak_reserved": phase["cuda_peak_reserved"], **process_memory(phase["pid"])})
                samples.write(json.dumps(row) + "\n")
                if row["available"] < 6 * GIB or time.monotonic() - started > 600:
                    reason = "host_below_6_gib" if row["available"] < 6 * GIB else "case_timeout"
                    write_json(case / "stop-request.json", {"reason": reason, "sample": row})
                    subprocess.run(["systemctl", "stop", unit + ".service"], check=True, timeout=20)
                    process.wait(timeout=30)
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                subprocess.run(["systemctl", "stop", unit + ".service"], check=True, timeout=20)
                process.wait(timeout=30)
        status = subprocess.run(["systemctl", "show", unit + ".service", "-p", "Result", "-p", "ExecMainCode",
                                 "-p", "ExecMainStatus"], capture_output=True, text=True, check=True).stdout
        result = {"returncode": process.returncode, "reason": reason, "sampled_peak_bytes": peak,
                  "elapsed_seconds": time.monotonic() - started, "systemd": status}
        write_json(case / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--counts", nargs="+", type=int, default=[500000, 1500000, 3000000])
    parser.add_argument("--paths", nargs="+", choices=("cancel", "final"), default=["final", "cancel"])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--count", type=int)
    parser.add_argument("--path", choices=("cancel", "final"))
    parser.add_argument("--unit")
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or ROOT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
        raise ValueError("profiling is restricted to the authorized remote repository")
    if args.worker:
        expected = f"0::/system.slice/{args.unit}.service"
        if expected not in Path("/proc/self/cgroup").read_text().splitlines():
            raise ValueError("profiling worker must be isolated from the panel")
        group = Path("/sys/fs/cgroup/system.slice") / (args.unit + ".service")
        if int((group / "memory.max").read_text()) != 12 * GIB or int((group / "memory.swap.max").read_text()) != 0:
            raise ValueError("profiling requires 12 GiB cgroup with no swap")
        from gsplat.distributed import cli
        cli(worker, args, verbose=True)
        return
    if args.output_dir.exists() or args.output_dir.is_symlink() or not args.output_dir.resolve().is_relative_to(ROOT / "outputs/analysis"):
        raise ValueError("profiling requires a new analysis directory")
    if any(n < 1 or n > 6000000 for n in args.counts):
        raise ValueError("profiling count outside approved bound")
    with FileLease(ROOT / "outputs/.gpu.lock"):
        inventory = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        if inventory.count("NVIDIA L2") != 2 or subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise ValueError("profiling requires two idle L2 GPUs")
        if shutil.disk_usage(ROOT).free < 20 * GIB:
            raise ValueError("profiling requires 20 GiB free disk")
        if available_host_bytes() < 18 * GIB:
            raise ValueError("profiling requires 18 GiB available host RAM")
        args.output_dir.mkdir()
        write_json(args.output_dir / "protocol.json", {"code": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "counts": args.counts, "paths": args.paths, "sample_seconds": 0.2, "cgroup_limit": 12 * GIB,
            "admission": 18 * GIB, "host_floor": 6 * GIB, "representative_only": True, "test_rgb": "not_loaded"})
        records = []
        for count in args.counts:
            for path in args.paths:
                if available_host_bytes() < 18 * GIB:
                    write_json(args.output_dir / "admission-blocked.json", {"available": available_host_bytes(), "count": count, "path": path})
                    raise RuntimeError("next diagnostic case lacks 18 GiB RAM")
                case = args.output_dir / f"{count}-{path}"
                case.mkdir()
                unit = f"image3d-save-{args.output_dir.name[-24:]}-{count}-{path}"
                command = ["systemd-run", "--wait", "--pipe", "--unit=" + unit,
                    "--property=MemoryMax=12G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
                    "--property=KillMode=control-group", "--property=OOMScoreAdjust=500", "--property=TimeoutStopSec=5",
                    "--working-directory=" + str(ROOT), "--setenv=PYTHONPATH=" + str(ROOT / "src") + ":" + str(ROOT / "scripts"),
                    "--setenv=CUDA_VISIBLE_DEVICES=0,1", "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
                    sys.executable, str(Path(__file__).resolve()), "--worker", "--output-dir", str(case),
                    "--count", str(count), "--path", path, "--unit", unit]
                result = monitor_case(case, unit, command)
                records.append({"count": count, "path": path, **result})
                print(json.dumps(records[-1]), flush=True)
        write_json(args.output_dir / "summary.json", {"cases": records, "real_scene_training": "not_run"})


if __name__ == "__main__":
    main()
