#!/usr/bin/env python3
"""Remote-only repeated 3M-Gaussian Validation under the memory-v2 budget."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import (
    TrainingMonitor, memory_telemetry, quality_cgroup, record_training_phase, run_stage, write_json,
)
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from run_mcmc_stability_ablation import HOST_POLICY, LIMITS, PROJECT, code_hash


UNIT = "image3d-mcmc-memory-gate-20261013-v1.service"


def worker(local_rank, rank, world_size, args):
    import torch

    from image3d_scenegraph.gaussian.model_io import load_model_snapshot
    from image3d_scenegraph.gaussian.runtime import load_training_views
    from image3d_scenegraph.gaussian.trainer import _checkpoint_rank_bytes, evaluate_views

    assert world_size == 2 and local_rank == rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    source = args.source_job
    checkpoint = source / "gaussian/attempts/train-001/checkpoints/iteration_000030000"
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    component = checkpoint / metadata["files"]["model"]["path"]
    assert sha256_file(component) == metadata["files"]["model"]["sha256"]
    contract = json.loads((source / "gaussian/replay/dataset.json").read_text())
    config = json.loads((source / "gaussian/preparation/train-001/effective_config.json").read_text())["effective_config"]
    assert config["strategy"]["name"] == "mcmc_v1" and config["resolution"]["longest_edge"] == 2048
    with memory_telemetry(args.output_dir / "memory", local_rank, rank):
        model = load_model_snapshot(_checkpoint_rank_bytes(component.read_bytes(), rank, world_size), device)
        gc.collect()
        assert model.count == 1500000 and model.max_sh_degree == 3
        optimizers = model.optimizers(config["learning_rate"])
        for name, parameter in model.params.items():
            optimizers[name].state[parameter] = {
                "step": torch.tensor(10000.), "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        views = {split: load_training_views(contract, source / "gaussian/replay", split=split,
                 longest_edge=2048, device=torch.device("cpu")) for split in ("train", "validation")}
        assert len(views["train"]) == 2808 and len(views["validation"]) == 351
        passes = []
        reference = None
        for repeat in range(3):
            record_training_phase("cache_warmup", repeat)
            for index in range(96):
                views["train"][index]
            cached = views["train"].cached_bytes
            assert cached > 0
            views["train"].clear_cache()
            assert views["train"].cached_bytes == 0 and len(views["train"]) == 2808
            rng = torch.cuda.get_rng_state(device).clone()
            started = time.monotonic()
            result = evaluate_views(model, views["validation"], config, distributed=True,
                                    progress_events=[{"iteration": repeat}])
            elapsed = time.monotonic() - started
            assert result["status"] == "complete" and result["successful_views"] == 351
            assert elapsed < 600 and model.count == 1500000
            assert torch.equal(rng, torch.cuda.get_rng_state(device))
            assert views["validation"].cached_bytes == 0 and len(views["validation"]) == 351
            metrics = [view["psnr"] for view in result["per_view"]]
            if reference is None:
                reference = metrics
            maximum_delta = max(abs(a - b) for a, b in zip(metrics, reference))
            assert maximum_delta < 1e-4
            passes.append({"elapsed_seconds": elapsed, "successful_views": 351,
                           "train_cache_bytes_released": cached,
                           "maximum_per_view_psnr_delta": maximum_delta})
        record_training_phase("complete", 3)
        write_json(args.output_dir / f"rank-{rank}.json", {
            "rank": rank, "status": "passed", "passes": passes,
            "source_model_component_sha256": metadata["files"]["model"]["sha256"],
            "training_code_hash": code_hash(), "local_gaussians": model.count,
            "optimizer_states_allocated": True, "optimizer_steps": 0,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.source_job = args.source_job.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.source_job.name != "20261010_172759_d4dda4b3":
        raise ValueError("memory gate uses only the audited old apartment MCMC checkpoint")
    if args.worker:
        from gsplat.distributed import cli
        cli(worker, args, verbose=True)
        return
    group = quality_cgroup(policy=HOST_POLICY, unit=UNIT)
    if (group / "memory.high").read_text().strip() != "max":
        raise ValueError("the memory gate must not throttle at memory.high")
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        monitor = TrainingMonitor(args.output_dir / "unused-progress.jsonl", args.output_dir / "memory",
                                  updates=0, main_stage=False, limits=LIMITS)
        run_stage(args.output_dir, "validation", [sys.executable, __file__, "--worker",
                  "--source-job", str(args.source_job), "--output-dir", str(args.output_dir)],
                  cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),),
                  host_group=group, host_policy=HOST_POLICY)
        records = [json.loads((args.output_dir / f"rank-{rank}.json").read_text()) for rank in range(2)]
        assert all(r["status"] == "passed" and len(r["passes"]) == 3 for r in records)
        samples = [json.loads(line) for line in (args.output_dir / "validation.memory.jsonl").read_text().splitlines()]
        peak = max(sample["task_current_bytes"] for sample in samples)
        assert peak < HOST_POLICY["stop_current_bytes"]
        assert all(sample["memory_events"].get("high", 0) == 0 for sample in samples)
        write_json(args.output_dir / "summary.json", {
            "status": "passed", "training_code_hash": code_hash(), "host_policy": HOST_POLICY,
            "global_gaussians": 3000000, "validation_views": 351, "passes": 3,
            "peak_sampled_task_bytes": peak, "ranks": records,
            "test_rgb": "not_loaded", "scene_training": "not_run",
            "scope": "Historical 30k model, allocated Adam states and repeated 2048px Validation; not an exact replay of the failed 10k heap or a quality comparison.",
        })


if __name__ == "__main__":
    main()
