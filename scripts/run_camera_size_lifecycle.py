#!/usr/bin/env python3
"""Bounded native MCMC lifecycle for original and target-camera-size environments."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import socket
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import TrainingMonitor, run_stage, write_json
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from scripts.prepare_gsplat_camera_sizes import verify_overlay
from scripts.run_mcmc_stability_ablation import PROJECT, LIMITS, unlimited_host_cgroup

UNIT = "image3d-camera-size-lifecycle-20261021-v1.service"
OVERLAY = PROJECT / "outputs/analysis/camera-size-repair-20261019-v1/camera-size-overlay"
OVERLAY_SHA = "fc3334aaf303ac1bebbfac12098fb2e3e306f0edf4d3b5527d030c6a03f28c42"


def lifecycle_config():
    from image3d_scenegraph.gaussian.config import resolve_internal_config
    return resolve_internal_config("mcmc_v1", {
        "iterations": 16, "resolution": {"longest_edge": 96},
        "sh_schedule": {"initial_degree": 3, "max_degree": 3, "increase_every_iterations": 1},
        "densification": {"start_iteration": 2, "end_iteration": 10, "every_iterations": 2},
        "opacity_reset": {"every_iterations": 16, "recovery_prune": {"window_iterations": 2}},
        "evaluation": {"validation_iterations": [6, 16]},
    })


def verify_pair(records):
    if set(records) != {"baseline", "repair"}:
        raise ValueError("both lifecycle environments are required")
    a, b = records["baseline"], records["repair"]
    for field in ("training_core", "dataset_hash", "config_hash", "initialization_sha256", "camera_sequence_sha256"):
        if a[field] != b[field]:
            raise ValueError(f"lifecycle identity differs: {field}")
    if a["environment_hash"] == b["environment_hash"]:
        raise ValueError("renderer environments must have distinct identities")
    if any(not r["passed"] or r["updates"] != 12 or r["nominal_iterations"] != 16 for r in records.values()):
        raise ValueError("incomplete lifecycle")


def worker(local_rank, rank, world_size, args):
    import numpy as np
    import torch
    from unittest.mock import patch
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    from image3d_scenegraph.gaussian.checkpoint import load_checkpoint, CheckpointContractError
    from image3d_scenegraph.gaussian.dataset import contract_hash, write_contract
    from image3d_scenegraph.gaussian.initialization import InitializationResult
    from image3d_scenegraph.gaussian.runtime import load_training_views
    from image3d_scenegraph.gaussian.training_math import exponential_learning_rate
    from scripts.smoke_gaussian_trainer import generate_scene
    from scripts.smoke_distributed_camera_sizes import tensor_hash

    assert world_size == 2 and local_rank == rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    root = args.output_root
    rank_dir = root / f"rank-{rank}"
    rank_dir.mkdir()
    with memory_telemetry(root / "memory", local_rank, rank), failure_record(rank_dir, args.arm, "native_lifecycle"):
        dataset = root.parent.parent / "synthetic-dataset"
        if rank == 0 and args.arm == "baseline":
            contract = generate_scene(dataset, image_sizes=[(64, 64), (96, 80)] * 6)
            contract["splits"] = {"train": [str(i) for i in range(1, 9)], "validation": ["9", "10"], "test": ["11", "12"]}
            contract["dataset_hash"] = contract_hash(contract)
            write_contract(dataset / "dataset.json", contract)
        torch.distributed.barrier()
        contract = json.loads((dataset / "dataset.json").read_text())
        cfg = lifecycle_config()
        initialization = InitializationResult(
            points=np.random.RandomState(7).uniform(-.15, .15, (60, 3)).astype(np.float32),
            colors=np.full((60, 3), 128, dtype=np.uint8), scales=np.full(60, .02, dtype=np.float32),
            diagnostics={"kind": "synthetic_camera_size_lifecycle"})
        provenance = trainer.training_provenance(dataset_hash=contract["dataset_hash"],
            effective_config_hash=cfg.effective_config_hash, world_size=2)
        observations = {}
        original_writer, original_merge = trainer._write_streaming_checkpoint, trainer._merge_model_shards

        def writer(*positional, **kwargs):
            expected = tensor_hash(dict(kwargs["model"].params.items()))
            rng = trainer._torch_load(trainer._rng_bytes(), torch.device("cpu"))
            record = original_writer(*positional, **kwargs)
            after = trainer._torch_load(trainer._rng_bytes(), torch.device("cpu"))
            assert rng["python"] == after["python"]
            assert np.array_equal(rng["numpy"][1], after["numpy"][1]) and rng["numpy"][2:] == after["numpy"][2:]
            assert torch.equal(rng["torch"], after["torch"])
            assert all(torch.equal(a, b) for a, b in zip(rng["cuda"], after["cuda"]))
            loaded = load_checkpoint(positional[0], "train-001", 12, expected_provenance=provenance)
            state = trainer._load_model(trainer._checkpoint_rank_bytes(loaded.state.model, rank, 2), device)
            assert tensor_hash(dict(state.params.items())) == expected
            payload = trainer._torch_load(trainer._checkpoint_rank_bytes(loaded.state.optimizer, rank, 2), device)
            for name, opt in kwargs["optimizers"].items():
                actual = payload[name]
                expected_opt = opt.state_dict()
                assert actual["param_groups"] == expected_opt["param_groups"]
                for index, values in expected_opt["state"].items():
                    assert set(actual["state"][index]) == set(values)
                    for field, value in values.items():
                        assert torch.equal(actual["state"][index][field].cpu(), value.cpu())
                    assert float(values["step"]) == 12
            dense = trainer._torch_load(trainer._checkpoint_rank_bytes(loaded.state.densification, rank, 2), device)
            assert dense["camera_order"] == kwargs["camera_order"] and dense["camera_cursor"] == kwargs["camera_cursor"]
            for key, value in kwargs["strategy_state"].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(dense["strategy_state"][key], value)
            saved_rng = trainer._torch_load(trainer._checkpoint_rank_bytes(loaded.state.rng, rank, 2), torch.device("cpu"))
            assert set(saved_rng) == {"python", "numpy", "torch", "cuda"}
            assert torch.equal(saved_rng["torch"], rng["torch"])
            assert all(torch.equal(a, b) for a, b in zip(saved_rng["cuda"], rng["cuda"]))
            assert saved_rng["python"] == rng["python"] and np.array_equal(saved_rng["numpy"][1], rng["numpy"][1])
            assert json.loads(loaded.state.scheduler) == {"iteration": 12, "nominal_iterations": 16, "requested_stop_iteration": 12}
            rejected = False
            try:
                load_checkpoint(positional[0], "train-001", 12, expected_provenance=replace(provenance, environment_hash="0"*64))
            except CheckpointContractError:
                rejected = True
            assert rejected
            expected_lr = exponential_learning_rate(.00016, .0000016, 12, 16, delay_multiplier=.01) * trainer._training_scene_scale(contract)
            assert abs(payload["means"]["param_groups"][0]["lr"]-expected_lr) < 1e-15
            observations.update(checkpoint_state_equal=True, rng_unchanged=True, environment_mismatch_rejected=True,
                                final_position_lr=expected_lr)
            return record

        def merge(paths, destination):
            models = [trainer._load_model(p, torch.device("cpu")) for p in paths]
            expected = {name: torch.cat([m.params[name].detach() for m in models]) for name in models[0].params}
            merged = original_merge(paths, destination)
            assert all(torch.equal(value, merged.params[name]) for name, value in expected.items())
            observations["best_merge_exact"] = True
            return merged

        with patch.object(trainer, "_write_streaming_checkpoint", writer), patch.object(trainer, "_merge_model_shards", merge):
            result = trainer.train_gaussians(contract=contract, dataset_root=dataset, initialization=initialization,
                resolved_config=cfg, run_dir=root / "trainer", local_rank=local_rank, world_rank=rank,
                world_size=2, stop_after_iteration=12, save_intermediate_previews=False)
        assert result.iteration == 12 and result.nominal_iterations == 16 and result.termination_reason == "execution_budget"
        events = [json.loads(line) for line in (root / "trainer" / result.progress_path).open()]
        steps = [e for e in events if "loss" in e]
        assert len(steps) == 12 and any(e.get("mcmc_added_count", 0) > 0 for e in steps)
        assert [e["iteration"] for e in events if e.get("event") == "validation"] == [6, 12]
        sizes = {str(i["image_id"]): (i["width"], i["height"]) for i in contract["images"]}
        assert any(sizes[e["batch_view_ids"][0]] != sizes[e["batch_view_ids"][1]] for e in steps)
        if rank == 0:
            model = trainer._load_model(root / "trainer" / result.model_path, device)
            views = load_training_views(contract, dataset, split="validation", longest_edge=96, device=torch.device("cpu"))
            evaluation = trainer.evaluate_views(model, views, cfg.effective_config, distributed=False)
            assert evaluation["successful_views"] == 2
            write_json(root / "merged-evaluation.json", evaluation)
        write_json(rank_dir / "summary.json", {"passed": True, "rank": rank, "updates": 12,
            "nominal_iterations": 16, "training_core": provenance.code_hash, "environment": trainer._gsplat_version(),
            "environment_hash": provenance.environment_hash, "dataset_hash": contract["dataset_hash"],
            "config_hash": cfg.effective_config_hash, "initialization_sha256": tensor_hash({
                "points": torch.from_numpy(initialization.points), "colors": torch.from_numpy(initialization.colors),
                "scales": torch.from_numpy(initialization.scales)}), "checks": observations,
            "result": result.__dict__})


def run(args):
    unlimited_host_cgroup(UNIT)
    if sha256_file(OVERLAY / "overlay.json") != OVERLAY_SHA:
        raise ValueError("previously verified overlay changed")
    verify_overlay(OVERLAY, OVERLAY / "gsplat")
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        if shutil.disk_usage(PROJECT).free < 20*1024**3:
            raise ValueError("requires 20 GiB free disk")
        args.output_root.mkdir(parents=True, exist_ok=False)
        from image3d_scenegraph.gaussian.trainer import training_provenance, _gsplat_version
        core = training_provenance(dataset_hash="0"*64, effective_config_hash="0"*64, world_size=2).code_hash
        write_json(args.output_root / "protocol.json", {"training_core": core, "overlay_sha256": OVERLAY_SHA,
            "base_environment": _gsplat_version(), "script_sha256": sha256_file(Path(__file__)),
            "config": lifecycle_config().effective_config, "updates": 12, "nominal_iterations": 16,
            "world_size": 2, "initial_points": 60, "image_sizes": [[64, 64], [96, 80]],
            "host_limits": None, "reserved_limit_each": 18*1024**3, "stage_seconds": 600, "unit_seconds": 1800})
        records = {}
        for arm in ("baseline", "repair"):
            require_idle_gpu()
            stage = args.output_root / arm
            stage.mkdir()
            output = stage / "probe"
            prefix = ["/usr/bin/env", "-u", "PYTHONPATH"] if arm == "baseline" else ["/usr/bin/env", "PYTHONPATH=" + str(OVERLAY)]
            monitor = TrainingMonitor(output / "trainer/attempts/train-001/artifacts/progress.jsonl", output / "memory",
                updates=12, main_stage=True, limits={**LIMITS, "max_main_stage_wall_seconds": 600})
            command = prefix + [sys.executable, str(Path(__file__).resolve()), "--output-root", str(output), "--arm", arm]
            run_stage(stage, "lifecycle", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
            ranks = [json.loads((output / f"rank-{r}/summary.json").read_text()) for r in (0, 1)]
            assert [r["rank"] for r in ranks] == [0, 1]
            assert all(r["passed"] and r["training_core"] == core and r["updates"] == 12 for r in ranks)
            for field in ("dataset_hash", "config_hash", "environment_hash", "initialization_sha256"):
                assert ranks[0][field] == ranks[1][field]
            records[arm] = {**ranks[0], "ranks": ranks, **monitor.record(),
                            "camera_sequence_sha256": monitor.record()["camera_sequence_sha256"]}
        verify_pair(records)
        from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance, load_checkpoint, CheckpointContractError
        for arm, other in (("baseline", "repair"), ("repair", "baseline")):
            row = records[arm]
            expected = CheckpointProvenance(row["dataset_hash"], row["config_hash"], row["training_core"], records[other]["environment_hash"])
            try:
                load_checkpoint(args.output_root / arm / "probe/trainer", "train-001", 12, expected_provenance=expected)
            except CheckpointContractError:
                continue
            raise ValueError("cross-renderer checkpoint was accepted")
        write_json(args.output_root / "summary.json", {"status": "passed", "arms": records,
            "overlay_sha256": OVERLAY_SHA, "cross_environment_rejected": True, "test_rgb": "not_loaded"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", choices=("baseline", "repair"))
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅在已授权远端运行生命周期，不运行本机模型")
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(PROJECT / "outputs/analysis"):
        raise ValueError("independent output required")
    if args.arm:
        import torch
        from gsplat.distributed import cli
        if torch.cuda.device_count() != 2:
            raise ValueError("two visible GPUs required")
        args.output_root.mkdir(parents=True, exist_ok=False)
        cli(worker, args, verbose=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
