#!/usr/bin/env python3
"""Remote-only CUDA gates for isolated MCMC relocation and per-rank diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gsplat.distributed import cli
from gsplat.strategy import ops
from gsplat.cuda._wrapper import _make_lazy_cuda_func
from gsplat.relocation import compute_relocation

from image3d_scenegraph.gaussian.config import resolve_internal_config
from image3d_scenegraph.gaussian.initialization import InitializationResult
from image3d_scenegraph.gaussian.trainer import (
    _gather_rank_health, _gsplat_version, _mcmc_parameter_health, train_gaussians,
    training_provenance,
)
from smoke_gaussian_trainer import generate_scene


ATOL = 2e-6
RTOL = 2e-4


def relocation_reference(alpha: float, ratio: int, minimum: float) -> tuple[float, float]:
    ratio = min(51, max(1, ratio))
    opacity = min(1. - 2.**-23, max(minimum, 1. - (1. - alpha)**(1. / ratio)))
    denominator = math.fsum(
        (-1.)**(k + 1) * math.comb(ratio, k) * opacity**k / math.sqrt(k)
        for k in range(1, ratio + 1)
    )
    return opacity, alpha / denominator


def relocation_checks(device: torch.device) -> dict:
    import gsplat.relocation as relocation

    if getattr(relocation, "PROJECT_MCMC_RELOCATION_VERSION", None) != 1:
        raise RuntimeError("the separately built relocation overlay is required")
    binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(51)]
                           for n in range(51)], dtype=torch.float32, device=device)
    alpha = torch.tensor([.006, .01, .05, .0001, .5, .99], device=device)
    ratios = torch.tensor([2, 10, 51, 75, 2, 1], device=device)
    scales = torch.full((len(alpha), 3), .2, device=device)
    expected = torch.tensor([relocation_reference(float(a), int(n), .005)
                             for a, n in zip(alpha, ratios)], device=device)
    actual_opacity, actual_scale = compute_relocation(alpha, scales, ratios.clone(), binoms)
    torch.testing.assert_close(actual_opacity, expected[:, 0], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual_scale, scales * expected[:, 1, None], atol=ATOL, rtol=RTOL)
    assert torch.isfinite(actual_scale).all() and (actual_scale > 0).all()
    old_opacity, old_scale = _make_lazy_cuda_func("relocation")(
        alpha, scales, ratios.clamp(1, 51).int(), binoms, 51,
    )
    torch.testing.assert_close(actual_opacity[-2:], old_opacity[-2:], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual_scale[-2:], old_scale[-2:], atol=ATOL, rtol=RTOL)
    for minimum in (.005, .01):
        for operation in ("relocate", "sample_add"):
            params = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.zeros((2, 3), device=device)),
                "scales": torch.nn.Parameter(torch.full((2, 3), math.log(.2), device=device)),
                "opacities": torch.nn.Parameter(torch.logit(torch.tensor([.006, 1e-6], device=device))),
            })
            optimizers = {key: torch.optim.Adam([value], lr=.01) for key, value in params.items()}
            sampler = ops._multinomial_sample
            ops._multinomial_sample = lambda weights, count, **kwargs: torch.zeros(count, dtype=torch.long, device=device)
            try:
                if operation == "relocate":
                    ops.relocate(params, optimizers, {}, torch.tensor([False, True], device=device), binoms, minimum)
                else:
                    ops.sample_add(params, optimizers, {}, 1, binoms, minimum)
            finally:
                ops._multinomial_sample = sampler
            expected_opacity, coefficient = relocation_reference(.006, 2, minimum)
            indices = [0, 1] if operation == "relocate" else [0, 2]
            torch.testing.assert_close(params["opacities"].sigmoid()[indices],
                                       torch.full((2,), expected_opacity, device=device), atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(params["scales"].exp()[indices],
                                       torch.full((2, 3), .2 * coefficient, device=device), atol=ATOL, rtol=RTOL)
    return {"formula_cases": len(alpha), "operation_cases": 4,
            "no_floor_original_binary_equivalent": True,
            "scale_max_absolute_error": float((actual_scale - scales * expected[:, 1, None]).abs().max())}


def destination_moment_checks(device: torch.device) -> dict:
    if getattr(ops, "PROJECT_MCMC_OVERLAY_PROFILE", None) != "mcmc_destination_moments_overlay_v1":
        raise RuntimeError("the destination-moment-only overlay is required")
    binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(51)]
                           for n in range(51)], dtype=torch.float32, device=device)
    shapes = {"means": (3, 3), "scales": (3, 3), "quats": (3, 4),
              "sh0": (3, 1, 3), "shN": (3, 15, 3), "opacities": (3,)}
    params = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.zeros(shape, device=device))
                                     for key, shape in shapes.items()})
    with torch.no_grad():
        params["opacities"].copy_(torch.logit(torch.tensor([.5, 1e-6, .1], device=device)))
    optimizers = {key: torch.optim.Adam([value], lr=.01) for key, value in params.items()}
    for key, value in params.items():
        optimizers[key].state[value] = {"step": torch.tensor(100.),
                                      "exp_avg": torch.full_like(value, .2),
                                      "exp_avg_sq": torch.full_like(value, .04)}
    sampler = ops._multinomial_sample
    ops._multinomial_sample = lambda weights, count, **kwargs: torch.zeros(count, dtype=torch.long, device=device)
    try:
        ops.relocate(params, optimizers, {}, torch.tensor([False, True, False], device=device), binoms)
    finally:
        ops._multinomial_sample = sampler
    snapshots = {key: value.detach().clone() for key, value in params.items()}
    for key, value in params.items():
        state = optimizers[key].state[value]
        assert state["step"] == 100
        assert (state["exp_avg"][1] == 0).all() and (state["exp_avg_sq"][1] == 0).all()
        assert (state["exp_avg"][2] == .2).all()
    for _ in range(5):
        for key, value in params.items():
            value.grad = torch.zeros_like(value)
            optimizers[key].step()
    for key, value in params.items():
        torch.testing.assert_close(value[1], snapshots[key][1], atol=0, rtol=0)
        assert not torch.equal(value[2], snapshots[key][2])
        assert optimizers[key].state[value]["step"] == 105
    return {"parameter_groups": len(params), "zero_gradient_steps": 5,
            "destination_drift": 0., "unrelated_rows_keep_momentum": True}


def trainer_check(rank: int, world_size: int, output: Path) -> dict:
    dataset_root = output / "synthetic-dataset"
    if rank == 0:
        generate_scene(dataset_root)
    torch.distributed.barrier()
    contract = json.loads((dataset_root / "dataset.json").read_text())
    initialization = InitializationResult(
        points=np.random.RandomState(7).uniform(-.15, .15, (60, 3)).astype(np.float32),
        colors=np.full((60, 3), 128, dtype=np.uint8), scales=np.full(60, .02, dtype=np.float32),
        diagnostics={"kind": "synthetic_mcmc_stability"},
    )
    config = resolve_internal_config("mcmc_v1", {
        "iterations": 12, "resolution": {"longest_edge": 64},
        "sh_schedule": {"initial_degree": 3, "max_degree": 3, "increase_every_iterations": 1},
        "densification": {"start_iteration": 2, "end_iteration": 10, "every_iterations": 2},
        "opacity_reset": {"every_iterations": 12, "recovery_prune": {"window_iterations": 2}},
        "evaluation": {"validation_iterations": [6, 12]},
    })
    result = train_gaussians(
        contract=contract, dataset_root=dataset_root, initialization=initialization,
        resolved_config=config, run_dir=output / "trainer", local_rank=rank,
        world_rank=rank, world_size=world_size, save_intermediate_previews=False,
    )
    if rank != 0:
        return {"updates": 12}
    events = [json.loads(line) for line in (output / "trainer" / result.progress_path).read_text().splitlines()]
    steps = [event for event in events if "loss" in event]
    assert len(steps) == 12 and any(event.get("mcmc_added_count", 0) > 0 for event in steps)
    for event in events:
        due = event.get("event") == "validation" or event["iteration"] == 1 or event["iteration"] % 2 == 0
        if due:
            assert len(event["mcmc_health_by_rank"]) == world_size
            if "loss" in event:
                assert [item["rank"] for item in event["mcmc_health_by_rank"]] == list(range(world_size))
                for item in event["mcmc_health_by_rank"]:
                    assert item["rgb_range_before_optimizer"]["nonfinite_count"] == 0
                if "mcmc_deferred_count" in event:
                    assert event["mcmc_relocated_count"] == sum(
                        item["relocated_count"] for item in event["mcmc_health_by_rank"])
                    assert event["mcmc_deferred_count"] == event["mcmc_dead_global"] - event["mcmc_relocated_count"]
            else:
                assert all("rgb_range" in view for view in event["per_view"])
    return {"updates": 12, "result": result.__dict__, "diagnostics_checked": True}



def capacity_checks(device: torch.device) -> dict:
    from gsplat.strategy import MCMCStrategy

    if getattr(ops, "PROJECT_MCMC_OVERLAY_PROFILE", None) != "mcmc_capacity_overlay_v1":
        raise RuntimeError("the independent donor-capacity overlay is required")
    binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(51)]
                           for n in range(51)], dtype=torch.float32, device=device)

    def parameters(count, alpha):
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.zeros((count, 3), device=device)),
            "quats": torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device=device).repeat(count, 1)),
            "scales": torch.nn.Parameter(torch.full((count, 3), math.log(.01), device=device)),
            "opacities": torch.nn.Parameter(torch.logit(torch.tensor([alpha] + [1e-6] * (count - 1), device=device))),
            "sh0": torch.nn.Parameter(torch.zeros((count, 1, 3), device=device)),
            "shN": torch.nn.Parameter(torch.zeros((count, 15, 3), device=device)),
        })
        optimizers = {key: torch.optim.Adam([value], lr=.01) for key, value in params.items()}
        for key, value in params.items():
            optimizers[key].state[value] = {"step": torch.tensor(9300.),
                                          "exp_avg": torch.ones_like(value),
                                          "exp_avg_sq": torch.ones_like(value)}
        return params, optimizers

    params, optimizers = parameters(184, .5)
    strategy = MCMCStrategy(cap_max=1000)
    accepted = strategy._relocate_gs(params, optimizers, binoms)
    assert accepted == 50 and len(params["means"]) == 184
    opacity = params["opacities"].sigmoid()
    assert int((opacity > .005).sum()) == 51
    composed = -math.expm1(51 * math.log1p(-float(opacity[0])))
    assert abs(composed - .5) < ATOL
    assert torch.isfinite(params["scales"]).all() and (params["scales"].exp() > 0).all()
    for key, value in params.items():
        state = optimizers[key].state[value]
        assert state["step"] == 9300 and (state["exp_avg"][0] == 0).all()
        assert (state["exp_avg"][1:] == 1).all()
    params, optimizers = parameters(1, .5)
    added = ops.sample_add(params, optimizers, {}, 183, binoms)
    assert added == 50 and len(params["means"]) == 51
    assert (params["opacities"].sigmoid() > .005).all()
    for key, value in params.items():
        state = optimizers[key].state[value]
        assert state["step"] == 9300 and (state["exp_avg"][0] == 1).all()
        assert (state["exp_avg"][1:] == 0).all()
    params, optimizers = parameters(184, .006)
    snapshots = {key: value.detach().clone() for key, value in params.items()}
    rng = torch.cuda.get_rng_state(device).clone()
    for _ in range(40):
        assert strategy._relocate_gs(params, optimizers, binoms) == 0
        assert strategy._add_new_gs(params, optimizers, binoms) == 0
    assert torch.equal(rng, torch.cuda.get_rng_state(device))
    for key, value in params.items():
        assert torch.equal(value, snapshots[key])
        assert (optimizers[key].state[value]["exp_avg"] == 1).all()
    return {"requested_relocations": 183, "accepted_relocations": accepted,
            "deferred_relocations": 183 - accepted, "actual_total_copies": 51,
            "composed_center_opacity": composed, "requested_additions": 183,
            "accepted_additions": added, "near_dead_rounds_without_mutation": 40,
            "optimizer_policy_unchanged": True, "quality_status": "not_evaluated"}


def probe_worker(local_rank: int, rank: int, world_size: int, args) -> None:
    assert world_size == 2 and local_rank == rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    config = resolve_internal_config("mcmc_v1").effective_config
    opacity = torch.tensor(([.1] * 100 if rank == 0 else [1e-6] * 96 + [.1] * 4), device=device)
    shard = SimpleNamespace(count=100, opacity_logits=torch.logit(opacity),
                            log_scales=torch.full((100, 3), -5., device=device),
                            means=torch.zeros((100, 3), device=device))
    rng = torch.cuda.get_rng_state(device).clone()
    health = _gather_rank_health(_mcmc_parameter_health(shard, config), world_size)
    assert [item["dead_count"] for item in health] == [0, 96]
    assert torch.equal(rng, torch.cuda.get_rng_state(device))
    record = {"rank": rank, "environment": _gsplat_version(), "rank_health": health,
              "training_code_hash": training_provenance(dataset_hash="0" * 64,
                  effective_config_hash="0" * 64, world_size=2).code_hash}
    if args.relocation:
        record["relocation"] = relocation_checks(device)
    if args.destination_moments:
        record["destination_moments"] = destination_moment_checks(device)
    if args.capacity:
        record["capacity"] = capacity_checks(device)
    record["trainer"] = trainer_check(rank, world_size, args.output_dir)
    record["status"] = "passed"
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    torch.distributed.barrier()



def worker(local_rank: int, rank: int, world_size: int, args) -> None:
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry

    torch.cuda.set_device(local_rank)
    with memory_telemetry(args.resource_telemetry_dir, local_rank, rank):
        probe_worker(local_rank, rank, world_size, args)


def guarded_gate(args) -> None:
    from image3d_scenegraph.gaussian.absgrad_resources import TrainingMonitor, run_stage
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    from run_mcmc_stability_ablation import LIMITS, PROJECT, unlimited_host_cgroup

    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise ValueError("guarded MCMC gates require the authorized remote project")
    if args.relocation or args.destination_moments or args.resource_telemetry_dir is not None:
        raise ValueError("guarded gate supports original and capacity environments only")
    unlimited_host_cgroup("image3d-mcmc-capacity-20261017-v1.service")
    root = args.output_dir.resolve()
    if not root.is_relative_to(PROJECT / "outputs/analysis"):
        raise ValueError("guarded gate output must be an independent research directory")
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
            raise ValueError("guarded gate requires 20 GiB free disk")
        root.mkdir(parents=True, exist_ok=False)
        telemetry = root / "memory"
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--output-dir", str(root / "probe"),
                   "--resource-telemetry-dir", str(telemetry)]
        if args.capacity:
            command.append("--capacity")
        monitor = TrainingMonitor(root / "probe/trainer/attempts/train-001/artifacts/progress.jsonl",
                                  telemetry, updates=12, main_stage=True,
                                  limits={**LIMITS, "max_main_stage_wall_seconds": 300})
        run_stage(root, "probe", command, cwd=PROJECT, monitor=monitor,
                  pass_fds=(lease.fileno(),), host_group=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--resource-telemetry-dir", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--relocation", action="store_true")
    group.add_argument("--destination-moments", action="store_true")
    group.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    if args.guarded:
        guarded_gate(args)
        return
    if torch.cuda.device_count() != 2:
        raise SystemExit("this smoke requires exactly two visible GPUs")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cli(worker, args, verbose=True)
    records = [json.loads((args.output_dir / f"rank-{rank}.json").read_text()) for rank in range(2)]
    summary = {"status": "passed", "world_size": 2, "ranks": records,
               "atol": ATOL, "rtol": RTOL, "real_scene_training": "not_run", "test_rgb": "not_loaded"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
