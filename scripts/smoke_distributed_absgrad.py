#!/usr/bin/env python3
"""Bounded two-GPU AbsGrad equivalence and native-trainer smoke check."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, TrainingMonitor
from image3d_scenegraph.gaussian.config import resolve_internal_config
from image3d_scenegraph.gaussian.initialization import InitializationResult
from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.render import (
    RenderCamera,
    collect_distributed_absgrad,
    render_gaussians,
    require_distributed_absgrad,
)
from image3d_scenegraph.gaussian.trainer import _build_strategy, train_gaussians, TrainingCancelled
from smoke_gaussian_trainer import generate_scene


ATOL = 2e-6
RTOL = 2e-4


def camera(index: int, device: torch.device) -> RenderCamera:
    pose = torch.eye(4, device=device)
    pose[0, 3] = 0.15 * index
    return RenderCamera(
        str(index), pose,
        torch.tensor([[48.0, 0, 32], [0, 48.0, 32], [0, 0, 1]], device=device),
        64, 64,
    )


def model_for(indices: list[int], device: torch.device, invisible_owner: bool, sh_degree: int = 0) -> GaussianModel:
    points = torch.tensor([
        [-0.2, 0, 2], [0.2, 0.1, 2.2], [0, -0.1, 2.4],
        [100, 0, 2], [0, 0.2, 2], [-0.1, 0.1, 2.6], [0, 0, -2],
    ], device=device, dtype=torch.float32)
    if invisible_owner:
        points[1::2, 2] = -2
    colors = torch.tensor([[0.8, 0.3, 0.2]] * 7, device=device)
    model = GaussianModel.from_points(
        points[indices], colors[indices], torch.full((len(indices),), 0.12, device=device),
        initial_opacity=0.6, max_sh_degree=sh_degree,
    )
    if sh_degree:
        coefficients = torch.arange(7 * 15 * 3, device=device, dtype=torch.float32)
        with torch.no_grad():
            model.params["shN"].copy_(0.03 * coefficients.reshape(7, 15, 3)[indices].sin())
    return model


def loss(image: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.5, 1.5, 64, device=image.device)[None, :, None]
    return (image * weights).mean() / 2


def full_reference(device: torch.device, invisible_owner: bool, sh_degree: int):
    model = model_for(list(range(7)), device, invisible_owner, sh_degree)
    means = []
    absolute = []
    state = DefaultStrategy(absgrad=True).initialize_state()
    strategy = DefaultStrategy(absgrad=True)
    for index in range(2):
        cam = camera(index, device)
        image, _, info = rasterization(
            *model.activated(), cam.camera_from_normalized[None], cam.intrinsic[None],
            width=64, height=64, sh_degree=sh_degree, packed=False, absgrad=True,
        )
        info["means2d"].retain_grad()
        loss(image[0]).backward()
        means.append(info["means2d"].grad.detach().clone()[0])
        absolute.append(info["means2d"].absgrad.detach().clone()[0])
        strategy._update_state(model.params, state, info, packed=False)
    return model, torch.stack(means), torch.stack(absolute), state


def snapshot(model, rendered, state):
    return {
        "image": rendered.image.detach().cpu(),
        "signed": rendered.metadata["means2d"].grad.detach().cpu(),
        "grad2d": state["grad2d"].detach().cpu(),
        "count": state["count"].detach().cpu(),
        **{f"gradient_{key}": value.grad.detach().cpu() for key, value in model.params.items()},
    }


def compare(actual, expected, errors: dict, prefix: str):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    errors[prefix] = float((actual - expected).abs().max()) if actual.numel() else 0.0


def micro_checks(rank: int, args, device: torch.device) -> dict:
    errors = {}
    ids = list(range(rank, 7, 2))
    for invisible_owner in (False, True):
        case = "invisible_owner" if invisible_owner else "unequal_shards"
        reference_path = (args.output_dir if args.baseline_only else args.reference_dir) / f"{case}-{rank}.pt"
        baseline = None if args.baseline_only else torch.load(reference_path, map_location="cpu", weights_only=False)
        for absolute in ((False,) if args.baseline_only else (False, True)):
            config = resolve_internal_config(
                "absgrad_ablation_v1", {"densification": {"absgrad": absolute}}
            ).effective_config
            model = model_for(ids, device, invisible_owner, args.sh_degree)
            strategy = _build_strategy(DefaultStrategy, config)
            optimizers = model.optimizers(config["learning_rate"])
            state = strategy.initialize_state()
            strategy.check_sanity(model.params, optimizers)
            rendered = render_gaussians(
                model, camera(rank, device), sh_degree=args.sh_degree,
                distributed=True, gradient_statistics=absolute,
            )
            strategy.step_pre_backward(model.params, optimizers, state, 1, rendered.metadata)
            loss(rendered.image).backward()
            if absolute:
                collect_distributed_absgrad(rendered.metadata)
            model.validate_gradients()
            strategy.step_post_backward(model.params, optimizers, state, 1, rendered.metadata, packed=False)
            result = snapshot(model, rendered, state)
            if args.baseline_only:
                torch.save(result, reference_path)
                continue
            for key, value in result.items():
                if not absolute or key != "grad2d":
                    compare(value, baseline[key], errors, f"{case}/{absolute}/pristine/{key}")
            if absolute:
                ref_model, signed, abs_stats, ref_state = full_reference(device, invisible_owner, args.sh_degree)
                compare(rendered.metadata["means2d"].grad, signed[:, ids], errors, f"{case}/signed_reference")
                compare(rendered.metadata["means2d"].absgrad, abs_stats[:, ids], errors, f"{case}/absolute_reference")
                for key, value in model.params.items():
                    compare(value.grad, ref_model.params[key].grad[ids], errors, f"{case}/reference/{key}")
                for key in ("grad2d", "count"):
                    compare(state[key], ref_state[key][ids], errors, f"{case}/strategy/{key}")
                if invisible_owner and rank == 1:
                    assert torch.count_nonzero(state["count"]) == 0
                if not invisible_owner:
                    assert bool((state["grad2d"].cpu() > baseline["grad2d"] + 1e-5).any())
    return errors


def cancellation_check(device: torch.device) -> dict:
    model = GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0]], device=device),
        torch.tensor([[0.8, 0.3, 0.2]], device=device),
        torch.tensor([0.12], device=device), initial_opacity=0.6, max_sh_degree=0,
    )
    cam = camera(0, device)
    image, _, info = rasterization(
        *model.activated(), cam.camera_from_normalized[None], cam.intrinsic[None],
        width=64, height=64, sh_degree=0, absgrad=True, packed=False,
    )
    info["means2d"].retain_grad()
    image.sum().backward()
    signed = float(info["means2d"].grad.norm())
    absolute = float(info["means2d"].absgrad.norm())
    assert absolute > 100 * max(signed, 1e-8)
    return {"signed_norm": signed, "absolute_norm": absolute}


def trainer_smoke(rank: int, args, device: torch.device) -> dict:
    dataset_root = args.output_dir / "synthetic-dataset"
    if rank == 0:
        generate_scene(dataset_root)
    torch.distributed.barrier()
    contract = json.loads((dataset_root / "dataset.json").read_text())
    rng = np.random.RandomState(7)
    points = rng.uniform(-0.15, 0.15, (13, 3)).astype(np.float32)
    initialization = InitializationResult(
        points=points,
        colors=np.full((13, 3), 128, dtype=np.uint8),
        scales=np.full(13, 0.02, dtype=np.float32),
        diagnostics={"kind": "synthetic_distributed_absgrad"},
    )
    config = resolve_internal_config("absgrad_ablation_v1", {
        "iterations": 12,
        "resolution": {"longest_edge": 64},
        "sh_schedule": {"initial_degree": args.sh_degree, "max_degree": args.sh_degree, "increase_every_iterations": 1},
        "densification": {
            "absgrad": True, "start_iteration": 2, "end_iteration": 10,
            "every_iterations": 2, "gradient_threshold": 1e-7,
        },
        "opacity_reset": {"enabled": False, "every_iterations": 12,
                          "recovery_prune": {"window_iterations": 2}},
        "evaluation": {"validation_iterations": [6, 12]},
    })
    result = train_gaussians(
        contract=contract, dataset_root=dataset_root, initialization=initialization,
        resolved_config=config, run_dir=args.output_dir / "trainer",
        local_rank=rank, world_rank=rank, world_size=2, save_intermediate_previews=False,
    )
    lifecycle = {}
    if args.save_lifecycle_checks:
        calls = 0
        def cancel_after_two():
            nonlocal calls
            calls += 1
            return calls > 2
        try:
            train_gaussians(contract=contract, dataset_root=dataset_root, initialization=initialization,
                resolved_config=config, run_dir=args.output_dir / "cancel-trainer", local_rank=rank,
                world_rank=rank, world_size=2, save_intermediate_previews=False,
                cancel_requested=cancel_after_two)
        except TrainingCancelled:
            lifecycle["cancelled_after_two_updates"] = True
        else:
            raise RuntimeError("trainer did not honor normal cancellation")
        torch.distributed.barrier()
        assert not list((args.output_dir / "cancel-trainer").rglob("checkpoint.json"))
        lifecycle["cancel_checkpoint_written"] = False
        if rank == 0 and args.reference_trainer_dir:
            old = torch.load(args.reference_trainer_dir / "attempts/train-001/artifacts/model.pt", map_location="cpu", weights_only=True)
            new = torch.load(args.output_dir / "trainer" / result.model_path, map_location="cpu", weights_only=True)
            assert old["max_sh_degree"] == new["max_sh_degree"]
            assert old["state_dict"].keys() == new["state_dict"].keys()
            for key in old["state_dict"]:
                torch.testing.assert_close(old["state_dict"][key], new["state_dict"][key], rtol=RTOL, atol=ATOL)
            lifecycle["historical_trainer_model_equivalent"] = True
            lifecycle["model_max_abs_diff"] = max(float((old["state_dict"][key] - new["state_dict"][key]).abs().max()) for key in old["state_dict"])
    if rank == 0:
        progress = args.output_dir / "trainer" / result.progress_path
        events = [json.loads(line) for line in progress.read_text().splitlines()]
        steps = [event for event in events if "loss" in event]
        assert len(steps) == 12
        assert all(event["world_size"] == 2 for event in steps)
        assert any(event.get("topology_net_growth", 0) > 0 for event in steps)
        assert all(event["gaussian_count"] < 1000 for event in steps)
        return {"updates": 12, "camera_samples": 24,
                "initial_count": 13, "peak_count": max(e["gaussian_count"] for e in steps),
                "final_count": steps[-1]["gaussian_count"], "result": result.__dict__, "save_lifecycle": lifecycle}
    return {"updates": 12}


def worker(local_rank: int, rank: int, world_size: int, args) -> None:
    assert world_size == 2 and local_rank == rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if not args.baseline_only:
        require_distributed_absgrad()
    errors = micro_checks(rank, args, device)
    record = {"rank": rank, "status": "passed", "max_absolute_errors": errors,
              "atol": ATOL, "rtol": RTOL, "baseline_only": args.baseline_only}
    if not args.baseline_only:
        record["cancellation"] = cancellation_check(device)
        telemetry = args.output_dir / "memory" if args.resource_telemetry else None
        with memory_telemetry(telemetry, local_rank, rank):
            record["trainer"] = trainer_smoke(rank, args, device)
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    torch.distributed.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--resource-telemetry", action="store_true")
    parser.add_argument("--save-lifecycle-checks", action="store_true")
    parser.add_argument("--reference-trainer-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--sh-degree", type=int, choices=(0, 3), default=0)
    args = parser.parse_args()
    if args.reference_trainer_dir and not args.save_lifecycle_checks:
        parser.error("--reference-trainer-dir requires --save-lifecycle-checks")
    if args.baseline_only and args.save_lifecycle_checks:
        parser.error("lifecycle checks require the trainer smoke")
    if args.baseline_only and args.resource_telemetry:
        parser.error("resource telemetry requires the candidate trainer smoke")
    if torch.cuda.device_count() != 2:
        raise SystemExit("This smoke requires exactly two visible GPUs")
    if not args.baseline_only and args.reference_dir is None:
        raise SystemExit("--reference-dir is required for the patched run")
    if not args.baseline_only:
        reference = json.loads((args.reference_dir / "summary.json").read_text())
        if reference.get("sh_degree", 0) != args.sh_degree:
            raise SystemExit("reference SH degree does not match the requested check")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cli(worker, args, verbose=True)
    records = [json.loads((args.output_dir / f"rank-{rank}.json").read_text()) for rank in range(2)]
    summary = {"status": "passed", "world_size": 2, "sh_degree": args.sh_degree, "ranks": records,
               "real_scene_training": "not_run", "test_rgb": "not_loaded"}
    if args.resource_telemetry:
        monitor = TrainingMonitor(
            args.output_dir / "trainer" / records[0]["trainer"]["result"]["progress_path"],
            args.output_dir / "memory", updates=12, main_stage=False,
        )
        failure = monitor(0, final=True)
        if failure or not all(peak > 0 for peak in monitor.peak_reserved):
            raise RuntimeError(f"resource telemetry smoke failed: {failure}")
        summary["resource_telemetry"] = monitor.record()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
