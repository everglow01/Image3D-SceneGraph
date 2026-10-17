#!/usr/bin/env python3
"""Remote-only MCMC failure-mechanism probes; never a scene-quality pass."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess


def emit(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


def tile_estimate(means, radii, tile_size, tile_width, tile_height) -> dict:
    import torch

    count = 0
    projected = 0
    maximum = 0
    invalid = 0
    with torch.no_grad():
        means = means.detach().reshape(-1, 2)
        radii = radii.detach().reshape(-1, 2)
        for start in range(0, len(means), 65536):
            centers = means[start:start + 65536]
            radius = radii[start:start + 65536]
            valid = (radius > 0).all(dim=-1)
            finite = torch.isfinite(centers).all(dim=-1)
            invalid += int((valid & ~finite).sum())
            valid &= finite
            projected += int(valid.sum())
            maximum = max(maximum, int(radius.max()) if radius.numel() else 0)
            centers = centers[valid] / tile_size
            extent = radius[valid].float() / tile_size
            low = torch.floor(centers - extent)
            high = torch.ceil(centers + extent)
            dx = high[:, 0].clamp(0, tile_width) - low[:, 0].clamp(0, tile_width)
            dy = high[:, 1].clamp(0, tile_height) - low[:, 1].clamp(0, tile_height)
            count += int((dx.long() * dy.long()).sum())
    return {"bbox_intersections_estimate": count if not invalid else None,
            "projected_count": projected, "maximum_radius_pixels": maximum,
            "invalid_projected_centers": invalid,
            "scope": "post_all_to_all_bbox_estimate_not_cuda_first_pass"}


@contextmanager
def observe_gsplat(path: Path):
    from gsplat import rendering
    from gsplat.strategy import ops

    original_tiles = rendering.isect_tiles
    original_relocation = ops.compute_relocation

    def tiles(means, radii, depths, tile_size, tile_width, tile_height, **kwargs):
        estimate = tile_estimate(means, radii, tile_size, tile_width, tile_height)
        emit(path, {"event": "before_tile_allocation", **estimate})
        try:
            result = original_tiles(means, radii, depths, tile_size, tile_width, tile_height, **kwargs)
        except Exception as exc:
            try:
                emit(path, {"event": "tile_failure", "reason": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass
            raise
        actual = result[1].numel()
        emit(path, {"event": "after_tile_allocation", "actual_intersections": actual,
                    "sorted_index_arrays_bytes": 24 * actual,
                    "estimate_matches_actual": estimate["bbox_intersections_estimate"] == actual})
        return result

    def relocation(*args, **kwargs):
        ratios = kwargs["ratios"] if "ratios" in kwargs else args[2]
        emit(path, {"event": "before_relocation_formula", "entry_count": ratios.numel(),
                    "maximum_actual_ratio": int(ratios.max()) if ratios.numel() else 0,
                    "entries_above_51": int((ratios > 51).sum()),
                    "scope": "formula_input_entries_not_unique_sources"})
        return original_relocation(*args, **kwargs)

    rendering.isect_tiles = tiles
    ops.compute_relocation = relocation
    try:
        yield
    finally:
        rendering.isect_tiles = original_tiles
        ops.compute_relocation = original_relocation


def pressure_checks(device) -> dict:
    import torch
    from gsplat.relocation import compute_relocation

    rng = torch.cuda.get_rng_state(device).clone()
    adam = []
    for mature_step in (0, 9300):
        parameter = torch.nn.Parameter(torch.logit(torch.tensor([.005], device=device)))
        optimizer = torch.optim.Adam([parameter], lr=.05, eps=1e-15)
        optimizer.state[parameter] = {"step": torch.tensor(float(mature_step)),
                                      "exp_avg": torch.zeros_like(parameter),
                                      "exp_avg_sq": torch.zeros_like(parameter)}
        samples = []
        for step in range(1, 101):
            optimizer.zero_grad(set_to_none=True)
            (.01 / 3_000_000 * parameter.sigmoid().sum()).backward()
            optimizer.step()
            if step in (1, 2, 100):
                samples.append(float(parameter.sigmoid()))
        adam.append({"initial_step": mature_step, "opacity_at_steps_1_2_100": samples})
    assert math.isclose(adam[1]["opacity_at_steps_1_2_100"][-1], 2.384893991e-5, rel_tol=2e-4)
    assert adam[1]["opacity_at_steps_1_2_100"][0] > 1 / 255
    assert adam[1]["opacity_at_steps_1_2_100"][1] < 1 / 255
    assert adam[0]["opacity_at_steps_1_2_100"][-1] < 1 / 255

    binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(51)]
                           for n in range(51)], dtype=torch.float32, device=device)
    alpha = torch.tensor([.5], device=device)
    scales = torch.full((1, 3), .01, device=device)
    ratios = torch.tensor([184], device=device)
    opacity, _ = compute_relocation(alpha, scales, ratios.clone(), binoms)
    composed = -math.expm1(184 * math.log1p(-float(opacity[0])))
    assert composed > .9
    alpha.fill_(.006)
    ratios.fill_(51)
    first_invalid = None
    for step in range(1, 41):
        _, scales = compute_relocation(alpha, scales, ratios.clone(), binoms)
        if not bool(torch.isfinite(scales).all() & (scales > 0).all()):
            first_invalid = step
            break
    positive = scales[torch.isfinite(scales) & (scales > 0)]
    assert torch.equal(rng, torch.cuda.get_rng_state(device))
    revived = float(torch.logit(torch.tensor(.005, device=device)).sigmoid())
    return {"adam_regularizer_only": adam, "revival_roundtrip": revived,
            "revived_still_dead": revived <= .005,
            "actual_copies": 184, "composed_center_opacity": composed,
            "scale_stress": {"constant_input_opacity": .006, "ratio": 51, "rounds": step,
                             "first_nonpositive_or_nonfinite": first_invalid,
                             "minimum_positive_scale": float(positive.min()) if positive.numel() else None,
                             "scope": "repeated_formula_with_opacity_reset_not_scene_replay"},
            "rng_unchanged": True}


def worker(local_rank, rank, world_size, args):
    import torch
    from gsplat import rendering
    from gsplat.strategy import ops
    from image3d_scenegraph.gaussian.trainer import _gsplat_version, training_provenance
    from smoke_mcmc_stability import trainer_check

    assert world_size == 2 and local_rank == rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    identity = _gsplat_version()
    checks = pressure_checks(device)
    if "project_mcmc_relocation_v1" in identity:
        assert checks["scale_stress"]["first_nonpositive_or_nonfinite"] is not None
    else:
        assert checks["scale_stress"]["first_nonpositive_or_nonfinite"] is None
    means = torch.tensor([[[32., 32.], [-2., 5.], [90., 90.]]], device=device)
    radii = torch.tensor([[[2, 2], [5, 3], [0, 0]]], dtype=torch.int32, device=device)
    depths = torch.ones((1, 3), device=device)
    original = rendering.isect_tiles(means, radii, depths, 16, 4, 4, packed=False)
    rng = torch.cuda.get_rng_state(device).clone()
    with observe_gsplat(args.output_dir / f"trace-rank-{rank}.jsonl"):
        observed = rendering.isect_tiles(means, radii, depths, 16, 4, 4, packed=False)
        assert all(torch.equal(a, b) for a, b in zip(original, observed))
        assert torch.equal(rng, torch.cuda.get_rng_state(device))
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.zeros((184, 3), device=device)),
            "scales": torch.nn.Parameter(torch.full((184, 3), math.log(.01), device=device)),
            "opacities": torch.nn.Parameter(torch.logit(torch.tensor([.5] + [1e-6] * 183, device=device))),
        })
        optimizers = {key: torch.optim.Adam([value], lr=.01) for key, value in params.items()}
        binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(51)]
                              for n in range(51)], dtype=torch.float32, device=device)
        ops.relocate(params, optimizers, {}, params["opacities"].sigmoid() <= .005, binoms)
        assert params["opacities"].numel() == 184
        result = trainer_check(rank, world_size, args.output_dir)
    events = [json.loads(line) for line in (args.output_dir / f"trace-rank-{rank}.jsonl").read_text().splitlines()]
    assert any(e.get("maximum_actual_ratio") == 184 for e in events)
    tile_events = [e for e in events if e["event"] == "after_tile_allocation"]
    assert tile_events and all(e["estimate_matches_actual"] for e in tile_events)
    reserved = int(torch.cuda.max_memory_reserved(device))
    assert reserved < 18 * 1024**3
    core = training_provenance(dataset_hash="0" * 64, effective_config_hash="0" * 64,
                               world_size=world_size).code_hash
    record = {"rank": rank, "environment": identity, "training_code_hash": core, "peak_reserved_bytes": reserved, "checks": checks, "trainer": result,
              "tile_estimate_matches_small_cases": True, "status": "diagnostic_complete",
              "stability_gate": "blocked_by_reproduced_failures"}
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if socket.gethostname() != "i-94B8D131" or str(root) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅允许在已授权远端项目运行，禁止本机模型验证")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(root / "outputs/analysis") or args.output_dir.exists():
        raise SystemExit("需要全新独立研究目录，不覆盖或重试旧门禁")
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    from run_mcmc_stability_ablation import unlimited_host_cgroup

    unlimited_host_cgroup("image3d-mcmc-degeneration-20261017-v1.service")
    import torch
    from gsplat.distributed import cli

    with FileLease(root / "outputs/.gpu.lock"):
        subprocess.run(["nvidia-smi", "-L"], check=True)
        require_idle_gpu()
        if torch.cuda.device_count() != 2:
            raise SystemExit("需要现场确认的两张可见GPU")
        args.output_dir.mkdir(parents=True)
        try:
            cli(worker, args, verbose=True)
        except Exception as exc:
            emit(args.output_dir / "failure.jsonl", {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            raise
    records = [json.loads((args.output_dir / f"rank-{rank}.json").read_text()) for rank in range(2)]
    summary = {"status": "diagnostic_complete", "stability_gate": "blocked_by_reproduced_failures",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "reused_smoke_sha256": hashlib.sha256((root / "scripts/smoke_mcmc_stability.py").read_bytes()).hexdigest(),
               "ranks": records, "scene_training": "not_run", "test_rgb": "not_loaded"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
