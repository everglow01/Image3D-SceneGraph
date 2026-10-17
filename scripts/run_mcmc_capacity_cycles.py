#!/usr/bin/env python3
"""Remote-only repeated relocation controls; never submit a real-scene run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import sys
import time

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import TrainingMonitor, run_stage, write_json
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from scripts.run_mcmc_stability_ablation import LIMITS, PROJECT, code_hash, unlimited_host_cgroup


CASES = ("visible", "no_image_signal", "imbalanced")
UPDATES = 1000
START_STEP = 9300
LOCAL_POINTS = 128
CORE = "8df617cd7cc91cb4fdf44d66fbac41d186576c00aad97aa4d97f87c89bc8897d"
UNIT = "image3d-mcmc-capacity-cycles-20261017-v1.service"


def assess_pair(records):
    reasons, deltas = [], {}
    for case in CASES:
        baseline = records.get(f"baseline/{case}")
        candidate = records.get(f"capacity/{case}")
        if baseline is None or candidate is None:
            reasons.append(f"{case}: missing_result")
            continue
        for field in ("initialization_sha256", "reference_sha256"):
            if baseline[field] != candidate[field]:
                reasons.append(f"{case}: {field}_differs")
        if case != "no_image_signal":
            for arm, record in (("baseline", baseline), ("capacity", candidate)):
                if record["terminal"] != "completed" or record["completed_updates"] != UPDATES:
                    reasons.append(f"{case}/{arm}: {record['terminal']}")
                health = record.get("final_rank_health", [])
                if len(health) != 2 or any(row.get("effective_visible_count", 0) <= 0 for row in health):
                    reasons.append(f"{case}/{arm}: no_effective_visible_points_on_a_shard")
            if baseline["terminal"] == candidate["terminal"] == "completed":
                scores = [row["final_metrics"]["raw_psnr"] for row in (baseline, candidate)]
                if not all(math.isfinite(score) for score in scores):
                    reasons.append(f"{case}: nonfinite_score")
                    continue
                delta = scores[1] - scores[0]
                deltas[case] = delta
                if delta < -.5:
                    reasons.append(f"{case}: raw_psnr_regressed_more_than_0.5_db")
    return {"eligible_for_10k": not reasons, "blocking_reasons": reasons,
            "raw_psnr_deltas": deltas,
            "scope": "controlled_safety_veto_not_scene_quality_or_statistical_significance"}


class CycleMonitor(TrainingMonitor):
    def __call__(self, elapsed, *, final=False):
        report = self.progress.parent / "summary.json"
        if final and report.exists():
            value = json.loads(report.read_text())
            updates, terminal = value["completed_updates"], value["terminal"]
            exhausted = value["exhausted_ranks"]
            if (value["requested_updates"] != UPDATES or type(updates) is not int or not 1 <= updates <= UPDATES
                    or terminal not in {"completed", "source_exhausted"}
                    or (terminal == "completed" and (updates != UPDATES or exhausted))
                    or (terminal == "source_exhausted" and (updates % 100 or not exhausted
                        or any(rank not in (0, 1) for rank in exhausted)
                        or any(value["final_rank_health"][rank]["alive_count"] != 0 for rank in exhausted)))):
                raise ValueError("invalid controlled termination")
            self.updates = updates
        return super().__call__(elapsed, final=final)


def worker(local_rank, rank, world_size, args):
    import torch
    from image3d_scenegraph.gaussian.absgrad_resources import failure_record, memory_telemetry, record_training_phase
    from image3d_scenegraph.gaussian.config import resolve_internal_config
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.render import RenderCamera, render_gaussians
    from image3d_scenegraph.gaussian.trainer import (
        _build_mcmc_strategy, _gather_rank_health, _gsplat_version,
        _mcmc_parameter_health, _mcmc_regularizers, _publish_event,
        _update_position_learning_rate, training_provenance,
    )
    from image3d_scenegraph.gaussian.training_math import l1_ssim_loss
    from gsplat.strategy import MCMCStrategy
    from scripts.mcmc_capacity_ops import copy_budget

    assert world_size == 2 and local_rank == rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    output = args.output_root
    failure_dir = output / f"rank-{rank}"
    failure_dir.mkdir()
    with memory_telemetry(output / "memory", local_rank, rank), failure_record(failure_dir, args.case, "controlled_cycle"):
        assert _gsplat_version() == args.expected_environment
        provenance = training_provenance(dataset_hash="0" * 64, effective_config_hash="0" * 64, world_size=2)
        assert provenance.code_hash == CORE
        cfg = resolve_internal_config("mcmc_v1", {"seed": 20260729}).effective_config
        assert cfg["iterations"] == 30000 and cfg["densification"]["every_iterations"] == 100
        torch.manual_seed(20260729 + rank)
        grid = torch.tensor([[-.15, -.15, 2.], [-.15, .15, 2.], [.15, -.15, 2.], [.15, .15, 2.]], device=device)
        centers = grid.clone()
        centers[:, 0] += -.35 if rank == 0 else .35
        means = centers.repeat(LOCAL_POINTS // 4, 1)
        means[4:] += torch.randn_like(means[4:]) * .005
        hidden = args.case == "no_image_signal" or (args.case == "imbalanced" and rank == 1)
        if hidden:
            means[:, 2] = -2.
        color = torch.tensor([.8, .2, .2] if rank == 0 else [.2, .8, .3], device=device)
        model = GaussianModel.from_points(means, color.expand(LOCAL_POINTS, 3),
                                           torch.full((LOCAL_POINTS,), .03, device=device),
                                           initial_opacity=.5, max_sh_degree=3)
        with torch.no_grad():
            model.params["opacities"][4:] = torch.logit(torch.tensor(1e-6, device=device))
        optimizers = model.optimizers(cfg["learning_rate"], position_scale=1.)
        for name, parameter in model.params.items():
            gradient = torch.zeros_like(parameter)
            if name == "opacities":
                opacity = parameter.detach().sigmoid()
                gradient = float(cfg["loss"]["opacity_regularization"]) / 3_000_000 * opacity * (1 - opacity)
            optimizers[name].state[parameter] = {"step": torch.tensor(float(START_STEP)),
                                                "exp_avg": gradient.clone(), "exp_avg_sq": gradient.square()}
        digest = hashlib.sha256()
        for name, parameter in model.params.items():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
            for key, value in optimizers[name].state[parameter].items():
                digest.update(key.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
        initial_hashes = _gather_rank_health({"rank": rank, "sha256": digest.hexdigest()}, world_size)
        target_points = torch.cat([grid + torch.tensor([shift, 0., 0.], device=device) for shift in (-.35, .35)])
        target_colors = torch.tensor([[.8, .2, .2]] * 4 + [[.2, .8, .3]] * 4, device=device)
        target = GaussianModel.from_points(target_points, target_colors, torch.full((8,), .04, device=device),
                                           initial_opacity=.8, max_sh_degree=3)
        transform = torch.eye(4, device=device)
        transform[0, 3] = -.05 if rank == 0 else .05
        camera = RenderCamera(f"synthetic-{rank}", transform,
                              torch.tensor([[52., 0., 32.5], [0., 52., 32.5], [0., 0., 1.]], device=device), 64, 64)
        with torch.no_grad():
            reference = render_gaussians(target, camera, sh_degree=3).image.detach()
            if args.case == "no_image_signal":
                reference.zero_()
        reference_hashes = _gather_rank_health(
            {"rank": rank, "sha256": hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()}, world_size)
        del target
        strategy = _build_mcmc_strategy(MCMCStrategy, cfg, LOCAL_POINTS)
        state = strategy.initialize_state()
        strategy.check_sanity(model.params, optimizers)
        terminal, exhausted = "completed", []
        started = time.monotonic()
        record_training_phase("controlled_training", START_STEP + 1)
        for iteration in range(1, UPDATES + 1):
            if (output.parent / "cancel").exists():
                raise RuntimeError("controlled cycle cancelled by resource monitor")
            step = START_STEP + iteration
            if iteration % 100 == 0:
                record_training_phase("controlled_training", step)
            _update_position_learning_rate(optimizers["means"], cfg, step, 30000, 1.)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            rendered = render_gaussians(model, camera, sh_degree=3, distributed=True)
            strategy.step_pre_backward(model.params, optimizers, state, step, rendered.metadata)
            prediction = rendered.image.clamp(0, 1) if cfg["loss"]["clamp_render"] else rendered.image
            image_loss, _ = l1_ssim_loss(prediction, reference,
                                       l1_weight=cfg["loss"]["l1_weight"], ssim_weight=cfg["loss"]["ssim_weight"])
            opacity_reg, scale_reg = _mcmc_regularizers(model, cfg)
            # The small subset keeps the per-point regularizer of a global 3M model.
            regularizer = (opacity_reg + scale_reg) * (LOCAL_POINTS / 1_500_000)
            diagnostic = iteration == 1 or iteration % 100 == 0
            reg_gradient = None
            if diagnostic:
                reg_gradient = torch.autograd.grad(regularizer / world_size, model.opacity_logits, retain_graph=True)[0]
            loss = image_loss + regularizer
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite controlled loss")
            (loss / world_size).backward()
            model.validate_gradients()
            image_gradient_l1 = None
            if diagnostic:
                image_gradient_l1 = float((model.opacity_logits.grad - reg_gradient).abs().sum())
            for optimizer in optimizers.values():
                optimizer.step()
            before = _mcmc_parameter_health(model, cfg) if diagnostic else None
            if iteration % 100 == 0:
                live = torch.zeros(world_size, dtype=torch.int64, device=device)
                live[rank] = int((model.opacity_logits.sigmoid() > .005).sum())
                torch.distributed.all_reduce(live)
                exhausted = [index for index, value in enumerate(live.tolist()) if value == 0]
            if exhausted:
                terminal = "source_exhausted"
            else:
                strategy.step_post_backward(params=model.params, optimizers=optimizers, state=state,
                                            step=step, info=rendered.metadata,
                                            lr=float(optimizers["means"].param_groups[0]["lr"]))
            model.validate(max_count=LOCAL_POINTS)
            metrics = torch.tensor([float(loss.detach()), float((rendered.image.detach() - reference).square().mean()),
                                    float((rendered.image.detach().clamp(0, 1) - reference).square().mean())],
                                   device=device, dtype=torch.float64)
            torch.distributed.all_reduce(metrics)
            metrics /= world_size
            event = {"iteration": iteration, "absolute_iteration": step, "world_size": 2,
                     "batch_view_ids": ["synthetic-0", "synthetic-1"], "gaussian_count": LOCAL_POINTS * 2,
                     "loss": float(metrics[0]), "raw_mse_before_optimizer": float(metrics[1]),
                     "display_mse_before_optimizer": float(metrics[2]), "strategy_applied": not bool(exhausted)}
            if diagnostic:
                budget = copy_budget(model.opacity_logits.detach().sigmoid(), .005, 51)
                local = {"rank": rank, "before_strategy": before, "after_strategy": _mcmc_parameter_health(model, cfg),
                         "image_gradient_l1_before_optimizer": image_gradient_l1,
                         "projected_local_points_before_optimizer": int((rendered.metadata["radii"] > 0).all(dim=-1).any(dim=0).sum()),
                         "tile_intersections_before_optimizer": rendered.metadata["isect_ids"].numel(),
                         "admissible_donors_after_strategy": int((budget > 0).sum()),
                         "new_copy_capacity_after_strategy": int(budget.sum()),
                         "accepted_relocations": state.get("project_mcmc_relocated_count") if iteration % 100 == 0 and not exhausted else None}
                event["ranks"] = _gather_rank_health(local, world_size)
            if rank == 0:
                _publish_event(output / "progress.jsonl", event, None)
            if exhausted:
                break
        record_training_phase("controlled_" + terminal, START_STEP + iteration)
        with torch.no_grad():
            final = render_gaussians(model, camera, sh_degree=3, distributed=True)
            mse = torch.tensor([float((final.image - reference).square().mean()),
                                float((final.image.clamp(0, 1) - reference).square().mean())], device=device, dtype=torch.float64)
            torch.distributed.all_reduce(mse)
            mse /= world_size
        local_health = _mcmc_parameter_health(model, cfg)
        projected = (final.metadata["radii"] > 0).all(dim=-1).any(dim=0)
        local_health["effective_visible_count"] = int((projected & (model.opacity_logits.detach().sigmoid() > .005)).sum())
        health = _gather_rank_health(local_health, world_size)
        if rank == 0:
            write_json(output / "summary.json", {
                "case": args.case, "arm": args.arm, "terminal": terminal, "requested_updates": UPDATES,
                "completed_updates": iteration, "completed_refinements": iteration // 100 - int(bool(exhausted)),
                "exhausted_ranks": exhausted, "initialization_sha256": [row["sha256"] for row in initial_hashes],
                "reference_sha256": [row["sha256"] for row in reference_hashes],
                "environment": _gsplat_version(), "elapsed_seconds": time.monotonic() - started,
                "training_core": provenance.code_hash,
                "final_metrics": {"raw_mse": float(mse[0]), "raw_psnr": -10 * math.log10(max(float(mse[0]), 1e-12)),
                                  "display_mse": float(mse[1])}, "final_rank_health": health,
                "quality_scope": "two_fixed_synthetic_cameras_not_held_out_scene_quality", "test_rgb": "not_loaded"})
        torch.distributed.barrier()


def run(args):
    unlimited_host_cgroup(UNIT)
    root = args.output_root.resolve()
    gates = PROJECT / "outputs/analysis/mcmc-capacity-gates-20261017-v1"
    environments = {}
    for arm in ("baseline", "capacity"):
        gate = json.loads((gates / arm / "probe/summary.json").read_text())
        assert gate["status"] == "passed" and gate["world_size"] == 2 and len(gate["ranks"]) == 2
        assert all(row["training_code_hash"] == CORE for row in gate["ranks"])
        assert gate["ranks"][0]["environment"] == gate["ranks"][1]["environment"]
        environments[arm] = gate["ranks"][0]["environment"]
    overlay = gates / "capacity-overlay"
    assert environments["capacity"].endswith(sha256_file(overlay / "overlay.json"))
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        assert code_hash() == CORE
        if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
            raise ValueError("cycle gates require 20 GiB free disk")
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "protocol.json", {"cases": list(CASES), "updates_per_case": UPDATES,
            "start_optimizer_step": START_STEP, "local_points": LOCAL_POINTS, "initial_live_per_rank": 4,
            "nominal_lr_iterations": 30000, "refine_every": 100, "virtual_regularizer_global_count": 3000000,
            "initial_opacity_moments": "steady_regularizer_gradient_and_square", "other_initial_moments": "zero",
            "initial_scale": .03, "target_scale": .04, "scene_scale": 1., "seed": 20260729,
            "image_gradient_scope": "total_gradient_minus_autograd_regularizer",
            "randomness_scope": "same_initial_seed_not_common_noise_after_sampling_divergence",
            "script_sha256": sha256_file(Path(__file__)),
            "environments": environments, "overlay_sha256": sha256_file(overlay / "overlay.json"), "training_core": CORE,
            "host_memory_limits": None, "max_per_rank_reserved_bytes": 18 * 1024**3,
            "stage_seconds": 900, "unit_seconds": 6000,
            "promotion_rule": "visible_and_imbalanced_complete_1000_both_arms_with_effective_visible_points_on_both_shards; capacity_raw_psnr_delta>=-0.5dB; identical_initialization_and_reference",
            "negative_control": "no_image_signal_may_exhaust_sources_without_being_a_quality_regression",
            "no_automatic_10k_submission": True, "test_rgb": "not_loaded"})
        records = {}
        for arm in ("baseline", "capacity"):
            for case in CASES:
                require_idle_gpu()
                case_root = root / arm / case
                case_root.mkdir(parents=True)
                output = case_root / "probe"
                prefix = ["/usr/bin/env", "-u", "PYTHONPATH"] if arm == "baseline" else ["/usr/bin/env", "PYTHONPATH=" + str(overlay)]
                command = prefix + [sys.executable, str(Path(__file__).resolve()), "--output-root", str(output),
                                    "--case", case, "--arm", arm, "--expected-environment", environments[arm]]
                monitor = CycleMonitor(output / "progress.jsonl", output / "memory", updates=UPDATES, main_stage=True,
                    limits={**LIMITS, "max_observed_global_gaussians": LOCAL_POINTS * 2, "max_main_stage_wall_seconds": 900})
                try:
                    run_stage(case_root, "train", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
                    records[f"{arm}/{case}"] = json.loads((output / "summary.json").read_text())
                except Exception as exc:
                    write_json(case_root / "case.failure.json", {"reason": f"{type(exc).__name__}: {exc}"})
        assessment = assess_pair(records)
        write_json(root / "summary.json", {"cases": records, **assessment, "real_scene_training": "not_run"})
        if not assessment["eligible_for_10k"]:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--arm", choices=("baseline", "capacity"))
    parser.add_argument("--expected-environment")
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅在已授权远端项目执行，不运行本机模型")
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(PROJECT / "outputs/analysis"):
        raise ValueError("independent research output required")
    if args.case:
        if args.arm is None or args.expected_environment is None:
            raise ValueError("worker requires explicit arm and environment")
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
