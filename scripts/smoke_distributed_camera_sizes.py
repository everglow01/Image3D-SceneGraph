#!/usr/bin/env python3
"""Remote-only camera-size equivalence diagnostics; never changes the renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import TrainingMonitor, run_stage, write_json
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from scripts.run_mcmc_stability_ablation import LIMITS, PROJECT, unlimited_host_cgroup

ATOL, RTOL = 2e-6, 2e-4
UPDATES = 5
UNIT = "image3d-camera-size-equivalence-20261019-v1.service"
CASES = ("equal_small", "equal_large", "historical", "historical_camera_swap",
         "mixed", "mixed_camera_swap", "mixed_owner_swap", "unequal_shards", "invisible_owner")
CONTROLS = ("equal_small", "equal_large", "unequal_shards", "invisible_owner")
ORDER_CASES = CASES + ("equal_small_unique", "historical_unique", "mixed_unique")
ORDER_UNIT = "image3d-camera-size-order-20261019-v2.service"


def reference_mapping(spec, ordered):
    order = spec["ids"][0] + spec["ids"][1] if ordered else list(range(16))
    inverse = {global_id: index for index, global_id in enumerate(order)}
    return order, [[inverse[i] for i in ids] for ids in spec["ids"]]


def case_spec(name):
    original_name = name
    unique = name.endswith("_unique")
    if unique:
        name = name.removesuffix("_unique")
    if original_name not in ORDER_CASES:
        raise ValueError("unknown camera-size case")
    sizes = [[64, 64], [64, 64]]
    if name == "equal_large":
        sizes = [[1361, 2048], [1361, 2048]]
    elif name.startswith("historical"):
        sizes = [[1361, 2048], [1372, 2048]]
    elif name.startswith("mixed"):
        sizes = [[64, 64], [96, 80]]
    ids = [list(range(0, 16, 2)), list(range(1, 16, 2))]
    if name == "unequal_shards":
        ids = [list(range(7)), list(range(7, 16))]
    elif name == "mixed_owner_swap":
        ids.reverse()
    assignment = [1, 0] if name.endswith("camera_swap") else [0, 1]
    focal = min(sizes[0]) * .5
    cameras = [{"id": str(i), "width": w, "height": h,
                "fx": focal, "fy": focal, "cx": sizes[0][0] / 2,
                "cy": sizes[0][1] / 2, "tx": .04 * i, "ty": .02 * i}
               for i, (w, h) in enumerate(sizes)]
    points, scales = [], []
    for i in range(16):
        if i < 8:
            points.append([-.35 + .1 * i, -.2 + .12 * (i % 4), 2 + .1 * (i % 3)])
            scales.append(.025 + .003 * (i % 3))
        else:
            cam = cameras[((i - 8) // 2) % 2]
            x = (sizes[0][0] + sizes[1][0]) / 2
            y = (sizes[0][1] + sizes[1][1]) / 2 if i >= 12 else cam["cy"]
            points.append([(x - cam["cx"]) * 2 / focal - cam["tx"],
                           (y - cam["cy"]) * 2 / focal - cam["ty"], 2.])
            scales.append(.001)
    if name == "invisible_owner":
        for i in ids[1]:
            points[i][2] = -2.
    if unique:
        for i, point in enumerate(points):
            factor = (2 + .05 * i) / point[2]
            points[i] = [value * factor for value in point]
            scales[i] *= factor
    return {"name": original_name, "cameras": cameras, "camera_assignment": assignment,
            "ids": ids, "points": points, "scales": scales, "sh_degree": 3,
            "edge_band_pixels": 32, "synthetic_geometry_not_scene_replay": True}


def assess(records, *, ordered=False):
    cases = ORDER_CASES if ordered else CASES
    control_names = CONTROLS + ("equal_small_unique",) if ordered else CONTROLS
    missing = [name for name in cases if name not in records]
    incomplete = [name for name, row in records.items()
                  if row.get("completed_updates") != UPDATES or len(row.get("ranks", [])) != 2
                  or sorted(r.get("rank") for r in row.get("ranks", [])) != [0, 1]
                  or any(r.get("completed_updates") != UPDATES for r in row.get("ranks", []))]
    packed = [name for name, row in records.items() if not row.get("packed_control_passed", False)]
    controls = [name for name in control_names if name in records and not records[name].get("equivalent", False)]
    mismatches = [name for name, row in records.items() if not row.get("equivalent", False)]
    status = ("incomplete" if missing or incomplete else "control_blocked" if packed or controls
              else "not_equivalent" if mismatches else "equivalent")
    return {"status": status, "missing_cases": missing, "incomplete_cases": incomplete,
            "packed_control_failures": packed, "same_size_control_failures": controls,
            "non_equivalent_cases": mismatches, "passed": status == "equivalent"}


def comparison(actual, expected, *, image=False, adaptive_edge=False):
    import numpy as np

    if actual is None or expected is None:
        return {"passed": actual is None and expected is None,
                "actual_none": actual is None, "expected_none": expected is None}
    a, b = actual.detach().cpu().numpy(), expected.detach().cpu().numpy()
    if a.shape != b.shape:
        raise ValueError("comparison shape mismatch")
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.zeros(a.shape, dtype=np.float64)
    np.subtract(a, b, out=diff, where=finite, dtype=np.float64)
    absolute = np.abs(diff)
    bad = ~finite | (absolute > ATOL + RTOL * np.abs(b))
    maximum = int(absolute.argmax()) if absolute.size else 0
    result = {"passed": not bool(bad.any()), "shape": list(a.shape),
              "nonfinite_actual": int((~np.isfinite(a)).sum()),
              "nonfinite_expected": int((~np.isfinite(b)).sum()),
              "zero_actual": int((a == 0).sum()), "zero_expected": int((b == 0).sum()),
              "elements": int(a.size), "mismatches": int(bad.sum()),
              "mismatch_fraction": float(bad.mean()) if a.size else 0.,
              "max_abs_finite": float(absolute.max()) if a.size else 0.,
              "mse_finite": float((diff[finite] ** 2).mean()) if finite.any() else None,
              "max_abs_coordinate": [int(v) for v in np.unravel_index(maximum, a.shape)] if a.size else []}
    if image:
        edge = np.ones(a.shape[:2], dtype=bool)
        band = min(32, max(1, min(a.shape[:2]) // 4)) if adaptive_edge else 32
        edge[band:-band, band:-band] = False
        result["edge_band_pixels"] = band
        result.update(edge_mismatches=int(bad[edge].sum()), interior_mismatches=int(bad[~edge].sum()))
    return result


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def worker(local_rank, rank, world_size, args):
    import torch
    from gsplat.rendering import rasterization
    from image3d_scenegraph.gaussian.absgrad_resources import failure_record, memory_telemetry
    from image3d_scenegraph.gaussian.config import resolve_internal_config
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.render import RenderCamera, render_gaussians, NEAR_PLANE_NORMALIZED, FAR_PLANE_NORMALIZED
    from image3d_scenegraph.gaussian.trainer import training_provenance, _gsplat_version
    from image3d_scenegraph.gaussian.training_math import l1_ssim_loss

    if world_size != 2 or local_rank != rank:
        raise ValueError("exactly two local ranks required")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    root = args.output_root
    rank_dir = root / f"rank-{rank}"
    rank_dir.mkdir()
    spec = case_spec(args.case)
    ordered = args.profile == "order-v2"
    order, mappings = reference_mapping(spec, ordered)
    reference_indices = mappings[rank]
    canonical_indices = [order.index(i) for i in range(16)]
    cfg = resolve_internal_config("mcmc_v1").effective_config
    ids = spec["ids"][rank]
    camera_index = spec["camera_assignment"][rank]
    with memory_telemetry(root / "memory", local_rank, rank), failure_record(rank_dir, args.case, "camera_equivalence"):
        if ";" in _gsplat_version():
            raise ValueError("overlays are not allowed")
        def model(indices):
            full_points = torch.tensor(spec["points"], device=device, dtype=torch.float32)
            colors = torch.tensor([[.3 + .02*i, .25 + .01*(i % 5), .4] for i in range(16)], device=device)
            initial = GaussianModel.from_points(full_points[indices], colors[indices],
                torch.tensor(spec["scales"], device=device)[indices], initial_opacity=.6, max_sh_degree=3)
            with torch.no_grad():
                initial.params["quats"].copy_(torch.tensor([[1., .02*(i % 3), -.03, .04] for i in indices], device=device))
                initial.params["scales"][:, 1] += math.log(.8)
                initial.params["scales"][:, 2] += math.log(1.2)
                initial.params["shN"].copy_(torch.tensor([[[.01*math.sin(i*45+j*3+k) for k in range(3)]
                    for j in range(15)] for i in indices], device=device))
            return initial

        cameras, targets = [], []
        for cam in spec["cameras"]:
            pose = torch.eye(4, device=device)
            pose[0, 3], pose[1, 3] = cam["tx"], cam["ty"]
            intrinsic = torch.tensor([[cam["fx"], 0., cam["cx"]], [0., cam["fy"], cam["cy"]], [0., 0., 1.]], device=device)
            cameras.append(RenderCamera(cam["id"], pose, intrinsic, cam["width"], cam["height"]))
            x = torch.linspace(0, 1, cam["width"], device=device)[None, :].expand(cam["height"], -1)
            y = torch.linspace(0, 1, cam["height"], device=device)[:, None].expand(-1, cam["width"])
            targets.append(torch.stack([.15+.2*x, .2+.15*y, .25+.1*x*y], dim=-1))

        reference, shard = model(order), model(ids)
        ref_optim, optim = reference.optimizers(cfg["learning_rate"]), shard.optimizers(cfg["learning_rate"])
        initial_hash = tensor_hash({name: value[canonical_indices] for name, value in reference.params.items()})
        if args.case.endswith("_unique") and torch.unique(reference.means[:, 2]).numel() != 16:
            raise ValueError("unique-depth fixture contains float32 depth ties")
        checks = [comparison(value, reference.params[name][reference_indices]) for name, value in shard.params.items()]
        if not all(c["passed"] for c in checks):
            raise ValueError("initial shard differs from full reference")
        identity = {"rank": rank, "case": args.case, "ids": ids, "camera_index": camera_index,
                    "initial_full_sha256": initial_hash, "reference_global_ids": order,
                    "reference_indices": reference_indices, "profile": args.profile,
                    "targets_sha256": tensor_hash({str(i): t for i, t in enumerate(targets)}),
                    "spec": spec, "environment": _gsplat_version(),
                    "training_core": training_provenance(dataset_hash="0"*64, effective_config_hash="0"*64, world_size=2).code_hash,
                    "optimizer": {name: {k: group[k] for k in ("lr", "betas", "eps")}
                                  for name, opt in optim.items() for group in opt.param_groups}}
        write_json(rank_dir / "identity.json", identity)
        equivalent, packed_pass = True, True
        first_forward, first_gradient = True, True
        for step in range(1, UPDATES+1):
            if (root.parent / "cancel").exists():
                raise RuntimeError("resource monitor requested cancellation")
            for opt in [*optim.values(), *ref_optim.values()]:
                opt.zero_grad(set_to_none=True)
            measurements = {}
            reference_image = reference_alpha = None
            for i, cam in enumerate(cameras):
                image, alpha, _ = rasterization(*reference.activated(), cam.camera_from_normalized[None],
                    cam.intrinsic[None], width=cam.width, height=cam.height, sh_degree=3,
                    packed=False, distributed=False, near_plane=NEAR_PLANE_NORMALIZED, far_plane=FAR_PLANE_NORMALIZED)
                if i == camera_index:
                    reference_image, reference_alpha = image[0].detach(), alpha[0].detach()
                    if step == 1 and ordered and args.case == "equal_small":
                        canonical = model(list(range(16)))
                        with torch.no_grad():
                            other, other_alpha, _ = rasterization(*canonical.activated(), cam.camera_from_normalized[None],
                                cam.intrinsic[None], width=cam.width, height=cam.height, sh_degree=3,
                                packed=False, distributed=False, near_plane=NEAR_PLANE_NORMALIZED,
                                far_plane=FAR_PLANE_NORMALIZED)
                        measurements["order_probe_rgb"] = comparison(other[0], reference_image, image=True, adaptive_edge=True)
                        measurements["order_probe_alpha"] = comparison(other_alpha[0], reference_alpha, image=True, adaptive_edge=True)
                        del canonical, other, other_alpha
                    if step == 1:
                        with torch.no_grad():
                            packed = render_gaussians(reference, cam, sh_degree=3)
                        for label, a, b in [("rgb", packed.image, reference_image), ("alpha", packed.alpha, reference_alpha)]:
                            check = comparison(a, b, image=True, adaptive_edge=ordered)
                            measurements[f"packed_dense_{label}"] = check
                            packed_pass &= check["passed"]
                        del packed
                loss, _ = l1_ssim_loss(image[0].clamp(0, 1), targets[i], l1_weight=.8, ssim_weight=.2)
                (loss / 2).backward()
                del image, alpha, loss
            rendered = render_gaussians(shard, cameras[camera_index], sh_degree=3, distributed=True)
            for label, a, b in [("rgb", rendered.image, reference_image), ("alpha", rendered.alpha, reference_alpha)]:
                measurements[label] = comparison(a, b, image=True, adaptive_edge=ordered)
            loss, _ = l1_ssim_loss(rendered.image.clamp(0, 1), targets[camera_index], l1_weight=.8, ssim_weight=.2)
            (loss / 2).backward()
            for name, param in shard.params.items():
                ref_grad = reference.params[name].grad
                measurements[f"gradient_{name}"] = comparison(param.grad, None if ref_grad is None else ref_grad[reference_indices])
            if step == 1:
                first_forward = all(measurements[k]["passed"] for k in ("rgb", "alpha"))
                first_gradient = all(v["passed"] for k, v in measurements.items() if k.startswith("gradient_"))
            # Retain the first differing gradients before optimizer validation or another collective.
            with (rank_dir / "before-update.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "measurements": measurements}, allow_nan=False)+"\n")
            shard.validate_gradients()
            reference.validate_gradients()
            for opt in [*optim.values(), *ref_optim.values()]:
                opt.step()
            for name, param in shard.params.items():
                measurements[f"parameter_{name}"] = comparison(param, reference.params[name][reference_indices])
                a, b = optim[name].state.get(param, {}), ref_optim[name].state.get(reference.params[name], {})
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    expected = b.get(field)
                    measurements[f"adam_{name}_{field}"] = comparison(a.get(field),
                        expected if field == "step" or expected is None else expected[reference_indices])
            passed = all(v["passed"] for k, v in measurements.items() if not k.startswith(("packed_dense_", "order_probe_")))
            equivalent &= passed
            with (rank_dir / "comparisons.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "passed": passed, "measurements": measurements}, allow_nan=False)+"\n")
            shard.validate()
            reference.validate()
            torch.distributed.barrier()
            if rank == 0:
                with (root / "progress.jsonl").open("a") as f:
                    f.write(json.dumps({"iteration": step, "world_size": 2, "gaussian_count": 16,
                        "batch_view_ids": [str(i) for i in spec["camera_assignment"]]})+"\n")
            del rendered, reference_image, reference_alpha, loss
        write_json(rank_dir / "summary.json", {**identity, "completed_updates": UPDATES,
            "equivalent": equivalent, "packed_control_passed": packed_pass,
            "first_forward_passed": first_forward, "first_gradient_passed": first_gradient})


def run(args):
    ordered = args.profile == "order-v2"
    unit = ORDER_UNIT if ordered else UNIT
    cases = ORDER_CASES if ordered else CASES
    unlimited_host_cgroup(unit)
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        if shutil.disk_usage(PROJECT).free < 20*1024**3:
            raise ValueError("requires 20 GiB free disk")
        args.output_root.mkdir(parents=True, exist_ok=False)
        import gsplat
        from image3d_scenegraph.gaussian.trainer import training_provenance, _gsplat_version
        package = Path(gsplat.__file__).parent
        if ";" in _gsplat_version():
            raise ValueError("original gsplat required")
        hashes = {str(p.relative_to(package)): sha256_file(p) for p in package.rglob("*")
                  if p.is_file() and p.suffix in (".py", ".cu", ".cuh", ".so")}
        write_json(args.output_root / "protocol.json", {"profile": args.profile, "cases": [case_spec(c) for c in cases],
            "updates": UPDATES, "atol": ATOL, "rtol": RTOL, "unit": unit,
            "environment": _gsplat_version(), "package_hashes": hashes,
            "core": training_provenance(dataset_hash="0"*64, effective_config_hash="0"*64, world_size=2).code_hash,
            "script_sha256": sha256_file(Path(__file__)), "host_limits": None,
            "stage_seconds": 300, "unit_seconds": 4200 if ordered else 3600, "max_reserved_each": 18*1024**3,
            "reference": "full_model_dense_two_camera_mean_loss_one_fresh_Adam_update",
            "topology_noise_regularizers": "disabled", "test_rgb": "not_loaded"})
        records = {}
        for case in cases:
            require_idle_gpu()
            stage = args.output_root / case
            stage.mkdir()
            output = stage / "probe"
            monitor = TrainingMonitor(output / "progress.jsonl", output / "memory", updates=UPDATES,
                main_stage=True, limits={**LIMITS, "max_main_stage_wall_seconds": 300, "max_observed_global_gaussians": 16})
            command = ["/usr/bin/env", "-u", "PYTHONPATH", sys.executable, str(Path(__file__).resolve()),
                       "--output-root", str(output), "--case", case, "--profile", args.profile]
            try:
                run_stage(stage, "probe", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
                records[case] = json.loads((output / "summary.json").read_text())
            except Exception as exc:
                write_json(stage / "case.failure.json", {"reason": f"{type(exc).__name__}: {exc}"})
        unchanged = all(sha256_file(package / name) == value for name, value in hashes.items())
        summary = {**assess(records, ordered=ordered), "cases": records, "installed_package_unchanged": unchanged}
        if ordered:
            summary["repair_ready"] = unchanged and summary["status"] == "not_equivalent"
        if not unchanged:
            summary.update(status="environment_changed", passed=False)
        write_json(args.output_root / "summary.json", summary)
        if not summary["passed"]:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case", choices=ORDER_CASES)
    parser.add_argument("--profile", choices=("legacy-v1", "order-v2"), default="legacy-v1")
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("仅在已授权远端执行，不运行本机模型")
    if args.profile == "legacy-v1" and args.case is not None and args.case not in CASES:
        raise ValueError("unique-depth cases require the order-v2 protocol")
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(PROJECT / "outputs/analysis"):
        raise ValueError("independent research output required")
    if args.case:
        import torch
        from gsplat.distributed import cli
        if torch.cuda.device_count() != 2:
            raise ValueError("two visible GPUs required")
        args.output_root.mkdir(parents=True, exist_ok=False)
        cli(worker, args, verbose=True)
        ranks = [json.loads((args.output_root / f"rank-{r}/summary.json").read_text()) for r in (0, 1)]
        for field in ("initial_full_sha256", "targets_sha256", "spec", "environment", "training_core", "optimizer"):
            if ranks[0][field] != ranks[1][field]:
                raise ValueError(f"rank identity differs: {field}")
        write_json(args.output_root / "summary.json", {"ranks": ranks, "completed_updates": UPDATES,
            "equivalent": all(r["equivalent"] for r in ranks),
            "packed_control_passed": all(r["packed_control_passed"] for r in ranks)})
    else:
        run(args)


if __name__ == "__main__":
    main()
