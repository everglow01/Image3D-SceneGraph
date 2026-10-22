#!/usr/bin/env python3
"""Five fixed-context original rasterizer backwards per camera; no training or forward."""
from __future__ import annotations

import argparse
import fcntl
import json
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

from . import legacy
from . import tensor_analysis as analysis
from .legacy import (PROJECT, LIMITS, snapshots, boundary, digest_state, rng_state,
                     unchanged_rng, file_hashes, unlimited_host_cgroup)
from image3d_scenegraph.file_integrity import sha256_file
from .legacy import run_stage
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu

PROFILE = "mcmc_fixed_b0_five_repeats_workspace_v1"
SOURCE = PROJECT / "outputs/analysis/backward-boundary-20261022-v1"
SOURCE_SHA = "d17d6f12125e3d15fe11f02ce822c872c2663a073f619108f958748a5b53ad0c"
COMPARISON_SHA = "c5f7efd5bac8b73e16c7067d59ed6a6ea811f2287fab397817922bfeba9b7c24"
UNIT = "image3d-fixed-b0-workspace-v1.service"
MODULE = "experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0.run"
EXPERIMENT_REL = "experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e004_fixed_b0"
RUNS_ROOT = PROJECT / EXPERIMENT_REL / "runs"
EXECUTION_FILES = tuple(EXPERIMENT_REL + "/" + name for name in (
    "run.py", "tensor_analysis.py", "legacy.py", "gpu-source.json",
    "tests/test_fixed_b0_gpu_runner.py", "tests/test_fixed_b0_gpu_remote.py"))
DEPENDENCIES = legacy.LEGACY_FILES + EXECUTION_FILES
CONTRACT = {"profile": PROFILE, "legacy_source_manifest_sha256": legacy.MANIFEST_SHA256, "source": str(SOURCE), "source_protocol_sha256": SOURCE_SHA,
    "source_comparison_sha256": COMPARISON_SHA, "branch": "a", "ranks": [0, 1], "repeats_per_rank": analysis.REPEATS,
    "pairs_per_rank": [list(x) for x in analysis.PAIRS], "comparison_reference": "i_actual_j_reference_i_less_than_j",
    "selection": analysis.SELECTION, "anomalies": [list(x) for x in analysis.ANOMALIES],
    "atol": snapshots.ATOL, "rtol": snapshots.RTOL, "host_limits": None, "limits": LIMITS,
    "new_optimizer_updates": 0, "new_forward_calls": 0, "backward_calls": 10, "tensor_pair_count": 80,
    "rng_policy": "unchanged_post_initialization_rng_no_restore",
    "initialization": "audited_pinned_precompiled_import_without_warmup_v1",
    "old_gate1": "not_reclassified", "scope": "same_process_per_camera_five_repeats_not_global_error_bound",
    "propagation": "historical_AA_only_not_replayed_downstream", "test_boundary": "no_Test_or_Validation_read"}
read, write_new = legacy.read, snapshots.write_new


def verify_source():
    return legacy.verify_source()


def prepare(root):
    p = verify_source()
    source = analysis.Source(SOURCE)
    points = analysis.select_points(source)
    bound = {str(PROJECT / name): sha256_file(PROJECT / name) for name in DEPENDENCIES}
    for path in [SOURCE / "protocol.json", SOURCE / "comparison.json", *(SOURCE / b / "complete.json" for b in ("a", "b"))]:
        bound[str(path)] = sha256_file(path)
    for b in ("a", "b"):
        for r in (0, 1):
            for stage in boundary.ORDER:
                folder = SOURCE / b / f"rank-{r}" / stage
                manifest = snapshots.verify_snapshot_files(folder)
                bound[str(folder / "manifest.json")] = sha256_file(folder / "manifest.json")
                bound.update({str(folder / name): entry["sha256"] for name, entry in manifest["files"].items()})
    # Original T1 identifies why these three points were nominated; it is not new propagation data.
    prior = PROJECT / "outputs/analysis/first-divergence-t1-20261021-v1"
    for name in ("protocol.json", "comparison.json"):
        bound[str(prior / name)] = sha256_file(prior / name)
    if bound[str(prior / "protocol.json")] != "f693126c9fd27e2fab269dd3c488830b56e081654886f2baf39175fac1f5e203":
        raise ValueError("prior nomination evidence differs")
    record = {**CONTRACT, "bound_files": bound, "points": points, "core": legacy.CORE,
              "package": p["pair"]["source_package"], "package_hashes": p["pair"]["source_hashes"],
              "disk_admission_bytes": 20 * snapshots.GIB, "status": "prepared_not_executed"}
    verify_inputs(record)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "protocol.json", record)


def protocol_at(root, digest):
    legacy.verify()
    if not digest or sha256_file(snapshots.safe_file(root, "protocol.json")) != digest:
        raise ValueError("explicit protocol SHA required")
    p = read(root / "protocol.json")
    if any(p.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("fixed-B0 contract differs")
    for name in DEPENDENCIES:
        path = str(PROJECT / name)
        if p["bound_files"].get(path) != sha256_file(snapshots.safe_file(PROJECT, name)):
            raise ValueError("execution dependency binding differs: " + name)
    required = {str(PROJECT / n) for n in DEPENDENCIES} | {str(SOURCE / "protocol.json"), str(SOURCE / "comparison.json")}
    required |= {str(SOURCE / b / "complete.json") for b in ("a", "b")}
    required |= {str(PROJECT / "outputs/analysis/first-divergence-t1-20261021-v1" / name) for name in ("protocol.json", "comparison.json")}
    for b in ("a", "b"):
        for r in (0, 1):
            for stage in boundary.ORDER:
                folder = SOURCE / b / f"rank-{r}" / stage
                required.add(str(folder / "manifest.json"))
                manifest = read(folder / "manifest.json")
                for name, entry in manifest["files"].items():
                    path = str(snapshots.safe_file(folder, name))
                    if p["bound_files"].get(path) != entry["sha256"]:
                        raise ValueError("missing source tensor binding")
    if (p["bound_files"].get(str(SOURCE / "protocol.json")) != SOURCE_SHA
            or p["bound_files"].get(str(SOURCE / "comparison.json")) != COMPARISON_SHA):
        raise ValueError("source identity differs")
    if not required <= p["bound_files"].keys() or p["disk_admission_bytes"] != 20 * snapshots.GIB:
        raise ValueError("missing bindings or disk admission")
    points = p["points"]
    if (len(points) != 6 or len({(x["owner"], x["slot"]) for x in points}) != 6
            or [(x["owner"], x["slot"], x["parameter"]) for x in points if x["role"] == "prior_anomaly"] != list(analysis.ANOMALIES)):
        raise ValueError("frozen point selection differs")
    controls = [x for x in points if x["role"] == "matched_control"]
    if (len(controls) != 3 or any(x["owner"] not in (0, 1) or not 0 <= x["slot"] < analysis.OWNER_SIZE for x in points)
            or [(x["owner"], x["matched_to"], x["parameter"]) for x in controls] != list(analysis.ANOMALIES)):
        raise ValueError("invalid matched controls")
    return p


def verify_inputs(p):
    legacy.verify()
    if legacy.core_hash() != p["core"] or p["core"] != legacy.CORE:
        raise ValueError("core differs")
    source_protocol = read(SOURCE / "protocol.json")
    if p["package"] != source_protocol["pair"]["source_package"] or p["package_hashes"] != source_protocol["pair"]["source_hashes"]:
        raise ValueError("package identity differs")
    if file_hashes(Path(p["package"])) != p["package_hashes"]:
        raise ValueError("original package differs")
    for name, digest in p["bound_files"].items():
        path = Path(name)
        if not path.is_relative_to(PROJECT) or sha256_file(snapshots.safe_file(PROJECT, path.relative_to(PROJECT))) != digest:
            raise ValueError(f"bound file changed: {name}")
    if shutil.disk_usage(PROJECT).free < 4 * snapshots.GIB:
        raise ValueError("disk reserve breached")


def validate_context(payload):
    import numpy as np
    x, g, settings = payload["inputs"], payload["image_gradients"], payload["settings"]
    width, height, tile = (settings[k] for k in ("width", "height", "tile_size"))
    if any(type(v) is not int or v <= 0 for v in (width, height)) or tile != 16:
        raise ValueError("invalid image/tile dimensions")
    n = x["means2d"].shape[1]
    shapes = {"means2d": (1, n, 2), "conics": (1, n, 3), "colors": (1, n, 3), "opacities": (1, n),
              "render_alphas": (1, height, width, 1), "last_ids": (1, height, width),
              "isect_offsets": (1, (height + tile - 1) // tile, (width + tile - 1) // tile)}
    for key, shape in shapes.items():
        dtype = np.int32 if key in ("last_ids", "isect_offsets") else np.float32
        if x[key].shape != shape or x[key].dtype != dtype or not np.isfinite(x[key]).all():
            raise ValueError(f"invalid input structure: {key}")
    for key, channels in (("rgb", 3), ("alpha", 1)):
        if g[key].shape != (1, height, width, channels) or g[key].dtype != np.float32 or not np.isfinite(g[key]).all():
            raise ValueError("invalid image gradient")
    ids, offsets = x["flatten_ids"], x["isect_offsets"].reshape(-1)
    if (ids.ndim != 1 or ids.dtype != np.int32 or not ids.size or ids.min() < 0 or ids.max() >= n
            or offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) < 0) or offsets[-1] > ids.size):
        raise ValueError("invalid tile indices")
    alpha, last = x["render_alphas"][0, :, :, 0], x["last_ids"][0]
    if (alpha < 0).any() or (alpha >= 1).any():
        raise ValueError("invalid saved alpha")
    ends = np.append(offsets[1:], ids.size)
    tile_ids = (np.arange(height) // tile)[:, None] * shapes["isect_offsets"][2] + np.arange(width)[None, :] // tile
    if ((alpha > 0) & ((last < offsets[tile_ids]) | (last >= ends[tile_ids]))).any():
        raise ValueError("last intersection outside own tile")


def load_context(folder, rank):
    import numpy as np
    import torch
    record = snapshots.load_snapshot(folder)
    boundary.validate_boundary_schema(record, "B0")
    fields = {k["value"]: v for k, v in record["tree"]["items"]}
    cpu = {k: analysis.decode(fields[k], folder) for k in ("inputs", "image_gradients", "settings")}
    validate_context(cpu)

    def to_cuda(array, info):
        if info["kind"] != "torch" or info["device"] != f"cuda:{rank}":
            raise ValueError("input is not from this camera GPU")
        value = torch.from_numpy(np.array(array, copy=True, order="C")).to(f"cuda:{rank}")
        if (not value.is_contiguous() or list(value.stride()) != info["stride"]
                or snapshots.array_hash(value.detach().cpu().numpy()) != info["canonical_sha256"]):
            raise ValueError("CUDA content/layout identity differs")
        return value

    payload = {k: analysis.decode(fields[k], folder, to_cuda) for k in ("inputs", "image_gradients", "settings")}
    settings = payload["settings"]
    if (settings["absgrad"] or settings["needs_input_grad"] != (True,) * 4 + (False,) * 8
            or payload["inputs"]["backgrounds"] is not None or payload["inputs"]["masks"] is not None):
        raise ValueError("unsupported frozen rasterizer context")
    ctx = SimpleNamespace(saved_tensors=tuple(payload["inputs"][k] for k in boundary.SAVED), **settings)
    return ctx, payload


def storage_ids(values):
    return {(str(v.device), v.untyped_storage().data_ptr()) for v in values if v is not None and v.numel()}


def initialize_backend():
    if sha256_file(SOURCE / "protocol.json") != SOURCE_SHA:
        raise ValueError("initialization source protocol differs")
    pair = read(SOURCE / "protocol.json")["pair"]
    package = Path(pair["source_package"])
    before = rng_state()
    before_hash = digest_state(before)
    components_before = {k: digest_state(v) for k, v in before.items()}
    spec = find_spec("gsplat.csrc")
    if spec is None or not isinstance(spec.loader, ExtensionFileLoader) or not spec.origin:
        raise ValueError("pinned precompiled extension required; no JIT fallback")
    path = snapshots.safe_file(package, Path(spec.origin).relative_to(package))
    expected = pair["source_hashes"].get(str(path.relative_to(package)))
    if expected is None or sha256_file(path) != expected:
        raise ValueError("precompiled extension identity differs")
    # Import the installed binary first: a missing binary fails instead of entering backend JIT.
    extension = import_module("gsplat.csrc")
    backend = import_module("gsplat.cuda._backend")
    if (backend._C is not extension or Path(extension.__file__).resolve() != path.resolve()
            or not callable(getattr(extension, "rasterize_to_pixels_3dgs_bwd", None))
            or sha256_file(path) != expected):
        raise ValueError("loaded backend identity differs")
    after = rng_state()
    components_after = {k: digest_state(v) for k, v in after.items()}
    return {"method": CONTRACT["initialization"], "extension": str(path), "extension_sha256": expected,
            "rng_before": components_before, "rng_after": components_after,
            "rng_before_sha256": before_hash, "rng_after_sha256": digest_state(after),
            "changed_rng_components": [k for k in components_before if components_before[k] != components_after[k]],
            "warmup_forward_calls": 0, "warmup_backward_calls": 0}


def replay_five(folder, output, rank):
    import torch
    from gsplat.cuda._wrapper import _RasterizeToPixels
    ctx, payload = load_context(folder, rank)
    output.mkdir(exist_ok=False)
    initialization = initialize_backend()
    write_new(output / "initialization.json", initialization)
    inputs = [*ctx.saved_tensors, *payload["image_gradients"].values()]
    torch.cuda.synchronize(rank)
    expected = digest_state(payload)
    rng = digest_state(rng_state())
    if rng != initialization["rng_after_sha256"]:
        raise ValueError("RNG changed between initialization and measurement")
    previous, records = None, []
    for trial in range(analysis.REPEATS):
        torch.cuda.synchronize(rank)
        if digest_state(payload) != expected or digest_state(rng_state()) != rng:
            raise ValueError("input/RNG changed before backward")
        if shutil.disk_usage(output).free < 4 * snapshots.GIB:
            raise ValueError("disk reserve breached")
        with unchanged_rng():
            result = _RasterizeToPixels.backward(ctx, payload["image_gradients"]["rgb"], payload["image_gradients"]["alpha"])
            torch.cuda.synchronize(rank)
            if len(result) != 12 or any(v is not None for v in result[4:]):
                raise ValueError("unexpected backward signature")
            values = result[:4]
            if any(v.shape != x.shape or v.dtype != x.dtype or v.device != x.device for v, x in zip(values, ctx.saved_tensors[:4])):
                raise ValueError("backward output structure differs")
            if len(storage_ids(values)) != 4 or storage_ids(values) & storage_ids(inputs + list(previous or ())):
                raise ValueError("backward output aliases live storage")
            if any(not torch.isfinite(v).all().item() for v in values):
                raise ValueError("nonfinite backward output")
            if digest_state(payload) != expected:
                raise ValueError("backward changed fixed inputs")
            payload_out = {"outputs": dict(zip(boundary.OUTPUTS, result[:5])), "rng": rng_state()}
            before_save = digest_state(payload_out)
            digest = snapshots.save_snapshot(output / f"trial-{trial}", payload_out)
            if digest_state(payload_out) != before_save:
                raise ValueError("snapshot changed outputs")
        if digest_state(payload) != expected or digest_state(rng_state()) != rng:
            raise ValueError("snapshot changed inputs/RNG")
        records.append({"trial": trial, "manifest_sha256": digest, "input_sha256": expected, "rng_sha256": rng})
        previous = values
    return {"rank": rank, "status": "replay_complete", "original_backward_calls": len(records),
            "source_manifest_sha256": sha256_file(folder / "manifest.json"), "trials": records,
            "initialization_sha256": sha256_file(output / "initialization.json"),
            "new_optimizer_updates": 0, "new_forward_calls": 0}


def verify_worker(stage, rank):
    record = read(stage / f"worker-rank-{rank}.json")
    if (record["rank"], record["status"], record["original_backward_calls"], record["new_optimizer_updates"], record["new_forward_calls"]) != (
            rank, "replay_complete", 5, 0, 0):
        raise ValueError("worker identity or budget differs")
    if [r["trial"] for r in record["trials"]] != list(range(5)):
        raise ValueError("missing/duplicate/reordered trial")
    if record["source_manifest_sha256"] != sha256_file(SOURCE / "a" / f"rank-{rank}" / "B0/manifest.json"):
        raise ValueError("wrong B0 source")
    if len({(r["input_sha256"], r["rng_sha256"]) for r in record["trials"]}) != 1:
        raise ValueError("input/RNG was not fixed")
    init_path = snapshots.safe_file(stage, f"rank-{rank}/initialization.json")
    if sha256_file(init_path) != record["initialization_sha256"]:
        raise ValueError("initialization audit changed")
    initialization = read(init_path)
    if (initialization["method"] != CONTRACT["initialization"]
            or initialization["warmup_forward_calls"] != 0 or initialization["warmup_backward_calls"] != 0
            or initialization["rng_after_sha256"] != record["trials"][0]["rng_sha256"]):
        raise ValueError("initialization/measurement boundary differs")
    for trial in record["trials"]:
        folder = stage / f"rank-{rank}" / f"trial-{trial['trial']}"
        if sha256_file(snapshots.safe_file(folder, "manifest.json")) != trial["manifest_sha256"]:
            raise ValueError("trial manifest changed")
        boundary.validate_boundary_schema(snapshots.load_snapshot(folder), "B1")
    return record


def worker(rank, args):
    import torch
    import gsplat
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    unlimited_host_cgroup(UNIT)
    if torch.cuda.device_count() != 2:
        raise ValueError("exactly two GPUs required")
    torch.cuda.set_device(rank)
    p = protocol_at(args.output_root, args.protocol_sha256)
    if Path(gsplat.__file__).resolve().parent != Path(p["package"]).resolve():
        raise ValueError("not original gsplat")
    runtime = snapshots.runtime_environment({}, rank)
    historical = analysis.Source(SOURCE).get("a", rank, "P0", "runtime")
    for field in runtime:
        if field != "optimizers" and json.dumps(runtime[field], sort_keys=True) != json.dumps(historical[field], sort_keys=True):
            raise ValueError(f"runtime setting differs: {field}")
    stage = args.output_root / "replay"
    failure = stage / f"failure-rank-{rank}"
    failure.mkdir(exist_ok=False)
    with memory_telemetry(stage / "memory", rank, rank), failure_record(failure, "fixed_B0", str(rank)):
        record = replay_five(SOURCE / "a" / f"rank-{rank}" / "B0", stage / f"rank-{rank}", rank)
        write_new(stage / f"worker-rank-{rank}.json", {**record, "runtime": runtime})


class Monitor:
    def __init__(self, stage):
        self.stage, self.telemetry = stage, stage / "memory"
        self.peaks = [0, 0]
        self.completed = False

    def __call__(self, elapsed, *, final=False):
        if elapsed > LIMITS["max_main_stage_wall_seconds"]:
            return "replay_timeout"
        for rank in (0, 1):
            path = self.telemetry / f"rank-{rank}.json"
            if path.exists():
                self.peaks[rank] = max(self.peaks[rank], read(path)["peak_reserved_bytes"])
                if self.peaks[rank] > LIMITS["max_per_rank_peak_reserved_bytes"][rank]:
                    return "gpu_reserved_limit"
            elif final:
                return "missing_memory_telemetry"
        if final:
            for rank in (0, 1):
                verify_worker(self.stage, rank)
            self.completed = True

    def record(self):
        return {"original_backward_calls": 10 if self.completed else None, "new_optimizer_updates": 0,
                "new_camera_samples": 0, "per_rank_peak_reserved_bytes": self.peaks}


def run(args, p):
    unlimited_host_cgroup(UNIT)
    verify_inputs(p)
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        if shutil.disk_usage(PROJECT).free < p["disk_admission_bytes"]:
            raise ValueError("insufficient disk admission")
        stage = args.output_root / "replay"
        stage.mkdir(exist_ok=False)
        monitor = Monitor(stage)
        command = ["/usr/bin/env", "-u", "PYTHONPATH", sys.executable, "-m", MODULE,
                   "--output-root", str(args.output_root), "--worker", "--authorize-execution",
                   "--protocol-sha256", args.protocol_sha256, "--lease-fd", str(lease.fileno())]
        run_stage(stage, "replay", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
        verify_inputs(p)
        files = [stage / f"worker-rank-{r}.json" for r in (0, 1)] + [stage / "replay.exit.json"]
        write_new(stage / "complete.json", {"status": "replay_complete", "protocol_sha256": args.protocol_sha256,
            "resources": monitor.record(), "files": {f.name: sha256_file(f) for f in files}})


def compare(root, p, digest):
    import numpy as np
    verify_inputs(p)
    stage = root / "replay"
    complete = read(stage / "complete.json")
    if (complete["status"], complete["protocol_sha256"], complete["resources"]["original_backward_calls"]) != ("replay_complete", digest, 10):
        raise ValueError("incomplete replay")
    required = {"worker-rank-0.json", "worker-rank-1.json", "replay.exit.json"}
    if set(complete["files"]) != required:
        raise ValueError("missing completed worker binding")
    for name, h in complete["files"].items():
        if sha256_file(snapshots.safe_file(stage, name)) != h:
            raise ValueError("completion binding differs")
    for rank in (0, 1):
        verify_worker(stage, rank)
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    ranks, selected = {}, []
    source = analysis.Source(SOURCE)
    for rank in (0, 1):
        rank_results = {}
        records = [snapshots.load_snapshot(stage / f"rank-{rank}" / f"trial-{t}") for t in range(5)]
        for name in boundary.OUTPUTS[:4]:
            arrays = []
            for trial, record in enumerate(records):
                node = snapshots.snapshot_leaves(record)[("outputs", name)]
                arrays.append(analysis.decode(node, stage / f"rank-{rank}" / f"trial-{trial}"))
            rank_results[name] = analysis.repeated_statistics(arrays, folder, f"rank-{rank}-{name}")
            historical = source.get("a", rank, "B1", "outputs", name)
            rank_results[name]["native_a_reference_only"] = [snapshots.compare_arrays(a, historical) for a in arrays]
            for point in p["points"]:
                index = point["owner"] * analysis.OWNER_SIZE + point["slot"]
                samples = np.stack([a[0, index] for a in arrays]).astype(np.float64)
                selected.append({"camera_rank": rank, "owner": point["owner"], "slot": point["slot"], "output": name,
                    "source": "new_fixed_B0_repeats_B1_only", "values": samples.tolist(),
                    "range": (samples.max(0) - samples.min(0)).tolist()})
        ranks[str(rank)] = rank_results
    historical = analysis.historical_points(source, p["points"])
    verify_inputs(p)
    summaries = [v["summary"] for rank in ranks.values() for v in rank.values()]
    observed_status = ("numerically_inconsistent" if any(any(s["above_tolerance_pair_counts"]) for s in summaries) else
                       "numerically_equivalent" if any(any(s["strict_unequal_pair_counts"]) for s in summaries) else
                       "bitwise_equal" if all(not any(s["bitwise_unequal_ratios"]) for s in summaries) else "numerically_equivalent")
    report = {**CONTRACT, "observed_pairwise_status": observed_status, "protocol_sha256": digest, "complete_sha256": sha256_file(stage / "complete.json"),
              "status": "analysis_complete", "ranks": ranks, "repeat_point_outputs": selected,
              "initializations": {str(r): read(stage / f"rank-{r}/initialization.json") for r in (0, 1)},
              "historical_point_analysis": historical, "global_boundedness_proven": False,
              "new_P2_P3_computed": False, "MCMC_long_term_causality_proven": False}
    write_new(root / "comparison.json", report)


def checked_output_root(output, *, preparing=False):
    output = Path(output).absolute()
    try:
        relative = output.relative_to(RUNS_ROOT)
    except ValueError as exc:
        raise ValueError("output must be inside this experiment's runs") from exc
    path = snapshots.safe_file(RUNS_ROOT, relative).resolve()
    if path == RUNS_ROOT.resolve():
        raise ValueError("独立输出目录必需")
    if preparing and path.exists():
        raise FileExistsError("output already exists")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "worker", "compare"):
        modes.add_argument("--" + name, action="store_true")
    for name in ("prepare", "execution", "comparison"):
        parser.add_argument("--authorize-" + name, action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
        raise SystemExit("本机禁止Torch/真实张量执行；仅允许指定远端")
    auth = args.authorize_prepare if args.prepare else args.authorize_comparison if args.compare else args.authorize_execution
    if not auth:
        raise SystemExit("未获显式授权")
    args.output_root = checked_output_root(args.output_root, preparing=args.prepare)
    if args.prepare:
        prepare(args.output_root)
        return
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.run:
        run(args, p)
    elif args.compare:
        compare(args.output_root, p, args.protocol_sha256)
    else:
        unlimited_host_cgroup(UNIT)
        if args.lease_fd is None or os.readlink(f"/proc/self/fd/{args.lease_fd}") != str(PROJECT / "outputs/.gpu.lock"):
            raise ValueError("worker requires inherited GPU lease")
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(args,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
