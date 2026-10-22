#!/usr/bin/env python3
"""Level-one A/A boundary probe; no loss-only or fixed-input replay."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from unittest.mock import patch

from scripts import run_mcmc_first_divergence as native
from scripts import mcmc_first_divergence as snapshots
from scripts.mcmc_lifecycle_observer import digest_state, rng_state, unchanged_rng

PROFILE = "mcmc_backward_boundary_v1"
BOUNDARIES = ("B0", "B1")
ORDER = ("P0", "P1", "B0", "B1", "P2", "P3", "P4")
SAVED = ("means2d", "conics", "colors", "opacities", "backgrounds", "masks",
         "isect_offsets", "flatten_ids", "render_alphas", "last_ids")
OUTPUTS = ("v_means2d", "v_conics", "v_colors", "v_opacities", "v_backgrounds")
CONTRACT = {"boundaries": list(BOUNDARIES), "order": list(ORDER), "saved_tensors": list(SAVED),
            "outputs": list(OUTPUTS), "original_backward_calls": 1, "fixed_input_replay": "not_authorized_or_implemented",
            "loss_only_replay": "not_authorized_or_implemented", "attribution_requires": "bitwise_equal_complete_B0_inputs"}
_NATIVE_WORKER = native.worker
_NATIVE_VERIFY = native.verify_worker
_NATIVE_PROTOCOL = native.protocol_at
_NATIVE_WRITE = native.write_new
_NATIVE_CGROUP = native.unlimited_host_cgroup


class BoundaryProbe(snapshots.FirstDivergenceProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = []
        self.boundary_manifests = {}
        self.raster_backward_calls = 0

    def capture(self, stage, extra=None):
        super().capture(stage, extra)
        self.order.append(stage)

    def capture_boundary(self, stage, data):
        import torch
        if stage != BOUNDARIES[len(self.boundary_manifests)] or self.stages != ["P0", "P1"] or self.steps:
            raise ValueError("boundary outside native backward")
        torch.cuda.synchronize(self.rank)
        with unchanged_rng():
            payload = {**data, "rng": rng_state()}
            state_hash = digest_state(self.full_state())
            payload_hash = digest_state(payload)
            folder = self.root / f"rank-{self.rank}" / stage
            sha = snapshots.save_snapshot(folder, payload)
            if digest_state(self.full_state()) != state_hash or digest_state(payload) != payload_hash:
                raise ValueError("boundary snapshot changed live tensors")
        self.boundary_manifests[stage] = sha
        self.order.append(stage)

    @contextmanager
    def installed(self):
        from gsplat.cuda._wrapper import _RasterizeToPixels
        original = _RasterizeToPixels.backward

        def backward(ctx, v_render_colors, v_render_alphas):
            if self.raster_backward_calls or len(ctx.saved_tensors) != len(SAVED):
                raise ValueError("unexpected rasterizer backward invocation")
            self.capture_boundary("B0", {
                "inputs": dict(zip(SAVED, ctx.saved_tensors)),
                "image_gradients": {"rgb": v_render_colors, "alpha": v_render_alphas},
                "settings": {"width": ctx.width, "height": ctx.height, "tile_size": ctx.tile_size,
                             "absgrad": ctx.absgrad, "needs_input_grad": tuple(ctx.needs_input_grad)},
            })
            result = original(ctx, v_render_colors, v_render_alphas)
            self.raster_backward_calls += 1
            if len(result) != 12 or any(v is not None for v in result[5:]):
                raise ValueError("rasterizer output signature differs")
            self.capture_boundary("B1", {"outputs": dict(zip(OUTPUTS, result[:5]))})
            return result

        with super().installed(), patch.object(_RasterizeToPixels, "backward", staticmethod(backward)):
            yield self

    def finish(self):
        if self.order != list(ORDER) or self.raster_backward_calls != 1 or list(self.boundary_manifests) != list(BOUNDARIES):
            raise ValueError("incomplete backward boundaries")
        snapshots.write_new(self.root / f"boundary-rank-{self.rank}.json", {
            "rank": self.rank, "iteration": self.iteration, "order": self.order,
            "original_raster_backward_calls": self.raster_backward_calls, "manifests": self.boundary_manifests})
        super().finish()


def validate_boundary_schema(record, stage):
    leaves = snapshots.snapshot_leaves(record)
    required = [("rng", "torch")]
    if stage == "B0":
        required += [("inputs", k) for k in SAVED if k not in ("backgrounds", "masks")]
        required += [("image_gradients", k) for k in ("rgb", "alpha")]
        for key in ("backgrounds", "masks"):
            node = leaves.get(("inputs", key))
            if node is None or node["type"] not in ("tensor", "NoneType"):
                raise ValueError("missing optional rasterizer input")
        for key in ("width", "height", "tile_size", "absgrad"):
            if ("settings", key) not in leaves:
                raise ValueError("missing rasterizer setting")
        if not all(("settings", "needs_input_grad", i) in leaves for i in range(12)):
            raise ValueError("missing autograd input flags")
    elif stage == "B1":
        required += [("outputs", k) for k in OUTPUTS if k != "v_backgrounds"]
        node = leaves.get(("outputs", "v_backgrounds"))
        if node is None or node["type"] not in ("tensor", "NoneType"):
            raise ValueError("missing optional background gradient")
    else:
        raise ValueError("unknown boundary")
    if any(k not in leaves or leaves[k]["type"] != "tensor" for k in required):
        raise ValueError("missing boundary tensor")
    if not all(any(path[:2] == ("rng", k) for path in leaves) for k in ("python", "numpy", "cuda")):
        raise ValueError("missing boundary RNG")


def verify_worker(stage, rank):
    worker_record = _NATIVE_VERIFY(stage, rank)
    record = native.read(stage / f"boundary-rank-{rank}.json")
    if (record["rank"], record["iteration"], record["order"], record["original_raster_backward_calls"], list(record["manifests"])) != (
            rank, snapshots.ITERATION, list(ORDER), 1, list(BOUNDARIES)):
        raise ValueError("boundary identity differs")
    for boundary in BOUNDARIES:
        folder = stage / f"rank-{rank}" / boundary
        if native.sha256_file(snapshots.safe_file(folder, "manifest.json")) != record["manifests"][boundary]:
            raise ValueError("boundary manifest changed")
        validate_boundary_schema(snapshots.verify_snapshot_files(folder), boundary)
    return worker_record


def protocol_at(root, digest):
    p = _NATIVE_PROTOCOL(root, digest)
    if p.get("backward_boundary") != CONTRACT:
        raise ValueError("backward boundary contract differs")
    return p


def write_new(path, value):
    value = dict(value)
    if path.name == "protocol.json":
        value["backward_boundary"] = CONTRACT
    elif path.name == "complete.json":
        value["files"] = dict(value["files"])
        for rank in (0, 1):
            name = f"boundary-rank-{rank}.json"
            value["files"][name] = native.sha256_file(path.parent / name)
    _NATIVE_WRITE(path, value)


def attribution(p_stages, boundaries):
    def exact(rows):
        return bool(rows) and all(v["status"] == "bitwise_equal" for v in rows.values())
    if not all(exact(p_stages[s]) for s in ("P0", "P1")):
        return "P0_P1_prerequisite_not_bitwise_equal"
    b0, b1 = boundaries["B0"], boundaries["B1"]
    if not b0 or not b1:
        raise ValueError("missing boundary comparison")
    inputs = {k: v for k, v in b0.items() if not k.startswith("image_gradients/")}
    gradients = {k: v for k, v in b0.items() if k.startswith("image_gradients/")}
    if not exact(inputs):
        return "saved_forward_inputs_or_boundary_state_differ"
    if not exact(gradients):
        return "image_loss_backward_differs_SSIM_not_yet_isolated"
    if not exact({k: v for k, v in b1.items() if k.startswith("rng/")}):
        return "rasterizer_boundary_rng_differs"
    if not exact({k: v for k, v in b1.items() if k.startswith("outputs/")}):
        return "rasterizer_backward_differs_with_bitwise_equal_inputs"
    if not exact(p_stages["P2"]):
        return "difference_downstream_of_rasterizer_backward"
    return "backward_difference_not_reproduced"


def compare(root, p, digest):
    native.verify_inputs(p)
    for branch in native.BRANCHES:
        complete = native.completed(root, branch, digest)
        if not {f"boundary-rank-{r}.json" for r in (0, 1)} <= complete["files"].keys():
            raise ValueError("boundary completion not bound")
    stages = {str(r): {s: snapshots.compare_snapshots(root / "a" / f"rank-{r}" / s,
             root / "b" / f"rank-{r}" / s) for s in snapshots.STAGES} for r in (0, 1)}
    boundaries = {str(r): {s: snapshots.compare_snapshots(root / "a" / f"rank-{r}" / s,
                 root / "b" / f"rank-{r}" / s) for s in BOUNDARIES} for r in (0, 1)}
    report = snapshots.summarize_comparison(stages)
    report.update(profile=PROFILE, protocol_sha256=digest, stages=stages, boundaries=boundaries,
        attribution={r: attribution(stages[r], boundaries[r]) for r in stages},
        complete_sha256={b: native.sha256_file(root / b / "complete.json") for b in native.BRANCHES},
        atol=snapshots.ATOL, rtol=snapshots.RTOL, new_updates=2, new_camera_samples=4,
        second_level="not_run_or_automatically_authorized", old_gate1="not_reclassified")
    _NATIVE_WRITE(root / "comparison.json", report)


def cgroup(unit):
    expected = {f"image3d-first-divergence-t1-{b}-v1.service": f"image3d-backward-boundary-{b}-v1.service" for b in native.BRANCHES}
    if unit not in expected:
        raise ValueError("unexpected unit")
    return _NATIVE_CGROUP(expected[unit])


def worker(local_rank, rank, world_size, args):
    with adapted_runner():
        _NATIVE_WORKER(local_rank, rank, world_size, args)


@contextmanager
def adapted_runner():
    # The frozen T1 protocol binds its source bytes, so reuse it without editing that file.
    replacements = {"PROFILE": PROFILE, "DEPENDENCIES": native.DEPENDENCIES + ("scripts/run_mcmc_backward_boundary.py",),
        "__file__": __file__, "FirstDivergenceProbe": BoundaryProbe, "verify_worker": verify_worker,
        "protocol_at": protocol_at, "write_new": write_new, "compare": compare, "worker": worker,
        "unlimited_host_cgroup": cgroup}
    with ExitStack() as stack:
        for key, value in replacements.items():
            stack.enter_context(patch.object(native, key, value))
        yield


if __name__ == "__main__":
    with adapted_runner():
        native.main()
