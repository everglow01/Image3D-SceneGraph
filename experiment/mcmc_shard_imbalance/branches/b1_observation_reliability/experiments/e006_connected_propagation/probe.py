"""One native graph, optional frozen B1 substitution, stop immediately after Adam."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

from ..e004_fixed_b0 import legacy

snapshots = legacy.snapshots
STAGES = ("P0", "P1", "P2", "P3")
ORDER = ("P0", "P1", "B0", "B1", "P2", "P3")
OUTPUTS = legacy.boundary.OUTPUTS


def require_exact(rows):
    if not rows or any(v["status"] != "bitwise_equal" for v in rows.values()):
        raise ValueError("prerequisite is not bitwise equal")


def require_finite(folder):
    import numpy as np
    record = snapshots.load_snapshot(folder)
    for node in snapshots.snapshot_leaves(record).values():
        if node["type"] == "nonfinite":
            raise ValueError("nonfinite scalar")
        if node["type"] == "tensor":
            a = np.load(snapshots.safe_file(folder, node["file"]), mmap_mode="r", allow_pickle=False)
            if not np.isfinite(a).all():
                raise ValueError("nonfinite tensor")
    return record


def validate_result(result, ctx, rank, gradients):
    import torch
    from ..e004_fixed_b0.run import storage_ids
    if not isinstance(result, tuple) or len(result) != 12 or any(v is not None for v in result[4:]):
        raise ValueError("rasterizer return signature differs")
    for value, source in zip(result[:4], ctx.saved_tensors[:4]):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != source.shape or str(value.device) != f"cuda:{rank}"
                or not value.is_contiguous() or not torch.isfinite(value).all()):
            raise ValueError("invalid B1 shape/dtype/device/layout/finiteness")
    if storage_ids(result[:4]) & storage_ids([*ctx.saved_tensors, *gradients]):
        raise ValueError("B1 aliases live input")


def injected_result(folder, ctx, rank, gradients):
    import numpy as np
    import torch
    from ..e004_fixed_b0.tensor_analysis import decode
    record = require_finite(folder)
    legacy.boundary.validate_boundary_schema(record, "B1")
    leaves = snapshots.snapshot_leaves(record)
    values = []
    for name in OUTPUTS[:4]:
        node = leaves[("outputs", name)]
        if node["kind"] != "torch" or node["device"] != f"cuda:{rank}":
            raise ValueError("B1 belongs to another camera rank")
        array = decode(node, folder)
        value = torch.from_numpy(np.array(array, copy=True, order="C")).to(f"cuda:{rank}")
        if list(value.stride()) != node["stride"] or snapshots.array_hash(value.cpu().numpy()) != node["canonical_sha256"]:
            raise ValueError("injected content/layout differs")
        values.append(value)
    if leaves[("outputs", "v_backgrounds")]["type"] != "NoneType":
        raise ValueError("background gradient must be None")
    result = (*values, *(None,) * 8)
    validate_result(result, ctx, rank, gradients)
    return result


class PropagationProbe(snapshots.FirstDivergenceProbe):
    def __init__(self, root, rank, source, reference, injection, train_ids, **kwargs):
        super().__init__(root, rank, **kwargs)
        self.source = Path(source)
        self.reference = Path(reference) if reference is not None else None
        self.injection = Path(injection) if injection is not None else None
        self.train_ids = set(map(str, train_ids))
        self.order, self.boundary_manifests = [], {}
        self.forward_calls = self.dispatch_calls = self.native_calls = 0
        self.loaded_train_ids = []

    def consensus(self, label, action):
        import torch
        result, error = None, None
        try:
            result = action()
        except Exception as exc:
            error = exc
        failed = torch.tensor([int(error is not None)], device=f"cuda:{self.rank}", dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            snapshots.write_new(self.root / f"rejected-{label}-rank-{self.rank}.json",
                                {"stage": label, "rank": self.rank, "error": str(error) if error else "peer rejected"})
            raise ValueError(f"{label} prerequisite failed on a rank") from error
        return result

    def check_prerequisite(self, stage):
        current = self.root / f"rank-{self.rank}" / stage
        require_exact(snapshots.compare_snapshots(current, self.source / f"rank-{self.rank}" / stage))
        if self.reference is not None:
            require_exact(snapshots.compare_snapshots(current, self.reference / f"rank-{self.rank}" / stage))

    def capture(self, stage, extra=None):
        def take():
            super(PropagationProbe, self).capture(stage, extra)
            require_finite(self.root / f"rank-{self.rank}" / stage)
            if stage in ("P0", "P1"):
                self.check_prerequisite(stage)
            self.order.append(stage)
        self.consensus(stage, take)
        if stage == "P3":
            raise snapshots.ProbeComplete() from None

    def capture_boundary(self, stage, data):
        # Import was primed before checkpoint restore in the worker, never during measurement.
        from scripts.run_mcmc_backward_boundary import BoundaryProbe
        BoundaryProbe.capture_boundary(self, stage, data)
        require_finite(self.root / f"rank-{self.rank}" / stage)

    @contextmanager
    def installed(self):
        from gsplat.cuda._wrapper import _RasterizeToPixels
        from image3d_scenegraph.gaussian import trainer, runtime
        forward, backward = _RasterizeToPixels.forward, _RasterizeToPixels.backward
        load = runtime._ViewSource.load

        def forward_once(*args, **kwargs):
            self.forward_calls += 1
            if self.forward_calls != 1:
                raise ValueError("extra rasterizer forward")
            return forward(*args, **kwargs)

        def dispatch(ctx, rgb, alpha):
            def take_inputs():
                if self.dispatch_calls or self.forward_calls != 1 or len(ctx.saved_tensors) != len(legacy.boundary.SAVED):
                    raise ValueError("unexpected rasterizer dispatch")
                self.capture_boundary("B0", {"inputs": dict(zip(legacy.boundary.SAVED, ctx.saved_tensors)),
                    "image_gradients": {"rgb": rgb, "alpha": alpha},
                    "settings": {"width": ctx.width, "height": ctx.height, "tile_size": ctx.tile_size,
                                 "absgrad": ctx.absgrad, "needs_input_grad": tuple(ctx.needs_input_grad)}})
                self.check_prerequisite("B0")
            self.consensus("B0", take_inputs)

            def take_outputs():
                with legacy.unchanged_rng():
                    before = legacy.digest_state([*ctx.saved_tensors, rgb, alpha])
                    state = legacy.digest_state(self.full_state())
                    if self.injection is None:
                        result = backward(ctx, rgb, alpha)
                        self.native_calls += 1
                    else:
                        result = injected_result(self.injection, ctx, self.rank, (rgb, alpha))
                    validate_result(result, ctx, self.rank, (rgb, alpha))
                    if (legacy.digest_state([*ctx.saved_tensors, rgb, alpha]) != before
                            or legacy.digest_state(self.full_state()) != state):
                        raise ValueError("rasterizer dispatch changed inputs or training state")
                    self.dispatch_calls += 1
                    self.capture_boundary("B1", {"outputs": dict(zip(OUTPUTS, result[:5]))})
                    return result
            return self.consensus("B1", take_outputs)

        def train_only(source):
            if str(source.camera.image_id) not in self.train_ids:
                raise ValueError("Test/Validation image decode forbidden")
            self.loaded_train_ids.append(str(source.camera.image_id))
            return load(source)

        def forbidden(*args, **kwargs):
            raise ValueError("evaluation/export/checkpoint forbidden in P3 probe")

        with super().installed(), ExitStack() as stack:
            stack.enter_context(patch.object(_RasterizeToPixels, "forward", staticmethod(forward_once)))
            stack.enter_context(patch.object(_RasterizeToPixels, "backward", staticmethod(dispatch)))
            stack.enter_context(patch.object(runtime._ViewSource, "load", train_only))
            for name in ("evaluate_views", "_write_streaming_checkpoint", "save_model_snapshot"):
                stack.enter_context(patch.object(trainer, name, forbidden))
            yield self

    def finish(self):
        expected_calls = {"restore": 1, "batch": 1, "render": 1, "backward": 1, "strategy": 0}
        if (self.stages != list(STAGES) or self.order != list(ORDER)
                or list(self.manifests) != list(STAGES) or list(self.boundary_manifests) != ["B0", "B1"]
                or self.steps != list(snapshots.PARAMETERS) or self.calls != expected_calls
                or self.forward_calls != 1 or self.dispatch_calls != 1
                or self.native_calls != int(self.injection is None)
                or not self.loaded_train_ids or not set(self.loaded_train_ids) <= self.train_ids):
            raise ValueError("incomplete P3-only probe")
        snapshots.write_new(self.root / f"worker-rank-{self.rank}.json", {
            "status": "probe_complete", "rank": self.rank, "iteration": self.iteration,
            "stages": self.stages, "order": self.order, "calls": self.calls, "optimizer_steps": self.steps,
            "new_updates": 1, "forward_calls": self.forward_calls, "rasterizer_dispatch_calls": self.dispatch_calls,
            "original_backward_calls": self.native_calls, "injected_backward_returns": int(self.injection is not None),
            "injection": str(self.injection) if self.injection is not None else None,
            "manifests": self.manifests, "boundary_manifests": self.boundary_manifests,
            "loaded_train_ids": self.loaded_train_ids, "native_training_complete": False, "validation_evaluated": False})
