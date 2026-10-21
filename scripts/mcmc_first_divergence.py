"""One-step diagnostic snapshots; never import Torch at module load."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from scripts.mcmc_lifecycle_observer import digest_state, rng_state, unchanged_rng
from image3d_scenegraph.file_integrity import sha256_file

STAGES = ("P0", "P1", "P2", "P3", "P4")
PARAMETERS = ("means", "scales", "quats", "opacities", "sh0", "shN")
ATOL, RTOL = 2e-6, 2e-4
ITERATION = 12001
GIB = 1024**3


class ProbeComplete(BaseException):
    pass


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def safe_file(root, relative):
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe snapshot path")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe snapshot symlink")
    return path


def numeric_pair(a, b, *, exact=False):
    finite_a, finite_b = math.isfinite(a), math.isfinite(b)
    finite = finite_a and finite_b
    error = abs(a - b) if finite else None
    unbounded = finite and b == 0 and error != 0
    relative = None if not finite or unbounded else (error / abs(b) if b else 0.)
    return {"strict_unequal": a != b, "nonfinite_a": not finite_a, "nonfinite_b": not finite_b,
            "abs_error": error, "rel_error": relative, "relative_unbounded": unbounded,
            "mismatch": not finite or (a != b if exact else error > ATOL + RTOL * abs(b))}


def array_hash(array):
    import numpy as np
    array = np.asarray(array)
    h = hashlib.sha256(json.dumps([str(array.dtype), list(array.shape)]).encode())
    raw = memoryview(np.ascontiguousarray(array).reshape(-1)).cast("B")
    for offset in range(0, len(raw), 4 * 1024**2):
        h.update(raw[offset:offset + 4 * 1024**2])
    return h.hexdigest()


def save_snapshot(directory, payload):
    import numpy as np
    import torch
    directory.mkdir(exist_ok=False)
    files = {}

    def encode(value):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if isinstance(value, torch.Tensor):
                if value.layout != torch.strided:
                    raise ValueError("only dense snapshot tensors are supported")
                cpu = value.detach().to(device="cpu", copy=True).contiguous()
                array = cpu.numpy()
                origin = {"kind": "torch", "device": str(value.device), "stride": list(value.stride())}
            else:
                array = np.array(value, copy=True, order="C")
                origin = {"kind": "numpy", "device": "cpu", "stride": list(value.strides)}
            if array.dtype.kind not in "biuf":
                raise ValueError("unsupported snapshot dtype")
            name = f"tensor-{len(files):05d}.npy"
            if shutil.disk_usage(directory).free < array.nbytes + 4 * GIB:
                raise ValueError("snapshot would breach disk reserve")
            with (directory / name).open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            record = {"file": name, "dtype": str(array.dtype), "shape": list(array.shape),
                      "sha256": sha256_file(directory / name), "bytes": (directory / name).stat().st_size,
                      "canonical_sha256": array_hash(array), **origin}
            files[name] = record
            return {"type": "tensor", **record}
        if isinstance(value, dict):
            return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in sorted(value.items(), key=lambda x: (type(x[0]).__name__, str(x[0]))) ]}
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "items": [encode(v) for v in value]}
        if type(value) is float and not math.isfinite(value):
            return {"type": "nonfinite", "value": repr(value)}
        if value is None or type(value) in (str, bool, int, float):
            return {"type": type(value).__name__, "value": value}
        raise TypeError(f"unsupported snapshot value: {type(value)}")

    tree = encode(payload)
    record = {"tree": tree, "files": files, "tree_sha256": digest_state(tree)}
    write_new(directory / "manifest.json", record)
    return sha256_file(directory / "manifest.json")


def snapshot_leaves(record):
    leaves = {}
    used = set()

    def walk(node, path):
        kind = node["type"]
        if kind == "dict":
            keys = [json.dumps(k, sort_keys=True) for k, _ in node["items"]]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate snapshot keys")
            for key, value in node["items"]:
                if key["type"] not in ("str", "int"):
                    raise ValueError("unsupported snapshot key")
                walk(value, (*path, key["value"]))
        elif kind in ("list", "tuple"):
            for i, value in enumerate(node["items"]):
                walk(value, (*path, i))
        else:
            if kind == "tensor":
                name = node["file"]
                if name in used or {k: v for k, v in node.items() if k != "type"} != record["files"].get(name):
                    raise ValueError("snapshot tree/file identity differs")
                used.add(name)
            elif kind not in ("str", "int", "float", "bool", "NoneType", "nonfinite"):
                raise ValueError("unsupported snapshot node")
            leaves[path] = node
    walk(record["tree"], ())
    if used != set(record["files"]):
        raise ValueError("unreferenced snapshot file")
    return leaves


def validate_stage_schema(record, stage):
    leaves = snapshot_leaves(record)
    required = set()
    if stage in ("P0", "P3", "P4"):
        required |= {("parameters", name) for name in PARAMETERS}
        required |= {("adam", name, "state", field) for name in PARAMETERS for field in ("exp_avg", "exp_avg_sq", "step")}
        if not all(any(path[:3] == ("adam", name, "param_groups") for path in leaves) for name in PARAMETERS):
            raise ValueError("missing optimizer parameter groups")
    elif stage == "P1":
        required |= {("render", key) for key in ("rgb", "alpha", "backward_loss", "ground_truth", "camera_matrix", "intrinsic")}
        required |= {("render", "loss_terms", key) for key in ("reconstruction", "opacity", "scale")}
        if not any(path[:2] == ("render", "metadata") for path in leaves):
            raise ValueError("missing renderer metadata")
    elif stage == "P2":
        if not all(("gradients", name) in leaves for name in PARAMETERS):
            raise ValueError("missing gradient presence records")
        required |= {("gradients", name) for name in PARAMETERS if leaves[("gradients", name)]["type"] != "NoneType"}
        if ("parameters_adam_sha256",) not in leaves:
            raise ValueError("missing P2 state digest")
    else:
        raise ValueError("unknown stage")
    required.add(("rng", "torch"))
    if any(key not in leaves or leaves[key]["type"] != "tensor" for key in required):
        raise ValueError("missing required stage tensors")
    if not all(any(path[:2] == ("rng", key) for path in leaves) for key in ("python", "numpy", "cuda")):
        raise ValueError("missing RNG state")
    if stage in ("P0", "P3", "P4") and not any(path[:1] == ("strategy",) for path in leaves):
        raise ValueError("missing strategy state")
    if stage in ("P0", "P1", "P3", "P4"):
        if ("camera", "cursor") not in leaves or not any(path[:2] == ("camera", "order") for path in leaves):
            raise ValueError("missing camera state")
    if stage == "P0":
        if not all(("gradient_presence", name) in leaves for name in PARAMETERS):
            raise ValueError("missing restored gradient presence")
        if not all(("runtime", key) in leaves for key in ("torch", "gsplat", "cuda", "gpu", "rank", "deterministic_algorithms", "matmul_tf32")):
            raise ValueError("missing runtime identity")
        if not all(("runtime", "optimizers", name, "class") in leaves for name in PARAMETERS):
            raise ValueError("missing optimizer implementation identity")
    return leaves


def verify_snapshot_files(directory):
    record = json.loads(safe_file(directory, "manifest.json").read_text())
    if digest_state(record["tree"]) != record["tree_sha256"]:
        raise ValueError("snapshot tree changed")
    snapshot_leaves(record)
    for name, info in record["files"].items():
        path = safe_file(directory, name)
        if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError("snapshot file changed")
    return record


def load_snapshot(directory):
    import numpy as np
    record = verify_snapshot_files(directory)
    for name, info in record["files"].items():
        array = np.load(safe_file(directory, name), mmap_mode="r", allow_pickle=False)
        if str(array.dtype) != info["dtype"] or list(array.shape) != info["shape"] or array_hash(array) != info["canonical_sha256"]:
            raise ValueError("snapshot tensor metadata changed")
    return record


def compare_arrays(a, b, *, exact=False):
    import numpy as np
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"status": "structure_mismatch", "sha_equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape),
                "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    exact = exact or a.dtype.kind in "biu"
    n = a.size
    result = {"elements": int(n), "num_strict_unequal": 0, "num_bitwise_unequal": 0,
              "num_mismatch": 0, "num_nonfinite_a": 0, "num_nonfinite_b": 0, "num_nonfinite": 0,
              "finite_pairs": 0, "relative_unbounded_count": 0, "first_mismatch_index": None,
              "first_strict_unequal_index": None, "max_abs_error": 0., "max_rel_error": 0.}
    total_error = 0.
    af, bf = a.reshape(-1), b.reshape(-1)
    for start in range(0, n, 1024**2):
        x, y = np.asarray(af[start:start + 1024**2]), np.asarray(bf[start:start + 1024**2])
        fa, fb = np.isfinite(x), np.isfinite(y)
        finite = fa & fb
        strict = x != y
        bits = (np.ascontiguousarray(x).view(np.uint8).reshape(-1, a.dtype.itemsize) !=
                np.ascontiguousarray(y).view(np.uint8).reshape(-1, b.dtype.itemsize)).any(axis=1)
        error = np.zeros(x.size, dtype=np.float64)
        if a.dtype.kind in "iu":
            # Integer subtraction must not lose low bits above 2**53 or overflow int64.
            error[:] = np.abs(x.astype(object) - y.astype(object)).astype(np.float64)
        else:
            np.subtract(x, y, out=error, where=finite, dtype=np.float64)
            np.abs(error, out=error)
        bad = ~finite | (strict if exact else error > ATOL + RTOL * np.abs(y.astype(np.float64)))
        unbounded = finite & (y == 0) & (error != 0)
        relative = np.zeros(x.size, dtype=np.float64)
        np.divide(error, np.abs(y.astype(np.float64)), out=relative, where=finite & (y != 0))
        result["num_strict_unequal"] += int(strict.sum())
        result["num_bitwise_unequal"] += int(bits.sum())
        result["num_mismatch"] += int(bad.sum())
        result["num_nonfinite_a"] += int((~fa).sum())
        result["num_nonfinite_b"] += int((~fb).sum())
        result["num_nonfinite"] += int((~finite).sum())
        result["finite_pairs"] += int(finite.sum())
        result["relative_unbounded_count"] += int(unbounded.sum())
        result["max_abs_error"] = max(result["max_abs_error"], float(error.max(initial=0)))
        result["max_rel_error"] = max(result["max_rel_error"], float(relative.max(initial=0)))
        total_error += float(error.sum())
        for key, mask in (("first_mismatch_index", bad), ("first_strict_unequal_index", strict)):
            if result[key] is None and mask.any():
                result[key] = [int(i) for i in np.unravel_index(start + int(np.flatnonzero(mask)[0]), a.shape)]
    result.update(sha_equal=array_hash(a) == array_hash(b),
                  strict_unequal_ratio=result["num_strict_unequal"] / n if n else 0.,
                  mismatch_ratio=result["num_mismatch"] / n if n else 0.,
                  mean_abs_error=total_error / result["finite_pairs"] if result["finite_pairs"] else None,
                  relative_unbounded=bool(result["relative_unbounded_count"]), exact_required=exact)
    if not result["finite_pairs"] and n:
        result["max_abs_error"] = result["max_rel_error"] = None
    if result["relative_unbounded"]:
        result["max_rel_error"] = None
    result["status"] = ("numerically_inconsistent" if result["num_mismatch"] else
                        "bitwise_equal" if result["sha_equal"] else "numerically_equivalent")
    return result


def compare_snapshots(left, right):
    import numpy as np
    a, b = load_snapshot(left), load_snapshot(right)
    results = {}
    used = [set(), set()]

    def visit(x, y, path):
        key = "/".join(str(k) for k in path)
        if x["type"] != y["type"]:
            results[key] = {"status": "structure_mismatch"}
        elif x["type"] == "tensor":
            for side, node, record in ((0, x, a), (1, y, b)):
                if {k: v for k, v in node.items() if k != "type"} != record["files"].get(node["file"]):
                    raise ValueError("tensor tree/file identity differs")
                used[side].add(node["file"])
            result = compare_arrays(np.load(safe_file(left, x["file"]), mmap_mode="r", allow_pickle=False),
                                    np.load(safe_file(right, y["file"]), mmap_mode="r", allow_pickle=False),
                                    exact="rng" in path or path[-1:] == ("step",))
            result["layout_equal"] = all(x[k] == y[k] for k in ("kind", "device", "stride"))
            if not result["layout_equal"]:
                result["status"] = "structure_mismatch"
            results[key] = result
        elif x["type"] == "dict":
            if [k for k, _ in x["items"]] != [k for k, _ in y["items"]]:
                results[key] = {"status": "structure_mismatch"}
                return
            for (k, v), (_, w) in zip(x["items"], y["items"]):
                visit(v, w, (*path, k.get("value", json.dumps(k, sort_keys=True))))
        elif x["type"] in ("tuple", "list"):
            if len(x["items"]) != len(y["items"]):
                results[key] = {"status": "structure_mismatch"}
                return
            for index, (v, w) in enumerate(zip(x["items"], y["items"])):
                visit(v, w, (*path, index))
        elif x["type"] == "float":
            results[key] = compare_arrays(np.asarray(x["value"]), np.asarray(y["value"]), exact=not path or path[0] != "render")
        else:
            equal = digest_state(x) == digest_state(y)
            results[key] = {"sha_equal": equal, "status": "bitwise_equal" if equal else "structure_mismatch"}
            if x["type"] == "nonfinite":
                results[key].update(status="numerically_inconsistent", num_nonfinite=1)
            if path[:1] == ("gradient_presence",) and (x.get("value") or y.get("value")):
                results[key]["status"] = "structure_mismatch"
            if "gradients" in path and x["type"] == "NoneType":
                results[key]["status"] = "missing_gradient"
    visit(a["tree"], b["tree"], ())
    if any(r["status"] == "structure_mismatch" for r in results.values()):
        return results
    if used != [set(a["files"]), set(b["files"])]:
        raise ValueError("unreferenced snapshot tensor")
    return results


def summarize_comparison(ranks):
    failures = {"structure_mismatch", "numerically_inconsistent", "missing_gradient"}
    summary = {}
    for rank, stages in ranks.items():
        if list(stages) != list(STAGES) or any(not rows for rows in stages.values()):
            raise ValueError("missing or reordered comparison stages")
        first = next((s for s in STAGES if any(r["status"] != "bitwise_equal" for r in stages[s].values())), None)
        bad = next((s for s in STAGES if any(r["status"] in failures for r in stages[s].values())), None)
        primary = {s: {k: v for k, v in rows.items() if not k.startswith("render/metadata/")}
                   for s, rows in stages.items()}
        summary[rank] = {"first_nonbitwise_stage": first, "first_failed_tolerance_or_structure_stage": bad,
                         "first_primary_nonbitwise_stage": next((s for s in STAGES if any(r["status"] != "bitwise_equal" for r in primary[s].values())), None),
                         "first_primary_failed_tolerance_or_structure_stage": next((s for s in STAGES if any(r["status"] in failures for r in primary[s].values())), None),
                         "p0_exact": first != "P0"}
    first = min((STAGES.index(r["first_nonbitwise_stage"]) for r in summary.values() if r["first_nonbitwise_stage"]), default=None)
    bad = min((STAGES.index(r["first_failed_tolerance_or_structure_stage"]) for r in summary.values() if r["first_failed_tolerance_or_structure_stage"]), default=None)
    p0_passed = all(r["p0_exact"] for r in summary.values())
    return {"ranks": summary, "first_nonbitwise_stage": STAGES[first] if first is not None else None,
            "first_failed_tolerance_or_structure_stage": STAGES[bad] if bad is not None else None,
            "first_primary_nonbitwise_stage": next((s for s in STAGES if any(r["first_primary_nonbitwise_stage"] == s for r in summary.values())), None),
            "first_primary_failed_tolerance_or_structure_stage": next((s for s in STAGES if any(r["first_primary_failed_tolerance_or_structure_stage"] == s for r in summary.values())), None),
            "p0_prerequisite_passed": p0_passed,
            "stage_localization_valid": p0_passed,
            "status": "restored_state_mismatch" if not p0_passed else "numerically_inconsistent" if bad is not None else "numerically_equivalent" if first is not None else "bitwise_equal",
            "interpretation": "first_observed_interval_only; synchronized_probe_not_original_throughput_or_causal_proof"}


def runtime_environment(optimizers, rank):
    import torch
    import gsplat
    return {"torch": str(torch.__version__), "gsplat": str(gsplat.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(rank), "rank": rank,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "threads": torch.get_num_threads(),
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                            "NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "PYTHONPATH")},
            "optimizers": {k: {"class": f"{type(v).__module__}.{type(v).__name__}", "defaults": v.defaults}
                           for k, v in optimizers.items()}}


class FirstDivergenceProbe:
    def __init__(self, root, rank, *, iteration=ITERATION, expected_count=1500000):
        self.root, self.rank = Path(root), rank
        self.iteration, self.expected_count = iteration, expected_count
        self.params = self.optimizers = self.state = None
        self.camera = {}
        self.rendered = self.view = None
        self.losses = {}
        self.stages = []
        self.manifests = {}
        self.steps = []
        self.handles = []
        self.views = []
        self.calls = dict.fromkeys(("restore", "batch", "render", "backward", "strategy"), 0)
        self.sanity_calls = 0
        self.dense_loaded = False

    def full_state(self):
        if set(self.params) != set(PARAMETERS) or tuple(self.optimizers) != PARAMETERS:
            raise ValueError("parameter/optimizer groups differ")
        if len(self.params["means"]) != self.expected_count:
            raise ValueError("fixed owner capacity differs")
        adam = {}
        for name, optimizer in self.optimizers.items():
            if len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != 1 or optimizer.param_groups[0]["params"][0] is not self.params[name]:
                raise ValueError("optimizer parameter mapping differs")
            state = optimizer.state[self.params[name]]
            if not {"exp_avg", "exp_avg_sq", "step"} <= state.keys():
                raise ValueError("restored Adam state incomplete")
            adam[name] = {"state": state, "param_groups": [{k: v for k, v in optimizer.param_groups[0].items() if k != "params"}]}
        return {"parameters": dict(self.params.items()), "adam": adam, "strategy": self.state,
                "camera": self.camera, "rng": rng_state()}

    def capture(self, stage, extra=None):
        import torch
        if stage != STAGES[len(self.stages)]:
            raise ValueError("unexpected probe stage")
        torch.cuda.synchronize(self.rank)
        with unchanged_rng():
            state = self.full_state()
            state_digest = digest_state(state)
            if stage == "P1":
                payload = {"render": extra, "camera": self.camera, "rng": state["rng"],
                           "learning_rates": {k: v.param_groups[0]["lr"] for k, v in self.optimizers.items()}}
            elif stage == "P2":
                payload = {"gradients": {k: v.grad for k, v in self.params.items()}, "rng": state["rng"],
                           "parameters_adam_sha256": digest_state({k: state[k] for k in ("parameters", "adam")})}
            else:
                payload = state
                if stage == "P0":
                    payload = {**state, "gradient_presence": {k: v.grad is not None for k, v in self.params.items()},
                               "runtime": runtime_environment(self.optimizers, self.rank)}
            folder = self.root / f"rank-{self.rank}" / stage
            digest = save_snapshot(folder, payload)
            if digest_state(self.full_state()) != state_digest:
                raise ValueError("snapshot changed live state")
        self.manifests[stage] = digest
        self.stages.append(stage)
        with (self.root / f"phases-rank-{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": stage, "iteration": self.iteration, "manifest_sha256": digest}) + "\n")

    @contextmanager
    def installed(self):
        import torch
        from image3d_scenegraph.gaussian import trainer
        originals = {k: getattr(trainer, k) for k in ("_build_mcmc_strategy", "_torch_load", "_restore_rng", "_next_camera_batch",
                     "_render_visible_training_view", "l1_ssim_loss", "_mcmc_regularizers", "load_training_views")}
        original_backward = torch.Tensor.backward

        def loaded(*args, **kwargs):
            result = originals["_torch_load"](*args, **kwargs)
            if isinstance(result, dict) and {"strategy_state", "camera_order", "camera_cursor"} <= result.keys():
                if self.dense_loaded:
                    raise ValueError("multiple densification restores")
                self.state = result["strategy_state"]
                self.dense_loaded = True
            return result

        def restored(*args, **kwargs):
            result = originals["_restore_rng"](*args, **kwargs)
            self.calls["restore"] += 1
            return result

        def optimizer_done(name, optimizer, args, kwargs):
            if self.stages != ["P0", "P1", "P2"] or name != PARAMETERS[len(self.steps)]:
                raise ValueError("optimizer execution order differs")
            self.steps.append(name)
            if len(self.steps) == len(PARAMETERS):
                self.capture("P3")

        def build(*args, **kwargs):
            strategy = originals["_build_mcmc_strategy"](*args, **kwargs)
            sanity, pre, post = strategy.check_sanity, strategy.step_pre_backward, strategy.step_post_backward

            def checked(params, optimizers):
                result = sanity(params, optimizers)
                self.sanity_calls += 1
                self.params, self.optimizers = params, optimizers
                return result

            def before(params, optimizers, state, step, info):
                if step != self.iteration or self.stages != ["P0"] or state is not self.state:
                    raise ValueError("unexpected pre-backward state")
                return pre(params, optimizers, state, step, info)

            def after(*a, **kw):
                if self.stages != ["P0", "P1", "P2", "P3"] or kw.get("step") != self.iteration:
                    raise ValueError("strategy executed before all optimizers")
                if trainer._mcmc_refinement_due(args[1], self.iteration):
                    raise ValueError("one-step probe must not refine")
                post(*a, **kw)
                self.calls["strategy"] += 1
                self.capture("P4")
                raise ProbeComplete() from None

            strategy.check_sanity, strategy.step_pre_backward, strategy.step_post_backward = checked, before, after
            return strategy

        def batch(count, order, cursor, batch_size):
            if self.calls["restore"] != 1 or self.sanity_calls != 2 or not self.dense_loaded or self.calls["batch"]:
                raise ValueError("P0 requires one complete native restore")
            self.camera = {"order": list(order), "cursor": cursor}
            self.capture("P0")
            for name in PARAMETERS:
                self.handles.append(self.optimizers[name].register_step_post_hook(
                    lambda opt, a, kw, name=name: optimizer_done(name, opt, a, kw)))
            result = originals["_next_camera_batch"](count, order, cursor, batch_size)
            self.camera = {"order": list(result[0]), "cursor": result[1], "indices": list(result[2])}
            self.calls["batch"] += 1
            return result

        def render(*args, **kwargs):
            if self.calls["render"] or self.stages != ["P0"]:
                raise ValueError("unexpected extra render")
            result = originals["_render_visible_training_view"](*args, **kwargs)
            self.view, self.rendered = result
            self.camera["image_id"] = str(self.view.camera.image_id)
            self.camera["sh_degree"] = args[3]
            self.calls["render"] += 1
            return result

        def reconstruction(*args, **kwargs):
            result = originals["l1_ssim_loss"](*args, **kwargs)
            self.losses["reconstruction"], self.losses["terms"] = result
            return result

        def regularizers(*args, **kwargs):
            result = originals["_mcmc_regularizers"](*args, **kwargs)
            self.losses["opacity"], self.losses["scale"] = result
            return result

        def backward(tensor, *args, **kwargs):
            if self.calls["backward"] or self.calls["render"] != 1 or set(self.losses) != {"reconstruction", "terms", "opacity", "scale"}:
                raise ValueError("unexpected backward")
            camera = self.view.camera
            self.capture("P1", {"rgb": self.rendered.image, "alpha": self.rendered.alpha,
                "backward_loss": tensor, "loss_world_size_divisor": 2, "loss_terms": self.losses,
                "metadata": self.rendered.metadata, "ground_truth": self.view.image,
                "camera_matrix": camera.camera_from_normalized, "intrinsic": camera.intrinsic,
                "width": camera.width, "height": camera.height})
            result = original_backward(tensor, *args, **kwargs)
            self.calls["backward"] += 1
            self.capture("P2")
            return result

        def views(*args, **kwargs):
            if kwargs["split"] == "test":
                raise ValueError("Test is forbidden")
            result = originals["load_training_views"](*args, **kwargs)
            self.views.append(result)
            return result

        self.root.joinpath(f"rank-{self.rank}").mkdir(exist_ok=False)
        try:
            with ExitStack() as stack:
                replacements = {"_torch_load": loaded, "_restore_rng": restored, "_build_mcmc_strategy": build,
                    "_next_camera_batch": batch, "_render_visible_training_view": render, "l1_ssim_loss": reconstruction,
                    "_mcmc_regularizers": regularizers, "load_training_views": views}
                for name, replacement in replacements.items():
                    stack.enter_context(patch.object(trainer, name, replacement))
                stack.enter_context(patch.object(torch.Tensor, "backward", backward))
                yield self
        finally:
            for handle in self.handles:
                handle.remove()
            trainer._release_training_views(*self.views)

    def finish(self):
        if self.stages != list(STAGES) or self.steps != list(PARAMETERS) or any(v != 1 for v in self.calls.values()):
            raise ValueError("incomplete one-step probe")
        write_new(self.root / f"worker-rank-{self.rank}.json", {"status": "probe_complete", "rank": self.rank,
            "iteration": self.iteration, "new_updates": 1, "calls": self.calls, "optimizer_steps": self.steps,
            "manifests": self.manifests, "native_training_complete": False, "validation_evaluated": False})
