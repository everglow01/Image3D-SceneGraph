"""Offline opacity diagnostics only; tensor libraries are imported at call time."""
from __future__ import annotations

import gzip
import json

from ..e006_connected_propagation import analyze as connected

snapshots = connected.snapshots
METHODS = ("cpu_float64", "cuda_original_dtype")
DEATH_KEYS = ("dead_before", "dead_after", "crossed_to_dead", "crossed_to_alive")
MASK_KEYS = (*DEATH_KEYS, "below_cutoff_before", "below_cutoff_after")
PAIRS = {"native": connected.NATIVE_PAIRS, "same": connected.SAME_PAIRS, "varied": connected.VARIED_PAIRS}


def validate_logits(value, *, count):
    import numpy as np
    if value.dtype != np.dtype("float32") or value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError("opacity logits must be finite float32 with the frozen owner size")


def opacity_leaf(folder, manifest_sha, count):
    """Verify the signed manifest but only load its required opacity array."""
    import numpy as np
    path = snapshots.safe_file(folder, "manifest.json")
    if snapshots.sha256_file(path) != manifest_sha:
        raise ValueError("opacity manifest changed")
    record = json.loads(path.read_text())
    if snapshots.digest_state(record["tree"]) != record["tree_sha256"]:
        raise ValueError("snapshot tree changed")
    leaf = snapshots.snapshot_leaves(record)[("parameters", "opacities")]
    file = snapshots.safe_file(folder, leaf["file"])
    if file.stat().st_size != leaf["bytes"] or snapshots.sha256_file(file) != leaf["sha256"]:
        raise ValueError("opacity file changed")
    value = np.load(file, mmap_mode="r", allow_pickle=False)
    validate_logits(value, count=count)
    if (str(value.dtype) != leaf["dtype"] or list(value.shape) != leaf["shape"]
            or snapshots.array_hash(value) != leaf["canonical_sha256"]):
        raise ValueError("opacity metadata or canonical hash changed")
    return value


def activated_masks(before, after):
    import numpy as np
    if before.shape != after.shape or before.ndim != 1 or before.size == 0:
        raise ValueError("opacity shape differs")
    if not all(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all() for x in (before, after)):
        raise ValueError("activated opacity must be finite within [0,1]")
    a, b = before <= .005, after <= .005
    return {"dead_before": a, "dead_after": b, "crossed_to_dead": ~a & b,
            "crossed_to_alive": a & ~b,
            "below_cutoff_before": before < 1 / 255, "below_cutoff_after": after < 1 / 255}


def diagnostics(before, after, rank):
    import numpy as np
    import torch
    validate_logits(before, count=before.size)
    validate_logits(after, count=before.size)
    with torch.no_grad(), torch.cuda.device(rank):
        # Copy the read-only mmap: never expose a shared source file to a writable tensor.
        a = torch.from_numpy(np.array(before, copy=True)).to(f"cuda:{rank}").sigmoid()
        b = torch.from_numpy(np.array(after, copy=True)).to(f"cuda:{rank}").sigmoid()
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            raise ValueError("nonfinite CUDA sigmoid output")
        dead_a, dead_b = a <= .005, b <= .005
        native = {"dead_before": dead_a, "dead_after": dead_b,
                  "crossed_to_dead": ~dead_a & dead_b, "crossed_to_alive": dead_a & ~dead_b,
                  "below_cutoff_before": a < 1 / 255, "below_cutoff_after": b < 1 / 255}
        native = {key: value.cpu().numpy() for key, value in native.items()}
    return {"cpu_float64": activated_masks(connected.opacity(before), connected.opacity(after)),
            "cuda_original_dtype": native}


def save_xor(root, name, left, right, reserve):
    import numpy as np
    if left.dtype != np.bool_ or right.dtype != np.bool_ or left.shape != right.shape or left.ndim != 1:
        raise ValueError("mask dtype or shape differs")
    delta = np.logical_xor(left, right)
    payload = gzip.compress(np.packbits(delta, bitorder="little").tobytes(), mtime=0)
    reserve(len(payload))
    path = snapshots.safe_file(root, name + ".bits.gz")
    with path.open("xb") as handle:
        handle.write(payload)
    return {"xor_count": int(delta.sum()), "slot_count": delta.size, "path": path.name,
            "encoding": "gzip_numpy_packbits_little_zero_based_slots",
            "sha256": snapshots.sha256_file(path), "bytes": len(payload)}


def verify_bitset(root, row):
    path = snapshots.safe_file(root, row["path"])
    if (path.stat().st_size != row["bytes"] or snapshots.sha256_file(path) != row["sha256"]
            or row["encoding"] != "gzip_numpy_packbits_little_zero_based_slots"):
        raise ValueError("bitset identity changed")
    size = row["slot_count"]
    if type(size) is not int or not 0 < size <= 1500000:
        raise ValueError("invalid bitset slot count")
    with gzip.open(path, "rb") as stream:
        raw = stream.read((size + 7) // 8 + 1)
    bits = int.from_bytes(raw, "little")
    if len(raw) != (size + 7) // 8 or bits.bit_length() > size or bits.bit_count() != row["xor_count"]:
        raise ValueError("bitset length, padding or count differs")


def direction(pairs, methods):
    def changed(rows):
        return any(row[key]["xor_count"] for row in rows for key in DEATH_KEYS)
    same = changed([row for owner in pairs.values() for row in owner["same"].values()])
    if same:
        raise ValueError("same-input death masks differ despite bitwise-equal source parameters")
    return {"varied_death_mask_difference": changed([row for owner in pairs.values() for row in owner["varied"].values()]),
            "native_death_mask_difference": changed([row for owner in pairs.values() for row in owner["native"].values()]),
            "method_death_mask_disagreement": changed(methods),
            "scope": "offline_single_step_diagnostic_not_executed_strategy_or_long_term_causality"}


def analyze(source, output, inventory, reserve, progress):
    import numpy as np
    pairs, summaries, cross_method = {}, {}, {}
    for rank in (0, 1):
        cached = {}
        first = None
        for name, entry in inventory["cases"].items():
            arrays = [opacity_leaf(source / name / f"rank-{rank}" / stage,
                                   entry[str(rank)][stage], inventory["owner_size"])
                      for stage in ("P0", "P3")]
            if first is None:
                first = np.array(arrays[0], copy=True)
            elif not np.array_equal(first, arrays[0]):
                raise ValueError("restored opacity differs between cases")
            cached[name] = diagnostics(*arrays, rank)
            for method, values in cached[name].items():
                key = f"owner-{rank}/{name}/{method}"
                summaries[key] = {k: int(v.sum()) for k, v in values.items()}
                if method == "cpu_float64":
                    expected = inventory["counts"][name][str(rank)]
                    if any(summaries[key][k] != expected[k] for k in DEATH_KEYS):
                        raise ValueError("offline float64 counts disagree with frozen E006 comparison")
            progress({"phase": "opacity_loaded", "owner": rank, "case": name})
        for method in METHODS:
            owner = {}
            for group, comparisons in PAIRS.items():
                owner[group] = {}
                for left, right in comparisons:
                    prefix = f"r{rank}-{method}-{group}-{left}__{right}"
                    owner[group][f"{left}__{right}"] = {
                        key: save_xor(output, prefix + "-" + key, cached[left][method][key],
                                      cached[right][method][key], reserve) for key in MASK_KEYS}
            pairs[f"owner-{rank}/{method}"] = owner
        for name in cached:
            cross_method[f"owner-{rank}/{name}"] = {
                key: save_xor(output, f"r{rank}-{name}-methods-{key}", cached[name][METHODS[0]][key],
                              cached[name][METHODS[1]][key], reserve) for key in MASK_KEYS}
        progress({"phase": "owner_compared", "owner": rank})
    conclusion = direction(pairs, cross_method.values())
    if conclusion["method_death_mask_disagreement"]:
        priority = "threshold_method_sensitive_keep_interpretations_separate"
    elif conclusion["varied_death_mask_difference"]:
        priority = "retain_numerical_trigger_candidate_not_causal_proof"
    elif conclusion["native_death_mask_difference"]:
        priority = "native_difference_only_no_varied_trigger_evidence"
    else:
        priority = "lower_numerical_trigger_priority_for_this_single_step"
    return {"status": "analysis_complete", "counts": summaries, "pairs": pairs,
            "method_comparison": cross_method, "conclusion": {**conclusion, "priority": priority},
            "cutoff_note": "1/255 is a separate diagnostic, not the .005 death threshold",
            "stage_B_authorized": False, "old_gate1": "not_reclassified", "long_term_causality_proven": False}
