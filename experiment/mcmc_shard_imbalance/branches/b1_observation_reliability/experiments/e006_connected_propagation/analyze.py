"""CPU comparisons of connected outputs, never historical P2/P3 substitution."""
from __future__ import annotations

from itertools import combinations

from ..e004_fixed_b0 import legacy
from ..e004_fixed_b0.tensor_analysis import decode

snapshots = legacy.snapshots
PASS = {"bitwise_equal", "numerically_equivalent"}
SAME_PAIRS = tuple(combinations(("same-0", "same-1", "same-2"), 2))
NATIVE_PAIRS = tuple(combinations(("native-0", "native-1", "native-2"), 2))
VARIED_PAIRS = tuple((f"varied-{i}", "same-0") for i in range(1, 5))


def acceptable(stages):
    return bool(stages) and all(rows and all(v["status"] in PASS for v in rows.values()) for rows in stages.values())


def compare_stages(root, left, right, rank):
    return {s: snapshots.compare_snapshots(root / left / f"rank-{rank}" / s, root / right / f"rank-{rank}" / s)
            for s in ("P2", "P3")}


def same_gate(root):
    pairs = {f"{a}__{b}": {str(r): {"stages": compare_stages(root, a, b, r),
                                  "parameter_updates": updates(root, a, b, r)} for r in (0, 1)} for a, b in SAME_PAIRS}
    passed = all(acceptable(row["stages"]) and row["parameter_updates"]
                 and all(v["status"] in PASS for v in row["parameter_updates"].values())
                 for ranks in pairs.values() for row in ranks.values())
    return {"passed": passed,
            "comparison_reference": "left_actual_right_reference", "pairs": pairs,
            "atol": snapshots.ATOL, "rtol": snapshots.RTOL, "scope": "three_same_input_runs_not_global_bound"}


def state(folder):
    record = snapshots.load_snapshot(folder)
    return decode(record["tree"], folder)


def opacity(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    q = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + q), q / (1 + q))


def updates(root, left, right, rank):
    import numpy as np
    a0, a3, b0, b3 = (state(root / name / f"rank-{rank}" / stage)
                      for name, stage in ((left, "P0"), (left, "P3"), (right, "P0"), (right, "P3")))
    result = {}
    for name in snapshots.PARAMETERS:
        a = np.subtract(a3["parameters"][name], a0["parameters"][name], dtype=np.float64)
        b = np.subtract(b3["parameters"][name], b0["parameters"][name], dtype=np.float64)
        result[name] = snapshots.compare_arrays(a, b)
    return result


def point_and_opacity_summary(root, name, rank, points):
    import numpy as np
    before, gradients, after = (state(root / name / f"rank-{rank}" / s) for s in ("P0", "P2", "P3"))
    a, b = opacity(before["parameters"]["opacities"]), opacity(after["parameters"]["opacities"])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite opacity")
    selected = []
    for point in points:
        if point["owner"] != rank:
            continue
        slot = point["slot"]
        groups = {}
        for group in snapshots.PARAMETERS:
            x, y = before["parameters"][group][slot], after["parameters"][group][slot]
            grad = gradients["gradients"][group]
            if grad is None:
                raise ValueError("missing connected gradient")
            groups[group] = {"gradient": grad[slot].tolist(), "before": x.tolist(), "after": y.tolist(),
                "update": np.subtract(y, x, dtype=np.float64).tolist(),
                "adam_before": {k: v[slot].tolist() for k, v in before["adam"][group]["state"].items() if k in ("exp_avg", "exp_avg_sq")},
                "adam_after": {k: v[slot].tolist() for k, v in after["adam"][group]["state"].items() if k in ("exp_avg", "exp_avg_sq")}}
        selected.append({**point, "groups": groups, "opacity_before": float(a[slot]), "opacity_after": float(b[slot]),
            "distance_to_death_before": float(a[slot] - .005), "distance_to_death_after": float(b[slot] - .005)})
    return {"opacity_method": "CPU_float64_sigmoid_diagnostic_not_native_strategy_death_mask",
        "dead_before": int((a <= .005).sum()), "dead_after": int((b <= .005).sum()),
        "crossed_to_dead": int(((a > .005) & (b <= .005)).sum()),
        "crossed_to_alive": int(((a <= .005) & (b > .005)).sum()),
        "min_abs_distance_before": float(np.min(np.abs(a - .005))),
        "min_abs_distance_after": float(np.min(np.abs(b - .005))), "selected": selected}


def compare(root, protocol):
    gate = same_gate(root)
    if not gate["passed"]:
        raise ValueError("same-input gate not passed")
    pairs = {}
    for kind, comparisons in (("native", NATIVE_PAIRS), ("same", SAME_PAIRS), ("varied", VARIED_PAIRS)):
        pairs[kind] = {}
        for a, b in comparisons:
            pairs[kind][f"{a}__{b}"] = {str(rank): {"stages": compare_stages(root, a, b, rank),
                "parameter_updates": updates(root, a, b, rank)} for rank in (0, 1)}
    points = {c["name"]: {str(r): point_and_opacity_summary(root, c["name"], r, protocol["points"])
                          for r in (0, 1)} for c in protocol["cases"]}
    # Keep raw comparisons: a finite baseline is not a calibrated global noise bound.
    different = any(not acceptable(row["stages"]) or any(v["status"] not in PASS for v in row["parameter_updates"].values())
                    for ranks in pairs["varied"].values() for row in ranks.values())
    return {"status": "analysis_complete", "same_input_gate_passed": True,
        "varied_has_above_tolerance_outputs": different, "pairs": pairs, "point_and_opacity_summary": points,
        "comparison_reference": "left_actual_right_reference", "atol": snapshots.ATOL, "rtol": snapshots.RTOL,
        "source_B1_comparison": {"path": protocol["B1_root"] + "/comparison.json", "sha256": protocol["B1_comparison_sha256"],
            "pairs_per_camera": [list(p) for p in combinations(range(5), 2)], "recomputed": False},
        "new_P2_P3_computed": True, "historical_P2_P3_used_as_outputs": False,
        "interpretation": "descriptive_connected_single_step; compare_same_and_native_variation_before_attribution"}
