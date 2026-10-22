"""CPU-only analysis of fixed-B0 repeats and separately labelled historical updates."""
from __future__ import annotations

from itertools import combinations
import math
import shutil
import struct

from .legacy import snapshots

REPEATS = 5
PAIRS = tuple(combinations(range(REPEATS), 2))
ANOMALIES = ((0, 223955, "quats"), (1, 709801, "quats"), (1, 1488267, "shN"))
OWNER_SIZE = 1_500_000
SELECTION = "same_owner_visibility_P2_P3_within_tolerance_nearest_log_features_slot_tiebreak"


def decode(node, folder, tensor=None):
    kind = node["type"]
    if kind == "dict":
        return {k["value"]: decode(v, folder, tensor) for k, v in node["items"]}
    if kind in ("list", "tuple"):
        values = [decode(v, folder, tensor) for v in node["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "tensor":
        import numpy as np
        value = np.load(snapshots.safe_file(folder, node["file"]), mmap_mode="r", allow_pickle=False)
        return tensor(value, node) if tensor else value
    if kind == "nonfinite":
        raise ValueError("nonfinite snapshot scalar")
    return node["value"]


class Source:
    def __init__(self, root):
        self.root, self.records = root, {}

    def get(self, branch, rank, stage, *keys):
        folder = self.root / branch / f"rank-{rank}" / stage
        if folder not in self.records:
            self.records[folder] = snapshots.load_snapshot(folder)
        node = self.records[folder]["tree"]
        for key in keys:
            node = (next(v for k, v in node["items"] if k["value"] == key)
                    if node["type"] == "dict" else node["items"][key])
        return decode(node, folder)


def ordered_float32(bits):
    return 0x80000000 - (bits & 0x7fffffff) if bits & 0x80000000 else 0x80000000 + bits


def scalar_ulp(a, b):
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("ULP requires finite inputs")
    def bits(x):
        return struct.unpack("<I", struct.pack("<f", x))[0]
    return abs(ordered_float32(bits(a)) - ordered_float32(bits(b)))


def nearest_control(target, candidates):
    # candidates: (slot, feature vector); Python tuple order supplies the slot tie-break.
    ranked = ((sum((x - y) ** 2 for x, y in zip(target, features)), slot, features)
              for slot, features in candidates)
    try:
        return min(ranked)
    except ValueError as exc:
        raise ValueError("no eligible control") from exc


def adam_scalar(parameter, gradient, m, v, step, group):
    if (group.get("weight_decay", 0) != 0 or group.get("amsgrad", False)
            or group.get("maximize", False) or step < 0 or int(step) != step or v < 0):
        raise ValueError("unsupported historical Adam configuration")
    beta1, beta2 = group["betas"]
    new_m = beta1 * m + (1 - beta1) * gradient
    new_v = beta2 * v + (1 - beta2) * gradient * gradient
    t = step + 1
    corrected_sigma = math.sqrt(new_v / (1 - beta2 ** t))
    update = -group["lr"] * (new_m / (1 - beta1 ** t)) / (corrected_sigma + group["eps"])
    return {"parameter": parameter + update, "update": update, "exp_avg": new_m, "exp_avg_sq": new_v,
            "corrected_sigma": corrected_sigma, "sigma_over_eps": corrected_sigma / group["eps"]}


def geometry(source, owner):
    import numpy as np
    def get(stage, *keys):
        return source.get("a", owner, stage, *keys)
    radii = get("P1", "render", "metadata", "radii")
    if radii.shape != (2, OWNER_SIZE, 2):
        raise ValueError("owner projection layout differs")
    visible = (radii > 0).all(-1)
    tiles = np.stack([source.get("a", c, "P1", "render", "metadata", "tiles_per_gauss")[0,
                     owner * OWNER_SIZE:(owner + 1) * OWNER_SIZE] for c in (0, 1)])
    if tiles.shape != visible.shape or ((~visible) & (tiles > 0)).any():
        raise ValueError("camera/owner mapping differs")
    quats = get("P0", "parameters", "quats").astype(np.float64)
    scales = get("P0", "parameters", "scales").astype(np.float64)
    opacity_logits = get("P0", "parameters", "opacities").astype(np.float64).reshape(-1)
    norm = np.linalg.norm(quats, axis=1)
    features = np.column_stack((np.log1p(tiles.sum(0)), scales.max(1) - scales.min(1),
                                np.log(norm), -np.logaddexp(0., -opacity_logits)))
    if not np.isfinite(features).all():
        raise ValueError("nonfinite selection features")
    return features, visible, tiles


def select_points(source):
    import numpy as np
    selected, excluded = [], {(o, s) for o, s, _ in ANOMALIES}
    for owner in (0, 1):
        features, visible, _ = geometry(source, owner)
        for o, slot, group in ANOMALIES:
            if o != owner:
                continue
            eligible = (visible == visible[:, slot, None]).all(0)
            for stage, key in (("P2", "gradients"), ("P3", "parameters")):
                a = source.get("a", owner, stage, key, group).reshape(OWNER_SIZE, -1)
                b = source.get("b", owner, stage, key, group).reshape(OWNER_SIZE, -1)
                for start in range(0, OWNER_SIZE, 32768):
                    x, y = a[start:start + 32768].astype(np.float64), b[start:start + 32768].astype(np.float64)
                    valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x - y) <= snapshots.ATOL + snapshots.RTOL * np.abs(y))
                    eligible[start:start + len(x)] &= valid.all(1)
            for oo, ss in excluded:
                if oo == owner:
                    eligible[ss] = False
            distance, control, control_features = nearest_control(features[slot],
                ((int(i), features[i].tolist()) for i in np.flatnonzero(eligible)))
            common = {"owner": owner, "parameter": group, "visibility": visible[:, slot].tolist()}
            selected += [{**common, "role": "prior_anomaly", "slot": slot, "features": features[slot].tolist()},
                         {**common, "role": "matched_control", "slot": control, "matched_to": slot,
                          "features": control_features, "squared_distance": distance, "eligible_count": int(eligible.sum())}]
            excluded.add((owner, control))
    return selected


def historical_points(source, points):
    import numpy as np
    result = []
    for point in points:
        owner, slot, group = point["owner"], point["slot"], point["parameter"]
        features, visible, tiles = geometry(source, owner)
        cameras = []
        for c in (0, 1):
            conic = source.get("a", owner, "P1", "render", "metadata", "conics")[c, slot].astype(np.float64)
            eigen = np.linalg.eigvalsh([[conic[0], conic[1]], [conic[1], conic[2]]]) if visible[c, slot] else None
            cameras.append({"camera": c, "visible": bool(visible[c, slot]), "tiles": int(tiles[c, slot]),
                "depth": float(source.get("a", owner, "P1", "render", "metadata", "depths")[c, slot]),
                "conic_eigenvalues": eigen.tolist() if eigen is not None else None,
                "conic_condition": float(eigen[1] / eigen[0]) if eigen is not None and eigen[0] > 0 else None})
        groups = {}
        for group in snapshots.PARAMETERS:
            branches = {}
            for branch in ("a", "b"):
                def get(stage, *keys):
                    return source.get(branch, owner, stage, *keys)
                old = get("P0", "parameters", group)[slot].reshape(-1).astype(np.float64)
                gradient = get("P2", "gradients", group)[slot].reshape(-1).astype(np.float64)
                actual = get("P3", "parameters", group)[slot].reshape(-1).astype(np.float64)
                m = get("P0", "adam", group, "state", "exp_avg")[slot].reshape(-1)
                v = get("P0", "adam", group, "state", "exp_avg_sq")[slot].reshape(-1)
                step = float(get("P0", "adam", group, "state", "step"))
                config = get("P3", "adam", group, "param_groups", 0)
                formulas = [adam_scalar(float(p), float(g), float(mm), float(vv), step, config)
                            for p, g, mm, vv in zip(old, gradient, m, v)]
                branches[branch] = {"P0_parameter": old.tolist(), "P2_gradient": gradient.tolist(),
                    "P0_exp_avg": m.tolist(), "P0_exp_avg_sq": v.tolist(), "P0_step": step,
                    "P3_parameter": actual.tolist(), "actual_update": (actual - old).tolist(),
                    "P3_exp_avg": get("P3", "adam", group, "state", "exp_avg")[slot].reshape(-1).tolist(),
                    "P3_exp_avg_sq": get("P3", "adam", group, "state", "exp_avg_sq")[slot].reshape(-1).tolist(),
                    "P3_step": float(get("P3", "adam", group, "state", "step")), "update_param_group": config,
                    "P0_param_group": get("P0", "adam", group, "param_groups", 0),
                    "float64_scalar_reference": formulas,
                    "parameter_reference_error": (actual - [f["parameter"] for f in formulas]).tolist()}
                if branches[branch]["P3_step"] != step + 1:
                    raise ValueError("historical Adam step differs")
            differences = {key: (np.asarray(branches["a"][key]) - np.asarray(branches["b"][key])).tolist()
                           for key in ("P2_gradient", "P3_parameter", "actual_update", "P3_exp_avg", "P3_exp_avg_sq")}
            groups[group] = {"branches": branches, "a_minus_b": differences}
        result.append({**point, "cameras": cameras, "raw_quaternion_norm": math.exp(features[slot, 2]),
            "scale_axis_ratio": math.exp(features[slot, 1]), "opacity": math.exp(features[slot, 3]),
            "scale_axes": np.exp(source.get("a", owner, "P0", "parameters", "scales")[slot].astype(np.float64)).tolist(),
            "historical_source": str(source.root), "historical_AA_by_parameter": groups})
    return result


def repeated_statistics(arrays, folder, name):
    """Ten within-camera pairs, with bounded temporary memory and durable per-element frequencies."""
    import numpy as np
    if len(arrays) != REPEATS or any(a.dtype != np.float32 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("five matching float32 outputs required")
    shape, size = arrays[0].shape, arrays[0].size
    if not size:
        raise ValueError("empty repeat output")
    if shutil.disk_usage(folder).free < size * 11 + 4 * snapshots.GIB:
        raise ValueError("comparison would breach disk reserve")
    paths = {k: folder / f"{name}-{k}.npy" for k in ("strict_frequency", "bitwise_frequency", "failure_frequency", "range")}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("comparison output already exists")
    maps = {k: np.lib.format.open_memmap(p, mode="w+", dtype=np.float64 if k == "range" else np.uint8, shape=shape)
            for k, p in paths.items()}
    pairs = {f"{i}-{j}": {**snapshots.compare_arrays(arrays[i], arrays[j]), "ulp_histogram": {},
                         "max_abs_coordinate": None, "max_ulp": -1, "max_ulp_coordinate": None}
             for i, j in PAIRS}
    if any(p["num_nonfinite"] for p in pairs.values()):
        raise ValueError("nonfinite replay output")
    flat = [a.reshape(-1) for a in arrays]
    flatmaps = {k: a.reshape(-1) for k, a in maps.items()}
    for start in range(0, size, 262144):
        x = np.stack([a[start:start + 262144] for a in flat])
        f = x.astype(np.float64)
        bits = x.view(np.uint32).astype(np.int64)
        ordered = np.where(bits & 0x80000000, 0x80000000 - (bits & 0x7fffffff), 0x80000000 + bits)
        counts = {k: np.zeros(x.shape[1], dtype=np.uint8) for k in paths if k != "range"}
        for i, j in PAIRS:
            p = pairs[f"{i}-{j}"]
            error, ulp = np.abs(f[i] - f[j]), np.abs(ordered[i] - ordered[j])
            counts["strict_frequency"] += x[i] != x[j]
            counts["bitwise_frequency"] += bits[i] != bits[j]
            counts["failure_frequency"] += error > snapshots.ATOL + snapshots.RTOL * np.abs(f[j])
            values, nums = np.unique(ulp, return_counts=True)
            for value, count in zip(values, nums):
                key = str(int(value))
                p["ulp_histogram"][key] = p["ulp_histogram"].get(key, 0) + int(count)
            for metric, coordinate, target in ((error, "max_abs_coordinate", p["max_abs_error"]),
                                               (ulp, "max_ulp_coordinate", int(ulp.max()))):
                index = int(metric.argmax())
                take = ((coordinate == "max_abs_coordinate" and p[coordinate] is None and metric[index] == target)
                        or (coordinate == "max_ulp_coordinate" and target > p["max_ulp"]))
                if take:
                    p[coordinate] = {"index": list(map(int, np.unravel_index(start + index, shape))),
                                     "a": float(x[i, index]), "b": float(x[j, index]),
                                     "abs_error": float(error[index]), "ulp": int(ulp[index])}
                    if coordinate == "max_ulp_coordinate":
                        p["max_ulp"] = target
        for key, count in counts.items():
            flatmaps[key][start:start + x.shape[1]] = count
        flatmaps["range"][start:start + x.shape[1]] = f.max(0) - f.min(0)
    for a in maps.values():
        a.flush()
    frequencies = {k: np.bincount(v.reshape(-1), minlength=len(PAIRS) + 1).tolist() for k, v in maps.items() if k != "range"}
    extrema = {}
    for p in pairs.values():
        key = str(p["max_abs_coordinate"]["index"])
        extrema[key] = extrema.get(key, 0) + 1
    maximum = int(flatmaps["range"].argmax())
    summary = {"max_abs_error": max(p["max_abs_error"] for p in pairs.values()),
               "max_mean_abs_error": max(p["mean_abs_error"] for p in pairs.values()),
               "max_ulp": max(p["max_ulp"] for p in pairs.values()),
               "strict_unequal_pair_counts": [p["num_strict_unequal"] for p in pairs.values()],
               "above_tolerance_pair_counts": [p["num_mismatch"] for p in pairs.values()],
               "bitwise_unequal_ratios": [p["num_bitwise_unequal"] / size for p in pairs.values()],
               "unique_changed_elements": size - frequencies["strict_frequency"][0],
               "unique_above_tolerance_elements": size - frequencies["failure_frequency"][0],
               "not_independent_Gaussian_counts": True, "extreme_ties": "first_flat_coordinate"}
    return {"summary": summary, "pairs": pairs, "frequency_histograms": frequencies, "max_abs_coordinate_pair_frequency": extrema,
            "observed_max_range": float(flatmaps["range"][maximum]),
            "max_range_coordinate": list(map(int, np.unravel_index(maximum, shape))),
            "element_count": size, "pair_count": len(PAIRS),
            "array_files": {k: {"file": p.name, "sha256": snapshots.sha256_file(p)} for k, p in paths.items()}}
