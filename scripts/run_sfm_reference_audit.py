#!/usr/bin/env python3
"""Bounded ALIKED/LightGlue reference audit on frozen pairs; never train or download."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from unittest.mock import patch

import numpy as np

from analyze_sfm_small_matches import PAIRS, read_pair


def compare_pairs(first, second):
    return {
        "a_count": len(first), "b_count": len(second), "intersection": len(first & second),
        "jaccard": len(first & second) / len(first | second) if first | second else 1.0,
    }


def database_features(connection, name):
    row = connection.execute(
        "SELECT k.rows,k.cols,k.data,d.cols,d.data,c.width,c.height "
        "FROM images i JOIN keypoints k ON k.image_id=i.image_id "
        "JOIN descriptors d ON d.image_id=i.image_id "
        "JOIN cameras c ON c.camera_id=i.camera_id WHERE i.name=?", (name,),
    ).fetchone()
    count, kcols, kblob, dcols, dblob, width, height = row
    xy = np.frombuffer(kblob, dtype="<f4").reshape(count, kcols)[:, :2].copy() - 0.5
    desc = np.frombuffer(dblob, dtype="<f4").reshape(count, dcols // 4).copy()
    return {
        "keypoints": xy[None], "descriptors": desc[None],
        "image_size": np.array([[width, height]], dtype=np.float32),
    }


def compare_features(xy0, desc0, xy1, desc1):
    from scipy.spatial import cKDTree

    if not len(xy0) or not len(xy1):
        return {"a_count": len(xy0), "b_count": len(xy1), "status": "empty"}
    distance, index = cKDTree(xy1).query(xy0)
    close = distance < 0.5
    cosine = np.sum(desc0[close] * desc1[index[close]], axis=1)
    return {
        "a_count": len(xy0), "b_count": len(xy1),
        "fraction_within_half_pixel": float(np.mean(close)),
        "nearest_distance_median_px": float(np.median(distance)),
        "descriptor_cosine_median_on_close_points": float(np.median(cosine)) if len(cosine) else None,
    }


def run(project: Path, experiment: Path, output: Path) -> None:
    import cv2
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    import torch
    from lightglue import ALIKED, LightGlue

    if output.exists():
        raise ValueError("refusing to overwrite reference results")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = project / "outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2"
    arm = source / "aliked-lightglue"
    selection = json.loads((arm / "selection.json").read_text())
    names = {int(Path(x["path"]).name.split("_")[1]): Path(x["path"]).name for x in selection["selected"]}
    report = {
        "profile": "sfm_reference_audit_v1", "torch": torch.__version__, "ort": ort.__version__,
        "device": torch.cuda.get_device_name(0), "training_started": False, "test_rgb_loaded": False,
        "matcher_records": [], "extractor_records": [], "source_revisions": {},
        "matcher_policy": {"flash": False, "depth_confidence": -1, "width_confidence": -1, "threshold": 0.1},
    }
    for repo in ("lightglue", "aliked"):
        path = project / "external" / repo
        report["source_revisions"][repo] = {
            "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
            "working_tree": subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True),
        }
    state = torch.load(experiment / "aliked_lightglue.pth", map_location="cpu", weights_only=True)
    for i in range(9):
        for kind in ("self_attn", "cross_attn"):
            state = {k.replace(f"{kind}.{i}", f"transformers.{i}.{kind}"): v for k, v in state.items()}
    matcher = LightGlue(features=None, input_dim=128, flash=False, depth_confidence=-1,
                        width_confidence=-1, filter_threshold=0.1).eval().cuda()
    # Older official checkpoints omit this deterministic, non-learned buffer.
    state.setdefault("confidence_thresholds", matcher.confidence_thresholds.detach().cpu())
    matcher.load_state_dict(state, strict=True)
    lg_path = project / "external/colmap-features/aliked-lightglue.onnx"
    aliked_path = project / "external/colmap-features/aliked-n16rot.onnx"
    graph = onnx.load(lg_path)
    report["named_onnx_weights_vs_reference"] = [
        {"name": t.name, "equal": bool(np.array_equal(numpy_helper.to_array(t), state[t.name].numpy()))}
        for t in graph.graph.initializer if t.name in state
    ]
    report["lightglue_weight_sha256"] = hashlib.sha256((experiment / "aliked_lightglue.pth").read_bytes()).hexdigest()
    report["onnx_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (lg_path, aliked_path)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(lg_path), sess_options=options, providers=["CPUExecutionProvider"])
    with sqlite3.connect((arm / "colmap/database.db").as_uri() + "?mode=ro", uri=True) as db:
        with torch.inference_mode():
            for first, second in PAIRS:
                left, right = names[first], names[second]
                a, b = database_features(db, left), database_features(db, right)
                stored = read_pair(db, left, right, "matches")
                if stored is None:
                    raise ValueError("missing frozen candidate matches")
                expected = set(map(tuple, stored.tolist()))
                def tensor(data):
                    return {k: torch.from_numpy(v).cuda() for k, v in data.items()}
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                result = matcher({"image0": tensor(a), "image1": tensor(b)})
                torch.cuda.synchronize()
                actual = set(map(tuple, result["matches"][0].cpu().numpy().tolist()))
                record = {
                    "frames": [first, second], "names": [left, right],
                    "reference_vs_stored_colmap": compare_pairs(actual, expected),
                    "reference_elapsed_seconds": time.monotonic() - start,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "reference_stop_layer": result["stop"],
                }
                if (first, second) in ((710, 713), (468, 469)):
                    feed = {"kpts0": a["keypoints"], "kpts1": b["keypoints"],
                            "desc0": a["descriptors"], "desc1": b["descriptors"],
                            "image_size0": a["image_size"], "image_size1": b["image_size"]}
                    start = time.monotonic()
                    indices, scores = session.run(None, feed)
                    indices, scores = indices.reshape(-1).astype(np.int64), scores.reshape(-1)
                    good = (indices >= 0) & (scores >= 0.1)
                    onnx_pairs = set(zip(np.flatnonzero(good).tolist(), indices[good].tolist()))
                    record["onnx_cpu_seconds"] = time.monotonic() - start
                    record["onnx_cpu_vs_stored_colmap"] = compare_pairs(onnx_pairs, expected)
                    record["reference_vs_onnx_cpu"] = compare_pairs(actual, onnx_pairs)
                report["matcher_records"].append(record)
                print("MATCHER", json.dumps(record), flush=True)
        del session, matcher, state, graph, result
        gc.collect()
        torch.cuda.empty_cache()

        weight_path = project / "external/aliked/models/aliked-n16rot.pth"
        report["aliked_weight_sha256"] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        def local_weights(url, **kwargs):
            if not url.endswith("/aliked-n16rot.pth"):
                raise ValueError("unexpected reference weight request")
            return torch.load(weight_path, map_location="cpu", weights_only=True)
        with patch("torch.hub.load_state_dict_from_url", side_effect=local_weights):
            extractor = ALIKED(model_name="aliked-n16rot", max_num_keypoints=8192,
                               detection_threshold=0.2).eval().cuda()
        session = ort.InferenceSession(str(aliked_path), sess_options=options, providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            for frame in (710, 468):
                name = names[frame]
                image = cv2.cvtColor(cv2.imread(str(arm / "input" / name)), cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                padded = np.pad(image, ((0, -height % 32), (0, -width % 32), (0, 0)), mode="edge")
                pixels = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                ref = extractor({"image": torch.from_numpy(pixels).cuda()})
                ref_xy = ref["keypoints"][0].cpu().numpy()
                ref_desc = ref["descriptors"][0].cpu().numpy()
                valid = (ref_xy[:, 0] < width - 0.5) & (ref_xy[:, 1] < height - 0.5)
                valid &= (ref_xy[:, 0] >= -0.5) & (ref_xy[:, 1] >= -0.5)
                ref_xy, ref_desc = ref_xy[valid], ref_desc[valid]
                stored = database_features(db, name)
                for budget in (2048, 8192):
                    start = time.monotonic()
                    xy, desc, scores = session.run(None, {"image": pixels, "max_keypoints": np.array(budget, dtype=np.int64),
                                                         "min_score": np.array(0.2, dtype=np.float32)})
                    xy = (xy[0] + 1) * np.array([padded.shape[1] - 1, padded.shape[0] - 1]) / 2
                    desc = desc[0]
                    valid = (xy[:, 0] < width - 0.5) & (xy[:, 1] < height - 0.5)
                    valid &= (xy[:, 0] >= -0.5) & (xy[:, 1] >= -0.5)
                    xy, desc = xy[valid], desc[valid]
                    record = {
                        "frame": frame, "name": name, "requested_budget": budget,
                        "reference_requested_budget": 8192, "reference_count": len(ref_xy),
                        "onnx_count": len(xy), "onnx_cpu_seconds": time.monotonic() - start,
                        "onnx_vs_reference": compare_features(xy, desc, ref_xy, ref_desc),
                        "onnx_vs_stored_colmap": compare_features(xy, desc, stored["keypoints"][0], stored["descriptors"][0]),
                        "image_sha256": hashlib.sha256((arm / "input" / name).read_bytes()).hexdigest(),
                    }
                    report["extractor_records"].append(record)
                    print("EXTRACTOR", json.dumps(record), flush=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print("report=" + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.experiment.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
