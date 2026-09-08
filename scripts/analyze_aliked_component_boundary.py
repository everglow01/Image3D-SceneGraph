#!/usr/bin/env python3
"""Read-only candidate/verification and RGB inspection of the disconnected 38-frame component."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from analyze_aliked_dense_gaps import MIN_PAIR_MATCHES, PAIR_BASE, distribution
from image3d_scenegraph.geometry.colmap import sha256_file
from image3d_scenegraph.geometry.view_graph import _connected_components


def crossing_pairs(pairs, component):
    return [p for p in pairs if (p["left"] in component) != (p["right"] in component)]


def spatial_support(xy, width, height):
    if not len(xy):
        return {"count": 0, "occupied_4x4_cells": 0, "bbox_fraction": 0.0}
    normalized = xy / [width, height]
    cells = np.clip(np.floor(normalized * 4), 0, 3).astype(int)
    return {"count": len(xy), "occupied_4x4_cells": len(set(map(tuple, cells))),
            "bbox_fraction": float(np.prod(np.ptp(normalized, axis=0)))}


def audit(root, output):
    root = root.resolve()
    prior_path = root / "gap-support-v1.json"
    prior = json.loads(prior_path.read_text())
    if prior.get("profile") != "aliked_dense_gap_support_v1":
        raise ValueError("expected dense-gap source audit")
    database = root / "positive/database.db"
    selection_path = root / "selection.json"
    sources = {prior_path: sha256_file(prior_path)}
    for p in (database, selection_path):
        sources[p] = prior["source_hashes"][str(p.relative_to(root))]
        if sha256_file(p) != sources[p]:
            raise ValueError("source differs from frozen audit")
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("nonempty source WAL")
    selection = json.loads(selection_path.read_text())["selected"]
    selected = {v["path"]: v for v in selection}
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        names = dict(db.execute("SELECT image_id,name FROM images"))
        times = {i: selected[n]["time_seconds"] for i, n in names.items()}
        counts = dict(db.execute("SELECT image_id,rows FROM keypoints"))
        pairs, adjacency = [], {i: set() for i in names}
        for pair_id, candidate, verified, config in db.execute(
            "SELECT m.pair_id,m.rows,COALESCE(t.rows,0),COALESCE(t.config,0) "
            "FROM matches m LEFT JOIN two_view_geometries t ON m.pair_id=t.pair_id ORDER BY m.pair_id"
        ):
            left, right = divmod(pair_id, PAIR_BASE)
            pairs.append({"pair_id": pair_id, "left": left, "right": right,
                          "left_name": names[left], "right_name": names[right],
                          "candidate": candidate, "verified": verified, "config": config,
                          "time_span_seconds": abs(times[left] - times[right])})
            if verified >= MIN_PAIR_MATCHES:
                adjacency[left].add(right)
                adjacency[right].add(left)
        components = _connected_components(adjacency)
        if list(map(len, components)) != [408, 38]:
            raise ValueError("expected frozen 408/38 component split")
        component = components[1]
        cross = crossing_pairs(pairs, component)
        begin, end = min(times[i] for i in component), max(times[i] for i in component)
        local = [p for p in cross if p["time_span_seconds"] <= 2.0]
        before = [p for p in cross if min(times[p["left"]], times[p["right"]]) < begin]
        after = [p for p in cross if max(times[p["left"]], times[p["right"]]) > end]
        nearest = [min(side, key=lambda p: (p["time_span_seconds"], p["pair_id"])) for side in (before, after)]
        strongest = sorted(cross, key=lambda p: (-p["candidate"], p["time_span_seconds"], p["pair_id"]))[:8]
        previews = list({p["pair_id"]: p for p in nearest + strongest[:3]}.values())
        canvas = Image.new("RGB", (640, 600 * len(previews)), "black")
        details = []
        for row, pair in enumerate(previews):
            arrays, dimensions, pictures = [], [], []
            for image_id in (pair["left"], pair["right"]):
                nrows, cols, blob = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
                if len(blob) != nrows * cols * 4 or cols < 2:
                    raise ValueError("invalid feature array")
                arrays.append(np.frombuffer(blob, dtype="<f4").reshape(nrows, cols)[:, :2])
                p = root / "images" / names[image_id]
                p.resolve().relative_to((root / "images").resolve())
                sources[p] = selected[names[image_id]]["sha256"]
                if sha256_file(p) != sources[p]:
                    raise ValueError("RGB hash mismatch")
                with Image.open(p) as image:
                    dimensions.append(image.size)
                    pictures.append(image.convert("RGB").resize((320, 568)))
            rows, cols, blob = db.execute("SELECT rows,cols,data FROM matches WHERE pair_id=?", (pair["pair_id"],)).fetchone()
            if rows and (cols != 2 or len(blob) != rows * 8):
                raise ValueError("invalid candidate match array")
            indices = np.frombuffer(blob or b"", dtype="<u4").reshape(-1, 2)
            if len(indices) and any(np.any(indices[:, k] >= len(arrays[k])) for k in (0, 1)):
                raise ValueError("candidate index out of bounds")
            xy = [arrays[k][indices[:, k]] for k in (0, 1)]
            detail = {**pair, "left_time": times[pair["left"]], "right_time": times[pair["right"]],
                      "left_features": counts[pair["left"]], "right_features": counts[pair["right"]],
                      "spatial_support": [spatial_support(xy[k], *dimensions[k]) for k in (0, 1)]}
            details.append(detail)
            y = row * 600
            canvas.paste(pictures[0], (0, y + 28))
            canvas.paste(pictures[1], (320, y + 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, y + 4), f'{pair["left"]} {detail["left_time"]:.3f}s | {pair["right"]} {detail["right_time"]:.3f}s | candidate={rows} verified={pair["verified"]}', fill="white")
            for j in range(min(rows, 50)):
                endpoints = [xy[k][j] / dimensions[k] * [320, 568] + [320 * k, y + 28] for k in (0, 1)]
                color = (255, 80 + j * 37 % 176, 40 + j * 71 % 216)
                draw.line([tuple(p) for p in endpoints], fill=color, width=1)
                for x, yy in endpoints:
                    draw.ellipse((x - 2, yy - 2, x + 2, yy + 2), outline=color)
        preview_path = output / "boundary-pairs.jpg"
        canvas.save(preview_path, quality=92)
    for p, expected in sources.items():
        if sha256_file(p) != expected:
            raise ValueError("source changed during boundary audit")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("source WAL changed during audit")
    return {
        "profile": "aliked_component_boundary_v1", "source_unchanged": True,
        "source_hashes": {str(p.relative_to(root)): h for p, h in sources.items()},
        "component_image_ids": sorted(component), "component_start_seconds": begin, "component_end_seconds": end,
        "component_feature_counts": distribution(counts[i] for i in component),
        "cross_pair_count": len(cross), "cross_candidate_nonempty_pairs": sum(p["candidate"] > 0 for p in cross),
        "cross_candidate_at_least_15_pairs": sum(p["candidate"] >= 15 for p in cross),
        "cross_verified_nonempty_pairs": sum(p["verified"] > 0 for p in cross),
        "cross_candidate_count": distribution(p["candidate"] for p in cross),
        "nearest_boundary_pairs": nearest, "strongest_candidate_pairs": strongest,
        "local_bridge_policy": "cross_component_time_span_at_most_2_seconds",
        "local_bridge_pairs": local, "preview_pairs": details,
        "preview": {"path": preview_path.name, "sha256": sha256_file(preview_path), "max_displayed_correspondences_per_pair": 50},
        "limitations": ["Candidate lines are unverified, not ground-truth correspondences.",
                        "Nonzero candidate count does not prove registration support or useful parallax."],
        "reconstruction_started": False, "training_started": False, "test_rgb_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = audit(args.root, args.output)
    record["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    path = args.output / "results.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(path), "sha256": sha256_file(path),
                      "cross_candidate_count": record["cross_candidate_count"],
                      "cross_candidate_at_least_15_pairs": record["cross_candidate_at_least_15_pairs"],
                      "local_pair_count": len(record["local_bridge_pairs"])}, indent=2))


if __name__ == "__main__":
    main()
