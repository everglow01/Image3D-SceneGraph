#!/usr/bin/env python3
"""Read a bounded set of frozen SfM matches without rerunning matching or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from image3d_scenegraph.geometry.grouping import (
    parse_colmap_images_with_points,
    qvec_to_rotmat,
)

MAX_IMAGE_ID = 2_147_483_647
PAIRS = ((61, 62), (87, 90), (710, 713), (728, 730), (729, 730), (827, 828),
         (468, 469), (575, 576))
ARMS = ("sift-bruteforce", "sift-lightglue", "aliked-bruteforce", "aliked-lightglue")


def read_pair(connection, left: str, right: str, table: str):
    if table not in {"matches", "two_view_geometries"}:
        raise ValueError("unsupported match table")
    ids = [connection.execute("SELECT image_id FROM images WHERE name=?", (name,)).fetchone()[0]
           for name in (left, right)]
    low, high = sorted(ids)
    row = connection.execute(
        f"SELECT rows, cols, data FROM {table} WHERE pair_id=?",
        (low * MAX_IMAGE_ID + high,),
    ).fetchone()
    if row is None:
        return None
    count, columns, blob = row
    if count == 0:
        return np.empty((0, 2), dtype=np.uint32)
    if columns != 2:
        raise ValueError("matches must have two columns")
    pairs = np.frombuffer(blob, dtype="<u4").reshape(count, columns)
    return pairs if ids[0] == low else pairs[:, ::-1]


def sampson_pixels(x0, x1, rotation, translation, focal):
    t = translation
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    essential = skew @ rotation
    a = np.column_stack((x0, np.ones(len(x0))))
    b = np.column_stack((x1, np.ones(len(x1))))
    ea, etb = a @ essential.T, b @ essential
    denominator = (ea[:, :2] ** 2).sum(axis=1) + (etb[:, :2] ** 2).sum(axis=1)
    if np.any(denominator <= 1e-20):
        raise ValueError("degenerate reference epipolar geometry")
    return np.abs((b * ea).sum(axis=1)) / np.sqrt(denominator) * focal


def audit(root: Path) -> dict:
    import cv2

    reference_dir = root / "sift-bruteforce/colmap/sparse_raw_txt"
    reference = {image.name: image for image in parse_colmap_images_with_points(reference_dir / "images.txt")}
    cameras = {}
    for line in (reference_dir / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "OPENCV":
            raise ValueError("this frozen audit requires OPENCV reference cameras")
        fx, fy, cx, cy, *distortion = map(float, fields[4:])
        cameras[int(fields[0])] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array(distortion))
    selection = json.loads((root / "sift-bruteforce/selection.json").read_text())
    names = [Path(item["path"]).name for item in selection["selected"]]
    by_frame = {int(name.split("_")[1]): name for name in names}
    records = []
    provenance = {}
    selection_hash = hashlib.sha256((root / "sift-bruteforce/selection.json").read_bytes()).hexdigest()
    for arm in ARMS:
        current_hash = hashlib.sha256((root / arm / "selection.json").read_bytes()).hexdigest()
        if current_hash != selection_hash:
            raise ValueError("frozen selections differ")
        contract_path = root / arm / "diagnostics/sfm_frontend_contract.json"
        provenance[arm] = {
            "selection_sha256": current_hash,
            "frontend_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "frontend_contract": json.loads(contract_path.read_text()),
        }
        path = root / arm / "colmap/database.db"
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for first, second in PAIRS:
                left, right = by_frame[first], by_frame[second]
                record = {"arm": arm, "names": [left, right], "frames": [first, second]}
                xy, counts, hashes = [], [], []
                for name in (left, right):
                    row = connection.execute(
                        "SELECT k.rows,k.cols,k.data FROM keypoints k JOIN images i ON i.image_id=k.image_id WHERE i.name=?", (name,)
                    ).fetchone()
                    counts.append(row[0])
                    hashes.append(hashlib.sha256(row[2]).hexdigest())
                    xy.append(np.frombuffer(row[2], dtype="<f4").reshape(row[0], row[1])[:, :2])
                record.update(keypoint_counts=counts, keypoint_sha256=hashes)
                candidate = read_pair(connection, left, right, "matches")
                verified = read_pair(connection, left, right, "two_view_geometries")
                record["candidate_count"] = None if candidate is None else len(candidate)
                record["verified_count"] = None if verified is None else len(verified)
                record["sift_reference_available"] = left in reference and right in reference
                if record["sift_reference_available"] and verified is not None and len(verified):
                    i0, i1 = reference[left], reference[right]
                    k0, d0 = cameras[i0.camera_id]
                    k1, d1 = cameras[i1.camera_id]
                    x0 = cv2.undistortPoints(xy[0][verified[:, 0]].reshape(-1, 1, 2), k0, d0).reshape(-1, 2)
                    x1 = cv2.undistortPoints(xy[1][verified[:, 1]].reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
                    r0, r1 = qvec_to_rotmat(i0.qvec), qvec_to_rotmat(i1.qvec)
                    relative = r1 @ r0.T
                    t = i1.tvec - relative @ i0.tvec
                    try:
                        errors = sampson_pixels(x0, x1, relative, t, float((k0[0, 0] + k1[0, 0]) / 2))
                        record["sift_reference_sampson_px"] = {
                            "median": float(np.median(errors)), "p90": float(np.quantile(errors, .9)),
                            "fraction_above_4px": float(np.mean(errors > 4)),
                        }
                    except ValueError:
                        record["sift_reference_error"] = "degenerate_epipolar_geometry"
                records.append(record)
    return {
        "profile": "sfm_small_match_audit_v1", "source": str(root),
        "source_contracts": provenance,
        "reference": "sift_bruteforce_estimated_poses_not_ground_truth",
        "reference_implementation_execution": "not_run",
        "matching_started": False, "training_started": False, "test_rgb_loaded": False,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing audit")
    record = audit(args.experiment.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(f"report={args.output}")
    print("matching_started=false training_started=false")


if __name__ == "__main__":
    main()
