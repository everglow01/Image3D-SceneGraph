"""Pixel reprojection evidence computed from fixed COLMAP text geometry, not ERROR."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.geometry.grouping import qvec_to_rotmat


def project_pixels(points: np.ndarray, model: str, params: list[float]) -> np.ndarray:
    sizes = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8}
    if model not in sizes or len(params) != sizes[model]:
        raise ValueError(f"unsupported or invalid reprojection camera: {model}")
    if not np.isfinite(params).all() or np.any(points[:, 2] <= 0):
        raise ValueError("invalid calibration or non-positive observation depth")
    if model in {"PINHOLE", "OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        fx, cx, cy = params[:3]
        fy = fx
    if min(fx, fy) <= 0:
        raise ValueError("non-positive focal length")
    x, y = (points[:, :2] / points[:, 2, None]).T
    r2 = x * x + y * y
    k1 = k2 = p1 = p2 = 0.0
    if model == "SIMPLE_RADIAL":
        k1 = params[3]
    elif model == "RADIAL":
        k1, k2 = params[3:]
    elif model == "OPENCV":
        k1, k2, p1, p2 = params[4:]
    radial = 1 + k1 * r2 + k2 * r2 * r2
    pixels = np.column_stack((
        fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx,
        fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy,
    ))
    if not np.isfinite(pixels).all():
        raise ValueError("non-finite pixel projection")
    return pixels


def pixel_reprojection_errors(model_dir: Path) -> tuple[dict[int, float], dict]:
    """Mean Euclidean pixel error per point; aggregate gives equal weight to points."""
    cameras = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        cid = int(values[0])
        if cid in cameras:
            raise ValueError("duplicate camera id")
        cameras[cid] = (values[1], list(map(float, values[4:])))
    ids, points = [], []
    with (model_dir / "points3D.txt").open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            ids.append(int(values[0]))
            points.append(list(map(float, values[1:4])))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty point cloud or duplicate point id")
    order = np.argsort(ids)
    point_ids = np.asarray(ids, dtype=np.int64)[order]
    xyz = np.asarray(points, dtype=np.float64)[order]
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite point coordinates")
    sums = np.zeros(len(ids))
    counts = np.zeros(len(ids), dtype=np.int64)
    image_ids = set()
    with (model_dir / "images.txt").open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            iid = int(fields[0])
            if iid in image_ids:
                raise ValueError("duplicate image id")
            image_ids.add(iid)
            tokens = handle.readline().split()
            if len(tokens) % 3:
                raise ValueError("invalid observation row")
            refs = np.asarray([int(x) for x in tokens[2::3]], dtype=np.int64)
            if not len(refs):
                continue
            xy = np.column_stack((np.asarray(tokens[0::3], float), np.asarray(tokens[1::3], float)))
            valid = refs != -1
            refs, xy = refs[valid], xy[valid]
            if not len(refs):
                continue
            indices = np.searchsorted(point_ids, refs)
            if np.any(indices >= len(ids)) or not np.array_equal(point_ids[indices], refs):
                raise ValueError("observation references missing point")
            q = np.asarray(fields[1:5], float)
            if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-6):
                raise ValueError("invalid camera quaternion")
            camera_points = xyz[indices] @ qvec_to_rotmat(q).T + np.asarray(fields[5:8], float)
            model, params = cameras[int(fields[8])]
            errors = np.linalg.norm(project_pixels(camera_points, model, params) - xy, axis=1)
            if not np.isfinite(errors).all():
                raise ValueError("non-finite observed pixel residual")
            np.add.at(sums, indices, errors)
            np.add.at(counts, indices, 1)
    if np.any(counts == 0):
        raise ValueError("point has no registered observations")
    means = sums / counts
    return dict(zip(point_ids.tolist(), means.tolist())), {
        "schema_version": 1,
        "profile": "fixed_model_pixel_reprojection_v1",
        "units": "pixels",
        "aggregation": "mean_euclidean_error_per_point_then_unweighted_point_summary",
        "point_count": len(ids),
        "observation_count": int(counts.sum()),
        "mean_reprojection_error_pixels": float(means.mean()),
        "median_reprojection_error_pixels": float(np.median(means)),
        "p90_reprojection_error_pixels": float(np.percentile(means, 90)),
    }


def normalize_pixel_errors(model_dir: Path) -> dict:
    """Correct a newly converted text model, preserving the original ERROR bytes."""
    path = model_dir / "points3D.txt"
    original = model_dir / "points3D.source-error.txt"
    record_path = model_dir / "pixel-reprojection.json"
    temporary = model_dir / ".points3D.pixels.tmp"
    if original.exists() or record_path.exists() or temporary.exists():
        raise ValueError("pixel normalization destination already exists")
    errors, record = pixel_reprojection_errors(model_dir)
    record["source_files_sha256"] = {
        name: sha256_file(model_dir / name) for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    with path.open() as source, temporary.open("x") as dest:
        for line in source:
            if line.strip() and not line.lstrip().startswith("#"):
                fields = list(re.finditer(r"\S+", line))
                token = fields[7]
                error = errors[int(fields[0].group())]
                line = line[:token.start()] + format(error, ".17g") + line[token.end():]
            dest.write(line)
        dest.flush()
        os.fsync(dest.fileno())
    record["normalized_points_sha256"] = sha256_file(temporary)
    os.rename(path, original)
    os.rename(temporary, path)
    with record_path.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    return record
