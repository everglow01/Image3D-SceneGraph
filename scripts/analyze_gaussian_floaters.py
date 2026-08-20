"""Census free-space floater populations in an exported Gaussian splat.

Axis-free measurement: nearest distance of every gaussian to the SfM point
cloud and to the camera centers, both in colmap_world. The gaussian PLY lives
in the canonical normalized frame, so the sibling export.json
world_from_normalized transform is applied to positions when present; scales
stay in the normalized training frame, where the veil thresholds are
meaningful. The SfM neighbor spacing (median / p90 of nearest-neighbor
distances on a seeded subsample) calibrates the hugging and free-space radii.
These census numbers gate the floater mitigation experiments in the codex.md
decision log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from analyze_pointcloud import read_ply_points_and_colors
from image3d_scenegraph.gaussian.dataset import qvec_to_rotmat
from image3d_scenegraph.gaussian.export import read_gaussian_ply

SCHEMA_VERSION = 1
NN_CALIBRATION_SAMPLE = 50_000
NN_CALIBRATION_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Census free-space floater populations in a Gaussian splat."
    )
    parser.add_argument("--gaussian-ply", required=True, type=Path, help="Canonical exported scene.ply (normalized frame).")
    parser.add_argument("--points", required=True, type=Path, help="SfM points.ply in colmap_world.")
    parser.add_argument("--cameras", required=True, type=Path, help="geometry/cameras.json.")
    parser.add_argument("--output", type=Path, help="Output report JSON. Prints JSON when omitted.")
    parser.add_argument("--haze-threshold", type=float, default=0.05, help="Opacity below this is haze.")
    parser.add_argument("--thick-threshold", type=float, default=0.25, help="Opacity at/above this is thick.")
    parser.add_argument("--free-camera-distance", type=float, default=1.0, help="World-unit camera distance required for free space.")
    args = parser.parse_args()
    if not 0.0 < args.haze_threshold < args.thick_threshold < 1.0:
        raise SystemExit("thresholds must satisfy 0 < haze < thick < 1")
    try:
        report = analyze_floaters(
            gaussian_ply=args.gaussian_ply,
            points=args.points,
            cameras=args.cameras,
            haze_threshold=args.haze_threshold,
            thick_threshold=args.thick_threshold,
            free_camera_distance=args.free_camera_distance,
        )
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))

    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    populations = report["populations"]
    haze = populations["haze"]
    print(f"gaussian_count={report['gaussian_count']}")
    print(f"sfm_nn_median={report['thresholds']['hug_radius']:.6f} sfm_nn_p90={report['thresholds']['free_radius']:.6f}")
    for name in ("haze", "core", "thick"):
        entry = populations[name]
        print(f"{name}_count={entry['count']} {name}_fraction={entry['fraction']:.4f}")
        if entry["count"]:
            print(f"{name}_hugging_fraction={entry['hugging_fraction']:.4f} {name}_free_space_count={entry['free_space_count']} {name}_free_space_fraction={entry['free_space_fraction']:.4f}")
    if haze["count"] and haze["free_space_count"]:
        print(f"free_haze_scale_median={haze['free_space_scale_median']:.6f} veil_count_gt_0.01={haze['veil_count_gt_0.01']} veil_count_gt_0.03={haze['veil_count_gt_0.03']}")


def analyze_floaters(
    *,
    gaussian_ply: Path,
    points: Path,
    cameras: Path,
    haze_threshold: float = 0.05,
    thick_threshold: float = 0.25,
    free_camera_distance: float = 1.0,
) -> dict[str, Any]:
    params = read_gaussian_ply(gaussian_ply)
    positions = np.column_stack((params["x"], params["y"], params["z"])).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-params["opacity"].astype(np.float64)))
    scales = np.exp(
        np.column_stack((params["scale_0"], params["scale_1"], params["scale_2"])).astype(np.float64)
    )
    transform = _load_world_transform(gaussian_ply)
    if transform is not None:
        _uniform_scale(transform)
        positions = positions @ transform[:3, :3].T + transform[:3, 3]
    max_scale = scales.max(axis=1)

    sfm_points, _ = read_ply_points_and_colors(points)
    sfm_points = sfm_points[np.isfinite(sfm_points).all(axis=1)].astype(np.float64)
    if len(sfm_points) < 2:
        raise ValueError(f"SfM point cloud has fewer than two finite points: {points}")
    camera_centers = _camera_centers(cameras)

    hug_radius, free_radius = _calibrate_nn_spacing(sfm_points)
    sfm_tree = cKDTree(sfm_points)
    camera_tree = cKDTree(camera_centers)

    bands = {
        "haze": opacity < haze_threshold,
        "core": (opacity >= haze_threshold) & (opacity < thick_threshold),
        "thick": opacity >= thick_threshold,
    }
    populations: dict[str, Any] = {}
    for name, mask in bands.items():
        entry: dict[str, Any] = {"count": int(mask.sum()), "fraction": float(mask.mean())}
        if entry["count"]:
            distance_to_sfm, _ = sfm_tree.query(positions[mask])
            distance_to_camera, _ = camera_tree.query(positions[mask])
            free_mask = (distance_to_sfm > free_radius) & (distance_to_camera > free_camera_distance)
            entry.update(
                {
                    "distance_to_sfm_median": float(np.median(distance_to_sfm)),
                    "distance_to_sfm_p90": float(np.percentile(distance_to_sfm, 90)),
                    "distance_to_camera_median": float(np.median(distance_to_camera)),
                    "hugging_fraction": float((distance_to_sfm < hug_radius).mean()),
                    "free_space_count": int(free_mask.sum()),
                    "free_space_fraction": float(free_mask.mean()),
                }
            )
            if name == "haze" and free_mask.any():
                free_scales = max_scale[mask][free_mask]
                entry.update(
                    {
                        "free_space_scale_median": float(np.median(free_scales)),
                        "free_space_scale_p99": float(np.percentile(free_scales, 99)),
                        "free_space_scale_max": float(free_scales.max()),
                        "veil_count_gt_0.01": int((free_scales > 0.01).sum()),
                        "veil_count_gt_0.03": int((free_scales > 0.03).sum()),
                    }
                )
        populations[name] = entry

    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "gaussian_ply": str(gaussian_ply),
            "points": str(points),
            "cameras": str(cameras),
            "world_transform": "world_from_normalized" if transform is not None else None,
        },
        "gaussian_count": int(len(opacity)),
        "sfm_point_count": int(len(sfm_points)),
        "camera_count": int(len(camera_centers)),
        "thresholds": {
            "haze": haze_threshold,
            "thick": thick_threshold,
            "free_camera_distance": free_camera_distance,
            "hug_radius": hug_radius,
            "free_radius": free_radius,
            "nn_calibration_sample": NN_CALIBRATION_SAMPLE,
            "nn_calibration_seed": NN_CALIBRATION_SEED,
        },
        "populations": populations,
    }


def _load_world_transform(gaussian_ply: Path) -> np.ndarray | None:
    export_path = gaussian_ply.parent / "export.json"
    if not export_path.is_file():
        print(f"warning: no export.json next to {gaussian_ply}; assuming a shared coordinate frame")
        return None
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    transform = np.asarray(payload["world_from_normalized"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"world_from_normalized is not a finite 4x4 matrix: {export_path}")
    return transform


def _uniform_scale(transform: np.ndarray) -> float:
    linear = transform[:3, :3]
    scale = float(np.mean(np.diag(linear)))
    expected = scale * np.eye(3)
    if scale <= 0 or not np.allclose(linear, expected, rtol=1e-4, atol=1e-6):
        raise ValueError("world_from_normalized must be a uniform scale plus translation")
    return scale


def _camera_centers(cameras: Path) -> np.ndarray:
    payload = json.loads(cameras.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not images:
        raise ValueError(f"cameras.json has no images: {cameras}")
    centers = []
    for image in images:
        rotation = qvec_to_rotmat(image["qvec"])
        centers.append(-rotation.T @ np.asarray(image["tvec"], dtype=np.float64))
    return np.asarray(centers, dtype=np.float64)


def _calibrate_nn_spacing(sfm_points: np.ndarray) -> tuple[float, float]:
    if len(sfm_points) > NN_CALIBRATION_SAMPLE:
        rng = np.random.RandomState(NN_CALIBRATION_SEED)
        indices = rng.choice(len(sfm_points), size=NN_CALIBRATION_SAMPLE, replace=False)
        sample = sfm_points[indices]
    else:
        sample = sfm_points
    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=2)
    nearest = distances[:, 1]
    return float(np.median(nearest)), float(np.percentile(nearest, 90))


if __name__ == "__main__":
    main()
