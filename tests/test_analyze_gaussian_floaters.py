from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.gaussian.export import PLY_FIELDS


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_gaussian_floaters as floaters  # noqa: E402


def logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def write_gaussian_ply(path: Path, rows: list[dict[str, float]]) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header.extend(f"property float {name}" for name in PLY_FIELDS)
    header.append("end_header")
    payload = b"".join(
        struct.pack(
            f"<{len(PLY_FIELDS)}f",
            *[row.get(name, 0.0) for name in PLY_FIELDS],
        )
        for row in rows
    )
    path.write_bytes("\n".join(header).encode("ascii") + b"\n" + payload)


def write_sfm_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x} {y} {z}" for x, y, z in points]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def write_cameras(path: Path, tvecs: list[tuple[float, float, float]]) -> None:
    images = [
        {
            "image_id": index + 1,
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": list(tvec),
            "camera_id": 1,
            "name": f"frame_{index:03d}.jpg",
        }
        for index, tvec in enumerate(tvecs)
    ]
    payload = {"coordinate_system": "colmap_world", "cameras": [], "images": images}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def gaussian(
    x: float, y: float, z: float, opacity: float, scale: float = 0.01
) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "z": z,
        "opacity": logit(opacity),
        "scale_0": float(np.log(scale)),
        "scale_1": float(np.log(scale)),
        "scale_2": float(np.log(scale)),
    }


def build_scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    gaussian_path = tmp_path / "scene.ply"
    # SfM grid spacing is 0.1, so hug_radius and free_radius both calibrate to 0.1.
    write_sfm_ply(tmp_path / "points.ply", [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.1, 0.1, 0.0)])
    # Identity rotation, center at (3, 4, 0), distance 5 from the origin.
    write_cameras(tmp_path / "cameras.json", [(-3.0, -4.0, 0.0)])
    write_gaussian_ply(
        gaussian_path,
        [
            gaussian(0.001, 0.0, 0.0, 0.9),
            gaussian(0.0, 0.101, 0.0, 0.5),
            gaussian(0.1, 0.0, 0.001, 0.1),
            gaussian(0.0, 0.0, 5.0, 0.01, scale=0.001),
            gaussian(0.0, 0.0, -5.0, 0.02, scale=0.001),
            {**gaussian(0.0, 0.0, 6.0, 0.02, scale=0.002), "scale_0": float(np.log(0.05))},
        ],
    )
    return gaussian_path, tmp_path / "points.ply", tmp_path / "cameras.json"


def test_census_classifies_populations_and_free_space(tmp_path):
    gaussian_path, points_path, cameras_path = build_scene(tmp_path)

    report = floaters.analyze_floaters(
        gaussian_ply=gaussian_path, points=points_path, cameras=cameras_path
    )

    assert report["schema_version"] == 1
    assert report["gaussian_count"] == 6
    assert report["sfm_point_count"] == 4
    assert report["camera_count"] == 1
    assert report["thresholds"]["hug_radius"] == pytest.approx(0.1)
    assert report["thresholds"]["free_radius"] == pytest.approx(0.1)

    populations = report["populations"]
    haze, core, thick = populations["haze"], populations["core"], populations["thick"]
    assert haze["count"] == 3
    assert core["count"] == 1
    assert thick["count"] == 2
    assert haze["hugging_fraction"] == 0.0
    assert haze["free_space_count"] == 3
    assert core["hugging_fraction"] == 1.0
    assert core["free_space_count"] == 0
    assert thick["hugging_fraction"] == 1.0
    assert thick["free_space_count"] == 0

    assert haze["free_space_scale_median"] == pytest.approx(0.001, rel=1e-6)
    assert haze["free_space_scale_max"] == pytest.approx(0.05, rel=1e-6)
    assert haze["veil_count_gt_0.01"] == 1
    assert haze["veil_count_gt_0.03"] == 1


def test_export_json_world_transform_is_applied(tmp_path):
    # Translation of +2 on z only: the lone haze gaussian moves from the
    # origin onto the SfM grid, flipping it from free space to hugging.
    gaussian_path = tmp_path / "scene.ply"
    points_path = tmp_path / "points.ply"
    cameras_path = tmp_path / "cameras.json"
    write_gaussian_ply(gaussian_path, [gaussian(0.0, 0.0, 0.0, 0.01)])
    write_sfm_ply(points_path, [(0.0, 0.0, 2.0), (0.1, 0.0, 2.0)])
    write_cameras(cameras_path, [(-10.0, 0.0, 0.0)])
    export_payload = {
        "schema_version": 1,
        "world_from_normalized": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    (gaussian_path.parent / "export.json").write_text(
        json.dumps(export_payload, indent=2) + "\n", encoding="utf-8"
    )

    report = floaters.analyze_floaters(
        gaussian_ply=gaussian_path, points=points_path, cameras=cameras_path
    )

    assert report["inputs"]["world_transform"] == "world_from_normalized"
    haze = report["populations"]["haze"]
    assert haze["count"] == 1
    assert haze["hugging_fraction"] == 1.0
    assert haze["free_space_count"] == 0
