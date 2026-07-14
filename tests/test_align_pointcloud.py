from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import align_pointcloud as align_module  # noqa: E402


def test_align_pointcloud_rotates_dominant_plane_to_z_axis(tmp_path):
    input_path = tmp_path / "tilted.ply"
    output_path = tmp_path / "aligned.ply"
    alignment_path = tmp_path / "alignment.json"
    aligned_diagnostics_path = tmp_path / "aligned_diagnostics.json"

    points = []
    for x_index in range(18):
        for y_index in range(12):
            x = x_index / 10
            y = y_index / 10
            z = 0.35 * x + 0.2 * y + 1.0
            points.append((x, y, z, 10, 120, 230))
    points.extend([(0.0, 0.0, 3.0, 255, 0, 0), (2.0, 2.0, 3.5, 255, 0, 0)])
    write_ascii_ply(input_path, points)

    subprocess.run(
        [
            sys.executable,
            "scripts/align_pointcloud.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--diagnostics-output",
            str(alignment_path),
            "--plane-distance",
            "0.03",
            "--ransac-iterations",
            "150",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pointcloud.py",
            "--input",
            str(output_path),
            "--output",
            str(aligned_diagnostics_path),
            "--plane-distance",
            "0.03",
            "--ransac-iterations",
            "150",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(aligned_diagnostics_path.read_text(encoding="utf-8"))
    plane = diagnostics["dominant_planes"][0]

    assert alignment["status"] == "aligned"
    assert alignment["colors_preserved"] is True
    assert plane["inlier_ratio"] > 0.9
    assert abs(plane["normal"][2]) > 0.99
    assert abs(plane["centroid"][2]) < 0.05


def test_align_pointcloud_selects_strongest_candidate_by_default(tmp_path, monkeypatch):
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "aligned.ply"
    write_ascii_ply(
        input_path,
        [
            (0.0, 0.0, 0.0, 255, 0, 0),
            (1.0, 0.0, 0.0, 0, 255, 0),
            (0.0, 1.0, 0.0, 0, 0, 255),
        ],
    )
    candidates = [
        {
            "index": 0,
            "normal": [1.0, 0.0, 0.0],
            "offset": 0.0,
            "inlier_count": 21,
            "inlier_ratio": 0.07,
            "centroid": [0.0, 0.0, 0.0],
        },
        {
            "index": 1,
            "normal": [0.0, 1.0, 0.0],
            "offset": 0.0,
            "inlier_count": 36,
            "inlier_ratio": 0.12,
            "centroid": [0.0, 0.0, 0.0],
        },
    ]
    monkeypatch.setattr(
        align_module,
        "analyze_pointcloud",
        lambda *args, **kwargs: {"dominant_planes": candidates},
    )

    result = align_module.align_pointcloud(
        input_path=input_path,
        output_path=output_path,
        min_plane_inlier_ratio=0.08,
    )

    assert result["source_plane"]["index"] == 1
    assert result["source_plane"]["inlier_ratio"] == 0.12
    assert output_path.exists()


def write_ascii_ply(path: Path, points: list[tuple[float, float, float, int, int, int]]) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body = [f"{x} {y} {z} {r} {g} {b}" for x, y, z, r, g, b in points]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
