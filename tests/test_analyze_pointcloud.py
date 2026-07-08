from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_analyze_pointcloud_detects_dominant_plane(tmp_path):
    ply_path = tmp_path / "plane.ply"
    output_path = tmp_path / "diagnostics.json"
    points = []
    for x_index in range(20):
        for y_index in range(10):
            x = x_index / 10
            y = y_index / 10
            z = 0.01 if (x_index + y_index) % 2 else 0.0
            points.append((x, y, z, 200, 180, 120))
    points.extend([(0.0, 0.0, 1.0, 255, 0, 0), (1.0, 1.0, 1.5, 255, 0, 0)])
    write_ascii_ply(ply_path, points)

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pointcloud.py",
            "--input",
            str(ply_path),
            "--output",
            str(output_path),
            "--sample-size",
            "1000",
            "--ransac-iterations",
            "120",
            "--plane-distance",
            "0.02",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    diagnostics = json.loads(output_path.read_text(encoding="utf-8"))

    assert diagnostics["num_points"] == len(points)
    assert diagnostics["finite_points"] == len(points)
    assert diagnostics["dominant_planes"]
    plane = diagnostics["dominant_planes"][0]
    assert plane["inlier_count"] >= 190
    assert plane["inlier_ratio"] > 0.9
    assert abs(plane["normal"][2]) > 0.99
    assert "no_dominant_plane" not in diagnostics["quality_flags"]


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
