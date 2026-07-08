from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
