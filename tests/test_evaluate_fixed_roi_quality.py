from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fixed_roi_quality_is_deterministic_and_detects_layers(tmp_path):
    ply_path = tmp_path / "layers.ply"
    roi_path = tmp_path / "rois.json"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    points = []
    for x_index in range(30):
        for y_index in range(30):
            x = x_index / 30
            y = y_index / 30
            points.append((x, y, 0.0))
            points.append((x, y, 0.12))
    write_ascii_ply(ply_path, points)
    roi_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {
                    "sample_size": 2_000,
                    "ransac_iterations": 100,
                    "plane_distance": 0.01,
                    "min_points": 100,
                    "grid_size": 10,
                    "local_min_points": 5,
                    "layer_bin_width": 0.01,
                    "layer_min_peak_ratio": 0.1,
                    "layer_min_separation": 0.03,
                },
                "rois": [
                    {"name": "layers", "min": [-0.1, -0.1, -0.1], "max": [1.1, 1.1, 0.2]},
                    {"name": "empty", "min": [2, 2, 2], "max": [3, 3, 3]},
                ],
            }
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "scripts/evaluate_fixed_roi_quality.py",
        "--input",
        str(ply_path),
        "--rois",
        str(roi_path),
        "--output",
        str(first_output),
    ]
    root = Path(__file__).resolve().parents[1]
    subprocess.run(command, cwd=root, check=True)
    command[-1] = str(second_output)
    subprocess.run(command, cwd=root, check=True)

    assert first_output.read_bytes() == second_output.read_bytes()
    diagnostics = json.loads(first_output.read_text(encoding="utf-8"))
    layers = diagnostics["rois"][0]
    assert layers["status"] == "ok"
    assert layers["selected_points"] == 1_800
    assert layers["robust_plane"]["inlier_count"] == 900
    assert layers["parallel_layer_count"] == 2
    assert layers["coverage"] == 1.0
    assert diagnostics["rois"][1] == {
        "name": "empty",
        "status": "empty_roi",
        "selected_points": 0,
    }


def test_fixed_roi_quality_reports_insufficient_points(tmp_path):
    ply_path = tmp_path / "few.ply"
    roi_path = tmp_path / "rois.json"
    output_path = tmp_path / "diagnostics.json"
    write_ascii_ply(ply_path, [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])
    roi_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {"min_points": 3},
                "rois": [{"name": "few", "min": [-1, -1, -1], "max": [1, 1, 1]}],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_fixed_roi_quality.py",
            "--input",
            str(ply_path),
            "--rois",
            str(roi_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    diagnostics = json.loads(output_path.read_text(encoding="utf-8"))
    assert diagnostics["rois"] == [{"name": "few", "status": "insufficient_points", "selected_points": 2}]


def write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    path.write_text("\n".join(header + [f"{x} {y} {z}" for x, y, z in points]) + "\n", encoding="utf-8")
