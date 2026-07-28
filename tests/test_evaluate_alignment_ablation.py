from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_alignment_ablation as ablation  # noqa: E402
from analyze_pointcloud import read_ply_points_and_colors  # noqa: E402


def test_tilted_manhattan_frame_maps_up_and_planes_to_cardinal_axes(tmp_path):
    inputs = write_inputs(tmp_path)
    output_path = tmp_path / "manhattan.ply"

    result = ablation.evaluate_alignment_ablation(
        *inputs,
        output_point_cloud_path=output_path,
    )

    assert result["selection"]["selected_strategy"] == "manhattan"
    assert result["selection"]["fallback"] is False
    assert result["selection"]["horizontal_axis_semantics_assigned"] is False
    assert result["metrics"]["manhattan"]["selected_up"]["vector"] == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-12
    )
    assert result["metrics"]["manhattan"]["max_angular_residual_degrees"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert result["metrics"]["single_plane"]["max_angular_residual_degrees"] > 5.0

    points, colors = read_ply_points_and_colors(output_path)
    source_points, source_colors = read_ply_points_and_colors(inputs[0])
    transform = np.asarray(result["selection"]["transform"])
    expected = ablation.apply_transform(source_points, transform)
    assert points == pytest.approx(expected, abs=1e-6)
    assert np.array_equal(colors, source_colors)
    assert result["output_point_cloud"]["sha256"] == sha256(output_path)
    serialized = json.dumps(result)
    assert '"ground_plane"' not in serialized
    assert '"floor"' not in serialized
    assert result["protocol"]["metric_scale_recovered"] is False
    assert result["protocol"]["local_geometry_changed"] is False


def test_auto_falls_back_without_writing_candidate_when_manhattan_is_ambiguous(tmp_path):
    inputs = list(write_inputs(tmp_path))
    manhattan = json.loads(inputs[2].read_text(encoding="utf-8"))
    manhattan.update(status="ambiguous", best_candidate=None, ambiguity_reasons=["multiple_candidates"])
    inputs[2].write_text(json.dumps(manhattan), encoding="utf-8")
    output_path = tmp_path / "must-not-exist.ply"

    result = ablation.evaluate_alignment_ablation(
        *inputs,
        output_point_cloud_path=output_path,
    )

    assert result["selection"] == {
        "requested_strategy": "auto",
        "selected_strategy": "single_plane",
        "fallback": True,
        "fallback_reasons": ["manhattan_frame_not_unambiguous"],
    }
    assert result["output_point_cloud"] is None
    assert result["metrics"]["manhattan"] is None
    assert not output_path.exists()


def test_auto_falls_back_when_gravity_sign_is_missing(tmp_path):
    inputs = list(write_inputs(tmp_path))
    gravity = json.loads(inputs[3].read_text(encoding="utf-8"))
    gravity["selection"].update(up_sign_status="ambiguous", up_sign=None, up_vector=None)
    inputs[3].write_text(json.dumps(gravity), encoding="utf-8")

    result = ablation.evaluate_alignment_ablation(
        *inputs,
        output_point_cloud_path=tmp_path / "must-not-exist.ply",
    )

    assert result["selection"]["selected_strategy"] == "single_plane"
    assert result["selection"]["fallback_reasons"] == ["gravity_sign_not_selected"]
    assert result["metrics"]["single_plane"] is None


def test_mismatched_hashes_axes_and_malformed_transform_are_rejected(tmp_path):
    inputs = list(write_inputs(tmp_path))
    gravity = json.loads(inputs[3].read_text(encoding="utf-8"))
    gravity["inputs"]["point_cloud"]["sha256"] = "0" * 64
    inputs[3].write_text(json.dumps(gravity), encoding="utf-8")
    with pytest.raises(ablation.AlignmentAblationError, match="does not match the gravity report"):
        ablation.evaluate_alignment_ablation(
            *inputs, output_point_cloud_path=tmp_path / "candidate.ply"
        )

    inputs = list(write_inputs(tmp_path, stem="axes"))
    gravity = json.loads(inputs[3].read_text(encoding="utf-8"))
    gravity["manhattan_candidate"]["axes"][0][0] += 0.1
    inputs[3].write_text(json.dumps(gravity), encoding="utf-8")
    with pytest.raises(ablation.AlignmentAblationError, match="different axes"):
        ablation.evaluate_alignment_ablation(
            *inputs, output_point_cloud_path=tmp_path / "candidate-axes.ply"
        )

    inputs = list(write_inputs(tmp_path, stem="transform"))
    single = json.loads(inputs[1].read_text(encoding="utf-8"))
    single["transform"][0][0] = 2.0
    inputs[1].write_text(json.dumps(single), encoding="utf-8")
    with pytest.raises(ablation.AlignmentAblationError, match="not orthonormal"):
        ablation.evaluate_alignment_ablation(
            *inputs, output_point_cloud_path=tmp_path / "candidate-transform.ply"
        )


def test_cli_check_reproduces_identical_report_and_point_cloud(tmp_path):
    inputs = write_inputs(tmp_path)
    output_cloud = tmp_path / "candidate.ply"
    output_report = tmp_path / "report.json"
    command = [
        sys.executable,
        "scripts/evaluate_alignment_ablation.py",
        "--point-cloud",
        str(inputs[0]),
        "--single-alignment",
        str(inputs[1]),
        "--manhattan-report",
        str(inputs[2]),
        "--gravity-report",
        str(inputs[3]),
        "--output-point-cloud",
        str(output_cloud),
        "--output",
        str(output_report),
    ]

    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    cloud_hash = sha256(output_cloud)
    report_text = output_report.read_text(encoding="utf-8")
    subprocess.run([*command, "--check"], cwd=Path(__file__).resolve().parents[1], check=True)

    assert sha256(output_cloud) == cloud_hash
    assert output_report.read_text(encoding="utf-8") == report_text


def write_inputs(tmp_path: Path, stem: str = "base") -> tuple[Path, Path, Path, Path]:
    angle = np.deg2rad(20.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )
    cloud_path = tmp_path / f"{stem}.ply"
    points = [
        (-1.0, -1.0, -1.0, 255, 0, 0),
        (1.0, -1.0, -1.0, 0, 255, 0),
        (-1.0, 1.0, 1.0, 0, 0, 255),
        (1.0, 1.0, 1.0, 255, 255, 0),
    ]
    write_ascii_ply(cloud_path, points)
    cloud_hash = sha256(cloud_path)
    planes = [
        plane(0, axes[2], axes[2] * -1.0, 0.30),
        plane(1, axes[0], axes[0] * -1.0, 0.20),
        plane(2, axes[1], axes[1] * -1.0, 0.10),
    ]
    candidate = {
        "cluster_ids": [10, 11, 12],
        "plane_indices": [0, 1, 2],
        "orthonormal_axes": axes.tolist(),
        "pairwise": [],
        "mean_orthogonality_residual_degrees": 0.0,
        "max_orthogonality_residual_degrees": 0.0,
        "support_score": 0.6,
    }
    manhattan = {
        "status": "candidate",
        "ambiguity_reasons": [],
        "input": {"sha256": cloud_hash},
        "reliable_plane_indices": [0, 1, 2],
        "plane_diagnostics": {"dominant_planes": planes},
        "normal_clusters": [
            {"id": 10, "plane_indices": [0]},
            {"id": 11, "plane_indices": [1]},
            {"id": 12, "plane_indices": [2]},
        ],
        "best_candidate": candidate,
    }
    gravity = {
        "status": "selected",
        "inputs": {"point_cloud": {"sha256": cloud_hash}},
        "manhattan_candidate": {"cluster_ids": [10, 11, 12], "axes": axes.tolist()},
        "selection": {
            "status": "selected",
            "winner_axis_index": 2,
            "winner_cluster_id": 12,
            "axis": axes[2].tolist(),
            "up_sign_status": "selected",
            "up_sign": 1,
            "up_vector": axes[2].tolist(),
        },
    }
    single = {
        "num_points": len(points),
        "colors_preserved": True,
        "transform": np.eye(4).tolist(),
    }
    single_path = tmp_path / f"{stem}-single.json"
    manhattan_path = tmp_path / f"{stem}-manhattan.json"
    gravity_path = tmp_path / f"{stem}-gravity.json"
    single_path.write_text(json.dumps(single), encoding="utf-8")
    manhattan_path.write_text(json.dumps(manhattan), encoding="utf-8")
    gravity_path.write_text(json.dumps(gravity), encoding="utf-8")
    return cloud_path, single_path, manhattan_path, gravity_path


def plane(index: int, normal: np.ndarray, centroid: np.ndarray, ratio: float) -> dict:
    return {
        "index": index,
        "normal": normal.tolist(),
        "centroid": centroid.tolist(),
        "inlier_ratio": ratio,
        "area_estimate": ratio * 10,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
