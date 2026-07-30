from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_gravity_axis as gravity  # noqa: E402


AXES = np.eye(3)


def test_synthetic_room_selects_vertical_axis_and_sign():
    rng = np.random.default_rng(4)
    points = rng.uniform([-5.0, -4.0, -1.0], [5.0, 4.0, 1.0], size=(4_000, 3))
    camera_centers = rng.uniform([-3.0, -2.0, -0.3], [3.0, 2.0, 0.3], size=(30, 3))
    camera_ups = np.tile([0.0, 0.0, 1.0], (30, 1))
    camera_ups += rng.normal(0.0, 0.01, size=camera_ups.shape)
    camera_ups /= np.linalg.norm(camera_ups, axis=1, keepdims=True)
    planes = [
        plane(0, [0, 0, 1], [0, 0, -1], 0.30),
        plane(1, [0, 0, 1], [0, 0, 1], 0.25),
        plane(2, [1, 0, 0], [-5, 0, 0], 0.15),
    ]

    result = gravity.evaluate_gravity_axes(
        AXES,
        [10, 11, 12],
        points,
        planes,
        {0, 1, 2},
        camera_centers,
        camera_ups,
        np.empty((0, 3)),
    )

    assert result["selection"]["status"] == "selected"
    assert result["selection"]["winner_axis_index"] == 2
    assert result["selection"]["winner_cluster_id"] == 12
    assert result["selection"]["up_sign"] == 1
    assert result["selection"]["up_vector"] == [0.0, 0.0, 1.0]
    assert set(result["selection"]["available_sources"]) == {
        "camera_orientation",
        "camera_center_span",
        "point_span",
        "plane_ordering",
    }
    for source in result["selection"]["available_sources"]:
        assert len(result["evidence"][source]["scores"]) == 3


def test_isotropic_conflicting_evidence_remains_ambiguous():
    rng = np.random.default_rng(6)
    points = rng.normal(size=(5_000, 3))
    camera_centers = rng.normal(size=(60, 3))
    camera_ups = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (20, 1)),
            np.tile([0.0, 1.0, 0.0], (20, 1)),
            np.tile([0.0, 0.0, 1.0], (20, 1)),
        ]
    )

    result = gravity.evaluate_gravity_axes(
        AXES,
        [0, 1, 2],
        points,
        [],
        set(),
        camera_centers,
        camera_ups,
        np.empty((0, 3)),
    )

    assert result["selection"]["status"] == "ambiguous"
    assert "no_reliable_directional_evidence" in result["selection"]["ambiguity_reasons"]
    assert "winner_margin_below_threshold" in result["selection"]["ambiguity_reasons"]
    assert result["selection"]["axis"] is None


def test_imu_metadata_rejects_bad_records_and_selects_from_valid_ones(tmp_path):
    path = tmp_path / "imu.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "opencv_camera",
                "records": {
                    "a.jpg": {"gravity": [0, 0, -1]},
                    "b.jpg": {"up": [0, 0, 1]},
                    "c.jpg": {"up": [0, 0, 1]},
                    "missing.jpg": {"up": [0, 0, 1]},
                    "bad.jpg": {"up": [0, 0, 0]},
                },
            }
        ),
        encoding="utf-8",
    )
    cameras = {
        name: {"rotation": np.eye(3)}
        for name in ("a.jpg", "b.jpg", "c.jpg", "bad.jpg", "no-imu.jpg")
    }

    loaded = gravity.load_imu_up_vectors(path, cameras)
    evidence = gravity.orientation_evidence(
        loaded["world_up_vectors"], AXES, min_records=3, min_winner_score=0.7
    )

    assert loaded["status"] == "available"
    assert loaded["valid_count"] == 3
    assert loaded["missing_count"] == 2
    assert loaded["rejected_count"] == 2
    assert evidence["winner_axis_index"] == 2
    assert evidence["mean_signed_dot"][2] == 1.0


def test_missing_and_malformed_metadata_have_nonfatal_fallback(tmp_path):
    missing = gravity.load_imu_up_vectors(None, {})
    malformed_path = tmp_path / "imu.json"
    malformed_path.write_text("not-json", encoding="utf-8")
    malformed = gravity.load_imu_up_vectors(malformed_path, {})
    cameras = gravity.load_colmap_cameras(None)

    assert missing["status"] == "unavailable"
    assert missing["reason"] == "imu_file_not_supplied"
    assert malformed["status"] == "invalid"
    assert malformed["reason"].startswith("imu_file_unreadable")
    assert cameras["status"] == "unavailable"
    assert cameras["centers"].shape == (0, 3)


def test_ambiguous_manhattan_report_does_not_force_axis(tmp_path):
    cloud_path = tmp_path / "cloud.ply"
    write_ascii_ply(cloud_path, [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    report_path = tmp_path / "manhattan.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "ambiguous",
                "best_candidate": None,
                "input": {"sha256": sha256(cloud_path)},
            }
        ),
        encoding="utf-8",
    )

    result = gravity.analyze_gravity_axis(report_path, cloud_path, sample_size=0)

    assert result["status"] == "ambiguous"
    assert result["ambiguity_reasons"] == ["manhattan_frame_not_unambiguous"]
    assert result["selection"] is None
    serialized = json.dumps(result)
    assert "ground_plane" not in serialized
    assert '"transform"' not in serialized


def test_cli_check_reproduces_identical_report(tmp_path):
    cloud_path = tmp_path / "room.ply"
    points = [
        (x, y, z)
        for x in (-3.0, 3.0)
        for y in (-2.0, 2.0)
        for z in (-0.5, 0.5)
    ]
    write_ascii_ply(cloud_path, points)
    report_path = tmp_path / "manhattan.json"
    report_path.write_text(
        json.dumps(manhattan_report(cloud_path)),
        encoding="utf-8",
    )
    output_path = tmp_path / "gravity.json"
    command = [
        sys.executable,
        "scripts/analyze_gravity_axis.py",
        "--manhattan-report",
        str(report_path),
        "--point-cloud",
        str(cloud_path),
        "--output",
        str(output_path),
        "--sample-size",
        "0",
    ]

    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    subprocess.run([*command, "--check"], cwd=Path(__file__).resolve().parents[1], check=True)

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "ambiguous"
    assert result["protocol"]["alignment_changed"] is False
    assert result["metadata"]["imu"]["status"] == "unavailable"


def plane(index, normal, centroid, ratio):
    return {
        "index": index,
        "normal": normal,
        "centroid": centroid,
        "inlier_ratio": ratio,
    }


def manhattan_report(cloud_path):
    planes = [
        plane(0, [0, 0, 1], [0, 0, -0.5], 0.3),
        plane(1, [1, 0, 0], [-3, 0, 0], 0.2),
        plane(2, [0, 1, 0], [0, -2, 0], 0.1),
    ]
    return {
        "status": "candidate",
        "input": {"sha256": sha256(cloud_path)},
        "best_candidate": {
            "cluster_ids": [0, 1, 2],
            "orthonormal_axes": AXES.tolist(),
        },
        "reliable_plane_indices": [0, 1, 2],
        "plane_diagnostics": {"dominant_planes": planes},
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ascii_ply(path, points):
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    path.write_text(
        "\n".join(header + [f"{x} {y} {z}" for x, y, z in points]) + "\n",
        encoding="utf-8",
    )
