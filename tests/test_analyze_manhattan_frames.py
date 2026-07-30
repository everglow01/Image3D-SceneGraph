from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_manhattan_frames as manhattan  # noqa: E402


def plane(index: int, normal: list[float], ratio: float) -> dict[str, object]:
    return {
        "index": index,
        "normal": normal,
        "inlier_count": int(ratio * 1_000),
        "inlier_ratio": ratio,
        "area_estimate": ratio * 100,
    }


def test_cubic_room_recovers_unordered_axes_from_point_cloud(tmp_path):
    input_path = tmp_path / "room.ply"
    rng = np.random.default_rng(7)
    points = []
    for axis in range(3):
        for _ in range(600):
            point = rng.uniform(-2.0, 2.0, size=3)
            point[axis] = rng.normal(0.0, 0.002)
            points.append(tuple(point))
    points.extend(tuple(point) for point in rng.uniform(-2.0, 2.0, size=(120, 3)))
    write_ascii_ply(input_path, points)

    result = manhattan.analyze_manhattan_frames(
        input_path,
        sample_size=0,
        ransac_iterations=700,
        max_planes=6,
        plane_distance=0.015,
        min_inlier_ratio=0.08,
        seed=9,
    )

    assert result["status"] == "candidate"
    assert result["frame_evidence"] == "full"
    candidate = result["best_candidate"]
    assert candidate is not None
    recovered = np.abs(np.asarray(candidate["orthonormal_axes"]))
    expected = np.eye(3)
    assert all(np.max(recovered @ axis) > 0.999 for axis in expected)
    assert "ground_plane_index" not in result
    assert "up_axis" not in result


def test_duplicate_parallel_and_weak_planes_remain_ambiguous():
    result = manhattan.evaluate_planes(
        [
            plane(0, [1.0, 0.0, 0.0], 0.30),
            plane(1, [-0.999, 0.02, 0.0], 0.20),
            plane(2, [0.0, 1.0, 0.0], 0.15),
            plane(3, [0.0, 0.0, 1.0], 0.07),
        ]
    )

    assert result["status"] == "ambiguous"
    assert result["frame_evidence"] == "partial"
    assert len(result["normal_clusters"]) == 2
    assert result["normal_clusters"][0]["plane_indices"] == [0, 1]
    assert result["candidates"] == []
    assert result["ambiguity_reasons"] == ["fewer_than_three_reliable_directions"]


def test_non_orthogonal_and_multiple_frames_are_ambiguous():
    non_orthogonal = manhattan.evaluate_planes(
        [
            plane(0, [1.0, 0.0, 0.0], 0.30),
            plane(1, [0.7, 0.7, 0.0], 0.20),
            plane(2, [0.0, 0.0, 1.0], 0.15),
        ]
    )
    multiple = manhattan.evaluate_planes(
        [
            plane(0, [1.0, 0.0, 0.0], 0.40),
            plane(1, [0.0, 1.0, 0.0], 0.30),
            plane(2, [0.0, 0.0, 1.0], 0.20),
            plane(3, [0.0, 0.10, 0.995], 0.10),
        ],
        cluster_angle_degrees=5.0,
    )

    assert non_orthogonal["status"] == "ambiguous"
    assert non_orthogonal["candidates"] == []
    assert non_orthogonal["ambiguity_reasons"] == [
        "no_orthogonal_three_direction_candidate"
    ]
    assert multiple["status"] == "ambiguous"
    assert len(multiple["candidates"]) == 2
    assert multiple["best_candidate"] is None


def test_largest_plane_is_not_assigned_ground():
    result = manhattan.evaluate_planes(
        [
            plane(0, [1.0, 0.0, 0.0], 0.50),
            plane(1, [0.0, 1.0, 0.0], 0.20),
            plane(2, [0.0, 0.0, 1.0], 0.10),
        ]
    )

    assert result["status"] == "candidate"
    assert result["best_candidate"]["cluster_ids"] == [0, 1, 2]
    serialized = json.dumps(result)
    assert "ground" not in serialized
    assert "up_axis" not in serialized
    assert "target_axis" not in serialized


def test_cli_check_reproduces_identical_report(tmp_path):
    input_path = tmp_path / "parallel.ply"
    output_path = tmp_path / "report.json"
    points = [(x / 10, y / 10, 0.0) for x in range(10) for y in range(10)]
    write_ascii_ply(input_path, points)
    command = [
        sys.executable,
        "scripts/analyze_manhattan_frames.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--sample-size",
        "0",
        "--ransac-iterations",
        "100",
        "--plane-distance",
        "0.01",
    ]

    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    subprocess.run(
        [*command, "--check"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "ambiguous"
    assert report["protocol"]["alignment_changed"] is False


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
    body = [f"{x} {y} {z}" for x, y, z in points]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
