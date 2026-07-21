from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_eth3d_scene import (  # noqa: E402
    build_evaluator_command,
    colmap_camera_center,
    ensure_empty_output_dir,
    index_images_by_basename,
    parse_evaluator_output,
)
from geometry_utils import (  # noqa: E402
    decompose_similarity_transform,
    estimate_similarity_transform,
    estimate_similarity_transform_ransac,
    transform_points,
)


def test_estimate_similarity_transform_recovers_known_transform() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    angle = np.deg2rad(32.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    scale = 2.4
    translation = np.array([3.0, -2.0, 0.75])
    target = scale * (source @ rotation.T) + translation

    transform = estimate_similarity_transform(source, target)
    actual_scale, actual_rotation, actual_translation = decompose_similarity_transform(transform)

    assert actual_scale == pytest.approx(scale)
    assert np.allclose(actual_rotation, rotation)
    assert np.allclose(actual_translation, translation)
    assert np.allclose(transform_points(source, transform), target)


def test_similarity_ransac_rejects_pose_outlier_deterministically() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
            [-0.5, 0.4, 1.2],
        ]
    )
    target = 1.7 * source + np.array([0.2, -0.3, 0.5])
    target[-1] += np.array([3.0, -2.0, 1.0])

    first = estimate_similarity_transform_ransac(
        source, target, threshold=1e-5, iterations=200, min_inliers=5, seed=42
    )
    second = estimate_similarity_transform_ransac(
        source, target, threshold=1e-5, iterations=200, min_inliers=5, seed=42
    )

    assert first.inliers.tolist() == [True, True, True, True, True, False]
    assert np.array_equal(first.inliers, second.inliers)
    assert np.allclose(first.transform, second.transform)
    assert np.allclose(transform_points(source[:5], first.transform), target[:5])


@pytest.mark.parametrize(
    "source,target,error",
    [
        (np.zeros((2, 3)), np.zeros((2, 3)), "At least three"),
        (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]]), "collinear"),
        (np.array([[0, 0, 0], [1, 0, 0], [0, 1, np.nan]]), np.eye(3), "finite"),
    ],
)
def test_similarity_transform_rejects_invalid_correspondences(
    source: np.ndarray, target: np.ndarray, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        estimate_similarity_transform(source, target)


def test_colmap_camera_center_uses_world_to_camera_pose() -> None:
    image = {
        "qvec": [1.0, 0.0, 0.0, 0.0],
        "tvec": [-1.0, -2.0, -3.0],
    }
    assert np.allclose(colmap_camera_center(image), [1.0, 2.0, 3.0])


def test_duplicate_image_basename_is_rejected() -> None:
    images = [{"name": "a/DSC.JPG"}, {"name": "b/DSC.JPG"}]
    with pytest.raises(ValueError, match="Duplicate reconstruction"):
        index_images_by_basename(images, "reconstruction")


def test_parse_evaluator_output_uses_labelled_rows() -> None:
    stdout = """
Loading scans...
Tolerances: 0.01 0.02 0.05
Completenesses: 0.2 0.4 0.6
progress 100%
Accuracies: 0.3 0.5 0.7
F1-scores: 0.24 0.444444 0.646154
"""
    rows = parse_evaluator_output(stdout)
    assert rows[1] == {
        "tolerance": 0.02,
        "completeness": 0.4,
        "accuracy": 0.5,
        "f1": 0.444444,
    }


def test_parse_evaluator_output_rejects_missing_or_mismatched_rows() -> None:
    with pytest.raises(ValueError, match="missing labelled rows"):
        parse_evaluator_output("Tolerances: 0.01\n")
    with pytest.raises(ValueError, match="inconsistent"):
        parse_evaluator_output(
            "Tolerances: 0.01 0.02\nCompletenesses: 0.1\nAccuracies: 0.2 0.3\nF1-scores: 0.1 0.2\n"
        )


def test_build_evaluator_command_uses_official_flags(tmp_path: Path) -> None:
    command = build_evaluator_command(
        evaluator_bin=tmp_path / "evaluator",
        reconstruction_ply=tmp_path / "aligned.ply",
        ground_truth_mlp=tmp_path / "scan_alignment.mlp",
        tolerances=[0.01, 0.02],
        official_dir=tmp_path / "official",
        write_visualizations=True,
    )
    assert command == [
        str(tmp_path / "evaluator"),
        "--reconstruction_ply_path",
        str(tmp_path / "aligned.ply"),
        "--ground_truth_mlp_path",
        str(tmp_path / "scan_alignment.mlp"),
        "--tolerances",
        "0.01,0.02",
        "--accuracy_cloud_output_path",
        str(tmp_path / "official/accuracy"),
        "--completeness_cloud_output_path",
        str(tmp_path / "official/completeness"),
    ]
    assert not any("icp" in value.lower() for value in command)


def test_nonempty_evaluation_output_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"status": "old"}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="absent or empty"):
        ensure_empty_output_dir(output)
