from pathlib import Path
import runpy

import numpy as np
import pytest

from image3d_scenegraph.geometry.grouping import ColmapImage


def test_overlap_camera_fit_preserves_rotation_and_checks_unfitted_orientation(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "analyze_aliked_model_overlap.py"))
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 2, 3]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 2 * source @ rotation.T + [3, 4, 5]
    reference, candidate = {}, {}
    for i, (left, right) in enumerate(zip(target, source)):
        name = f"{i}.jpg"
        reference[name] = ColmapImage(i, np.array([1, 0, 0, 0]), -left, 1, name, [])
        candidate[name] = ColmapImage(i, np.array([2 ** -0.5, 0, 0, 2 ** -0.5]), -rotation @ right, 1, name, [])
    transform, _, result = audit["compare_cameras"](reference, candidate)
    assert transform[:3, :3] == pytest.approx(2 * rotation)
    assert result["center_residual_to_reference_common_radius"]["max"] < 1e-12
    assert result["orientation_residual_degrees"]["max"] < 1e-5
    assert max(x["excluded_camera_residual_to_radius"] for x in result["leave_one_camera_out"]) < 1e-12
    orientation_fit = audit["orientation_similarity"](target, source, [rotation] * len(source))
    assert orientation_fit == pytest.approx(transform)
    observations = {1: {i: i for i in range(len(source))}, 2: {i: i for i in range(len(source))}}
    cross = audit["cross_check_alignment"](
        [reference, candidate], [observations, observations],
        [dict(enumerate(target)), dict(enumerate(source))], transform, 1,
    )
    for fit in cross.values():
        assert fit["status"] == "evaluated"
        assert fit["camera_center_residual_to_radius"]["max"] < 1e-12
        assert fit["camera_orientation_residual_degrees"]["max"] < 1e-5
        assert fit["shared_point_residual_to_radius"]["max"] < 1e-12
    with pytest.raises(ValueError, match="not positive"):
        audit["orientation_similarity"](-source, source, [np.eye(3)] * len(source))
    candidate["0.jpg"] = ColmapImage(0, np.array([1, 0, 0, 0]), -source[0], 1, "0.jpg", [])
    _, _, result = audit["compare_cameras"](reference, candidate)
    assert result["orientation_residual_degrees"]["max"] == pytest.approx(90)
    assert result["center_residual_to_reference_common_radius"]["max"] < 1e-12


def test_overlap_points_use_feature_indices_and_independent_images(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "analyze_aliked_model_overlap.py"))
    reference = {1: {0: 100, 1: 100}, 2: {0: 100, 2: 200}, 3: {3: 300}}
    candidate = {1: {0: 9, 1: 9}, 2: {0: 9, 2: 8}, 3: {3: 8}}
    result = audit["compare_points"](
        reference, candidate, {100: [0, 0, 5], 200: [1, 0, 5], 300: [2, 0, 5]},
        {9: [0, 0, 5], 8: [1, 0, 5]}, np.eye(4), 1,
    )
    assert result["shared_feature_observation_count"] == 5
    assert result["candidate_points_with_multiple_reference_partners"] == 1
    assert result["groups"]["all_shared_point_pairs"]["point_pair_count"] == 3
    unique = result["groups"]["mutually_unique_seen_in_two_common_images"]
    assert unique["point_pair_count"] == 1
    assert unique["common_image_support"]["max"] == 2
    assert unique["camera_fitted_residual_to_reference_common_radius"]["max"] == 0
