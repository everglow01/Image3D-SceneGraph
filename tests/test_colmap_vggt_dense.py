from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_colmap_vggt_dense import (  # noqa: E402
    ColmapImage,
    DepthScaleEstimate,
    FusionFrame,
    FusionCamera,
    build_fusion_camera,
    build_covisibility_graph,
    derive_consistency_relative_threshold,
    estimate_depth_scale,
    map_original_pixel_to_vggt,
    undistort_radial_coordinates,
    unproject_depth_with_colmap_pose,
    valid_depth_canvas_mask,
    validate_cross_view_consistency,
)


def test_build_fusion_camera_maps_colmap_intrinsics_into_vggt_canvas():
    camera = build_fusion_camera(
        colmap_camera={
            "camera_id": 1,
            "model": "SIMPLE_RADIAL",
            "width": 1000,
            "height": 500,
            "params": [1000.0, 500.0, 250.0, 0.02],
        },
        original_size=(1000, 500),
        image_shape=(518, 518),
    )

    assert camera.model == "SIMPLE_RADIAL"
    assert camera.radial_distortion == (0.02,)
    assert np.allclose(camera.intrinsic, [[518.0, 0.0, 259.0], [0.0, 504.0, 259.0], [0.0, 0.0, 1.0]])

    u, v = map_original_pixel_to_vggt(725.0, 375.0, (1000, 500), (518, 518))
    assert np.isclose((u - camera.intrinsic[0, 2]) / camera.intrinsic[0, 0], 0.225)
    assert np.isclose((v - camera.intrinsic[1, 2]) / camera.intrinsic[1, 1], 0.125)


def test_undistort_radial_coordinates_inverts_colmap_radial_model():
    undistorted_x = np.array([0.05, 0.2, -0.35], dtype=np.float32)
    undistorted_y = np.array([0.1, -0.15, 0.25], dtype=np.float32)
    coefficients = (0.08, -0.01)
    radius_squared = undistorted_x * undistorted_x + undistorted_y * undistorted_y
    radial = 1 + coefficients[0] * radius_squared + coefficients[1] * radius_squared * radius_squared

    recovered_x, recovered_y = undistort_radial_coordinates(
        undistorted_x * radial,
        undistorted_y * radial,
        coefficients,
    )

    assert np.allclose(recovered_x, undistorted_x, atol=1e-6)
    assert np.allclose(recovered_y, undistorted_y, atol=1e-6)


def test_unproject_depth_uses_colmap_camera_rays_and_pose():
    points = unproject_depth_with_colmap_pose(
        depth=np.full((2, 2), 2.0, dtype=np.float32),
        camera=FusionCamera(
            model="PINHOLE",
            intrinsic=np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            radial_distortion=(),
        ),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
    )

    assert np.allclose(points[1, 1], [1.0, 0.5, 2.0])


def test_estimate_depth_scale_reports_sparse_observation_diagnostics():
    estimate = estimate_depth_scale(
        colmap_image=ColmapImage(
            image_id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros(3),
            camera_id=1,
            name="frame.jpg",
            observations=[(7.0, 7.0, 1)],
        ),
        points3d={1: np.array([0.0, 0.0, 10.0])},
        depth=np.full((14, 14), 2.0, dtype=np.float32),
        image_shape=(14, 14),
        original_size=(14, 14),
        min_observations=1,
    )

    assert estimate is not None
    assert estimate.observation_count == 1
    assert np.isclose(estimate.scale, 5.0)
    assert np.isclose(estimate.log_mad, 0.0)


def test_vggt_canvas_mask_excludes_white_padding_pixels():
    mask = valid_depth_canvas_mask((1000, 500), (518, 518))

    assert mask.shape == (518, 518)
    assert not mask[0, 259]
    assert mask[133, 0]
    assert mask[384, 517]
    assert not mask[385, 259]


def test_covisibility_graph_uses_shared_sparse_tracks():
    frames = [
        make_frame(1, [(0.0, 0.0, 10), (1.0, 1.0, 11)]),
        make_frame(2, [(0.0, 0.0, 10), (1.0, 1.0, 11)]),
        make_frame(3, [(0.0, 0.0, 99)]),
    ]

    graph = build_covisibility_graph(frames, max_neighbors=2, min_shared_points=2)

    assert [(edge.target_image_id, edge.shared_points) for edge in graph[1]] == [(2, 2)]
    assert [(edge.target_image_id, edge.shared_points) for edge in graph[2]] == [(1, 2)]
    assert graph[3] == []


def test_cross_view_validation_rejects_conflicts_but_keeps_occlusions():
    neighbor = make_frame(2, [])
    validation = validate_cross_view_consistency(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        neighbors=[neighbor],
        confidence_threshold=0.5,
        relative_threshold=0.1,
    )

    assert validation.accepted.tolist() == [True, True, False]
    assert validation.support_counts.tolist() == [1, 0, 0]
    assert validation.visible_counts.tolist() == [1, 0, 1]
    assert validation.occluded_counts.tolist() == [0, 1, 0]


def test_consistency_threshold_uses_robust_scale_dispersion_with_bounds():
    threshold = derive_consistency_relative_threshold(
        [DepthScaleEstimate(1.0, 100, 0.006), DepthScaleEstimate(1.0, 100, 0.006)],
        min_threshold=0.02,
        max_threshold=0.08,
    )

    assert np.isclose(threshold, 0.02)


def make_frame(image_id: int, observations: list[tuple[float, float, int]]) -> FusionFrame:
    return FusionFrame(
        image_path=Path(f"frame_{image_id}.jpg"),
        colmap_image=ColmapImage(
            image_id=image_id,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros(3),
            camera_id=1,
            name=f"frame_{image_id}.jpg",
            observations=observations,
        ),
        camera=FusionCamera(
            model="PINHOLE",
            intrinsic=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            radial_distortion=(),
        ),
        depth=np.full((14, 14), 2.0, dtype=np.float32),
        confidence=np.ones((14, 14), dtype=np.float32),
        colors=np.zeros((14, 14, 3), dtype=np.uint8),
        scale=1.0,
        image_shape=(14, 14),
        original_size=(14, 14),
    )
