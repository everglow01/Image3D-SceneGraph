from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_colmap_vggt_dense import (  # noqa: E402
    ColmapImage,
    CovisibilityEdge,
    DepthScaleEstimate,
    FusionFrame,
    FusionCamera,
    apply_point_budget,
    apply_support_policy,
    build_covisibility_graph,
    build_fusion_camera,
    build_vggt_groups,
    compute_confidence_thresholds,
    compute_frame_confidence_thresholds,
    derive_consistency_relative_threshold,
    derive_tsdf_parameters,
    estimate_depth_scale,
    filter_points_by_cross_view_consistency,
    factorial_arm_name,
    fuse_frames_tsdf,
    map_original_pixel_to_vggt,
    optimize_depth_scale_graph,
    order_images_by_covisibility,
    undistort_radial_coordinates,
    undistort_to_pinhole,
    unproject_depth_with_colmap_pose,
    valid_depth_canvas_mask,
    validate_cross_view_consistency,
    validate_tsdf_output,
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

    graph = build_covisibility_graph(
        [frame.colmap_image for frame in frames], max_neighbors=2, min_shared_points=2
    )

    assert [(edge.target_image_id, edge.shared_points) for edge in graph[1]] == [(2, 2)]
    assert [(edge.target_image_id, edge.shared_points) for edge in graph[2]] == [(1, 2)]
    assert graph[3] == []


def test_cross_view_validation_rejects_conflicts_but_keeps_occlusions():
    neighbor = make_frame(2, [])
    validation = validate_cross_view_consistency(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        neighbors=[neighbor],
        confidence_thresholds={neighbor.colmap_image.image_id: 0.5},
        relative_threshold=0.1,
    )

    assert validation.accepted.tolist() == [True, True, False]
    assert validation.support_counts.tolist() == [1, 0, 0]
    assert validation.visible_counts.tolist() == [1, 0, 1]
    assert validation.occluded_counts.tolist() == [0, 1, 0]



def test_adaptive_support_policy_requires_two_when_two_views_are_visible():
    support_counts = np.array([0, 1, 1, 2, 2], dtype=np.int16)
    visible_counts = np.array([0, 1, 2, 2, 3], dtype=np.int16)

    any_support = apply_support_policy(
        support_counts,
        visible_counts,
        policy="any_support",
    )
    adaptive = apply_support_policy(
        support_counts,
        visible_counts,
        policy="adaptive_two",
    )

    assert any_support.tolist() == [True, True, True, True, True]
    assert adaptive.tolist() == [True, True, False, True, True]


def test_adaptive_support_policy_does_not_count_occluded_or_low_confidence_views():
    supporting = make_frame(2, [])
    occluding = make_frame(3, [])
    low_confidence = make_frame(4, [])
    occluding = FusionFrame(
        image_path=occluding.image_path,
        colmap_image=occluding.colmap_image,
        camera=occluding.camera,
        depth=np.full((14, 14), 1.0, dtype=np.float32),
        confidence=occluding.confidence,
        colors=occluding.colors,
        scale=occluding.scale,
        image_shape=occluding.image_shape,
        original_size=occluding.original_size,
    )
    low_confidence = FusionFrame(
        image_path=low_confidence.image_path,
        colmap_image=low_confidence.colmap_image,
        camera=low_confidence.camera,
        depth=low_confidence.depth,
        confidence=np.zeros((14, 14), dtype=np.float32),
        colors=low_confidence.colors,
        scale=low_confidence.scale,
        image_shape=low_confidence.image_shape,
        original_size=low_confidence.original_size,
    )

    validation = validate_cross_view_consistency(
        np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        neighbors=[supporting, occluding, low_confidence],
        confidence_thresholds={2: 0.5, 3: 0.5, 4: 0.5},
        relative_threshold=0.1,
        support_policy="adaptive_two",
    )

    assert validation.support_counts.tolist() == [1]
    assert validation.visible_counts.tolist() == [1]
    assert validation.occluded_counts.tolist() == [1]
    assert validation.accepted.tolist() == [True]


def test_adaptive_support_policy_rejects_single_support_with_visible_conflict():
    supporting = make_frame(2, [])
    conflicting = make_frame(3, [])
    conflicting = FusionFrame(
        image_path=conflicting.image_path,
        colmap_image=conflicting.colmap_image,
        camera=conflicting.camera,
        depth=np.full((14, 14), 3.0, dtype=np.float32),
        confidence=conflicting.confidence,
        colors=conflicting.colors,
        scale=conflicting.scale,
        image_shape=conflicting.image_shape,
        original_size=conflicting.original_size,
    )
    source_points = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    thresholds = {2: 0.5, 3: 0.5}

    baseline = validate_cross_view_consistency(
        source_points,
        neighbors=[supporting, conflicting],
        confidence_thresholds=thresholds,
        relative_threshold=0.1,
        support_policy="any_support",
    )
    adaptive = validate_cross_view_consistency(
        source_points,
        neighbors=[supporting, conflicting],
        confidence_thresholds=thresholds,
        relative_threshold=0.1,
        support_policy="adaptive_two",
    )

    assert baseline.support_counts.tolist() == [1]
    assert baseline.visible_counts.tolist() == [2]
    assert baseline.accepted.tolist() == [True]
    assert adaptive.accepted.tolist() == [False]


def test_frame_confidence_thresholds_are_independent_per_frame():
    first = make_frame(1, [])
    second = make_frame(2, [])
    first = FusionFrame(
        image_path=first.image_path,
        colmap_image=first.colmap_image,
        camera=first.camera,
        depth=first.depth,
        confidence=np.tile(np.array([1.0, 3.0], dtype=np.float32), (14, 7)),
        colors=first.colors,
        scale=first.scale,
        image_shape=first.image_shape,
        original_size=first.original_size,
    )
    second = FusionFrame(
        image_path=second.image_path,
        colmap_image=second.colmap_image,
        camera=second.camera,
        depth=second.depth,
        confidence=np.tile(np.array([100.0, 300.0], dtype=np.float32), (14, 7)),
        colors=second.colors,
        scale=second.scale,
        image_shape=second.image_shape,
        original_size=second.original_size,
    )

    thresholds = compute_frame_confidence_thresholds([first, second], 50.0)

    assert thresholds == {1: 2.0, 2: 200.0}


def test_global_confidence_scope_preserves_one_pooled_threshold():
    first = make_frame(1, [])
    second = make_frame(2, [])
    first = FusionFrame(
        image_path=first.image_path,
        colmap_image=first.colmap_image,
        camera=first.camera,
        depth=first.depth,
        confidence=np.ones((14, 14), dtype=np.float32),
        colors=first.colors,
        scale=first.scale,
        image_shape=first.image_shape,
        original_size=first.original_size,
    )
    second = FusionFrame(
        image_path=second.image_path,
        colmap_image=second.colmap_image,
        camera=second.camera,
        depth=second.depth,
        confidence=np.full((14, 14), 100.0, dtype=np.float32),
        colors=second.colors,
        scale=second.scale,
        image_shape=second.image_shape,
        original_size=second.original_size,
    )

    thresholds = compute_confidence_thresholds([first, second], 50.0, scope="global")

    assert thresholds == {1: 50.5, 2: 50.5}


def test_points_filter_uses_source_frame_thresholds():
    first = make_frame(1, [])
    second = make_frame(2, [])
    first = FusionFrame(
        image_path=first.image_path,
        colmap_image=first.colmap_image,
        camera=first.camera,
        depth=first.depth,
        confidence=np.tile(np.array([1.0, 3.0], dtype=np.float32), (14, 7)),
        colors=first.colors,
        scale=first.scale,
        image_shape=first.image_shape,
        original_size=first.original_size,
    )
    second = FusionFrame(
        image_path=second.image_path,
        colmap_image=second.colmap_image,
        camera=second.camera,
        depth=second.depth,
        confidence=np.tile(np.array([100.0, 300.0], dtype=np.float32), (14, 7)),
        colors=second.colors,
        scale=second.scale,
        image_shape=second.image_shape,
        original_size=second.original_size,
    )

    filtered = filter_points_by_cross_view_consistency(
        [first, second],
        covisibility_graph={1: [], 2: []},
        confidence_thresholds={1: 2.0, 2: 200.0},
        relative_threshold=0.1,
        stride=1,
    )

    assert filtered.candidate_points == 196
    assert [record["confidence_threshold"] for record in filtered.image_records] == [2.0, 200.0]


def test_cross_view_validation_uses_each_neighbor_threshold():
    neighbor = make_frame(2, [])
    neighbor = FusionFrame(
        image_path=neighbor.image_path,
        colmap_image=neighbor.colmap_image,
        camera=neighbor.camera,
        depth=neighbor.depth,
        confidence=np.full((14, 14), 10.0, dtype=np.float32),
        colors=neighbor.colors,
        scale=neighbor.scale,
        image_shape=neighbor.image_shape,
        original_size=neighbor.original_size,
    )
    source_points = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)

    accepted = validate_cross_view_consistency(
        source_points,
        neighbors=[neighbor],
        confidence_thresholds={2: 5.0},
        relative_threshold=0.1,
    )
    unverified = validate_cross_view_consistency(
        source_points,
        neighbors=[neighbor],
        confidence_thresholds={2: 20.0},
        relative_threshold=0.1,
    )

    assert accepted.support_counts.tolist() == [1]
    assert unverified.support_counts.tolist() == [0]
    assert unverified.visible_counts.tolist() == [0]
    assert unverified.accepted.tolist() == [True]

    threshold = derive_consistency_relative_threshold(
        [DepthScaleEstimate(1.0, 100, 0.006), DepthScaleEstimate(1.0, 100, 0.006)],
        min_threshold=0.02,
        max_threshold=0.08,
    )

    assert np.isclose(threshold, 0.02)


def test_order_images_by_covisibility_starts_at_hub_and_ends_isolated():
    images = [
        make_frame(1, [(0.0, 0.0, 10), (0.0, 0.0, 11), (0.0, 0.0, 12)]).colmap_image,
        make_frame(2, [(0.0, 0.0, 10), (0.0, 0.0, 11), (0.0, 0.0, 12), (0.0, 0.0, 20), (0.0, 0.0, 21)]).colmap_image,
        make_frame(3, [(0.0, 0.0, 20), (0.0, 0.0, 21)]).colmap_image,
        make_frame(4, [(0.0, 0.0, 99)]).colmap_image,
    ]
    graph = build_covisibility_graph(images, max_neighbors=8, min_shared_points=2)

    order = order_images_by_covisibility(images, graph)

    assert [image.image_id for image in order] == [2, 1, 3, 4]


def test_build_vggt_groups_sequential_matches_disjoint_chunks():
    paths = [Path(f"frame_{i}.jpg") for i in range(5)]
    registered_by_name = {path.name: make_frame(i, []).colmap_image for i, path in enumerate(paths)}

    groups = build_vggt_groups(
        registered_paths=paths,
        registered_by_name=registered_by_name,
        grouping="sequential",
        batch_size=2,
        overlap_size=2,
    )

    assert groups == [paths[0:2], paths[2:4], paths[4:5]]


def test_build_vggt_groups_covisibility_covers_every_image_with_overlap():
    paths = [Path(f"frame_{i}.jpg") for i in range(5)]
    registered_by_name = {}
    for i, path in enumerate(paths):
        shared = [(0.0, 0.0, i), (0.0, 0.0, i + 1)]  # chain: consecutive frames share a track
        registered_by_name[path.name] = make_frame(i, shared).colmap_image

    groups = build_vggt_groups(
        registered_paths=paths,
        registered_by_name=registered_by_name,
        grouping="covisibility",
        batch_size=3,
        overlap_size=1,
    )

    assert all(len(group) <= 3 for group in groups)
    assert {path.name for group in groups for path in group} == {path.name for path in paths}
    assert len(groups) >= 2  # sliding windows, not a single full pass


def test_undistort_to_pinhole_is_identity_without_distortion():
    depth = np.arange(9, dtype=np.float32).reshape(3, 3)
    color = np.zeros((3, 3, 3), dtype=np.uint8)
    camera = FusionCamera(
        model="PINHOLE",
        intrinsic=np.array([[3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        radial_distortion=(),
    )

    out_depth, out_color = undistort_to_pinhole(depth, color, camera)

    assert out_depth is depth
    assert out_color is color


def test_undistort_to_pinhole_preserves_principal_point_pixel():
    depth = np.arange(121, dtype=np.float32).reshape(11, 11)
    color = np.zeros((11, 11, 3), dtype=np.uint8)
    camera = FusionCamera(
        model="SIMPLE_RADIAL",
        intrinsic=np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        radial_distortion=(0.1,),
    )

    out_depth, out_color = undistort_to_pinhole(depth, color, camera)

    assert out_depth.shape == depth.shape
    # The principal point has zero radius, so distortion leaves it untouched.
    assert np.isclose(out_depth[5, 5], depth[5, 5])


def test_fuse_frames_tsdf_reconstructs_a_single_plane():
    frame = FusionFrame(
        image_path=Path("plane.jpg"),
        colmap_image=ColmapImage(
            image_id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros(3),
            camera_id=1,
            name="plane.jpg",
            observations=[],
        ),
        camera=FusionCamera(
            model="PINHOLE",
            intrinsic=np.array([[14.0, 0.0, 7.0], [0.0, 14.0, 7.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            radial_distortion=(),
        ),
        depth=np.full((14, 14), 1.0, dtype=np.float32),
        confidence=np.ones((14, 14), dtype=np.float32),
        colors=np.full((14, 14, 3), 128, dtype=np.uint8),
        scale=1.0,
        image_shape=(14, 14),
        original_size=(14, 14),
    )

    points, colors, stats = fuse_frames_tsdf(
        [frame],
        confidence_percentile=50.0,
        voxel_length=0.02,
        sdf_trunc=0.1,
        depth_trunc=5.0,
    )

    assert stats["integrated_frames"] == 1
    assert len(points) > 0
    assert colors.shape == points.shape
    assert np.all(np.isfinite(points))
    # The fused surface should sit at the plane depth (z = 1) in the camera frame.
    assert abs(float(np.median(points[:, 2])) - 1.0) < 0.1


def test_derive_tsdf_parameters_ignores_sparse_outliers():
    points3d = {
        index: np.array([index / 10.0, 0.0, 0.0])
        for index in range(1000)
    }
    points3d[1000] = np.array([10_000.0, 0.0, 0.0])

    parameters = derive_tsdf_parameters(
        points3d,
        [],
        voxel_length=0.0,
        sdf_trunc=0.0,
        depth_trunc=0.0,
    )

    assert parameters.full_diagonal > 9_000
    assert 90 < parameters.robust_diagonal < 110
    assert np.isclose(parameters.voxel_length, parameters.robust_diagonal / 1024.0)
    assert np.isclose(parameters.sdf_trunc, 5.0 * parameters.voxel_length)
    assert parameters.depth_trunc > 0


def test_random_point_budget_preserves_seeded_legacy_selection():
    points = np.arange(90, dtype=np.float32).reshape(30, 3)
    colors = np.arange(90, dtype=np.uint8).reshape(30, 3)
    expected_indices = np.random.default_rng(42).choice(30, size=10, replace=False)
    expected_indices.sort()

    result = apply_point_budget(points, colors, 10, 42, policy="random")

    assert result.applied
    assert result.policy == "random"
    assert np.array_equal(result.points, points[expected_indices])
    assert np.array_equal(result.colors, colors[expected_indices])
    assert result.spatial_quantization_bits is None
    assert result.occupied_spatial_codes is None


def test_spatially_balanced_point_budget_stratifies_spatial_order():
    points = np.column_stack(
        [
            np.arange(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ]
    )
    colors = np.arange(300, dtype=np.uint16).reshape(100, 3).astype(np.uint8)

    result = apply_point_budget(points, colors, 10, 42, policy="spatial_balanced")

    assert result.applied
    assert result.output_points == 10
    assert result.spatial_quantization_bits == 21
    assert result.occupied_spatial_codes == 100
    assert np.array_equal(result.points[:, 0], np.arange(5, 100, 10, dtype=np.float32))
    selected_color_rows = {tuple(row) for row in result.colors.tolist()}
    source_color_rows = {tuple(row) for row in colors.tolist()}
    assert selected_color_rows <= source_color_rows


def test_spatially_balanced_point_budget_is_deterministic_and_exact():
    points = np.random.default_rng(7).normal(size=(1_003, 3)).astype(np.float32)
    colors = np.arange(1_003 * 3, dtype=np.uint32).reshape(1_003, 3).astype(np.uint8)

    first = apply_point_budget(points, colors, 257, 42, policy="spatial_balanced")
    second = apply_point_budget(points, colors, 257, 999, policy="spatial_balanced")

    assert first.output_points == 257
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.colors, second.colors)


def test_spatially_balanced_point_budget_rejects_invalid_input():
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="matching lengths"):
        apply_point_budget(points, colors[:2], 2, 42, policy="spatial_balanced")
    points[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        apply_point_budget(points, colors, 2, 42, policy="spatial_balanced")


def test_point_budget_is_inactive_below_cap_for_both_policies():
    points = np.arange(18, dtype=np.float32).reshape(6, 3)
    colors = np.arange(18, dtype=np.uint8).reshape(6, 3)

    with pytest.raises(ValueError, match="Unknown point-budget policy"):
        apply_point_budget(points[:2], colors[:2], 10, 42, policy="unknown")

    for policy in ("random", "spatial_balanced"):
        result = apply_point_budget(points, colors, 10, 42, policy=policy)
        assert not result.applied
        assert result.points is points
        assert result.colors is colors


def test_factorial_arm_names_encode_phase_combinations():
    assert factorial_arm_name("global", "any_support", "random") == "baseline"
    assert factorial_arm_name("per_frame", "any_support", "random") == "phase1"
    assert factorial_arm_name("global", "adaptive_two", "random") == "phase2"
    assert factorial_arm_name("global", "any_support", "spatial_balanced") == "phase3"
    assert (
        factorial_arm_name("per_frame", "adaptive_two", "spatial_balanced")
        == "phase1_phase2_phase3"
    )


def test_validate_tsdf_output_rejects_sparse_or_incomplete_results():
    with pytest.raises(RuntimeError, match="skipped too many frames"):
        validate_tsdf_output({"input_frames": 100, "integrated_frames": 80, "num_points": 1_000_000})

    with pytest.raises(RuntimeError, match="implausibly sparse"):
        validate_tsdf_output({"input_frames": 100, "integrated_frames": 100, "num_points": 20_000})

    validate_tsdf_output({"input_frames": 100, "integrated_frames": 100, "num_points": 100_000})


def test_global_scale_graph_recovers_pairwise_scale_relation():
    frames = [make_frame(1, []), make_frame(2, [])]
    second = frames[1]
    frames[1] = FusionFrame(
        image_path=second.image_path,
        colmap_image=ColmapImage(
            image_id=2,
            qvec=second.colmap_image.qvec,
            tvec=np.array([-1.0, 0.0, 0.0]),
            camera_id=second.colmap_image.camera_id,
            name=second.colmap_image.name,
            observations=[],
        ),
        camera=second.camera,
        depth=np.full((14, 14), 1.0, dtype=np.float32),
        confidence=second.confidence,
        colors=second.colors,
        scale=1.0,
        image_shape=second.image_shape,
        original_size=second.original_size,
    )
    graph = {1: [CovisibilityEdge(target_image_id=2, shared_points=1, baseline=1.0)], 2: []}
    estimates = {"frame_1.jpg": DepthScaleEstimate(1.0, 100, 0.01)}

    result = optimize_depth_scale_graph(
        frames=frames,
        covisibility_graph=graph,
        scale_estimates=estimates,
        fallback_scale=1.0,
        confidence_threshold=0.5,
        relative_threshold=0.1,
        iterations=5,
        pair_weight=10.0,
        huber_delta=0.1,
        max_pairs_per_edge=32,
    )

    assert np.isclose(result.scales[1], 1.0, atol=1e-4)
    assert np.isclose(result.scales[2], 2.0, atol=1e-3)
    assert result.pair_constraint_count > 0
    assert result.component_count == 1
    assert result.fallback_image_count == 0


def test_global_scale_graph_pins_unanchored_component_to_fallback():
    frames = [make_frame(1, []), make_frame(2, [])]
    result = optimize_depth_scale_graph(
        frames=frames,
        covisibility_graph={1: [], 2: []},
        scale_estimates={"frame_1.jpg": DepthScaleEstimate(1.0, 100, 0.01)},
        fallback_scale=3.0,
        confidence_threshold=0.5,
        relative_threshold=0.1,
        iterations=3,
        pair_weight=0.0,
        huber_delta=0.1,
        max_pairs_per_edge=8,
    )

    assert np.isclose(result.scales[1], 1.0, atol=1e-4)
    assert np.isclose(result.scales[2], 3.0, atol=1e-4)
    assert result.component_count == 2
    assert result.fallback_image_count == 1


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
