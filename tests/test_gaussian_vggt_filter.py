from __future__ import annotations

import numpy as np
import pytest

from image3d_scenegraph.gaussian.vggt_filter import (
    DepthEvidence,
    GaussianVggtFilterError,
    OVERSIZED_CENTER_PROFILE_ID,
    PROFILE_ID,
    REASON_FRONT_FREE_SPACE,
    REASON_OUTSIDE_OVERSIZED,
    classify_gaussians,
)


def evidence(image_id: str, *, camera_x: float = 0.0) -> DepthEvidence:
    camera = np.eye(4)
    camera[0, 3] = camera_x
    intrinsic = np.array([[20, 0, 32], [0, 20, 32], [0, 0, 1]], dtype=np.float64)
    depth = np.full((64, 64), 5.0, dtype=np.float32)
    confidence = np.ones((64, 64), dtype=np.float32)
    return DepthEvidence(
        image_id=image_id,
        image_name=f"{image_id}.png",
        camera_from_normalized=camera,
        intrinsic=intrinsic,
        depth=depth,
        confidence=confidence,
        confidence_threshold=0.5,
        scale_x=1.0,
        scale_y=1.0,
        pad_left=0.0,
        pad_top=0.0,
        far_depth=5.0,
        scale_observations=30,
        scale_log_mad=0.01,
    )


def gaussian_arrays(count: int = 200):
    means = np.zeros((count, 3), dtype=np.float32)
    means[:, 2] = 5.0
    scales = np.full((count, 3), 0.01, dtype=np.float32)
    quaternions = np.zeros((count, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    opacities = np.full(count, 0.5, dtype=np.float32)
    return means, scales, quaternions, opacities


def test_multiview_free_space_floaters_are_removed_but_surface_is_kept():
    means, scales, quaternions, opacities = gaussian_arrays()
    means[:10, 2] = 2.0

    result = classify_gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        evidence=[evidence("1"), evidence("2", camera_x=0.01)],
    )

    assert (~result.keep[:10]).all()
    assert result.keep[10:].all()
    assert (result.reasons[:10] & REASON_FRONT_FREE_SPACE).all()
    assert result.diagnostics["profile"] == PROFILE_ID
    assert result.diagnostics["settings"]["oversized_depth_policy"] == "extent"
    assert result.diagnostics["validation_image_ids"] == []
    assert result.diagnostics["test_image_ids"] == []


def test_single_view_contradiction_and_occluded_points_are_kept():
    means, scales, quaternions, opacities = gaussian_arrays()
    means[0, 2] = 2.0
    means[1, 2] = 8.0
    second = evidence("2")
    second_confidence = second.confidence.copy()
    second_confidence[:, :] = 0.0
    second = DepthEvidence(
        **{**second.__dict__, "confidence": second_confidence}
    )

    result = classify_gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        evidence=[evidence("1"), second],
    )

    assert result.keep[0]
    assert result.keep[1]


def test_oversized_center_policy_rejects_extent_only_surface_support():
    means, scales, quaternions, opacities = gaussian_arrays()
    means[0, 2] = 8.0
    scales[0] = 1.1
    views = [evidence("1"), evidence("2", camera_x=0.01)]

    extent = classify_gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        evidence=views,
    )
    center = classify_gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        evidence=views,
        oversized_depth_policy="center",
    )

    assert extent.keep[0]
    assert extent.diagnostics["profile"] == PROFILE_ID
    assert not center.keep[0]
    assert center.reasons[0] & REASON_OUTSIDE_OVERSIZED
    assert center.diagnostics["profile"] == OVERSIZED_CENTER_PROFILE_ID
    assert center.diagnostics["settings"]["oversized_depth_policy"] == "center"


def test_oversized_center_policy_keeps_centered_surface_and_leaves_normal_rows_unchanged():
    means, scales, quaternions, opacities = gaussian_arrays()
    scales[0] = 1.1
    views = [evidence("1"), evidence("2", camera_x=0.01)]

    extent = classify_gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        evidence=views,
    )
    center = classify_gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        evidence=views,
        oversized_depth_policy="center",
    )

    assert center.keep[0]
    assert np.array_equal(center.keep[1:], extent.keep[1:])
    assert np.array_equal(center.reasons[1:], extent.reasons[1:])


def test_unknown_oversized_depth_policy_is_rejected():
    means, scales, quaternions, opacities = gaussian_arrays()

    with pytest.raises(GaussianVggtFilterError, match="unsupported oversized"):
        classify_gaussians(
            means=means,
            scales=scales,
            quaternions=quaternions,
            opacities=opacities,
            evidence=[evidence("1"), evidence("2")],
            oversized_depth_policy="unknown",
        )


def test_overaggressive_filter_fails_instead_of_publishing():
    means, scales, quaternions, opacities = gaussian_arrays()
    means[:100, 2] = 2.0

    with pytest.raises(GaussianVggtFilterError, match="safety limit"):
        classify_gaussians(
            means=means,
            scales=scales,
            quaternions=quaternions,
            opacities=opacities,
            evidence=[evidence("1"), evidence("2")],
        )
