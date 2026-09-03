from __future__ import annotations

import numpy as np
import pytest

from image3d_scenegraph.gaussian.config import (
    resolve_internal_config,
    resolve_mcmc_config,
)
from image3d_scenegraph.gaussian.dataset import camera_normalization
from image3d_scenegraph.gaussian.initialization import InitializationResult
from image3d_scenegraph.gaussian.readiness import (
    INITIAL_SCALE_FLOOR,
    GeometryReadinessError,
    build_geometry_readiness,
    project_initialization_keep_mask,
    require_geometry_readiness,
)
from scripts.run_gaussian_training import _print_readiness_summary


def _image(image_id: int, center: np.ndarray) -> dict:
    world_from_camera = np.eye(4)
    world_from_camera[:3, 3] = center
    return {
        "image_id": str(image_id),
        "path": f"images/{image_id}.jpg",
        "intrinsic": [[1_000.0, 0.0, 640.0], [0.0, 1_000.0, 360.0], [0.0, 0.0, 1.0]],
        "camera_from_world": np.linalg.inv(world_from_camera).tolist(),
        "world_from_camera": world_from_camera.tolist(),
    }


def _contract(centers: np.ndarray) -> dict:
    images = [_image(index, center) for index, center in enumerate(centers)]
    count = len(images)
    validation_start = max(1, count - 4)
    test_start = max(validation_start + 2, count - 2)
    return {
        "images": images,
        "normalization": camera_normalization(images),
        "splits": {
            "train": [str(index) for index in range(validation_start)],
            "validation": [str(index) for index in range(validation_start, test_start)],
            "test": [str(index) for index in range(test_start, count)],
        },
    }


def _initialization(scales: np.ndarray) -> InitializationResult:
    count = len(scales)
    points = np.column_stack(
        (
            np.linspace(-0.01, 0.01, count),
            np.zeros(count),
            np.linspace(0.001, 0.02, count),
        )
    ).astype(np.float32)
    return InitializationResult(
        points=points,
        colors=np.full((count, 3), 127, dtype=np.uint8),
        scales=scales.astype(np.float32),
        diagnostics={},
    )


def test_readiness_rejects_isolated_camera_and_scale_floor_collapse(capsys):
    angles = np.linspace(0.0, 2.0 * np.pi, 999, endpoint=False)
    body = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(999)))
    centers = np.vstack((body, [1_000.0, 0.0, 0.0]))
    initialized = _initialization(
        np.full(256, INITIAL_SCALE_FLOOR, dtype=np.float32)
    )

    record = build_geometry_readiness(
        _contract(centers),
        initialized,
        resolve_internal_config().effective_config,
        trainer_id="project",
    )

    assert record["status"] == "failed"
    assert record["reason_codes"] == [
        "unusable_camera_pose_outlier",
        "initialization_scale_floor_collapse",
    ]
    assert record["camera_centers"]["farthest_image_id"] == "999"
    assert record["camera_centers"]["largest_distances"][0]["image_id"] == "999"
    assert record["camera_centers"]["largest_distances"][0][
        "distance_to_median_ratio"
    ] > 900
    assert record["camera_centers"]["max_to_median"] > 900
    assert record["camera_centers"]["max_to_p99"] > 900
    assert record["initialization"]["scale_floor_fraction"] == 1.0
    assert record["projection_risk"]["status"] == "diagnostic_only"
    assert record["projection_risk"]["test_rgb_loaded"] is False
    assert record["projection_risk"]["highest_risk_views"]
    with pytest.raises(
        GeometryReadinessError,
        match="unusable_camera_pose_outlier,initialization_scale_floor_collapse",
    ):
        require_geometry_readiness(record)
    _print_readiness_summary(record)
    output = capsys.readouterr().out
    assert "readiness_status=failed" in output
    assert "camera_center_rank=01 image_id=999" in output


def test_readiness_rejects_small_dataset_camera_outlier():
    angles = np.linspace(0.0, 2.0 * np.pi, 11, endpoint=False)
    body = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(11)))
    centers = np.vstack((body, [1_000.0, 0.0, 0.0]))

    record = build_geometry_readiness(
        _contract(centers),
        _initialization(np.full(32, 0.01, dtype=np.float32)),
        resolve_internal_config().effective_config,
        trainer_id="project",
    )

    assert record["reason_codes"] == ["unusable_camera_pose_outlier"]
    assert record["camera_centers"]["max_to_p99"] > 100


def test_readiness_accepts_ring_and_linear_camera_trajectories():
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(20)))
    line = np.column_stack((np.arange(20), np.zeros(20), np.zeros(20)))
    initialized = _initialization(np.full(32, 0.01, dtype=np.float32))

    for centers in (ring, line):
        record = build_geometry_readiness(
            _contract(centers),
            initialized,
            resolve_internal_config().effective_config,
            trainer_id="project",
        )
        assert record["status"] == "passed"
        assert record["reason_codes"] == []
        assert record["projection_risk"] == {
            "status": "not_run_geometry_passed"
        }


def test_pre_render_world_scale_pruning_is_default_strategy_only():
    initialized = _initialization(np.array([0.05, 0.1, 0.1001], dtype=np.float32))
    project = resolve_internal_config().effective_config
    mcmc = resolve_mcmc_config().effective_config

    assert project_initialization_keep_mask(initialized, project).tolist() == [
        True,
        True,
        False,
    ]
    assert project_initialization_keep_mask(initialized, mcmc).tolist() == [
        True,
        True,
        True,
    ]

    record = build_geometry_readiness(
        _contract(
            np.column_stack(
                (np.arange(12), np.zeros(12), np.zeros(12))
            )
        ),
        initialized,
        project,
        trainer_id="project",
    )
    assert record["initialization"]["pre_render_world_scale_pruning"] == {
        "applied": True,
        "maximum_effective_scale": 0.1,
        "before": 3,
        "removed": 1,
        "after": 2,
    }
