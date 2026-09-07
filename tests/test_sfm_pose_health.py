from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from image3d_scenegraph.geometry.grouping import (
    ColmapImage,
    parse_colmap_images_with_points,
)
from image3d_scenegraph.geometry.sfm_pose_health import build_sfm_pose_health


def _images(centers: list[float]) -> list[ColmapImage]:
    return [
        ColmapImage(
            image_id=index + 1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([-center, 0.0, 0.0]),
            camera_id=1,
            name=f"frame-{index:04d}.jpg",
            observations=[(10.0, 10.0, 1)],
        )
        for index, center in enumerate(centers)
    ]


def _timestamps(images: list[ColmapImage]) -> dict[str, float]:
    return {image.name: float(index) for index, image in enumerate(images)}


def _database(path: Path, pairs: list[tuple[int, int, int, int]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE two_view_geometries(
                pair_id INTEGER PRIMARY KEY, rows INTEGER, config INTEGER
            );
            """
        )
        for left, right, candidates, verified in pairs:
            pair_id = min(left, right) * 2_147_483_647 + max(left, right)
            connection.execute("INSERT INTO matches VALUES (?, ?)", (pair_id, candidates))
            connection.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?)",
                (pair_id, verified, 2),
            )


def test_colmap_parser_preserves_empty_observation_rows(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 first.jpg\n\n"
        "2 1 0 0 0 -1 0 0 1 second.jpg\n"
        "10 20 7\n",
        encoding="utf-8",
    )

    images = parse_colmap_images_with_points(path)

    assert [image.name for image in images] == ["first.jpg", "second.jpg"]
    assert images[0].observations == []
    assert images[1].observations == [(10.0, 20.0, 7)]


def test_normal_long_trajectory_passes() -> None:
    images = _images([float(index) for index in range(100)])

    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
    )

    assert record["status"] == "passed"
    assert record["reason_codes"] == []
    assert record["temporal"]["registration_timeline"]["registration_rate"] == 1.0
    assert record["temporal"]["registration_timeline"]["temporal_coverage"] == 1.0
    assert record["temporal"]["rotation_jump_degrees"]["max"] == 0.0
    assert record["outlier_candidates"] == []
    assert record["automatic_repair"] == {
        "eligible": False,
        "reason": "pose_health_passed",
    }


def test_stationary_segment_does_not_create_pose_failure() -> None:
    images = _images([0.0] * 30 + [float(index) for index in range(70)])

    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
    )

    assert record["status"] == "passed"
    assert record["temporal"]["translation_speed_world_per_second"]["min"] == 0.0


def test_isolated_camera_pose_is_detected() -> None:
    centers = [float(index) for index in range(100)]
    centers[50] = 1_000_000.0
    images = _images(centers)

    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
    )

    assert "isolated_camera_pose_outlier" in record["reason_codes"]
    assert [item["image_id"] for item in record["outlier_candidates"]] == [51]
    assert record["outlier_candidates"][0]["observed_depth_world"]["p50"] == 10.0
    assert record["automatic_repair"]["eligible"] is True


def test_multiscale_branch_is_detected_with_bridge_evidence(tmp_path: Path) -> None:
    centers = [float(index) for index in range(100)]
    centers[50:53] = [10_000.0, 10_001.0, 10_002.0]
    images = _images(centers)
    database = tmp_path / "database.db"
    _database(database, [(50, 51, 80, 40), (51, 54, 120, 60)])

    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
        database_path=database,
    )

    assert "multiscale_camera_pose_branch" in record["reason_codes"]
    assert {item["image_id"] for item in record["outlier_candidates"]} == {
        51,
        52,
        53,
    }
    assert record["automatic_repair"]["eligible"] is True
    assert record["automatic_repair"]["outlier_fraction"] == 0.03
    assert record["covisibility"]["component_count"] == 1
    assert record["covisibility"]["mixed_core_outlier_component_count"] == 1
    assert record["covisibility"]["components"][0]["outlier_image_count"] == 3
    assert any(
        item["candidate_match_count"] is not None
        and item["verified_inlier_count"] is not None
        for item in record["bridge_pairs"]
    )


def test_automatic_repair_requires_video_and_at_most_ten_percent() -> None:
    centers = [float(index) for index in range(100)]
    centers[45:56] = [10_000.0 + index for index in range(11)]
    images = _images(centers)

    without_video = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
    )
    with_video = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
    )

    assert without_video["automatic_repair"]["reason"] == (
        "video_timestamps_unavailable"
    )
    assert with_video["automatic_repair"]["reason"] == (
        "outlier_fraction_exceeds_limit"
    )


def test_temporal_discontinuity_below_spatial_cutoff_is_rejected() -> None:
    centers = [float(index) for index in range(100)]
    centers[50] = 2_000.0
    images = _images(centers)
    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
    )
    assert record["profile"] == "sfm_pose_health_v2"
    assert record["camera_centers"]["max_to_median"] < 100
    assert record["reason_codes"] == ["temporal_pose_discontinuity"]
    assert record["temporal"]["discontinuity_count"] == 2
    assert record["outlier_candidates"] == []
    assert record["automatic_repair"]["eligible"] is False


def test_stationary_majority_and_fast_rotation_are_not_translation_failures() -> None:
    images = _images([0.0] * 95 + [float(i) for i in range(5)])
    images[50].qvec[:] = [0.0, 0.0, 1.0, 0.0]
    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=_timestamps(images),
    )
    assert "temporal_pose_discontinuity" not in record["reason_codes"]
    assert record["temporal"]["rotation_jump_degrees"]["max"] == 180
    assert record["temporal"]["discontinuity_count"] == 0


def test_unequal_sampling_and_world_scale_do_not_create_discontinuities() -> None:
    times = [float(i * i) for i in range(100)]
    for scale in (0.001, 1.0, 1000.0):
        images = _images([t * scale for t in times])
        record = build_sfm_pose_health(
            images=images,
            points3d={1: np.array([0.0, 0.0, 10.0])},
            selected_timestamps={image.name: t for image, t in zip(images, times)},
        )
        assert record["status"] == "passed"
        assert record["temporal"]["discontinuity_count"] == 0


def test_empty_sparse_geometry_cannot_pass_with_healthy_cameras() -> None:
    record = build_sfm_pose_health(images=_images(list(range(100))), points3d={})
    assert record["status"] == "failed"
    assert "empty_sfm_point_cloud" in record["reason_codes"]


def test_fast_local_step_without_scene_scale_jump_stays_diagnostic() -> None:
    images = _images([float(i) for i in range(100)])
    times = _timestamps(images)
    times[images[50].name] = 49.0001
    record = build_sfm_pose_health(
        images=images,
        points3d={1: np.array([0.0, 0.0, 10.0])},
        selected_timestamps=times,
    )
    assert record["status"] == "passed"
    assert record["temporal"]["discontinuity_count"] == 0
