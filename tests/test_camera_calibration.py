from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest
from PIL import Image

from image3d_scenegraph.geometry.camera_calibration import (
    CameraCalibrationError,
    build_camera_calibration_diagnostics,
    camera_calibration_metrics,
    prepare_camera_extraction,
)
from image3d_scenegraph.geometry.colmap import resolve_colmap_camera_calibration


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = (64, 48),
    make: str | None = "Camera Co",
    model: str | None = "Model A",
    lens: str | None = "Lens A",
    focal: int | None = 35,
    orientation: int = 1,
    include_gps: bool = False,
) -> None:
    exif = Image.Exif()
    if make is not None:
        exif[271] = make
    if model is not None:
        exif[272] = model
    if lens is not None:
        exif[42036] = lens
    if focal is not None:
        exif[37386] = focal
    exif[274] = orientation
    if include_gps:
        exif[34853] = {1: "N", 2: (1, 2, 3)}
    Image.new("RGB", size, (32, 64, 96)).save(path, exif=exif)


def _database(
    path: Path,
    *,
    model_id: int,
    params: list[float],
    images: list[tuple[int, str, int]],
    cameras: list[tuple[int, int, int, int, list[float], int]] | None = None,
) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE cameras(
          camera_id INTEGER PRIMARY KEY,
          model INTEGER NOT NULL,
          width INTEGER NOT NULL,
          height INTEGER NOT NULL,
          params BLOB,
          prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE images(
          image_id INTEGER PRIMARY KEY,
          name TEXT NOT NULL UNIQUE,
          camera_id INTEGER NOT NULL
        );
        """
    )
    camera_rows = cameras or [(1, model_id, 1000, 800, params, 1)]
    for camera_id, row_model, width, height, row_params, prior in camera_rows:
        connection.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                row_model,
                width,
                height,
                struct.pack(f"<{len(row_params)}d", *row_params),
                prior,
            ),
        )
    connection.executemany("INSERT INTO images VALUES (?, ?, ?)", images)
    connection.commit()
    connection.close()


def test_focal_aware_grouping_is_deterministic_and_conservative(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    first = image_root / "a.jpg"
    second = image_root / "b.jpg"
    zoomed = image_root / "c.jpg"
    missing = image_root / "d.jpg"
    rotated = image_root / "e.jpg"
    resized = image_root / "f.jpg"
    _write_image(first, include_gps=True)
    _write_image(second)
    _write_image(zoomed, focal=50)
    _write_image(missing, make=None, model=None, focal=None)
    _write_image(rotated, orientation=6)
    _write_image(resized, size=(80, 48))
    profile = resolve_colmap_camera_calibration(
        "auto_grouped_simple_radial_v1"
    )

    paths = [resized, missing, second, zoomed, first, rotated]
    plan = prepare_camera_extraction(
        profile, image_root, paths, tmp_path / "lists-a"
    )
    reversed_plan = prepare_camera_extraction(
        profile, image_root, list(reversed(paths)), tmp_path / "lists-b"
    )

    assert plan.record() == reversed_plan.record()
    assert plan.record()["planned_camera_count"] == 5
    assert [group["images"] for group in plan.groups] == [
        ["a.jpg", "b.jpg"],
        ["c.jpg"],
        ["d.jpg"],
        ["e.jpg"],
        ["f.jpg"],
    ]
    assert len(plan.batches) == 2
    assert plan.batches[0].image_names == ("a.jpg", "b.jpg")
    assert plan.batches[1].image_names == (
        "c.jpg",
        "d.jpg",
        "e.jpg",
        "f.jpg",
    )
    assert "--ImageReader.single_camera_per_image" in (
        plan.batches[1].image_reader_options
    )
    serialized = str(plan.record()).casefold()
    assert "gps" not in serialized
    assert "serial" not in serialized


def test_shared_profile_uses_one_extraction_batch_without_reading_exif(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = [image_root / "broken-a.jpg", image_root / "broken-b.jpg"]
    for path in paths:
        path.write_bytes(b"not an image")

    plan = prepare_camera_extraction(
        resolve_colmap_camera_calibration("shared_opencv_v1"),
        image_root,
        paths,
        tmp_path / "unused",
    )

    assert len(plan.groups) == 1
    assert len(plan.batches) == 1
    assert plan.batches[0].image_list_path is None
    assert plan.batches[0].image_reader_options[1] == "OPENCV"


def test_camera_diagnostics_compare_database_and_final_raw_model(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = [image_root / "a.jpg", image_root / "b.jpg"]
    for path in paths:
        path.write_bytes(b"fixture")
    plan = prepare_camera_extraction(
        resolve_colmap_camera_calibration("shared_opencv_v1"),
        image_root,
        paths,
        tmp_path / "unused",
    )
    database = tmp_path / "database.db"
    initial_params = [900.0, 900.0, 500.0, 400.0, 0.0, 0.0, 0.0, 0.0]
    _database(
        database,
        model_id=4,
        params=initial_params,
        images=[(1, "a.jpg", 1), (2, "b.jpg", 1)],
    )
    points = tmp_path / "points3D.txt"
    points.write_text(
        "# points\n1 0 0 1 255 255 255 0.5 1 0 2 0\n",
        encoding="utf-8",
    )

    diagnostics = build_camera_calibration_diagnostics(
        database_path=database,
        final_camera_payload={
            "cameras": [
                {
                    "camera_id": 1,
                    "model": "OPENCV",
                    "width": 1000,
                    "height": 800,
                    "params": [
                        1000.0,
                        1000.0,
                        500.0,
                        400.0,
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
            ],
            "images": [
                {"name": "a.jpg", "camera_id": 1},
                {"name": "b.jpg", "camera_id": 1},
            ],
        },
        points3d_path=points,
        plan=plan,
        colmap_build="COLMAP 4.0.0",
    )

    assert diagnostics["initial"]["prior_focal_camera_count"] == 1
    assert diagnostics["initial"]["groups"][0]["group_id"] == (
        "camera-group-0000"
    )
    assert diagnostics["final"]["camera_count"] == 1
    assert diagnostics["final"]["median_focal_length_ratio"] == 1.0
    assert diagnostics["final"]["cameras"][0][
        "relative_focal_length_change"
    ] == pytest.approx(1 / 9)
    assert diagnostics["sparse"] == {
        "point_count": 1,
        "observation_count": 2,
        "mean_reprojection_error_pixels": 0.5,
        "median_reprojection_error_pixels": 0.5,
        "mean_track_length": 2.0,
        "median_track_length": 2,
    }
    assert diagnostics["plausibility"]["warning_count"] == 0
    assert camera_calibration_metrics(diagnostics)[
        "sfm_camera_calibration_profile"
    ] == "shared_opencv_v1"


def test_camera_diagnostics_keep_plausibility_issues_soft(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "a.jpg"
    image.write_bytes(b"fixture")
    plan = prepare_camera_extraction(
        resolve_colmap_camera_calibration("shared_simple_radial_v1"),
        image_root,
        [image],
        tmp_path / "unused",
    )
    database = tmp_path / "database.db"
    _database(
        database,
        model_id=2,
        params=[1200.0, 500.0, 400.0, 0.0],
        images=[(1, "a.jpg", 1)],
    )
    points = tmp_path / "points3D.txt"
    points.write_text("# empty\n", encoding="utf-8")

    diagnostics = build_camera_calibration_diagnostics(
        database_path=database,
        final_camera_payload={
            "cameras": [
                {
                    "camera_id": 1,
                    "model": "SIMPLE_RADIAL",
                    "width": 1000,
                    "height": 800,
                    "params": [20_000.0, -1.0, 400.0, 2.0],
                }
            ],
            "images": [{"name": "a.jpg", "camera_id": 1}],
        },
        points3d_path=points,
        plan=plan,
        colmap_build="COLMAP 4.0.0",
    )

    assert diagnostics["plausibility"]["warning_count"] == 3
    assert diagnostics["plausibility"]["warnings_are_job_gates"] is False


def test_camera_diagnostics_reject_group_partition_drift(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    first = image_root / "a.jpg"
    second = image_root / "b.jpg"
    _write_image(first)
    _write_image(second)
    plan = prepare_camera_extraction(
        resolve_colmap_camera_calibration(
            "auto_grouped_simple_radial_v1"
        ),
        image_root,
        [first, second],
        tmp_path / "lists",
    )
    database = tmp_path / "database.db"
    _database(
        database,
        model_id=2,
        params=[1200.0, 32.0, 24.0, 0.0],
        images=[(1, "a.jpg", 1), (2, "b.jpg", 2)],
        cameras=[
            (1, 2, 64, 48, [76.8, 32.0, 24.0, 0.0], 1),
            (2, 2, 64, 48, [76.8, 32.0, 24.0, 0.0], 1),
        ],
    )
    points = tmp_path / "points3D.txt"
    points.write_text("# empty\n", encoding="utf-8")

    with pytest.raises(CameraCalibrationError, match="was split"):
        build_camera_calibration_diagnostics(
            database_path=database,
            final_camera_payload={"cameras": [], "images": []},
            points3d_path=points,
            plan=plan,
            colmap_build="COLMAP 4.0.0",
        )
