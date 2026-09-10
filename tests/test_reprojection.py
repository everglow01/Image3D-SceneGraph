from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.geometry.reprojection import (
    normalize_pixel_errors,
    pixel_reprojection_errors,
    project_pixels,
)
from image3d_scenegraph.geometry.camera_calibration import _sparse_summary
from image3d_scenegraph.gaussian.initialization import read_colmap_points_text


def text_model(root: Path, stored_error: float = 0.00001) -> None:
    root.mkdir(exist_ok=True)
    (root / "cameras.txt").write_text("1 PINHOLE 100 100 100 100 50 50\n")
    (root / "points3D.txt").write_text(f"# original\n5 0 0 2 1 2 3 {stored_error} 1 0 2 0\n")
    (root / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 first.jpg\n53 54 5\n"
        "2 1 0 0 0 0 0 0 1 second.jpg\n50 57 5\n"
    )


def test_errors_are_pixels_not_solver_field_and_aggregate_per_point(tmp_path):
    text_model(tmp_path)
    errors, report = pixel_reprojection_errors(tmp_path)
    assert errors == {5: 6.0}  # mean of 5px and 7px, not RMS and not stored ERROR
    assert report["median_reprojection_error_pixels"] == 6
    assert report["observation_count"] == 2
    source = (tmp_path / "points3D.txt").read_bytes()
    normalize_pixel_errors(tmp_path)
    assert (tmp_path / "points3D.source-error.txt").read_bytes() == source
    fields = (tmp_path / "points3D.txt").read_text().splitlines()[1].split()
    assert fields[:7] == source.decode().splitlines()[1].split()[:7]
    assert fields[8:] == ["1", "0", "2", "0"]
    assert _sparse_summary(tmp_path / "points3D.txt")["median_reprojection_error_pixels"] == 6
    assert read_colmap_points_text(tmp_path / "points3D.txt")[2].tolist() == [6.0]
    with pytest.raises(ValueError, match="already exists"):
        normalize_pixel_errors(tmp_path)


@pytest.mark.parametrize("model,params,expected", [
    ("SIMPLE_PINHOLE", [100, 50, 50], [70, 80]),
    ("PINHOLE", [100, 200, 50, 60], [70, 120]),
    ("SIMPLE_RADIAL", [100, 50, 50, 0.1], [70.26, 80.39]),
    ("RADIAL", [100, 50, 50, 0.1, 0.2], [70.3276, 80.4914]),
    ("OPENCV", [100, 200, 50, 60, 0.1, 0.2, 0.01, 0.02], [70.8676, 122.0828]),
])
def test_projection_camera_models(model, params, expected):
    result = project_pixels(np.array([[0.4, 0.6, 2]]), model, params)
    assert np.allclose(result[0], expected)


@pytest.mark.parametrize("defect", ["missing_point", "nonfinite", "behind", "no_observations"])
def test_invalid_geometry_cannot_be_normalized(tmp_path, defect):
    text_model(tmp_path)
    if defect == "missing_point":
        (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a.jpg\n50 50 9\n")
    elif defect == "no_observations":
        (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a.jpg\n\n")
    else:
        (tmp_path / "points3D.txt").write_text(
            f"5 0 0 {'nan' if defect == 'nonfinite' else '-2'} 1 2 3 0 1 0 2 0\n"
        )
    with pytest.raises(ValueError):
        normalize_pixel_errors(tmp_path)
    assert not (tmp_path / "points3D.source-error.txt").exists()


def test_point_summary_does_not_weight_long_tracks_more(tmp_path):
    text_model(tmp_path)
    with (tmp_path / "points3D.txt").open("a") as handle:
        handle.write("9 0 0 2 1 2 3 0.0001 2 1\n")
    path = tmp_path / "images.txt"
    path.write_text(path.read_text().replace("50 57 5", "50 57 5 50 70 9"))
    errors, report = pixel_reprojection_errors(tmp_path)
    assert errors == {5: 6, 9: 20}
    assert report["mean_reprojection_error_pixels"] == 13
    assert report["median_reprojection_error_pixels"] == 13
    assert report["observation_count"] == 3


def test_unsupported_camera_is_not_silently_approximated():
    with pytest.raises(ValueError, match="unsupported"):
        project_pixels(np.array([[0, 0, 2]]), "FISHEYE", [100, 100, 50, 50])
