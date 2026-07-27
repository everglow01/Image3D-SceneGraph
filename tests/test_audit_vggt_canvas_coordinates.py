from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_vggt_canvas_coordinates import audit_canvas_coordinates  # noqa: E402


def write_job(tmp_path: Path, *, bad_transform: bool = False) -> tuple[Path, Path]:
    job_dir = tmp_path / "job"
    sparse_dir = job_dir / "colmap_vggt" / "sparse_txt"
    diagnostics = job_dir / "diagnostics"
    sparse_dir.mkdir(parents=True)
    diagnostics.mkdir()
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1000, 500)).save(image_path)

    (sparse_dir / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 1000 500 1000 500 250 0.02\n", encoding="utf-8"
    )
    (sparse_dir / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame.png\n"
        "725 375 1 500 250 2\n",
        encoding="utf-8",
    )
    (sparse_dir / "points3D.txt").write_text(
        "1 2.25 1.25 10 0 0 0 0\n2 0 0 10 0 0 0 0\n", encoding="utf-8"
    )
    transform = {
        "scale_x": 0.518,
        "scale_y": 0.504,
        "pad_left": 0,
        "pad_top": 133,
        "pad_right": 0,
        "pad_bottom": 133,
        "resized_width": 518,
        "resized_height": 252,
    }
    if bad_transform:
        transform["pad_top"] = 132
    prediction = {
        "image": "frame.png",
        "image_path": image_path.as_posix(),
        "image_id": 1,
        "group_index": 0,
        "group_position": 0,
        "selected_for_first_wins": True,
        "prediction_file": "unused.npz",
        "image_shape": [518, 518],
        "original_size": [1000, 500],
        "canvas_transform": transform,
    }
    index_path = diagnostics / "vggt_window_predictions.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_enabled": True,
                "prediction_count": 1,
                "predictions": [prediction],
            }
        ),
        encoding="utf-8",
    )
    (diagnostics / "fusion.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image": "frame.png",
                        "fusion_intrinsic": [
                            [518.0, 0.0, 259.0],
                            [0.0, 504.0, 259.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "radial_distortion": [0.02],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return job_dir, index_path


def test_audit_round_trips_sparse_points_and_reports_pixel_center_offset(tmp_path):
    job_dir, index_path = write_job(tmp_path)

    result = audit_canvas_coordinates(job_dir=job_dir, index_path=index_path)

    assert result["round_trip_gate"]["passed"] is True
    assert result["round_trip_gate"]["sparse_point_count"] == 2
    assert result["round_trip_gate"]["pixel_error"]["max"] < 1e-3
    assert result["inventory"] == {
        "registered_image_count": 1,
        "captured_prediction_count": 1,
        "unique_captured_image_count": 1,
        "camera_models": ["SIMPLE_RADIAL"],
    }
    offsets = result["coordinate_convention"][
        "production_minus_pil_pixel_center_offset"
    ]
    assert np.isclose(offsets["x_absolute"]["max"], 0.241)
    assert np.isclose(offsets["y_absolute"]["max"], 0.248)
    assert result["conclusion"] == {
        "production_coordinate_chain_is_self_consistent": True,
        "pixel_center_convention_requires_candidate_evaluation": True,
        "production_fusion_changed": False,
    }


def test_audit_rejects_capture_transform_that_differs_from_production(tmp_path):
    job_dir, index_path = write_job(tmp_path, bad_transform=True)

    try:
        audit_canvas_coordinates(job_dir=job_dir, index_path=index_path)
    except ValueError as error:
        assert str(error) == "canvas pad_top mismatch for frame.png"
    else:
        raise AssertionError("expected mismatched canvas metadata to fail")
