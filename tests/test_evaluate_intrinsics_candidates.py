from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_intrinsics_candidates import (  # noqa: E402
    build_candidate_cameras,
    evaluate_intrinsics_candidates,
)
from run_colmap_vggt_dense import FusionCamera  # noqa: E402


def write_scene(tmp_path: Path, name: str, *, vggt_intrinsic: list[list[float]]) -> tuple[str, Path, Path]:
    job_dir = tmp_path / name
    sparse_dir = job_dir / "colmap_vggt" / "sparse_txt"
    diagnostics = job_dir / "diagnostics"
    sparse_dir.mkdir(parents=True)
    diagnostics.mkdir()
    image_path = tmp_path / f"{name}.png"
    Image.new("RGB", (1000, 500)).save(image_path)
    (sparse_dir / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 1000 500 1000 500 250\n", encoding="utf-8"
    )
    # Exact projections are x=725,y=375 and x=50,y=25. The stored observations
    # use the same points; the evaluator's target differs only by resize semantics.
    (sparse_dir / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame.png\n"
        "725 375 1 50 25 2\n",
        encoding="utf-8",
    )
    (sparse_dir / "points3D.txt").write_text(
        "1 2.25 1.25 10 0 0 0 0\n2 -4.5 -2.25 10 0 0 0 0\n",
        encoding="utf-8",
    )
    prediction = {
        "image": "frame.png",
        "image_path": image_path.as_posix(),
        "image_id": 1,
        "group_index": 0,
        "group_position": 0,
        "selected_for_first_wins": True,
        "prediction_file": "unused.npz",
        "intrinsic": vggt_intrinsic,
        "image_shape": [518, 518],
        "original_size": [1000, 500],
        "canvas_transform": {
            "scale_x": 0.518,
            "scale_y": 0.504,
            "pad_left": 0,
            "pad_top": 133,
            "pad_right": 0,
            "pad_bottom": 133,
            "resized_width": 518,
            "resized_height": 252,
        },
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
                        "radial_distortion": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return name, job_dir, index_path


def test_candidates_apply_pixel_center_and_invalid_vggt_fallback():
    production = FusionCamera(
        "PINHOLE",
        np.array([[518.0, 0.0, 259.0], [0.0, 504.0, 259.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (),
    )

    candidates, fallback = build_candidate_cameras(
        production=production,
        vggt_intrinsic=np.zeros((3, 3)),
        scale_x=0.518,
        scale_y=0.504,
    )

    assert np.isclose(candidates["pixel_center_colmap"].intrinsic[0, 2], 258.759)
    assert np.isclose(candidates["pixel_center_colmap"].intrinsic[1, 2], 258.752)
    assert np.array_equal(candidates["vggt"].intrinsic, production.intrinsic)
    assert fallback["vggt"] == "invalid_vggt_intrinsic_to_production_colmap"
    assert fallback["vggt_focal_colmap_center"] == "invalid_vggt_intrinsic_to_pixel_center_colmap"


def test_evaluator_uses_same_points_and_pixel_center_candidate_wins(tmp_path):
    scene = write_scene(
        tmp_path,
        "scene",
        vggt_intrinsic=[[518.0, 0.0, 259.0], [0.0, 504.0, 259.0], [0.0, 0.0, 1.0]],
    )

    result = evaluate_intrinsics_candidates([scene])
    candidates = result["scenes"][0]["candidates"]

    assert candidates["production_colmap"]["all"]["p50"] > 0.3
    assert candidates["pixel_center_colmap"]["all"]["max"] < 1e-4
    assert candidates["pixel_center_colmap"]["edge"]["max"] < 1e-4
    assert candidates["vggt"]["all"] == candidates["production_colmap"]["all"]
    assert result["decision_gate"]["g1_20_status"] == "blocked"
    assert result["protocol"]["ground_truth_used"] is False
