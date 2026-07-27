from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_vggt_overlap_predictions import (  # noqa: E402
    compare_prediction_pair,
    evaluate_overlap_predictions,
)


def prediction_record(
    *,
    filename: str,
    group_index: int,
    selected: bool,
    scale: float,
) -> dict[str, object]:
    return {
        "image": "frame.jpg",
        "image_path": "/ignored/frame.jpg",
        "image_id": 1,
        "group_index": group_index,
        "group_position": 0,
        "role": "reference" if group_index == 0 else "overlap",
        "group_selection": None,
        "selected_for_first_wins": selected,
        "prediction_file": filename,
        "image_shape": [8, 8],
        "original_size": [8, 4],
        "canvas_transform": {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "pad_left": 0,
            "pad_top": 2,
            "pad_right": 0,
            "pad_bottom": 2,
            "resized_width": 8,
            "resized_height": 4,
        },
        "sparse_scale_anchor": {
            "scale": scale,
            "observation_count": 100,
            "log_mad": 0.01,
        },
    }


def write_prediction(path: Path, depth: np.ndarray, confidence: np.ndarray) -> None:
    np.savez_compressed(
        path,
        depth=depth.astype(np.float32),
        confidence=confidence.astype(np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
    )


def test_pair_fit_recovers_scale_on_held_out_pixels_and_excludes_padding(tmp_path):
    prediction_dir = tmp_path / "predictions"
    heatmap_dir = tmp_path / "heatmaps"
    prediction_dir.mkdir()
    heatmap_dir.mkdir()
    first = prediction_record(filename="first.npz", group_index=0, selected=True, scale=1.0)
    second = prediction_record(filename="second.npz", group_index=1, selected=False, scale=1.0)
    first_depth = np.full((8, 8), 4.0)
    second_depth = np.full((8, 8), 2.0)
    first_depth[:2] = 1000.0
    second_depth[:2] = 0.001
    confidence = np.ones((8, 8))
    write_prediction(prediction_dir / "first.npz", first_depth, confidence)
    write_prediction(prediction_dir / "second.npz", second_depth, confidence)

    result = compare_prediction_pair(
        first_record=first,
        second_record=second,
        prediction_dir=prediction_dir,
        heatmap_dir=heatmap_dir,
        confidence_percentile=50.0,
        min_common_pixels=16,
        grid_size=4,
    )

    assert result["status"] == "evaluated"
    assert result["pixel_counts"]["padding_pixels"] == 32
    assert result["pixel_counts"]["common_reliable_pixels"] == 32
    assert np.isclose(result["scale_only_fit"]["second_depth_multiplier"], 2.0)
    assert result["scale_only_fit"]["held_out_log_depth"]["absolute_p90"] < 1e-6
    heatmaps = np.load(heatmap_dir / result["heatmap_file"])
    assert heatmaps.shape == (3, 4, 4)


def test_spatial_deformation_survives_scale_only_fit(tmp_path):
    prediction_dir = tmp_path / "predictions"
    heatmap_dir = tmp_path / "heatmaps"
    prediction_dir.mkdir()
    heatmap_dir.mkdir()
    first = prediction_record(filename="first.npz", group_index=0, selected=True, scale=1.0)
    second = prediction_record(filename="second.npz", group_index=1, selected=False, scale=1.0)
    first_depth = np.full((8, 8), 4.0)
    second_depth = np.full((8, 8), 4.0)
    second_depth[:, 4:] *= 1.5
    confidence = np.ones((8, 8))
    write_prediction(prediction_dir / "first.npz", first_depth, confidence)
    write_prediction(prediction_dir / "second.npz", second_depth, confidence)

    result = compare_prediction_pair(
        first_record=first,
        second_record=second,
        prediction_dir=prediction_dir,
        heatmap_dir=heatmap_dir,
        confidence_percentile=50.0,
        min_common_pixels=16,
        grid_size=4,
    )

    assert result["status"] == "evaluated"
    assert result["scale_only_fit"]["held_out_log_depth"]["absolute_p90"] > 0.19


def test_evaluator_is_deterministic_and_counts_all_same_image_pairs(tmp_path):
    index_dir = tmp_path / "diagnostics"
    prediction_dir = index_dir / "vggt_window_predictions"
    prediction_dir.mkdir(parents=True)
    records = [
        prediction_record(
            filename=f"prediction_{index}.npz",
            group_index=index,
            selected=index == 0,
            scale=1.0,
        )
        for index in range(3)
    ]
    for index, record in enumerate(records):
        write_prediction(
            prediction_dir / record["prediction_file"],
            np.full((8, 8), 2.0 + index),
            np.ones((8, 8)),
        )
    index_path = index_dir / "vggt_window_predictions.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_enabled": True,
                "fusion_policy": "first_wins",
                "grouping": "covisibility",
                "batch_size": 4,
                "requested_overlap_size": 2,
                "frames_chunk_size": 2,
                "group_count": 3,
                "registered_image_count": 1,
                "prediction_count": 3,
                "unique_image_count": 1,
                "overlap_image_count": 1,
                "max_predictions_per_image": 3,
                "predictions": records,
            }
        ),
        encoding="utf-8",
    )

    first = evaluate_overlap_predictions(
        index_path=index_path,
        heatmap_dir=tmp_path / "heatmaps",
        confidence_percentile=50.0,
        min_common_pixels=16,
        grid_size=4,
    )
    second = evaluate_overlap_predictions(
        index_path=index_path,
        heatmap_dir=tmp_path / "heatmaps",
        confidence_percentile=50.0,
        min_common_pixels=16,
        grid_size=4,
    )

    assert first == second
    assert first["aggregate"]["pair_count"] == 3
    assert first["aggregate"]["evaluated_pair_count"] == 3
    assert first["aggregate"]["first_wins_pair_count"] == 2
    assert first["decision_gate"]["scene_supports_scale_dominance"] is False
