from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_g1_17a_scene_pressure import (  # noqa: E402
    PressureAnalysisError,
    analyze,
    concentration_records,
    fraction,
    load_sidecar,
    main,
    support_strata,
)
from run_vggt_pointcloud import write_ply  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(tmp_path: Path) -> Path:
    capture = tmp_path / "capture"
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    prediction_index = capture / "diagnostics/vggt_window_predictions.json"
    write_json(
        prediction_index,
        {
            "schema_version": 1,
            "group_count": 1,
            "unique_image_count": 2,
            "predictions": [
                {"image": "a.jpg"},
                {"image": "a.jpg"},
                {"image": "b.jpg"},
            ],
        },
    )
    write_json(capture / "diagnostics/vggt_groups.json", {"schema_version": 1, "image_count": 2})
    write_json(
        capture / "diagnostics/visibility_graph.json",
        {
            "image_count": 2,
            "images": [
                {"image": "a.jpg", "neighbors": [{"image": "b.jpg"}]},
                {"image": "b.jpg", "neighbors": []},
            ],
        },
    )
    overlap = tmp_path / "overlap.json"
    write_json(
        overlap,
        {
            "schema_version": 1,
            "aggregate": {
                "evaluated_pair_count": 1,
                "pair_metric_medians": {
                    "anchored_absolute_p50": 0.01,
                    "anchored_absolute_p90": 0.03,
                },
            }
        },
    )

    diagnostics = baseline / "diagnostics"
    diagnostics.mkdir(parents=True)
    sidecar = diagnostics / "support_points.npz"
    np.savez_compressed(
        sidecar,
        support_counts=np.array([1, 2, 0, 1], dtype=np.uint16),
        contradicted_counts=np.array([1, 0, 0, 0], dtype=np.uint16),
        overlap_disagreement=np.array([0.03, np.nan, 0.01, 0.04], dtype=np.float32),
        source_image_index=np.array([0, 0, 1, 1], dtype=np.int32),
        source_group_index=np.array([2, 2, 3, 3], dtype=np.int32),
        source_window_role=np.array([1, 2, 3, 3], dtype=np.uint8),
    )
    write_json(
        diagnostics / "support_points.json",
        {
            "schema_version": 1,
            "point_order": "exactly matches geometry/points.ply vertex order",
            "point_count": 4,
            "sidecar": sidecar.name,
            "sidecar_sha256": sha256(sidecar),
            "source_prediction_index": str(prediction_index.resolve()),
            "window_role_codes": {"reference": 1, "overlap": 2, "fresh": 3},
            "images": [
                {"source_image_index": 0, "image": "a.jpg"},
                {"source_image_index": 1, "image": "b.jpg"},
            ],
        },
    )
    write_json(
        diagnostics / "consistency.json",
        {
            "relative_threshold": 0.02,
            "candidate_points": 5,
            "accepted_points": 4,
            "occluded_only_points": 1,
            "not_observed_only_points": 1,
            "contradicted_only_points": 1,
            "supported_and_contradicted_points": 1,
        },
    )
    points = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    colors = np.zeros((4, 3), dtype=np.uint8)
    (baseline / "geometry").mkdir()
    write_ply(baseline / "geometry/points.ply", points, colors)
    write_json(
        candidate / "diagnostics/g1_17_support_policy.json",
        {
            "schema_version": 1,
            "filter": {"accepted_points": 3},
        },
    )

    roi = tmp_path / "rois.json"
    write_json(
        roi,
        {
            "schema_version": 1,
            "rois": [{"name": "monitor", "min": [-1, -1, -1], "max": [1, 1, 1]}],
        },
    )
    alignment = tmp_path / "alignment.json"
    write_json(alignment, {"transform": np.eye(4).tolist()})
    g1 = tmp_path / "g1.json"
    write_json(
        g1,
        {
            "scenes": {
                "private": {
                    "baseline": {
                        "output_points": 4,
                        "rois": {"monitor": {"coverage": 1.0, "layers": 2, "thickness": 0.2}},
                    },
                    "contradiction_free": {
                        "accepted_points": 3,
                        "rois": {"monitor": {"coverage": 0.9, "layers": 1, "thickness": 0.1}},
                    },
                }
            }
        },
    )
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "schema_version": 1,
            "g1_17_summary": str(g1),
            "private_scene": "private",
            "private_roi_definition": str(roi),
            "private_alignment": str(alignment),
            "scenes": {
                "private": {
                    "capture_dir": str(capture),
                    "overlap_diagnostics": str(overlap),
                    "baseline_dir": str(baseline),
                    "candidate_dir": str(candidate),
                }
            },
        },
    )
    return config


def test_helpers_report_strata_and_concentration():
    assert fraction(1, 4) == 0.25
    assert fraction(1, 0) == 0.0
    assert support_strata(np.array([0, 1, 2, 4])) == {"0": 1, "1": 1, "2_plus": 2}
    result = concentration_records(np.array([2, 1, 2, 2]), names={1: "a", 2: "b"})
    assert result["top_1_fraction"] == 0.75
    assert result["top"][0] == {"id": 2, "name": "b", "count": 3, "fraction": 0.75}


def test_analysis_attributes_private_removals_and_rois(tmp_path):
    payload = analyze(build_fixture(tmp_path))

    scene = payload["scenes"]["private"]
    assert scene["multi_prediction_image_fraction"] == 0.5
    assert scene["final_points"]["support_strata"] == {"0": 1, "1": 2, "2_plus": 1}
    attribution = payload["private_removal_attribution"]
    assert attribution["removed_points"] == 1
    assert attribution["source_images"]["top"][0]["name"] == "a.jpg"
    assert attribution["fixed_rois"]["monitor"]["baseline_points"] == 3
    assert attribution["fixed_rois"]["monitor"]["removed_points"] == 1


def test_sidecar_hash_mismatch_fails(tmp_path):
    config = build_fixture(tmp_path)
    value = json.loads(config.read_text())
    index_path = Path(value["scenes"]["private"]["baseline_dir"]) / "diagnostics/support_points.json"
    index = json.loads(index_path.read_text())
    index["sidecar_sha256"] = "0" * 64
    write_json(index_path, index)

    with pytest.raises(PressureAnalysisError, match="SHA-256 mismatch"):
        analyze(config)


def test_check_mode_is_deterministic(tmp_path, monkeypatch):
    config = build_fixture(tmp_path)
    output = tmp_path / "result.json"
    output.write_text(json.dumps(analyze(config), indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_g1_17a_scene_pressure.py", "--config", str(config), "--output", str(output), "--check"],
    )

    assert main() == 0
