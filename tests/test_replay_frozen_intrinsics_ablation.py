from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from replay_frozen_intrinsics_ablation import (  # noqa: E402
    build_candidate_camera,
    read_runner_log,
    resolve_support_policy,
    selected_prediction_records,
)
from run_colmap_vggt_dense import FusionCamera  # noqa: E402


def test_pixel_center_candidate_changes_only_principal_point():
    production = FusionCamera(
        "SIMPLE_RADIAL",
        np.array(
            [[518.0, 0.0, 259.0], [0.0, 504.0, 259.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        (0.02,),
    )

    candidate = build_candidate_camera(
        production,
        scale_x=0.518,
        scale_y=0.504,
        candidate="pixel_center_colmap",
    )

    assert candidate.model == production.model
    assert candidate.radial_distortion == production.radial_distortion
    assert np.array_equal(candidate.intrinsic[:2, :2], production.intrinsic[:2, :2])
    assert np.array_equal(candidate.intrinsic[2], production.intrinsic[2])
    assert np.isclose(candidate.intrinsic[0, 2], 258.759)
    assert np.isclose(candidate.intrinsic[1, 2], 258.752)
    assert np.array_equal(production.intrinsic, [[518.0, 0.0, 259.0], [0.0, 504.0, 259.0], [0.0, 0.0, 1.0]])


def test_first_wins_inventory_rejects_duplicate_selection():
    index = {
        "unique_image_count": 1,
        "predictions": [
            {"image": "frame.png", "selected_for_first_wins": True},
            {"image": "frame.png", "selected_for_first_wins": True},
        ],
    }

    with pytest.raises(ValueError, match="multiple first-wins predictions"):
        selected_prediction_records(index)


def test_support_policy_override_is_explicit_and_limited():
    assert resolve_support_policy("adaptive_two", None) == "adaptive_two"
    assert (
        resolve_support_policy("adaptive_two", "contradiction_free")
        == "contradiction_free"
    )
    with pytest.raises(ValueError, match="unsupported support-policy candidate"):
        resolve_support_policy("adaptive_two", "supported_only")


def test_runner_log_preserves_frozen_point_cap_and_seed(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("max_points=30000000\nseed=42\n", encoding="utf-8")

    assert read_runner_log(path) == {"max_points": "30000000", "seed": "42"}
