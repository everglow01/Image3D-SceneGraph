from __future__ import annotations

import copy

import pytest

from scripts import evaluate_absgrad_pair as pair


def test_paired_global_metrics_reject_missing_duplicate_and_nonfinite_views():
    ids = [str(i) for i in range(377)]
    left = {"per_view": [{"image_id": i, "psnr": 20.0, "ssim": 0.8} for i in ids]}
    right = {"per_view": [{"image_id": i, "psnr": 20.6, "ssim": 0.81} for i in reversed(ids)]}
    result = pair.paired_validation(left, right, ids)
    assert result["global_delta"]["psnr"]["mean"] == pytest.approx(0.6)
    assert result["global_delta"]["ssim"]["p10"] == pytest.approx(0.01)
    assert len(result["per_view"]) == 377
    for kind in ("missing", "duplicate", "nan"):
        bad = copy.deepcopy(right)
        if kind == "missing":
            bad["per_view"].pop()
        elif kind == "duplicate":
            bad["per_view"][0]["image_id"] = bad["per_view"][1]["image_id"]
        else:
            bad["per_view"][0]["psnr"] = float("nan")
        with pytest.raises(ValueError):
            pair.paired_validation(left, bad, ids)


def test_all_regions_required_and_visual_review_never_automatically_passes():
    regions = ["chair_legs_and_wheels", "carpet_texture", "wall_corner", "window_chairs",
               "table_edges_and_chairs", "window_and_curtain"]
    rows = [{"split": split, "image_id": f"{region}-{i}", "name": region,
             "delta": {"psnr": 0.6, "ssim": 0.02}}
            for split, count in (("train", 4), ("validation", 1))
            for region in regions for i in range(count)]
    gates = {
        "train_primary": {"per_region_group_mean_min_delta_psnr_db": 0.5,
            "per_region_group_mean_min_delta_ssim": 0.01, "required_directional_improvement_views_out_of_four": 3},
        "validation_primary": {"each_primary_roi_min_delta_psnr_db": 0.5, "each_primary_roi_min_delta_ssim": 0.01},
        "normal_control_train_group_and_validation_roi": {"min_delta_psnr_db": -0.1, "min_delta_ssim": -0.002},
        "global_validation": {"mean_min_delta_psnr_db": -0.05, "mean_min_delta_ssim": -0.001,
            "p10_min_delta_psnr_db": -0.1, "p10_min_delta_ssim": -0.002},
    }
    validation = {"global_delta": {key: {"mean": 0.0, "p10": 0.0} for key in ("psnr", "ssim")}}
    result = pair.numerical_gate(rows, validation, gates)
    assert result["numerical_passed"] and not result["promotion_eligible"]
    assert result["quality_decision"] == "pending_visual_review"
    bad = copy.deepcopy(rows)
    for row in bad[:2]:
        row["delta"] = {"psnr": -0.01, "ssim": -0.001}
    assert pair.numerical_gate(bad, validation, gates)["numerical_passed"] is False
    bad = copy.deepcopy(rows)
    next(row for row in bad if row["split"] == "validation" and row["name"] == "wall_corner")["delta"]["psnr"] = -0.101
    assert pair.numerical_gate(bad, validation, gates)["numerical_passed"] is False
    with pytest.raises(ValueError):
        pair.numerical_gate(rows[:-1], validation, gates)
