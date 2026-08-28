from __future__ import annotations

import copy

from scripts.evaluate_video_v2_promotion import evaluate_promotion


STAGES = {
    "feature_extraction": 10.0,
    "feature_matching": 10.0,
    "mapping": 100.0,
    "initial_registration_expansion": 20.0,
    "registration_recovery": 20.0,
    "undistortion": 10.0,
    "point_cloud_conversion": 1.0,
    "text_conversion": 1.0,
}


def _selection(profile: str, *, recovery: bool = False) -> dict:
    candidate_count = 12 if profile.endswith("standard_v2") else 10
    candidates = [
        {
            "candidate_index": index,
            "pts": index * 10,
            "time_seconds": float(index),
            "quality_score": 0.5 + index / 100,
            "rejection_reason": None,
            "selected": index < 10 or recovery,
            "selection_reason": (
                "recovery_round_1" if recovery and index >= 10 else "base"
            ),
        }
        for index in range(candidate_count)
    ]
    selected = [
        {
            "candidate_index": item["candidate_index"],
            "pts": item["pts"],
            "time_seconds": item["time_seconds"],
            "path": f"frames/selected/frame_{item['candidate_index']:03d}.jpg",
            "width": 1280,
            "height": 720,
            "sha256": f"{item['candidate_index']:064x}",
            "selection_reason": item["selection_reason"],
        }
        for item in candidates
        if item["selected"]
    ]
    return {
        "schema_version": 2 if profile.endswith("standard_v2") else 1,
        "profile": profile,
        "source_sha256": "a" * 64,
        "candidates": candidates,
        "selected": selected,
        "selected_count": len(selected),
    }


def _keyframe_timing(profile: str, seconds: float) -> dict:
    return {
        "schema_version": 1,
        "profile": "video_keyframe_timing_v1",
        "video_profile": profile,
        "elapsed_seconds": seconds,
    }


def _colmap_timing(profile: str, seconds: float) -> dict:
    return {
        "schema_version": 1,
        "profile": "colmap_timing_v1",
        "video_profile": profile,
        "stage_elapsed_seconds": dict(STAGES),
        "total_elapsed_seconds": seconds,
    }


def _expansion() -> dict:
    return {
        "schema_version": 1,
        "profile": "video_initial_registration_expansion_v1",
        "initial": {
            "registered_count": 8,
            "sparse_point_count": 100,
        },
        "final": {
            "registered_count": 10,
            "sparse_point_count": 105,
        },
        "registered_camera_retention": {
            "initial_count": 8,
            "retained_count": 8,
            "lost_count": 0,
            "passed": True,
        },
    }


def _recovery() -> dict:
    return {
        "schema_version": 1,
        "initial_selected_count": 10,
        "recovery_selected_count": 2,
        "initial": {
            "registered_count": 10,
            "registration_rate": 0.9,
            "maximum_registered_gap_seconds": 5.0,
            "gap_violation_count": 1,
            "sparse_point_count": 100,
        },
        "final": {
            "registered_count": 12,
            "registration_rate": 0.96,
            "maximum_registered_gap_seconds": 1.5,
            "gap_violation_count": 0,
            "sparse_point_count": 105,
        },
        "registered_camera_retention": {
            "initial_count": 10,
            "retained_count": 10,
            "lost_count": 0,
            "passed": True,
        },
        "rounds": [{"round": 1}, {"round": 2}],
    }


def _inputs() -> dict:
    v1 = "video_keyframes_standard_v1"
    v2 = "video_keyframes_standard_v2"
    return {
        "baseline_selection": _selection(v1),
        "baseline_keyframe_timing": _keyframe_timing(v1, 10.0),
        "baseline_colmap_timing": _colmap_timing(v1, 90.0),
        "candidate_selection": _selection(v2, recovery=True),
        "repeat_selection": _selection(v2),
        "candidate_keyframe_timing": _keyframe_timing(v2, 15.0),
        "candidate_colmap_timing": _colmap_timing(v2, 180.0),
        "candidate_expansion": _expansion(),
        "candidate_recovery": _recovery(),
    }


def test_promotion_gate_passes_only_when_all_frozen_checks_pass() -> None:
    report = evaluate_promotion(**_inputs())

    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"])
    by_name = {check["name"]: check for check in report["checks"]}
    assert by_name["geometry_time_multiplier"]["actual"]["multiplier"] == 1.95
    assert by_name["selector_determinism"]["passed"] is True


def test_promotion_gate_reports_quality_and_time_failures() -> None:
    inputs = _inputs()
    inputs["candidate_recovery"]["final"].update(
        registration_rate=0.94,
        maximum_registered_gap_seconds=3.0,
        gap_violation_count=1,
    )
    inputs["candidate_colmap_timing"]["total_elapsed_seconds"] = 200.0

    report = evaluate_promotion(**inputs)

    assert report["passed"] is False
    failed = {
        check["name"] for check in report["checks"] if not check["passed"]
    }
    assert failed == {
        "registration_gap_count",
        "maximum_registered_gap_seconds",
        "registration_rate",
        "geometry_time_multiplier",
    }


def test_promotion_gate_measures_point_retention_from_mapper_input() -> None:
    inputs = _inputs()
    inputs["candidate_expansion"]["final"]["sparse_point_count"] = 91
    inputs["candidate_recovery"]["initial"]["sparse_point_count"] = 91
    inputs["candidate_recovery"]["final"]["sparse_point_count"] = 82

    report = evaluate_promotion(**inputs)

    check = next(
        check for check in report["checks"] if check["name"] == "sparse_point_retention"
    )
    assert check["passed"] is False
    assert check["actual"] == 0.82


def test_promotion_gate_detects_selector_nondeterminism() -> None:
    inputs = _inputs()
    inputs["repeat_selection"] = copy.deepcopy(inputs["repeat_selection"])
    inputs["repeat_selection"]["candidates"][3]["quality_score"] += 0.01

    report = evaluate_promotion(**inputs)

    assert report["passed"] is False
    check = next(
        check for check in report["checks"] if check["name"] == "selector_determinism"
    )
    assert check["passed"] is False
    assert check["actual"]["candidate_evidence_matches"] is False
