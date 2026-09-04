from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts.evaluate_sfm_frontend_factorial import (
    ARM_FRONTENDS,
    evaluate_factorial,
    load_arm,
)


def _arms(statuses: dict[str, bool]) -> dict[str, dict]:
    common = {
        "colmap_build": "COLMAP 4.0.0",
        "pairing": "exhaustive",
        "geometric_verification": {
            "profile": "default_v1",
            "guided_matching": False,
        },
        "camera_calibration_profile": "shared_opencv_v1",
        "requested_mapper": "incremental",
        "colmap_random_seed": 0,
        "video_profile": "video_keyframes_standard_v1",
        "initial_video_selection_sha256": "a" * 64,
        "v2_mapper_options": [],
        "v2_mapper_seed_count": 0,
        "primary_reason_codes": [],
        "global_recovery_passed": False,
    }
    return {
        name: {
            **deepcopy(common),
            "feature_profile": feature,
            "local_matcher_profile": matcher,
            "primary_pose_health_passed": statuses[name],
        }
        for name, (feature, matcher) in ARM_FRONTENDS.items()
    }


@pytest.mark.parametrize(
    ("statuses", "conclusion"),
    [
        (
            {
                "sift_bruteforce": True,
                "sift_lightglue": True,
                "aliked_bruteforce": True,
                "aliked_lightglue": False,
            },
            "aliked_lightglue_interaction_risk",
        ),
        (
            {
                "sift_bruteforce": True,
                "sift_lightglue": False,
                "aliked_bruteforce": True,
                "aliked_lightglue": False,
            },
            "lightglue_path_risk",
        ),
        (
            {
                "sift_bruteforce": True,
                "sift_lightglue": True,
                "aliked_bruteforce": False,
                "aliked_lightglue": False,
            },
            "aliked_feature_path_risk",
        ),
        ({name: False for name in ARM_FRONTENDS}, "common_pipeline_or_scene_risk"),
        ({name: True for name in ARM_FRONTENDS}, "failure_not_reproduced"),
    ],
)
def test_factorial_classifies_only_complete_frozen_matrix(
    statuses, conclusion
) -> None:
    report = evaluate_factorial(_arms(statuses))

    assert report["status"] == "complete"
    assert report["conclusion"] == conclusion
    assert report["test_rgb_loaded"] is False


def test_factorial_reports_global_solver_sensitivity() -> None:
    statuses = {name: True for name in ARM_FRONTENDS}
    statuses["aliked_lightglue"] = False
    arms = _arms(statuses)
    arms["aliked_lightglue"]["global_recovery_passed"] = True

    report = evaluate_factorial(arms)

    assert report["solver_sensitivity_evidence"] == (
        "aliked_lightglue_primary_failed_global_passed"
    )


def test_factorial_rejects_incomparable_contract_without_claim() -> None:
    arms = _arms({name: True for name in ARM_FRONTENDS})
    arms["aliked_lightglue"]["pairing"] = "sequential_loop"

    report = evaluate_factorial(arms)

    assert report["status"] == "inconclusive"
    assert report["conclusion"] == "incomparable_arm_contracts"
    assert report["contract_mismatches"] == [
        {"arm": "aliked_lightglue", "field": "pairing"}
    ]


def test_factorial_loads_failed_arm_without_final_timing(tmp_path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    contract = {
        "schema_version": 1,
        "profile": "sfm_frontend_contract_v1",
        "feature": {
            "profile": "aliked_n16rot_v1",
            "local_matcher_profile": "lightglue",
        },
        "colmap_build": "COLMAP 4.0.0",
        "pairing": "exhaustive",
        "geometric_verification": {"profile": "default_v1"},
        "camera_calibration": {"profile": "shared_opencv_v1"},
        "requested_mapper": "incremental",
        "colmap_random_seed": 0,
        "video_profile": "video_keyframes_standard_v2",
        "initial_video_selection_sha256": "a" * 64,
        "v2_mapper_options": ["bounded"],
        "v2_mapper_seed_count": 1000,
    }
    recovery = {
        "schema_version": 1,
        "profile": "sfm_pose_recovery_v1",
        "primary_candidates": [
            {
                "accepted": False,
                "pose_health": {
                    "status": "failed",
                    "reason_codes": ["multiscale_camera_pose_branch"],
                },
            }
        ],
        "recovery_candidates": [],
    }
    (diagnostics / "sfm_frontend_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (diagnostics / "sfm_pose_recovery.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )

    arm = load_arm(tmp_path)

    assert arm["primary_pose_health_passed"] is False
    assert arm["primary_reason_codes"] == ["multiscale_camera_pose_branch"]
    assert arm["feature_profile"] == "aliked_n16rot_v1"


def test_factorial_separates_pose_health_from_product_gates(tmp_path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    contract = {
        "schema_version": 1,
        "profile": "sfm_frontend_contract_v1",
        "feature": {
            "profile": "sift_v1",
            "local_matcher_profile": "bruteforce",
        },
        "colmap_build": "COLMAP 4.0.0",
        "pairing": "exhaustive",
        "geometric_verification": {"profile": "default_v1"},
        "camera_calibration": {"profile": "shared_opencv_v1"},
        "requested_mapper": "incremental",
        "colmap_random_seed": 0,
        "video_profile": "video_keyframes_standard_v1",
        "initial_video_selection_sha256": "a" * 64,
        "v2_mapper_options": [],
        "v2_mapper_seed_count": 0,
    }
    recovery = {
        "schema_version": 1,
        "profile": "sfm_pose_recovery_v1",
        "primary_candidates": [
            {
                "accepted": False,
                "gate_reason_codes": ["registration_rate_below_gate"],
                "pose_health": {"status": "passed", "reason_codes": []},
            }
        ],
        "recovery_candidates": [],
    }
    (diagnostics / "sfm_frontend_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (diagnostics / "sfm_pose_recovery.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )

    arm = load_arm(tmp_path)

    assert arm["primary_pose_health_passed"] is True
    assert arm["primary_product_gate_passed"] is False
    assert arm["primary_gate_reason_codes"] == ["registration_rate_below_gate"]


def test_factorial_rejects_mislabeled_frontend_arm() -> None:
    arms = _arms({name: True for name in ARM_FRONTENDS})
    arms["sift_lightglue"]["local_matcher_profile"] = "bruteforce"

    with pytest.raises(ValueError, match="wrong local matcher"):
        evaluate_factorial(arms)
