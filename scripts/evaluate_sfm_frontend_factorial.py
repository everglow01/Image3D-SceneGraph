#!/usr/bin/env python3
"""Evaluate a frozen SIFT/ALIKED × Brute-force/LightGlue SfM matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


ARM_FRONTENDS = {
    "sift_bruteforce": ("sift_v1", "bruteforce"),
    "sift_lightglue": ("sift_v1", "lightglue"),
    "aliked_bruteforce": ("aliked_n16rot_v1", "bruteforce"),
    "aliked_lightglue": ("aliked_n16rot_v1", "lightglue"),
}
COMMON_FIELDS = (
    "colmap_build",
    "pairing",
    "geometric_verification",
    "camera_calibration_profile",
    "requested_mapper",
    "colmap_random_seed",
    "video_profile",
    "initial_video_selection_sha256",
    "v2_mapper_options",
    "v2_mapper_seed_count",
)


def evaluate_factorial(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(arms) != set(ARM_FRONTENDS):
        raise ValueError("factorial requires exactly four named frontend arms")
    normalized = {
        name: _validate_arm(name, dict(arms[name])) for name in ARM_FRONTENDS
    }
    reference = normalized["sift_bruteforce"]
    mismatches = []
    for name, arm in normalized.items():
        for field in COMMON_FIELDS:
            if arm[field] != reference[field]:
                mismatches.append({"arm": name, "field": field})
    if mismatches:
        return {
            "schema_version": 1,
            "profile": "sfm_frontend_factorial_v1",
            "status": "inconclusive",
            "conclusion": "incomparable_arm_contracts",
            "contract_mismatches": mismatches,
            "arms": normalized,
            "test_rgb_loaded": False,
        }

    passed = {name: bool(arm["primary_pose_health_passed"]) for name, arm in normalized.items()}
    a = passed["sift_bruteforce"]
    b = passed["sift_lightglue"]
    c = passed["aliked_bruteforce"]
    d = passed["aliked_lightglue"]
    if a and b and c and not d:
        conclusion = "aliked_lightglue_interaction_risk"
    elif a and c and not b and not d:
        conclusion = "lightglue_path_risk"
    elif a and b and not c and not d:
        conclusion = "aliked_feature_path_risk"
    elif not any(passed.values()):
        conclusion = "common_pipeline_or_scene_risk"
    elif all(passed.values()):
        conclusion = "failure_not_reproduced"
    else:
        conclusion = "mixed_result_inconclusive"
    global_recovered = bool(
        normalized["aliked_lightglue"]["global_recovery_passed"]
    )
    return {
        "schema_version": 1,
        "profile": "sfm_frontend_factorial_v1",
        "status": "complete",
        "conclusion": conclusion,
        "solver_sensitivity_evidence": (
            "aliked_lightglue_primary_failed_global_passed"
            if not d and global_recovered
            else "not_established"
        ),
        "arms": normalized,
        "test_rgb_loaded": False,
    }


def load_arm(root: Path) -> dict[str, Any]:
    contract_path = root / "diagnostics" / "sfm_frontend_contract.json"
    provenance = _read_json(
        contract_path
        if contract_path.is_file()
        else root / "diagnostics" / "colmap_timing.json"
    )
    recovery = _read_json(root / "diagnostics" / "sfm_pose_recovery.json")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("profile")
        not in {"sfm_frontend_contract_v1", "colmap_timing_v1"}
        or recovery.get("schema_version") != 1
        or recovery.get("profile") != "sfm_pose_recovery_v1"
    ):
        raise ValueError(f"arm evidence schema is unsupported: {root}")
    camera = provenance.get("camera_calibration")
    feature = provenance.get("feature")
    if not isinstance(camera, dict) or not isinstance(feature, dict):
        raise ValueError(f"arm provenance is incomplete: {root}")
    primary = recovery.get("primary_candidates")
    recovered = recovery.get("recovery_candidates")
    if not isinstance(primary, list) or not isinstance(recovered, list):
        raise ValueError(f"arm pose recovery record is invalid: {root}")
    return {
        "feature_profile": feature.get("profile"),
        "local_matcher_profile": feature.get("local_matcher_profile"),
        "colmap_build": provenance.get("colmap_build"),
        "pairing": provenance.get("pairing"),
        "geometric_verification": provenance.get("geometric_verification"),
        "camera_calibration_profile": camera.get("profile"),
        "requested_mapper": provenance.get("requested_mapper"),
        "colmap_random_seed": provenance.get("colmap_random_seed"),
        "video_profile": provenance.get("video_profile"),
        "initial_video_selection_sha256": provenance.get(
            "initial_video_selection_sha256"
        ),
        "v2_mapper_options": provenance.get("v2_mapper_options"),
        "v2_mapper_seed_count": provenance.get("v2_mapper_seed_count"),
        "primary_pose_health_passed": any(
            candidate.get("accepted") is True for candidate in primary
        ),
        "primary_reason_codes": sorted(
            {
                str(reason)
                for candidate in primary
                for reason in candidate.get("pose_health", {}).get(
                    "reason_codes", []
                )
            }
        ),
        "global_recovery_passed": any(
            candidate.get("kind") == "global_recovery_v1"
            and candidate.get("accepted") is True
            for candidate in recovered
        ),
    }


def _validate_arm(name: str, arm: dict[str, Any]) -> dict[str, Any]:
    expected_feature, expected_matcher = ARM_FRONTENDS[name]
    if arm.get("feature_profile") != expected_feature:
        raise ValueError(f"{name} has the wrong feature profile")
    if arm.get("local_matcher_profile") != expected_matcher:
        raise ValueError(f"{name} has the wrong local matcher profile")
    if arm.get("initial_video_selection_sha256") in {None, ""}:
        raise ValueError(f"{name} has no frozen video selection hash")
    if arm.get("colmap_random_seed") != 0:
        raise ValueError(f"{name} has no frozen COLMAP random seed")
    return arm


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read factorial evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"factorial evidence must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARM_FRONTENDS:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    arms = {
        name: load_arm(getattr(args, name))
        for name in ARM_FRONTENDS
    }
    report = evaluate_factorial(arms)
    _write_json(args.output, report)
    print(f"factorial_status={report['status']}")
    print(f"factorial_conclusion={report['conclusion']}")
    print(
        "solver_sensitivity_evidence="
        + str(report.get("solver_sensitivity_evidence", "not_established"))
    )
    print(f"factorial_report={args.output}")
    print("test_rgb_loaded=false")


if __name__ == "__main__":
    main()
