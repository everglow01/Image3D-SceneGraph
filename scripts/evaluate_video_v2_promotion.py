#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MAX_REGISTERED_GAP_SECONDS = 2.0
MIN_REGISTRATION_RATE = 0.95
MIN_POINT_RETENTION = 0.90
MAX_RECOVERY_ROUNDS = 2
MAX_RECOVERY_FRACTION = 0.50
MAX_GEOMETRY_TIME_MULTIPLIER = 2.0
V1_PROFILE = "video_keyframes_standard_v1"
V2_PROFILE = "video_keyframes_standard_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen standard_v2 video geometry promotion gates."
    )
    parser.add_argument("--baseline-selection", required=True, type=Path)
    parser.add_argument("--baseline-keyframe-timing", required=True, type=Path)
    parser.add_argument("--baseline-colmap-timing", required=True, type=Path)
    parser.add_argument("--candidate-selection", required=True, type=Path)
    parser.add_argument("--repeat-selection", required=True, type=Path)
    parser.add_argument("--candidate-keyframe-timing", required=True, type=Path)
    parser.add_argument("--candidate-colmap-timing", required=True, type=Path)
    parser.add_argument("--candidate-recovery", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = evaluate_promotion(
            baseline_selection=_read_json(args.baseline_selection),
            baseline_keyframe_timing=_read_json(args.baseline_keyframe_timing),
            baseline_colmap_timing=_read_json(args.baseline_colmap_timing),
            candidate_selection=_read_json(args.candidate_selection),
            repeat_selection=_read_json(args.repeat_selection),
            candidate_keyframe_timing=_read_json(args.candidate_keyframe_timing),
            candidate_colmap_timing=_read_json(args.candidate_colmap_timing),
            candidate_recovery=_read_json(args.candidate_recovery),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["passed"]:
        raise SystemExit(1)


def evaluate_promotion(
    *,
    baseline_selection: dict[str, Any],
    baseline_keyframe_timing: dict[str, Any],
    baseline_colmap_timing: dict[str, Any],
    candidate_selection: dict[str, Any],
    repeat_selection: dict[str, Any],
    candidate_keyframe_timing: dict[str, Any],
    candidate_colmap_timing: dict[str, Any],
    candidate_recovery: dict[str, Any],
) -> dict[str, Any]:
    _validate_selection(baseline_selection, V1_PROFILE)
    _validate_selection(candidate_selection, V2_PROFILE)
    _validate_selection(repeat_selection, V2_PROFILE)
    _validate_keyframe_timing(baseline_keyframe_timing, V1_PROFILE)
    _validate_keyframe_timing(candidate_keyframe_timing, V2_PROFILE)
    _validate_colmap_timing(baseline_colmap_timing, V1_PROFILE)
    _validate_colmap_timing(candidate_colmap_timing, V2_PROFILE)
    if candidate_recovery.get("schema_version") != 1:
        raise ValueError("candidate recovery diagnostics use an unsupported schema")

    final = candidate_recovery.get("final")
    initial = candidate_recovery.get("initial")
    retention = candidate_recovery.get("registered_camera_retention")
    rounds = candidate_recovery.get("rounds")
    if not isinstance(final, dict) or not isinstance(initial, dict):
        raise ValueError("candidate recovery diagnostics have no initial/final timelines")
    if not isinstance(retention, dict):
        raise ValueError("candidate recovery diagnostics have no camera-retention record")
    if not isinstance(rounds, list):
        raise ValueError("candidate recovery diagnostics have no round records")

    baseline_seconds = _positive_finite(
        baseline_keyframe_timing["elapsed_seconds"],
        "baseline keyframe elapsed_seconds",
    ) + _positive_finite(
        baseline_colmap_timing["total_elapsed_seconds"],
        "baseline COLMAP total_elapsed_seconds",
    )
    candidate_seconds = _positive_finite(
        candidate_keyframe_timing["elapsed_seconds"],
        "candidate keyframe elapsed_seconds",
    ) + _positive_finite(
        candidate_colmap_timing["total_elapsed_seconds"],
        "candidate COLMAP total_elapsed_seconds",
    )
    deterministic, deterministic_details = _compare_v2_selections(
        candidate_selection,
        repeat_selection,
    )

    initial_selected_count = int(candidate_recovery["initial_selected_count"])
    recovery_selected_count = int(candidate_recovery["recovery_selected_count"])
    point_retention = int(final["sparse_point_count"]) / max(
        int(initial["sparse_point_count"]), 1
    )
    time_multiplier = candidate_seconds / baseline_seconds
    required_timing_stages = {
        "feature_extraction",
        "feature_matching",
        "mapping",
        "registration_recovery",
        "undistortion",
        "point_cloud_conversion",
        "text_conversion",
    }
    candidate_stages = candidate_colmap_timing["stage_elapsed_seconds"]

    checks = [
        _check(
            "selector_determinism",
            deterministic,
            deterministic_details,
            "candidate evidence and initial selected records are identical",
        ),
        _check(
            "registration_gap_count",
            int(final["gap_violation_count"]) == 0,
            int(final["gap_violation_count"]),
            0,
        ),
        _check(
            "maximum_registered_gap_seconds",
            float(final["maximum_registered_gap_seconds"])
            <= MAX_REGISTERED_GAP_SECONDS,
            float(final["maximum_registered_gap_seconds"]),
            f"<= {MAX_REGISTERED_GAP_SECONDS}",
        ),
        _check(
            "registration_rate",
            float(final["registration_rate"]) >= MIN_REGISTRATION_RATE,
            float(final["registration_rate"]),
            f">= {MIN_REGISTRATION_RATE}",
        ),
        _check(
            "registered_camera_retention",
            bool(retention.get("passed"))
            and int(retention.get("lost_count", -1)) == 0,
            retention,
            "no initial registered camera lost",
        ),
        _check(
            "sparse_point_retention",
            point_retention >= MIN_POINT_RETENTION,
            point_retention,
            f">= {MIN_POINT_RETENTION}",
        ),
        _check(
            "recovery_round_limit",
            len(rounds) <= MAX_RECOVERY_ROUNDS,
            len(rounds),
            f"<= {MAX_RECOVERY_ROUNDS}",
        ),
        _check(
            "recovery_frame_budget",
            recovery_selected_count
            <= math.ceil(initial_selected_count * MAX_RECOVERY_FRACTION),
            {
                "initial_selected_count": initial_selected_count,
                "recovery_selected_count": recovery_selected_count,
                "fraction": recovery_selected_count / initial_selected_count,
            },
            f"<= {MAX_RECOVERY_FRACTION}",
        ),
        _check(
            "timing_diagnostics_complete",
            required_timing_stages <= set(candidate_stages),
            sorted(candidate_stages),
            sorted(required_timing_stages),
        ),
        _check(
            "geometry_time_multiplier",
            time_multiplier <= MAX_GEOMETRY_TIME_MULTIPLIER,
            {
                "baseline_seconds": baseline_seconds,
                "candidate_seconds": candidate_seconds,
                "multiplier": time_multiplier,
            },
            f"<= {MAX_GEOMETRY_TIME_MULTIPLIER}",
        ),
    ]
    return {
        "schema_version": 1,
        "profile": "video_standard_v2_promotion_gate_v1",
        "passed": all(bool(check["passed"]) for check in checks),
        "thresholds": {
            "maximum_registered_gap_seconds": MAX_REGISTERED_GAP_SECONDS,
            "minimum_registration_rate": MIN_REGISTRATION_RATE,
            "minimum_sparse_point_retention": MIN_POINT_RETENTION,
            "maximum_recovery_rounds": MAX_RECOVERY_ROUNDS,
            "maximum_recovery_fraction": MAX_RECOVERY_FRACTION,
            "maximum_geometry_time_multiplier": MAX_GEOMETRY_TIME_MULTIPLIER,
        },
        "checks": checks,
    }


def _compare_v2_selections(
    candidate: dict[str, Any], repeat: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    candidate_evidence = [_candidate_evidence(item) for item in candidate["candidates"]]
    repeat_evidence = [_candidate_evidence(item) for item in repeat["candidates"]]
    candidate_selected = [
        _selected_evidence(item)
        for item in candidate["selected"]
        if not str(item.get("selection_reason", "")).startswith("recovery_round_")
    ]
    repeat_selected = [_selected_evidence(item) for item in repeat["selected"]]
    source_matches = candidate.get("source_sha256") == repeat.get("source_sha256")
    candidates_match = candidate_evidence == repeat_evidence
    selected_match = candidate_selected == repeat_selected
    details = {
        "source_sha256_matches": source_matches,
        "candidate_evidence_matches": candidates_match,
        "initial_selected_records_match": selected_match,
        "candidate_count": len(candidate_evidence),
        "repeat_candidate_count": len(repeat_evidence),
        "initial_selected_count": len(candidate_selected),
        "repeat_selected_count": len(repeat_selected),
        "candidate_evidence_sha256": _json_sha256(candidate_evidence),
        "repeat_candidate_evidence_sha256": _json_sha256(repeat_evidence),
        "initial_selected_sha256": _json_sha256(candidate_selected),
        "repeat_selected_sha256": _json_sha256(repeat_selected),
    }
    return source_matches and candidates_match and selected_match, details


def _candidate_evidence(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("selection candidate records must be objects")
    return {
        key: value
        for key, value in item.items()
        if key not in {"selected", "selection_reason"}
    }


def _selected_evidence(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("selected frame records must be objects")
    return item


def _validate_selection(selection: dict[str, Any], profile: str) -> None:
    if selection.get("profile") != profile:
        raise ValueError(f"expected selection profile {profile}")
    if not isinstance(selection.get("candidates"), list):
        raise ValueError("selection has no candidate records")
    if not isinstance(selection.get("selected"), list) or not selection["selected"]:
        raise ValueError("selection has no selected records")


def _validate_keyframe_timing(timing: dict[str, Any], profile: str) -> None:
    if (
        timing.get("schema_version") != 1
        or timing.get("profile") != "video_keyframe_timing_v1"
        or timing.get("video_profile") != profile
    ):
        raise ValueError(f"invalid keyframe timing diagnostics for {profile}")


def _validate_colmap_timing(timing: dict[str, Any], profile: str) -> None:
    if (
        timing.get("schema_version") != 1
        or timing.get("profile") != "colmap_timing_v1"
        or timing.get("video_profile") != profile
        or not isinstance(timing.get("stage_elapsed_seconds"), dict)
    ):
        raise ValueError(f"invalid COLMAP timing diagnostics for {profile}")


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _check(name: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


if __name__ == "__main__":
    main()
