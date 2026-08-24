from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

MAX_REGISTERED_GAP_SECONDS = 2.0
MIN_VIDEO_REGISTERED_COUNT = 12
MIN_VIDEO_REGISTRATION_RATE = 0.70
MIN_VIDEO_TEMPORAL_COVERAGE = 0.80


def registered_gap_violations(
    timestamps: Iterable[float],
    *,
    maximum_gap_seconds: float = MAX_REGISTERED_GAP_SECONDS,
) -> list[dict[str, float]]:
    ordered = sorted(float(value) for value in timestamps)
    if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
        raise ValueError("maximum registered gap must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("registered timestamps must be finite and non-negative")
    return [
        {
            "start_seconds": left,
            "end_seconds": right,
            "seconds": right - left,
        }
        for left, right in zip(ordered, ordered[1:])
        if right - left > maximum_gap_seconds
    ]


def analyze_registration_timeline(
    selected_timestamps: Mapping[str, float],
    registered_names: Iterable[str],
    *,
    maximum_gap_seconds: float = MAX_REGISTERED_GAP_SECONDS,
) -> dict[str, Any]:
    if not selected_timestamps:
        raise ValueError("video selection metadata has no selected frames")
    selected = {str(name): float(value) for name, value in selected_timestamps.items()}
    if any(
        not name or not math.isfinite(timestamp) or timestamp < 0
        for name, timestamp in selected.items()
    ):
        raise ValueError("video selection timestamps are invalid")
    registered = sorted(set(selected) & {str(name) for name in registered_names})
    selected_times = sorted(selected.values())
    registered_times = sorted(selected[name] for name in registered)
    selected_span = selected_times[-1] - selected_times[0]
    registered_span = (
        registered_times[-1] - registered_times[0]
        if registered_times
        else 0.0
    )
    gaps = [
        right - left
        for left, right in zip(registered_times, registered_times[1:])
    ]
    violations = registered_gap_violations(
        registered_times,
        maximum_gap_seconds=maximum_gap_seconds,
    )
    return {
        "selected_count": len(selected),
        "registered_count": len(registered),
        "registered_names": registered,
        "registration_rate": len(registered) / len(selected),
        "temporal_coverage": registered_span / selected_span if selected_span > 0 else 0.0,
        "maximum_registered_gap_seconds": max(gaps, default=0.0),
        "maximum_registered_gap_threshold_seconds": maximum_gap_seconds,
        "gap_violations": violations,
        "gap_violation_count": len(violations),
        "gap_violation_total_seconds": sum(item["seconds"] for item in violations),
        "gap_violation_excess_seconds": sum(
            item["seconds"] - maximum_gap_seconds for item in violations
        ),
    }
