from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image3d_scenegraph.video.keyframes import (
    MAX_CANDIDATES,
    MAX_KEYFRAMES,
    STANDARD_V1,
    STANDARD_V2,
    V2_PROFILE_ID,
    VideoKeyframeError,
    _estimate_sparse_motion,
    base_keyframe_count,
    extract_video_keyframes,
    score_frame,
    select_keyframes,
    target_keyframe_count,
)


def test_ten_minute_video_limits() -> None:
    assert target_keyframe_count(10) == 60
    assert target_keyframe_count(60) == 360
    assert target_keyframe_count(240) == MAX_KEYFRAMES == 1_000
    assert target_keyframe_count(360) == 1_000
    assert target_keyframe_count(600) == 1_000
    assert target_keyframe_count(606) == 1_000
    assert MAX_CANDIDATES == 3_636


def test_standard_v2_duration_budgets() -> None:
    assert (base_keyframe_count(10, STANDARD_V2), target_keyframe_count(10, STANDARD_V2)) == (40, 50)
    assert (base_keyframe_count(60, STANDARD_V2), target_keyframe_count(60, STANDARD_V2)) == (240, 300)
    assert (
        base_keyframe_count(400.817, STANDARD_V2),
        target_keyframe_count(400.817, STANDARD_V2),
    ) == (1_603, 2_004)
    assert (base_keyframe_count(606, STANDARD_V2), target_keyframe_count(606, STANDARD_V2)) == (
        2_424,
        3_030,
    )


def test_frame_quality_and_selection_are_bounded() -> None:
    assert score_frame(np.zeros((320, 320), dtype=np.uint8))["black_fraction"] == 1.0
    checker = (np.indices((320, 320)).sum(axis=0) % 2 * 255).astype(np.uint8)
    assert score_frame(checker)["sharpness"] > 100

    candidates = []
    for index in range(120):
        candidates.append(
            {
                "candidate_index": index,
                "pts": index * 3 + 1,
                "time_seconds": index * 0.5 + 0.25,
                "mean_luma": 0.5,
                "black_fraction": 0.0,
                "white_fraction": 0.0,
                "entropy": 6.0 + index / 1000,
                "edge_density": 0.2,
                "sharpness": 100.0 + index,
                "tenengrad": 20.0,
                "exposure_score": 0.9,
                "perceptual_hash": index * 0x0101010101010101,
                "descriptor": np.full((32, 32), (index * 11) % 256, dtype=np.float32),
            }
        )
    selected = select_keyframes(candidates, 60.0)
    assert len(selected) <= 120
    assert [item["pts"] for item in selected] == sorted(item["pts"] for item in selected)


def test_standard_v1_pts_regression() -> None:
    candidates = []
    for index in range(240):
        descriptor = (
            np.arange(1024, dtype=np.float32).reshape(32, 32) * ((index % 7) + 1)
            + index * 17
        ) % 256
        candidates.append(
            _candidate(
                index,
                time_seconds=index / 12 + 1 / 24,
                descriptor=descriptor,
            )
        )
    selected = select_keyframes(candidates, 20.0, profile=STANDARD_V1)
    pts_hash = hashlib.sha256(
        json.dumps([item["pts"] for item in selected]).encode()
    ).hexdigest()
    assert pts_hash == "9c0ae528c11b761e342cbffa7d543735cc214a49af3ab404b6d566fdd1d6b4b2"


def test_standard_v2_selection_is_deterministic_and_motion_bounded() -> None:
    candidates = []
    for index in range(360):
        descriptor = (
            np.arange(1024, dtype=np.float32).reshape(32, 32) * ((index % 7) + 1)
            + index * 17
        ) % 256
        candidate = _candidate(
            index,
            time_seconds=(index + 0.5) / 6,
            descriptor=descriptor,
        )
        candidate.update(
            {
                "motion_status": "ok",
                "motion_detected_count": 100,
                "motion_tracked_count": 90,
                "motion_tracked_fraction": 0.9,
                "motion_displacement_median_normalized": 0.001
                if index < 180
                else 0.02 + (index % 6) / 1000,
                "motion_displacement_p90_normalized": 0.002
                if index < 180
                else 0.03 + (index % 6) / 1000,
            }
        )
        if index % 60 == 0:
            candidate["motion_status"] = "insufficient_tracks"
        candidates.append(candidate)

    first_candidates = deepcopy(candidates)
    first = select_keyframes(first_candidates, 60.0, profile=STANDARD_V2)
    second = select_keyframes(deepcopy(candidates), 60.0, profile=STANDARD_V2)
    assert [item["pts"] for item in first] == [item["pts"] for item in second]
    assert sum(item["selection_reason"] == "base" for item in first) == 240
    base_times = [
        float(item["time_seconds"])
        for item in first
        if item["selection_reason"] == "base"
    ]
    assert max(right - left for left, right in zip(base_times, base_times[1:])) <= 0.5
    adaptive = [item for item in first if item["selection_reason"] == "adaptive_motion"]
    assert len(first) <= 300
    assert len({math.floor(float(item["time_seconds"])) for item in adaptive}) == len(adaptive)
    assert any(
        item["motion_score_source"] == "descriptor_novelty"
        for item in first_candidates
    )


def test_sparse_optical_flow_translation_static_and_weak_texture() -> None:
    rng = np.random.default_rng(20260824)
    previous = rng.integers(0, 256, size=(320, 320), dtype=np.uint8)
    translated = np.zeros_like(previous)
    translated[3:, 5:] = previous[:-3, :-5]

    moving = _estimate_sparse_motion(previous, translated)
    static = _estimate_sparse_motion(previous, previous.copy())
    weak = _estimate_sparse_motion(np.zeros_like(previous), np.zeros_like(previous))

    assert moving["motion_status"] == "ok"
    assert moving["motion_tracked_count"] >= 12
    assert moving["motion_displacement_median_normalized"] > 0.01
    assert static["motion_status"] == "ok"
    assert static["motion_displacement_median_normalized"] < 1e-5
    assert weak["motion_status"] == "insufficient_tracks"
    assert weak["motion_detected_count"] == 0
    assert weak["motion_tracked_count"] == 0


def _candidate(
    index: int,
    *,
    time_seconds: float,
    descriptor: np.ndarray,
) -> dict[str, object]:
    return {
        "candidate_index": index,
        "pts": index * 5 + 7,
        "time_seconds": time_seconds,
        "mean_luma": 0.5,
        "black_fraction": 0.0,
        "white_fraction": 0.0,
        "entropy": 6.0 + (index % 11) / 100,
        "edge_density": 0.2,
        "sharpness": 100.0 + (index % 17) * 3,
        "tenengrad": 20.0,
        "exposure_score": 0.9,
        "perceptual_hash": index * 0x0101010101010101,
        "descriptor": descriptor,
    }


def test_insufficient_video_keyframes_is_explicit() -> None:
    candidates = [
        {
            "candidate_index": index,
            "pts": index,
            "time_seconds": index / 4,
            "mean_luma": 0.0,
            "black_fraction": 1.0,
            "white_fraction": 0.0,
            "entropy": 0.0,
            "edge_density": 0.0,
            "sharpness": 0.0,
            "tenengrad": 0.0,
            "exposure_score": 0.0,
            "perceptual_hash": 0,
            "descriptor": np.zeros((32, 32), dtype=np.float32),
        }
        for index in range(40)
    ]
    with pytest.raises(VideoKeyframeError, match="insufficient_video_keyframes"):
        select_keyframes(candidates, 10.0)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg tools are unavailable",
)
def test_extracts_upright_video_with_truthful_exif(tmp_path: Path) -> None:
    source = tmp_path / "portrait.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=480x720:rate=10:duration=10.2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    result = extract_video_keyframes(source, tmp_path / "out", longest_edge=1280)
    assert result["selection"]["selected_count"] == target_keyframe_count(10.2)
    timing = json.loads(
        (tmp_path / "out" / "diagnostics" / "video_keyframe_timing.json").read_text()
    )
    assert timing["profile"] == "video_keyframe_timing_v1"
    assert timing["video_profile"] == "video_keyframes_standard_v1"
    assert timing["elapsed_seconds"] > 0
    assert result["metrics"]["video_keyframe_elapsed_seconds"] == timing[
        "elapsed_seconds"
    ]
    selected = sorted((tmp_path / "out" / "frames" / "selected").glob("*.jpg"))
    assert len(selected) == target_keyframe_count(10.2)
    with Image.open(selected[0]) as image:
        assert image.height > image.width
        assert image.getexif().get(274) == 1
        assert "video_keyframes_standard_v1" in image.getexif().get(305)

    v2_result = extract_video_keyframes(
        source,
        tmp_path / "out-v2",
        longest_edge=1280,
        profile=STANDARD_V2,
    )
    v2_selection = v2_result["selection"]
    assert v2_selection["schema_version"] == 2
    assert v2_selection["profile"] == V2_PROFILE_ID
    assert v2_selection["base_selected_count"] == base_keyframe_count(10.2, STANDARD_V2)
    assert v2_selection["selected_count"] <= target_keyframe_count(10.2, STANDARD_V2)
    assert v2_selection["selected_count"] == (
        v2_selection["base_selected_count"] + v2_selection["adaptive_selected_count"]
    )
    v2_paths = sorted((tmp_path / "out-v2" / "frames" / "selected").glob("*.jpg"))
    assert all(re.fullmatch(r"frame_c\d{6}_pts-?\d+\.jpg", path.name) for path in v2_paths)
    assert {item["selection_reason"] for item in v2_selection["selected"]} <= {
        "base",
        "adaptive_motion",
    }
    v2_repeat = extract_video_keyframes(
        source,
        tmp_path / "out-v2-repeat",
        longest_edge=1280,
        profile=STANDARD_V2,
    )["selection"]
    assert [item["pts"] for item in v2_selection["selected"]] == [
        item["pts"] for item in v2_repeat["selected"]
    ]
    assert [item["sha256"] for item in v2_selection["selected"]] == [
        item["sha256"] for item in v2_repeat["selected"]
    ]
    with Image.open(v2_paths[0]) as image:
        assert image.getexif().get(274) == 1
        assert V2_PROFILE_ID in image.getexif().get(305)
