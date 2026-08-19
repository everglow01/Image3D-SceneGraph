from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image3d_scenegraph.video.keyframes import (
    MAX_CANDIDATES,
    MAX_KEYFRAMES,
    VideoKeyframeError,
    extract_video_keyframes,
    score_frame,
    select_keyframes,
    target_keyframe_count,
)


def test_ten_minute_video_limits() -> None:
    assert target_keyframe_count(10) == 60
    assert target_keyframe_count(60) == 360
    assert target_keyframe_count(240) == 1440
    assert target_keyframe_count(360) == 2160
    assert target_keyframe_count(600) == 3600
    assert target_keyframe_count(606) == MAX_KEYFRAMES == 3_636
    assert MAX_CANDIDATES == 7_272


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
    selected = sorted((tmp_path / "out" / "frames" / "selected").glob("*.jpg"))
    assert len(selected) == target_keyframe_count(10.2)
    with Image.open(selected[0]) as image:
        assert image.height > image.width
        assert image.getexif().get(274) == 1
        assert "video_keyframes_standard_v1" in image.getexif().get(305)
