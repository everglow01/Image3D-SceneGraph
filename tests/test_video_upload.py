from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.main as api
from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput
from image3d_scenegraph.video import keyframes


@pytest.mark.parametrize("size", [2 * 1024**3 + 1, 12 * 1024**3, 16 * 1024**3, 16 * 1024**3 + 1])
def test_large_video_size_boundary_across_entrypoints(tmp_path, monkeypatch, size):
    assert api.MAX_VIDEO_BYTES == keyframes.MAX_VIDEO_BYTES == 16 * 1024**3
    store = JobStore(tmp_path / "jobs")
    uploaded = UploadedInput(filename="DSC_2385.MOV", staged_path=tmp_path / "source", size_bytes=size)
    source = MagicMock(spec=Path)
    source.name = "DSC_2385.MOV"
    source.is_file.return_value = True
    source.stat.return_value.st_size = size
    payload = {
        "streams": [{
            "codec_type": "video", "codec_name": "hevc", "duration": "601",
            "width": 3840, "height": 2160, "avg_frame_rate": "30/1",
        }],
        "format": {"format_name": "mov,mp4"},
    }
    monkeypatch.setattr(
        subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(payload)),
    )
    monkeypatch.setattr(keyframes, "sha256_file", lambda _path: "a" * 64)
    if size <= keyframes.MAX_VIDEO_BYTES:
        store._validate_request("video", [uploaded])
        probe = keyframes.probe_video(source, ffprobe="ffprobe")
        assert probe["source"]["size_bytes"] == size
        assert probe["codec"] == "hevc"
        assert (probe["display_width"], probe["display_height"]) == (3840, 2160)
    else:
        with pytest.raises(JobError, match="16 GiB"):
            store._validate_request("video", [uploaded])
        with pytest.raises(keyframes.VideoKeyframeError, match="16 GiB"):
            keyframes.probe_video(source, ffprobe="ffprobe")


@pytest.mark.parametrize("oversize", [False, True])
def test_video_upload_is_chunked_and_cleans_up_overflow(tmp_path, monkeypatch, oversize):
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(api, "MAX_VIDEO_BYTES", 16)
    chunks = [b"a" * 8, b"b" * 8, b"c" if oversize else b""]
    upload = MagicMock()
    upload.filename = "DSC_2385.MOV"
    upload.content_type = "video/quicktime"
    upload.read = AsyncMock(side_effect=chunks)
    if oversize:
        with pytest.raises(JobError, match="video exceeds"):
            asyncio.run(api._stage_video_upload(store, upload))
        assert list((store.output_root / ".uploads").iterdir()) == []
    else:
        staged = asyncio.run(api._stage_video_upload(store, upload))
        assert staged.content is None
        assert staged.size_bytes == 16
        assert staged.staged_path.read_bytes() == b"a" * 8 + b"b" * 8
        assert staged.sha256 == hashlib.sha256(b"a" * 8 + b"b" * 8).hexdigest()
    assert all(call.args == (8 * 1024 * 1024,) for call in upload.read.call_args_list)
