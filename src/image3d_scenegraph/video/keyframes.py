from __future__ import annotations

import json
import math
import os
import queue
import re
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from image3d_scenegraph.file_integrity import sha256_file

STANDARD_V1 = "standard_v1"
STANDARD_V2 = "standard_v2"
DEFAULT_VIDEO_PROFILE = STANDARD_V2
V1_PROFILE_ID = "video_keyframes_standard_v1"
V2_PROFILE_ID = "video_keyframes_standard_v2"
PROFILE_ID = V2_PROFILE_ID
VIDEO_PROFILES = {STANDARD_V1, STANDARD_V2}
MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 606.0
MAX_VIDEO_BYTES = 2 * 1024**3
CANDIDATE_FPS = 6
MAX_CANDIDATES = 3_636
MAX_KEYFRAMES = 1_000
MIN_KEYFRAMES = 24
V2_BASE_FPS = 4
V2_ADAPTIVE_MAX_FPS = 5
ANALYSIS_SIZE = 320
EXTRACTION_DEADLINE_SECONDS = 30 * 60
VALID_ROTATIONS = {"auto", "clockwise_90", "counterclockwise_90", "180"}
_SHOWINFO = re.compile(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:([^\s]+)")


class VideoKeyframeError(ValueError):
    """Raised when video input or selected keyframes violate the profile."""


def target_keyframe_count(duration_seconds: float, profile: str = DEFAULT_VIDEO_PROFILE) -> int:
    if not math.isfinite(duration_seconds):
        raise VideoKeyframeError("video duration must be finite")
    if profile == STANDARD_V1:
        return min(MAX_KEYFRAMES, max(MIN_KEYFRAMES, round(duration_seconds * 6)))
    if profile == STANDARD_V2:
        return max(MIN_KEYFRAMES, round(duration_seconds * V2_ADAPTIVE_MAX_FPS))
    raise VideoKeyframeError(f"unsupported video keyframe profile: {profile}")


def base_keyframe_count(duration_seconds: float, profile: str = DEFAULT_VIDEO_PROFILE) -> int:
    if not math.isfinite(duration_seconds):
        raise VideoKeyframeError("video duration must be finite")
    if profile == STANDARD_V1:
        return target_keyframe_count(duration_seconds, profile)
    if profile == STANDARD_V2:
        return max(MIN_KEYFRAMES, round(duration_seconds * V2_BASE_FPS))
    raise VideoKeyframeError(f"unsupported video keyframe profile: {profile}")


def _profile_id(profile: str) -> str:
    if profile == STANDARD_V1:
        return V1_PROFILE_ID
    if profile == STANDARD_V2:
        return V2_PROFILE_ID
    raise VideoKeyframeError(f"unsupported video keyframe profile: {profile}")


def resolve_video_tools() -> tuple[str, str]:
    ffmpeg_override = os.environ.get("IMAGE3D_FFMPEG_BIN")
    ffprobe_override = os.environ.get("IMAGE3D_FFPROBE_BIN")
    ffmpeg = shutil.which(ffmpeg_override or "ffmpeg")
    ffprobe = shutil.which(ffprobe_override or "ffprobe")
    if not ffmpeg or not ffprobe:
        raise VideoKeyframeError("ffmpeg and ffprobe executables are required for video input")
    return ffmpeg, ffprobe


def probe_video(
    source: Path,
    *,
    rotation_override: str = "auto",
    ffprobe: str | None = None,
    profile: str = DEFAULT_VIDEO_PROFILE,
) -> dict[str, Any]:
    profile_id = _profile_id(profile)
    if rotation_override not in VALID_ROTATIONS:
        raise VideoKeyframeError(f"unsupported video rotation: {rotation_override}")
    if not source.is_file():
        raise VideoKeyframeError(f"video input is missing: {source}")
    size_bytes = source.stat().st_size
    if size_bytes > MAX_VIDEO_BYTES:
        raise VideoKeyframeError("video exceeds the 2 GiB limit")
    if size_bytes == 0:
        raise VideoKeyframeError("video input is empty")
    if ffprobe is None:
        _, ffprobe = resolve_video_tools()
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=60
        )
        payload = json.loads(completed.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise VideoKeyframeError("ffprobe could not parse the uploaded video") from exc

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoKeyframeError("ffprobe returned no video streams")
    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
        and not int((stream.get("disposition") or {}).get("attached_pic", 0))
    ]
    if len(video_streams) != 1:
        raise VideoKeyframeError("video must contain exactly one primary video stream")
    stream = video_streams[0]
    duration = _finite_float(stream.get("duration"))
    if duration is None:
        duration = _finite_float((payload.get("format") or {}).get("duration"))
    if duration is None:
        raise VideoKeyframeError("video duration is unavailable")
    if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
        raise VideoKeyframeError("video duration must be between 10 seconds and 10 minutes")

    width = _positive_int(stream.get("width"), "video width")
    height = _positive_int(stream.get("height"), "video height")
    rotation = _resolve_rotation(stream, rotation_override)
    display_width, display_height = (
        (height, width) if rotation["quarter_turn"] else (width, height)
    )
    if min(display_width, display_height) < 480 or max(display_width, display_height) < 720:
        raise VideoKeyframeError("display resolution must be at least 480x720")
    if max(display_width, display_height) > 4096 or display_width * display_height > 12_000_000:
        raise VideoKeyframeError("video resolution exceeds the 4096px or 12MP limit")
    frame_rate = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    if frame_rate is None or not 1 <= frame_rate <= 120:
        raise VideoKeyframeError("average video frame rate must be between 1 and 120 fps")

    format_payload = payload.get("format") or {}
    container = str(format_payload.get("format_name", "")).split(",")[0]
    if container not in {"mov", "mp4", "matroska", "webm"}:
        raise VideoKeyframeError(f"unsupported video container: {container or 'unknown'}")
    tags = {**(format_payload.get("tags") or {}), **(stream.get("tags") or {})}
    safe_tags = {
        str(key): value
        for key, value in tags.items()
        if "location" not in str(key).lower() and "gps" not in str(key).lower()
    }
    location_hidden = len(safe_tags) != len(tags)
    return {
        "schema_version": 1,
        "profile": profile_id,
        "source": {
            "filename": source.name,
            "size_bytes": size_bytes,
            "sha256": sha256_file(source),
        },
        "container": container,
        "codec": stream.get("codec_name"),
        "duration_seconds": duration,
        "time_base": stream.get("time_base"),
        "average_frame_rate": frame_rate,
        "source_width": width,
        "source_height": height,
        "display_width": display_width,
        "display_height": display_height,
        "orientation": _orientation(display_width, display_height),
        "rotation": rotation,
        "sample_aspect_ratio": stream.get("sample_aspect_ratio"),
        "display_aspect_ratio": stream.get("display_aspect_ratio"),
        "color": {
            key: stream.get(key)
            for key in ("color_range", "color_space", "color_transfer", "color_primaries")
        },
        "metadata": safe_tags,
        "location_metadata": "present_but_hidden" if location_hidden else "absent",
        "ignored_stream_counts": {
            kind: sum(1 for item in streams if item.get("codec_type") == kind)
            for kind in ("audio", "subtitle")
        },
    }


def extract_video_keyframes(
    source: Path,
    output_root: Path,
    *,
    longest_edge: int,
    rotation_override: str = "auto",
    profile: str = DEFAULT_VIDEO_PROFILE,
    progress: Callable[[str, float], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile_id = _profile_id(profile)
    ffmpeg, ffprobe = resolve_video_tools()
    _progress(progress, "video_probing", 0.06)
    probe = probe_video(
        source,
        rotation_override=rotation_override,
        ffprobe=ffprobe,
        profile=profile,
    )
    diagnostics_dir = output_root / "diagnostics"
    frames_dir = output_root / "frames"
    selected_dir = frames_dir / "selected"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    probe_path = diagnostics_dir / "video_probe.json"
    _write_json(probe_path, probe)

    deadline = time.monotonic() + EXTRACTION_DEADLINE_SECONDS
    _progress(progress, "video_frame_scoring", 0.08)
    rotation_filter = _rotation_filter(probe["rotation"]["applied_degrees"])
    filters = [
        f"select='gte(t,(selected_n+0.5)/{CANDIDATE_FPS})'",
        "showinfo",
        *rotation_filter,
        f"scale={ANALYSIS_SIZE}:{ANALYSIS_SIZE}:flags=area",
    ]
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "info",
        "-noautorotate",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        ",".join(filters),
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    candidates = _score_stream(
        command,
        width=ANALYSIS_SIZE,
        height=ANALYSIS_SIZE,
        deadline=deadline,
        cancel_requested=cancel_requested,
        estimate_motion=profile == STANDARD_V2,
    )
    if len(candidates) > MAX_CANDIDATES:
        raise VideoKeyframeError(
            f"candidate frame limit exceeded ({len(candidates)} > {MAX_CANDIDATES})"
        )
    selected = select_keyframes(
        candidates,
        float(probe["duration_seconds"]),
        profile=profile,
    )

    _progress(progress, "video_frame_extraction", 0.12)
    output_width, output_height = _scaled_dimensions(
        int(probe["display_width"]), int(probe["display_height"]), longest_edge
    )
    _materialize_selected(
        ffmpeg,
        source,
        selected,
        selected_dir,
        rotation_filter=rotation_filter,
        width=output_width,
        height=output_height,
        deadline=deadline,
        cancel_requested=cancel_requested,
        profile_id=profile_id,
    )
    contact_sheet = diagnostics_dir / "video_keyframes.jpg"
    _write_contact_sheet(selected_dir, contact_sheet)

    selected_pts = {int(item["pts"]) for item in selected}
    records = []
    for candidate in candidates:
        record = {key: value for key, value in candidate.items() if key != "descriptor"}
        record["selected"] = int(candidate["pts"]) in selected_pts
        records.append(record)
    selected_records = []
    selected_by_pts = {int(item["pts"]): item for item in selected}
    for path in sorted(selected_dir.glob("*.jpg")):
        pts = int(path.stem.split("_pts")[-1])
        source_record = selected_by_pts[pts]
        with Image.open(path) as image:
            selected_record = {
                "candidate_index": source_record["candidate_index"],
                "pts": pts,
                "time_seconds": source_record["time_seconds"],
                "path": path.relative_to(output_root).as_posix(),
                "width": image.width,
                "height": image.height,
                "sha256": sha256_file(path),
                "exif": {
                    "orientation": image.getexif().get(274),
                    "software": image.getexif().get(305),
                },
            }
            if profile == STANDARD_V2:
                selected_record["selection_reason"] = source_record["selection_reason"]
            selected_records.append(selected_record)
    rejection_counts: dict[str, int] = {}
    for item in records:
        reason = item.get("rejection_reason")
        if reason:
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
    selection = {
        "schema_version": 1,
        "profile": profile_id,
        "source_sha256": probe["source"]["sha256"],
        "duration_seconds": probe["duration_seconds"],
        "candidate_fps": CANDIDATE_FPS,
        "candidate_count": len(candidates),
        "target_count": target_keyframe_count(
            float(probe["duration_seconds"]), profile
        ),
        "selected_count": len(selected_records),
        "rejection_counts": rejection_counts,
        "rotation": probe["rotation"],
        "candidates": records,
        "selected": selected_records,
    }
    if profile == STANDARD_V2:
        base_selected_count = sum(
            item.get("selection_reason") == "base" for item in selected_records
        )
        adaptive_selected_count = sum(
            item.get("selection_reason") == "adaptive_motion"
            for item in selected_records
        )
        selection.update(
            {
                "schema_version": 2,
                "profile_parameters": {
                    "candidate_fps": CANDIDATE_FPS,
                    "base_fps": V2_BASE_FPS,
                    "adaptive_max_fps": V2_ADAPTIVE_MAX_FPS,
                },
                "motion_estimator": {
                    "name": "sparse_lucas_kanade",
                    "version": 1,
                    "analysis_size": ANALYSIS_SIZE,
                    "opencv_version": cv2.__version__,
                    "robust_baseline_score": float(
                        np.median(
                            [
                                float(item["motion_score"])
                                for item in candidates
                                if "motion_score" in item
                            ]
                        )
                    ),
                },
                "base_target_count": base_keyframe_count(
                    float(probe["duration_seconds"]), profile
                ),
                "base_selected_count": base_selected_count,
                "adaptive_selected_count": adaptive_selected_count,
            }
        )
    selection_path = frames_dir / "selection.json"
    _write_json(selection_path, selection)
    metrics: dict[str, Any] = {
        "video_profile": profile_id,
        "video_duration_seconds": float(probe["duration_seconds"]),
        "video_orientation": str(probe["orientation"]),
        "video_rotation_degrees": int(probe["rotation"]["applied_degrees"]),
        "video_source_width": int(probe["source_width"]),
        "video_source_height": int(probe["source_height"]),
        "video_display_width": int(probe["display_width"]),
        "video_display_height": int(probe["display_height"]),
        "video_candidate_count": len(candidates),
        "video_selected_count": len(selected_records),
        "video_rejection_counts": json.dumps(rejection_counts, sort_keys=True),
    }
    if profile == STANDARD_V2:
        metrics.update(
            {
                "video_base_selected_count": base_selected_count,
                "video_adaptive_selected_count": adaptive_selected_count,
            }
        )
    timing_path = diagnostics_dir / "video_keyframe_timing.json"
    elapsed_seconds = time.perf_counter() - started_at
    _write_json(
        timing_path,
        {
            "schema_version": 1,
            "profile": "video_keyframe_timing_v1",
            "video_profile": profile_id,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    metrics["video_keyframe_elapsed_seconds"] = elapsed_seconds
    _progress(progress, "video_frame_extraction", 0.14)
    return {
        "probe": probe,
        "selection": selection,
        "assets": {
            "video_probe": probe_path.relative_to(output_root).as_posix(),
            "video_frame_selection": selection_path.relative_to(output_root).as_posix(),
            "video_keyframe_contact_sheet": contact_sheet.relative_to(output_root).as_posix(),
            "video_keyframe_timing": timing_path.relative_to(output_root).as_posix(),
        },
        "metrics": metrics,
    }


def materialize_video_candidates(
    source: Path,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    selection: dict[str, Any],
) -> list[Path]:
    if selection.get("profile") != V2_PROFILE_ID:
        raise VideoKeyframeError("incremental materialization requires standard_v2")
    selected = selection.get("selected")
    if not isinstance(selected, list) or not selected:
        raise VideoKeyframeError("video selection metadata has no selected frames")
    try:
        width = int(selected[0]["width"])
        height = int(selected[0]["height"])
        rotation_degrees = int(selection["rotation"]["applied_degrees"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoKeyframeError("video selection materialization metadata is invalid") from exc
    if not candidates:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg, _ = resolve_video_tools()
    _materialize_selected(
        ffmpeg,
        source,
        candidates,
        output_dir,
        rotation_filter=_rotation_filter(rotation_degrees),
        width=width,
        height=height,
        deadline=time.monotonic() + EXTRACTION_DEADLINE_SECONDS,
        cancel_requested=None,
        profile_id=V2_PROFILE_ID,
    )
    paths = [output_dir / candidate_frame_filename(item) for item in candidates]
    if any(not path.is_file() for path in paths):
        raise VideoKeyframeError("incremental keyframe materialization is incomplete")
    return paths


def candidate_frame_filename(candidate: dict[str, Any]) -> str:
    return f"frame_c{int(candidate['candidate_index']):06d}_pts{int(candidate['pts'])}.jpg"


def score_frame(gray: np.ndarray) -> dict[str, float]:
    values = gray.astype(np.float32)
    mean = float(values.mean() / 255.0)
    black_fraction = float(np.mean(values <= 4))
    white_fraction = float(np.mean(values >= 251))
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    gx = np.abs(np.diff(values, axis=1))
    gy = np.abs(np.diff(values, axis=0))
    edge_density = float((np.mean(gx > 20) + np.mean(gy > 20)) / 2)
    laplacian = (
        -4 * values[1:-1, 1:-1]
        + values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
    )
    sharpness = float(np.var(laplacian))
    tenengrad = float(np.mean(gx) + np.mean(gy))
    clipped = black_fraction + white_fraction
    exposure = max(0.0, 1.0 - abs(mean - 0.5) * 1.5 - clipped)
    return {
        "mean_luma": mean,
        "black_fraction": black_fraction,
        "white_fraction": white_fraction,
        "entropy": entropy,
        "edge_density": edge_density,
        "sharpness": sharpness,
        "tenengrad": tenengrad,
        "exposure_score": exposure,
    }


def select_keyframes(
    candidates: list[dict[str, Any]],
    duration_seconds: float,
    *,
    profile: str = DEFAULT_VIDEO_PROFILE,
) -> list[dict[str, Any]]:
    _profile_id(profile)
    if not candidates:
        raise VideoKeyframeError("video decoding produced no candidate frames")
    sharp_values = np.asarray([item["sharpness"] for item in candidates], dtype=np.float64)
    adaptive_blur = max(6.0, float(np.percentile(sharp_values, 10)) * 0.25)
    recent_accepted: list[dict[str, Any]] = []
    viable: list[dict[str, Any]] = []
    previous_descriptor: np.ndarray | None = None
    for item in candidates:
        reasons: list[str] = []
        if item["mean_luma"] < 0.02 or item["black_fraction"] > 0.80:
            reasons.append("near_black")
        if item["mean_luma"] > 0.98 or item["white_fraction"] > 0.80:
            reasons.append("near_white")
        if item["entropy"] < 1.5 and item["edge_density"] < 0.005:
            reasons.append("blank_or_textureless")
        if item["sharpness"] < adaptive_blur and item["tenengrad"] < 4.0:
            reasons.append("severe_blur")
        duplicate = any(_near_duplicate(item, prior) for prior in recent_accepted[-3:])
        if duplicate:
            reasons.append("duplicate")
        novelty = (
            1.0
            if previous_descriptor is None
            else min(1.0, float(np.mean(np.abs(item["descriptor"] - previous_descriptor))) / 48.0)
        )
        item["novelty"] = novelty
        if novelty > 0.75 and item["sharpness"] < adaptive_blur * 2:
            reasons.append("unstable_transition")
        item["rejection_reasons"] = reasons
        item["rejection_reason"] = reasons[0] if reasons else None
        previous_descriptor = item["descriptor"]
        if not reasons:
            viable.append(item)
            recent_accepted.append(item)

    if len(viable) < MIN_KEYFRAMES:
        raise VideoKeyframeError(
            f"insufficient_video_keyframes: {len(viable)} accepted, {MIN_KEYFRAMES} required"
        )
    _assign_quality_scores(viable)
    if profile == STANDARD_V1:
        selected = _select_uniform_keyframes(
            viable,
            min(target_keyframe_count(duration_seconds, profile), len(viable)),
            duration_seconds,
        )
    else:
        _assign_motion_scores(viable)
        base_target = min(
            base_keyframe_count(duration_seconds, profile),
            len(viable),
        )
        selected = _select_uniform_keyframes(
            viable,
            base_target,
            duration_seconds,
        )
        for item in selected:
            item["selection_reason"] = "base"
        selected_pts = {int(item["pts"]) for item in selected}
        adaptive_budget = min(
            target_keyframe_count(duration_seconds, profile),
            len(viable),
        ) - len(selected)
        if adaptive_budget > 0:
            adaptive = _select_adaptive_keyframes(
                [item for item in viable if int(item["pts"]) not in selected_pts],
                duration_seconds,
                adaptive_budget,
            )
            for item in adaptive:
                item["selection_reason"] = "adaptive_motion"
            selected.extend(adaptive)
            selected.sort(
                key=lambda item: (float(item["time_seconds"]), int(item["pts"]))
            )
    coverage = (float(selected[-1]["time_seconds"]) - float(selected[0]["time_seconds"])) / duration_seconds
    if coverage < 0.80:
        raise VideoKeyframeError(
            f"insufficient_video_temporal_coverage: {coverage:.3f} < 0.800"
        )
    return selected


def _select_uniform_keyframes(
    viable: list[dict[str, Any]],
    target: int,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(target)]
    for item in viable:
        index = min(
            target - 1,
            int(float(item["time_seconds"]) / duration_seconds * target),
        )
        buckets[index].append(item)
    selected = [max(bucket, key=_selection_key) for bucket in buckets if bucket]
    selected_pts = {int(item["pts"]) for item in selected}
    if len(selected) < target:
        remaining = [item for item in viable if int(item["pts"]) not in selected_pts]
        remaining.sort(key=_selection_key, reverse=True)
        selected.extend(remaining[: target - len(selected)])
    selected.sort(key=lambda item: (float(item["time_seconds"]), int(item["pts"])))
    return selected


def _assign_motion_scores(candidates: list[dict[str, Any]]) -> None:
    reliable = np.asarray(
        [
            float(item["motion_displacement_median_normalized"])
            for item in candidates
            if item.get("motion_status") == "ok"
        ],
        dtype=np.float64,
    )
    if reliable.size:
        low, high = np.percentile(reliable, [10, 90])
        denominator = max(float(high - low), 1e-9)
    else:
        low, denominator = 0.0, 1.0
    for item in candidates:
        if item.get("motion_status") == "ok":
            score = (float(item["motion_displacement_median_normalized"]) - float(low)) / denominator
            item["motion_score_source"] = "optical_flow"
        else:
            score = float(item["novelty"])
            item["motion_score_source"] = "descriptor_novelty"
        item["motion_score"] = min(1.0, max(0.0, score))
    baseline = float(np.median([float(item["motion_score"]) for item in candidates]))
    for item in candidates:
        item["motion_above_baseline"] = float(item["motion_score"]) > baseline


def _select_adaptive_keyframes(
    candidates: list[dict[str, Any]],
    duration_seconds: float,
    budget: int,
) -> list[dict[str, Any]]:
    by_second: dict[int, list[dict[str, Any]]] = {}
    for item in candidates:
        if not item["motion_above_baseline"]:
            continue
        second = min(
            max(0, math.floor(float(item["time_seconds"]))),
            max(0, math.ceil(duration_seconds) - 1),
        )
        by_second.setdefault(second, []).append(item)
    group_winners = [
        max(group, key=_adaptive_selection_key)
        for _, group in sorted(by_second.items())
    ]
    target = min(budget, len(group_winners))
    if target == 0:
        return []
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(target)]
    for item in group_winners:
        index = min(
            target - 1,
            int(float(item["time_seconds"]) / duration_seconds * target),
        )
        buckets[index].append(item)
    selected = [max(bucket, key=_adaptive_selection_key) for bucket in buckets if bucket]
    if len(selected) < target:
        selected_pts = {int(item["pts"]) for item in selected}
        remaining = [
            item for item in group_winners if int(item["pts"]) not in selected_pts
        ]
        remaining.sort(key=_adaptive_selection_key, reverse=True)
        selected.extend(remaining[: target - len(selected)])
    return selected


def _estimate_sparse_motion(previous: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=256,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        useHarrisDetector=False,
    )
    detected_count = 0 if corners is None else len(corners)
    if corners is None or detected_count < 12:
        return _empty_motion("insufficient_tracks", detected_count)
    tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or forward_status is None:
        return _empty_motion("insufficient_tracks", detected_count)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        tracked,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward is None or backward_status is None:
        return _empty_motion("insufficient_tracks", detected_count)
    forward = tracked.reshape(-1, 2)
    original = corners.reshape(-1, 2)
    reverse = backward.reshape(-1, 2)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(forward).all(axis=1)
        & np.isfinite(reverse).all(axis=1)
        & (np.linalg.norm(reverse - original, axis=1) <= 1.5)
    )
    displacement = np.linalg.norm(forward[valid] - original[valid], axis=1)
    tracked_count = int(displacement.size)
    if tracked_count < 12:
        return _empty_motion("insufficient_tracks", detected_count, tracked_count)
    diagonal = math.hypot(*previous.shape)
    return {
        "motion_status": "ok",
        "motion_detected_count": detected_count,
        "motion_tracked_count": tracked_count,
        "motion_tracked_fraction": tracked_count / detected_count,
        "motion_displacement_median_normalized": float(np.median(displacement) / diagonal),
        "motion_displacement_p90_normalized": float(np.percentile(displacement, 90) / diagonal),
    }


def _empty_motion(status: str, detected_count: int, tracked_count: int = 0) -> dict[str, Any]:
    return {
        "motion_status": status,
        "motion_detected_count": detected_count,
        "motion_tracked_count": tracked_count,
        "motion_tracked_fraction": tracked_count / detected_count if detected_count else 0.0,
        "motion_displacement_median_normalized": 0.0,
        "motion_displacement_p90_normalized": 0.0,
    }


def _score_stream(
    command: list[str],
    *,
    width: int,
    height: int,
    deadline: float,
    cancel_requested: Callable[[], bool] | None,
    estimate_motion: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    if estimate_motion:
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)

    def consume(index: int, pts: int, pts_time: float, frame: bytes) -> None:
        nonlocal previous_gray
        rgb = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
        gray = np.clip(
            rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114,
            0,
            255,
        ).astype(np.uint8)
        metrics = score_frame(gray)
        descriptor = gray.reshape(32, 10, 32, 10).mean(axis=(1, 3)).astype(np.float32)
        if estimate_motion:
            metrics.update(
                _empty_motion("no_previous_frame", 0)
                if previous_gray is None
                else _estimate_sparse_motion(previous_gray, gray)
            )
            previous_gray = gray
        candidates.append(
            {
                "candidate_index": index,
                "pts": pts,
                "time_seconds": pts_time,
                **metrics,
                "perceptual_hash": _difference_hash(descriptor),
                "descriptor": descriptor,
            }
        )

    _stream_raw_frames(
        command,
        width=width,
        height=height,
        deadline=deadline,
        cancel_requested=cancel_requested,
        consume=consume,
    )
    return candidates


def _materialize_selected(
    ffmpeg: str,
    source: Path,
    selected: list[dict[str, Any]],
    output_dir: Path,
    *,
    rotation_filter: list[str],
    width: int,
    height: int,
    deadline: float,
    cancel_requested: Callable[[], bool] | None,
    profile_id: str,
) -> None:
    expression = "+".join(f"eq(pts\\,{int(item['pts'])})" for item in selected)
    filters = [f"select='{expression}'", "showinfo", *rotation_filter, f"scale={width}:{height}:flags=lanczos"]
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "info",
        "-noautorotate",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        ",".join(filters),
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    expected = {int(item["pts"]): item for item in selected}
    written: set[int] = set()

    def consume(index: int, pts: int, _pts_time: float, frame: bytes) -> None:
        if pts not in expected or pts in written:
            raise VideoKeyframeError(f"unexpected selected video timestamp: {pts}")
        item = expected[pts]
        if profile_id == V1_PROFILE_ID:
            timestamp_ms = round(float(item["time_seconds"]) * 1000)
            filename = f"frame_{index + 1:06d}_t{timestamp_ms:010d}_pts{pts}.jpg"
        else:
            filename = candidate_frame_filename(item)
        path = output_dir / filename
        image = Image.fromarray(np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3), "RGB")
        exif = Image.Exif()
        exif[274] = 1
        exif[305] = f"Image3D-SceneGraph {profile_id}"
        image.save(path, format="JPEG", quality=95, subsampling=0, optimize=False, exif=exif)
        with Image.open(path) as check:
            if check.size != (width, height) or check.mode != "RGB":
                raise VideoKeyframeError(f"generated keyframe validation failed: {path.name}")
            if check.getexif().get(274) != 1:
                raise VideoKeyframeError(f"generated keyframe orientation is invalid: {path.name}")
        written.add(pts)

    _stream_raw_frames(
        command,
        width=width,
        height=height,
        deadline=deadline,
        cancel_requested=cancel_requested,
        consume=consume,
    )
    missing = set(expected) - written
    if missing:
        raise VideoKeyframeError(f"ffmpeg did not materialize {len(missing)} selected keyframes")


def _stream_raw_frames(
    command: list[str],
    *,
    width: int,
    height: int,
    deadline: float,
    cancel_requested: Callable[[], bool] | None,
    consume: Callable[[int, int, float, bytes], None],
) -> None:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise VideoKeyframeError("failed to open ffmpeg pipes")
    timestamps: queue.Queue[tuple[int, float]] = queue.Queue()
    stderr_tail: list[str] = []

    def read_stderr() -> None:
        for raw in iter(process.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            stderr_tail.append(line.rstrip())
            del stderr_tail[:-40]
            match = _SHOWINFO.search(line)
            if match:
                try:
                    timestamps.put((int(match.group(1)), float(match.group(2))))
                except ValueError:
                    continue

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    frame_size = width * height * 3
    index = 0
    try:
        while True:
            frame = _read_exact(
                process.stdout,
                frame_size,
                process=process,
                deadline=deadline,
                cancel_requested=cancel_requested,
            )
            if not frame:
                break
            try:
                pts, pts_time = timestamps.get(timeout=5)
            except queue.Empty as exc:
                raise VideoKeyframeError("ffmpeg emitted a frame without a PTS") from exc
            consume(index, pts, pts_time, frame)
            index += 1
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        if process.stdout:
            process.stdout.close()
    return_code = process.wait(timeout=30)
    stderr_thread.join(timeout=5)
    if return_code != 0:
        details = "\n".join(stderr_tail[-12:])
        raise VideoKeyframeError(f"ffmpeg video decoding failed:\n{details}")


def _read_exact(
    stream: Any,
    size: int,
    *,
    process: subprocess.Popen[bytes],
    deadline: float,
    cancel_requested: Callable[[], bool] | None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        if cancel_requested is not None and cancel_requested():
            raise VideoKeyframeError("video keyframe extraction cancelled")
        if time.monotonic() > deadline:
            raise VideoKeyframeError("video keyframe extraction exceeded 30 minutes")
        ready, _, _ = select.select([stream], [], [], 1.0)
        if not ready:
            if process.poll() is not None:
                break
            continue
        chunk = os.read(stream.fileno(), remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if payload and len(payload) != size:
        raise VideoKeyframeError("ffmpeg returned a partial video frame")
    return payload


def _assign_quality_scores(candidates: list[dict[str, Any]]) -> None:
    for metric in ("sharpness", "exposure_score", "entropy", "novelty"):
        values = np.asarray([float(item[metric]) for item in candidates], dtype=np.float64)
        low, high = np.percentile(values, [5, 95])
        denominator = max(float(high - low), 1e-9)
        for item in candidates:
            item[f"{metric}_normalized"] = min(
                1.0, max(0.0, (float(item[metric]) - float(low)) / denominator)
            )
    for item in candidates:
        item["quality_score"] = (
            0.4 * item["sharpness_normalized"]
            + 0.2 * item["exposure_score_normalized"]
            + 0.2 * item["entropy_normalized"]
            + 0.2 * item["novelty_normalized"]
        )


def _near_duplicate(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    distance = (int(current["perceptual_hash"]) ^ int(previous["perceptual_hash"])).bit_count()
    mse = float(np.mean(((current["descriptor"] - previous["descriptor"]) / 255.0) ** 2))
    return distance <= 4 and mse <= 0.0015


def _difference_hash(descriptor: np.ndarray) -> int:
    sample = descriptor[np.linspace(0, 31, 8, dtype=int)][:, np.linspace(0, 31, 9, dtype=int)]
    bits = sample[:, 1:] > sample[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _selection_key(item: dict[str, Any]) -> tuple[float, float, int]:
    return (float(item["quality_score"]), float(item["sharpness"]), -int(item["candidate_index"]))


def _adaptive_selection_key(item: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(item["motion_score"]),
        float(item["quality_score"]),
        -int(item["pts"]),
    )


def _resolve_rotation(stream: dict[str, Any], override: str) -> dict[str, Any]:
    metadata_degrees = 0.0
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            parsed = _finite_float(side_data.get("rotation"))
            if parsed is not None:
                metadata_degrees = parsed
                break
    else:
        parsed = _finite_float((stream.get("tags") or {}).get("rotate"))
        if parsed is not None:
            metadata_degrees = parsed
    normalized = metadata_degrees % 360
    nearest = min((0, 90, 180, 270), key=lambda value: abs(((normalized - value + 180) % 360) - 180))
    error = abs(((normalized - nearest + 180) % 360) - 180)
    if error > 1.0 and override == "auto":
        raise VideoKeyframeError("video display matrix is not a supported rigid quarter-turn")
    if override == "auto":
        # ffprobe reports display-matrix rotation in mathematical degrees; applying
        # the inverse produces the same upright pixels as ffmpeg autorotation.
        applied = (-nearest) % 360
    else:
        applied = {"clockwise_90": 90, "counterclockwise_90": 270, "180": 180}[override]
    return {
        "override": override,
        "metadata_degrees": metadata_degrees,
        "applied_degrees": applied,
        "quarter_turn": applied in {90, 270},
    }


def _rotation_filter(degrees: int) -> list[str]:
    return {
        0: [],
        90: ["transpose=clock"],
        180: ["hflip", "vflip"],
        270: ["transpose=cclock"],
    }[degrees]


def _scaled_dimensions(width: int, height: int, longest_edge: int) -> tuple[int, int]:
    if longest_edge < 1:
        raise VideoKeyframeError("Gaussian longest edge must be positive")
    scale = min(1.0, longest_edge / max(width, height))
    output_width = max(2, round(width * scale))
    output_height = max(2, round(height * scale))
    output_width -= output_width % 2
    output_height -= output_height % 2
    return output_width, output_height


def _write_contact_sheet(selected_dir: Path, output: Path) -> None:
    paths = sorted(selected_dir.glob("*.jpg"))
    if not paths:
        raise VideoKeyframeError("no selected keyframes were written")
    count = min(64, len(paths))
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    chosen = [paths[int(index)] for index in indices]
    thumb_width, thumb_height = 160, 100
    columns = 8
    rows = math.ceil(count / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 18)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(chosen):
        with Image.open(path) as source:
            thumb = source.copy()
        thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width + (thumb_width - thumb.width) // 2
        y = (index // columns) * (thumb_height + 18) + (thumb_height - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        draw.text((index % columns * thumb_width + 4, y + thumb_height + 1), path.stem[:22], fill="white")
    sheet.save(output, format="JPEG", quality=88)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VideoKeyframeError(f"{label} is invalid") from exc
    if parsed <= 0:
        raise VideoKeyframeError(f"{label} must be positive")
    return parsed


def _parse_rate(value: Any) -> float | None:
    text = str(value or "")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = _finite_float(denominator)
        numerator_value = _finite_float(numerator)
        if numerator_value is None or denominator_value in {None, 0.0}:
            return None
        return numerator_value / denominator_value
    return _finite_float(text)


def _orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _progress(callback: Callable[[str, float], None] | None, stage: str, value: float) -> None:
    if callback is not None:
        callback(stage, value)
