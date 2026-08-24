#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image3d_scenegraph.video.keyframes import (
    STANDARD_V1,
    VIDEO_PROFILES,
    VideoKeyframeError,
    extract_video_keyframes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bounded video keyframes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--longest-edge", type=int, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(VIDEO_PROFILES)),
        default=STANDARD_V1,
    )
    parser.add_argument(
        "--rotation",
        choices=("auto", "clockwise_90", "counterclockwise_90", "180"),
        default="auto",
    )
    parser.add_argument("--progress-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(stage: str, value: float) -> None:
        if args.progress_file is None:
            return
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.progress_file.with_suffix(args.progress_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"stage": stage, "progress": value}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.progress_file)

    try:
        result = extract_video_keyframes(
            args.input,
            args.output_dir,
            longest_edge=args.longest_edge,
            rotation_override=args.rotation,
            profile=args.profile,
            progress=progress,
        )
    except VideoKeyframeError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "profile": result["selection"]["profile"],
                "selected_count": result["selection"]["selected_count"],
                "candidate_count": result["selection"]["candidate_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
