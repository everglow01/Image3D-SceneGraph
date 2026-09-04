#!/usr/bin/env python3
"""Analyze raw COLMAP camera poses without modifying or training a Job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image3d_scenegraph.geometry.sfm_pose_health import (
    build_sfm_pose_health_from_text,
    selected_timestamps_from_payload,
    write_sfm_pose_health,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--video-selection", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    timestamps = None
    if args.video_selection is not None:
        payload = json.loads(args.video_selection.read_text(encoding="utf-8"))
        timestamps = selected_timestamps_from_payload(payload)
    record = build_sfm_pose_health_from_text(
        model_dir=args.model_dir,
        selected_timestamps=timestamps,
        database_path=args.database,
    )
    write_sfm_pose_health(args.output, record)
    print(f"pose_health_status={record['status']}")
    print("reason_codes=" + ",".join(record["reason_codes"]))
    repair = record["automatic_repair"]
    print(f"automatic_repair_eligible={str(repair['eligible']).lower()}")
    print(f"automatic_repair_reason={repair['reason']}")
    for rank, image in enumerate(record["outlier_candidates"], start=1):
        print(
            f"pose_outlier_rank={rank:02d} image_id={image['image_id']} "
            f"name={image['name']} time_seconds={image['time_seconds']} "
            f"median_ratio={image['distance_to_median_ratio']}"
        )
    for rank, pair in enumerate(record["bridge_pairs"], start=1):
        print(
            f"pose_bridge_rank={rank:02d} image_ids={pair['image_ids'][0]},{pair['image_ids'][1]} "
            f"tracks={pair['shared_final_tracks']} "
            f"candidates={pair['candidate_match_count']} "
            f"verified={pair['verified_inlier_count']} "
            f"time_span_seconds={pair['time_span_seconds']}"
        )
    print(f"pose_health_report={args.output}")
    print("training_started=false")


if __name__ == "__main__":
    main()
