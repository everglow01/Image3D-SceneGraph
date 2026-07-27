#!/usr/bin/env python3
"""Evaluate a deterministic VGGT group-selection candidate from a completed job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from image3d_scenegraph.geometry.grouping import (
    build_scale_disagreement_diagnostics,
    build_vggt_group_diagnostics,
    build_vggt_group_selection,
    parse_colmap_images_with_points,
)
from run_colmap_sparse import discover_images


def evaluate_group_selection(
    *,
    registered_paths: list[Path],
    registered_by_name: dict[str, Any],
    grouping: str,
    batch_size: int,
    overlap_size: int,
    scales_by_name: dict[str, float] | None = None,
) -> dict[str, Any]:
    selection = build_vggt_group_selection(
        registered_paths=registered_paths,
        registered_by_name=registered_by_name,
        grouping=grouping,
        batch_size=batch_size,
        overlap_size=overlap_size,
    )
    diagnostics = build_vggt_group_diagnostics(
        groups=selection.groups,
        registered_by_name=registered_by_name,
        grouping=grouping,
        batch_size=batch_size,
        requested_overlap_size=overlap_size,
        selection_records=selection.records,
    )
    payload = {
        "schema_version": 1,
        "grouping": grouping,
        "batch_size": batch_size,
        "requested_overlap_size": overlap_size,
        "selection_rule": "direct_sparse_covisibility_with_explicit_fallback",
        "diagnostics": diagnostics,
    }
    if scales_by_name is not None:
        payload["scale_disagreement"] = build_scale_disagreement_diagnostics(
            colmap_images=list(registered_by_name.values()),
            groups=selection.groups,
            scales_by_name=scales_by_name,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--vggt-grouping", choices=["covisibility", "sequential"], default="covisibility")
    parser.add_argument("--vggt-batch-size", type=int, default=4)
    parser.add_argument("--vggt-overlap-size", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sparse_images_path = args.job_dir / "colmap_vggt" / "sparse_txt" / "images.txt"
    images = parse_colmap_images_with_points(sparse_images_path)
    registered_by_name = {image.name: image for image in images}
    registered_paths = [
        path for path in discover_images(args.job_dir / "input" / "images")
        if path.name in registered_by_name
    ]
    if len(registered_paths) != len(registered_by_name):
        missing = sorted(set(registered_by_name) - {path.name for path in registered_paths})
        raise SystemExit("registered COLMAP images are missing from input/images: " + ", ".join(missing))
    fusion = json.loads((args.job_dir / "diagnostics" / "fusion.json").read_text(encoding="utf-8"))
    scales_by_name = {
        record["image"]: record["depth_scale"]
        for record in fusion["images"]
        if record["scale_source"] == "sparse_colmap"
    }

    payload = evaluate_group_selection(
        registered_paths=registered_paths,
        registered_by_name=registered_by_name,
        grouping=args.vggt_grouping,
        batch_size=args.vggt_batch_size,
        overlap_size=args.vggt_overlap_size,
        scales_by_name=scales_by_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
