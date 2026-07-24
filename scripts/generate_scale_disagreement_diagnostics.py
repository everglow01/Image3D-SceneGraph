#!/usr/bin/env python3
"""Generate scale-disagreement diagnostics from a completed COLMAP+VGGT job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_colmap_vggt_dense import (
    GROUPING_MIN_SHARED_POINTS,
    build_scale_disagreement_diagnostics,
    parse_colmap_images_with_points,
)


class ScaleDiagnosticsError(RuntimeError):
    """Raised when retained job evidence cannot reproduce scale diagnostics."""


def build_completed_job_diagnostics(job_dir: Path) -> dict[str, Any]:
    diagnostics_dir = job_dir / "diagnostics"
    try:
        fusion = json.loads((diagnostics_dir / "fusion.json").read_text(encoding="utf-8"))
        group_diagnostics = json.loads(
            (diagnostics_dir / "vggt_groups.json").read_text(encoding="utf-8")
        )
        colmap_images = parse_colmap_images_with_points(
            job_dir / "colmap_vggt" / "sparse_txt" / "images.txt"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ScaleDiagnosticsError(f"cannot read retained job evidence: {exc}") from exc

    scales_by_name = {
        record["image"]: record["depth_scale"]
        for record in fusion.get("images", [])
        if record.get("scale_source") == "sparse_colmap"
    }
    groups = [
        [Path(member["image"]) for member in group["members"]]
        for group in group_diagnostics.get("groups", [])
    ]
    if not colmap_images or not scales_by_name or not groups:
        raise ScaleDiagnosticsError("retained job evidence is incomplete")
    if group_diagnostics.get("strong_connection_min_shared_tracks") != GROUPING_MIN_SHARED_POINTS:
        raise ScaleDiagnosticsError("group diagnostics use a different strong-edge threshold")

    return build_scale_disagreement_diagnostics(
        colmap_images=colmap_images,
        groups=groups,
        scales_by_name=scales_by_name,
    )


def serialized(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output = args.job_dir / "diagnostics" / "scale_disagreement.json"
    try:
        payload = build_completed_job_diagnostics(args.job_dir)
        content = serialized(payload)
        if args.write:
            output.write_text(content, encoding="utf-8")
            print(f"wrote {output}")
        else:
            if not output.is_file() or output.read_text(encoding="utf-8") != content:
                raise ScaleDiagnosticsError(
                    f"scale diagnostics differ from retained evidence: {output}"
                )
            print(f"verified {output}")
        for partition in ("all", "within_group", "group_boundary"):
            summary = payload[partition]
            print(
                f"{partition}: edges={summary['edge_count']} "
                f"p50={summary['log_scale_difference_p50']} "
                f"p90={summary['log_scale_difference_p90']} "
                f"p95={summary['log_scale_difference_p95']}"
            )
    except ScaleDiagnosticsError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
