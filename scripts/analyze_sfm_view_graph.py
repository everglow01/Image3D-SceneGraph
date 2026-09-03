#!/usr/bin/env python3
"""Summarize the verified SfM view graph from retained diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from image3d_scenegraph.geometry.view_graph import ViewGraphError, summarize_view_graph


class ViewGraphAnalysisError(ValueError):
    """Raised when retained SfM diagnostics cannot be analyzed safely."""


def analyze_job(job_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    root = job_dir.resolve()
    manifest = _read_json(root / "diagnostics" / "sfm" / "manifest.json")
    if manifest.get("schema_version") not in {1, 2, 3, 4}:
        raise ViewGraphAnalysisError("SfM diagnostics schema is unsupported")
    images = manifest.get("images")
    runs = manifest.get("runs")
    if not isinstance(images, list) or not isinstance(runs, list) or not runs:
        raise ViewGraphAnalysisError("SfM diagnostics images or runs are invalid")
    selected_id = run_id or manifest.get("default_run_id")
    run = next(
        (
            item
            for item in runs
            if isinstance(item, dict) and item.get("run_id") == selected_id
        ),
        None,
    )
    if run is None:
        raise ViewGraphAnalysisError(f"SfM diagnostics run is missing: {selected_id}")
    pair_path_value = run.get("pair_index_path")
    if not isinstance(pair_path_value, str) or not pair_path_value:
        raise ViewGraphAnalysisError("SfM run pair index path is invalid")
    pair_path = (root / pair_path_value).resolve()
    try:
        pair_path.relative_to(root)
    except ValueError as exc:
        raise ViewGraphAnalysisError("SfM pair index escapes the job directory") from exc
    try:
        pair_index = json.loads(gzip.decompress(pair_path.read_bytes()))
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise ViewGraphAnalysisError(f"cannot read SfM pair index: {exc}") from exc
    if not isinstance(pair_index, dict) or pair_index.get("schema_version") not in {
        1,
        2,
    }:
        raise ViewGraphAnalysisError("SfM pair index schema is unsupported")
    pairs = pair_index.get("pairs")
    if not isinstance(pairs, list) or not all(isinstance(pair, dict) for pair in pairs):
        raise ViewGraphAnalysisError("SfM pair index records are invalid")
    try:
        return summarize_view_graph(images, pairs)
    except ViewGraphError as exc:
        raise ViewGraphAnalysisError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ViewGraphAnalysisError(f"cannot read SfM diagnostics: {exc}") from exc
    if not isinstance(value, dict):
        raise ViewGraphAnalysisError("SfM diagnostics must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        summary = analyze_job(args.job_dir, args.run_id)
    except ViewGraphAnalysisError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
