#!/usr/bin/env python3
"""Generate VGGT group diagnostics from a completed COLMAP+VGGT job."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from run_colmap_sparse import discover_images
from run_colmap_vggt_dense import (
    build_vggt_group_diagnostics,
    build_vggt_groups,
    parse_colmap_images_with_points,
)


class GroupDiagnosticsError(RuntimeError):
    """Raised when retained job evidence cannot reproduce the VGGT groups."""


def parse_run_configuration(path: Path) -> dict[str, Any]:
    runner = ""
    logged_values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise GroupDiagnosticsError(f"cannot read run log {path}: {exc}") from exc
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        logged_values.setdefault(key, value)
        if key == "runner":
            runner = value
    if not runner:
        raise GroupDiagnosticsError(f"missing runner= line in {path}")

    argv = shlex.split(runner)

    def option(name: str) -> str:
        try:
            return argv[argv.index(name) + 1]
        except (ValueError, IndexError) as exc:
            raise GroupDiagnosticsError(f"runner command is missing {name}") from exc

    try:
        configuration = {
            "grouping": option("--vggt-grouping"),
            "batch_size": int(option("--vggt-batch-size")),
            "overlap_size": int(option("--vggt-overlap-size")),
        }
    except ValueError as exc:
        raise GroupDiagnosticsError(f"runner grouping options are not integers: {exc}") from exc

    if configuration["grouping"] not in {"sequential", "covisibility"}:
        raise GroupDiagnosticsError(
            f"runner --vggt-grouping has unknown value: {configuration['grouping']}"
        )

    expected = {
        "vggt_grouping": configuration["grouping"],
        "vggt_batch_size": str(configuration["batch_size"]),
        "vggt_overlap_size": str(configuration["overlap_size"]),
    }
    for key, expected_value in expected.items():
        logged_value = logged_values.get(key)
        if logged_value is not None and logged_value != expected_value:
            raise GroupDiagnosticsError(
                f"runner {key}={expected_value} disagrees with logged {key}={logged_value}"
            )
    return configuration


def build_completed_job_diagnostics(job_dir: Path) -> dict[str, Any]:
    configuration = parse_run_configuration(job_dir / "logs" / "run.log")
    image_dir = job_dir / "input" / "images"
    sparse_images_path = job_dir / "colmap_vggt" / "sparse_txt" / "images.txt"
    if not image_dir.is_dir():
        raise GroupDiagnosticsError(f"missing job image directory: {image_dir}")
    if not sparse_images_path.is_file():
        raise GroupDiagnosticsError(f"missing COLMAP sparse images: {sparse_images_path}")

    colmap_images = parse_colmap_images_with_points(sparse_images_path)
    registered_by_name = {image.name: image for image in colmap_images}
    registered_paths = [
        path for path in discover_images(image_dir) if path.name in registered_by_name
    ]
    if not registered_paths:
        raise GroupDiagnosticsError("completed job has no registered input images")
    if len(registered_paths) != len(registered_by_name):
        missing = sorted(set(registered_by_name) - {path.name for path in registered_paths})
        raise GroupDiagnosticsError(
            "registered COLMAP images are missing from input/images: " + ", ".join(missing)
        )

    groups = build_vggt_groups(
        registered_paths=registered_paths,
        registered_by_name=registered_by_name,
        grouping=configuration["grouping"],
        batch_size=configuration["batch_size"],
        overlap_size=configuration["overlap_size"],
    )
    return build_vggt_group_diagnostics(
        groups=groups,
        registered_by_name=registered_by_name,
        grouping=configuration["grouping"],
        batch_size=configuration["batch_size"],
        requested_overlap_size=configuration["overlap_size"],
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

    output = args.job_dir / "diagnostics" / "vggt_groups.json"
    try:
        payload = build_completed_job_diagnostics(args.job_dir)
        content = serialized(payload)
        if args.write:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            print(f"wrote {output}")
        else:
            if not output.is_file():
                raise GroupDiagnosticsError(f"missing VGGT group diagnostics: {output}")
            if output.read_text(encoding="utf-8") != content:
                raise GroupDiagnosticsError(
                    f"VGGT group diagnostics differ from retained evidence: {output}"
                )
            print(f"verified {output}")
        print(f"group_count={payload['group_count']}")
        print(f"overlap_status={payload['overlap']['status']}")
    except GroupDiagnosticsError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
