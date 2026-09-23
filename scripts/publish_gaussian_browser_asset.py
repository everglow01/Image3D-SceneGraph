#!/usr/bin/env python3
"""Publish one optional K2 browser asset without changing the Job's source files."""

from __future__ import annotations

import argparse
from pathlib import Path

from image3d_scenegraph.gaussian.browser_assets import publish_browser_asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Job-relative scene_splat of the chosen variant")
    args = parser.parse_args()
    converter = Path(__file__).resolve().with_name("convert_gaussian_browser_ksplat.mjs")
    print(publish_browser_asset(args.job_dir.absolute(), args.source, converter))


if __name__ == "__main__":
    main()
