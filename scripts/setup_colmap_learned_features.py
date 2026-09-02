"""Install the pinned COLMAP models used by learned features and local matchers.

Dry-run by default. Jobs never download these assets at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from image3d_scenegraph.geometry.colmap import (
    COLMAP_FEATURE_ASSETS,
    colmap_feature_asset_root,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Download and verify the models. Default is dry-run.",
    )
    args = parser.parse_args()

    root = colmap_feature_asset_root()
    print("COLMAP learned feature setup")
    print(f"  destination: {root}")
    for asset in COLMAP_FEATURE_ASSETS.values():
        print(f"  asset: {asset.filename}")
        print(f"    url: {asset.url}")
        print(f"    size_bytes: {asset.size_bytes}")
        print(f"    sha256: {asset.sha256}")
    print()

    if not args.install:
        print("dry_run=true")
        print("Add --install to download the models.")
        return

    root.mkdir(parents=True, exist_ok=True)
    for asset in COLMAP_FEATURE_ASSETS.values():
        destination = root / asset.filename
        if destination.is_file():
            verify_asset(destination, asset.size_bytes, asset.sha256)
            print(f"model_exists={destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            for attempt in range(1, 4):
                try:
                    urllib.request.urlretrieve(asset.url, temporary)
                    verify_asset(temporary, asset.size_bytes, asset.sha256)
                    break
                except (urllib.error.URLError, SystemExit) as exc:
                    temporary.unlink(missing_ok=True)
                    if attempt == 3:
                        raise
                    print(
                        f"download_retry={attempt}/3 asset={asset.filename} "
                        f"error={exc}",
                        file=sys.stderr,
                    )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"model={destination}")
    print("COLMAP learned feature setup complete.")


def verify_asset(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"model size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"model SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
