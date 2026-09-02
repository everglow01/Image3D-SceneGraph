"""Install the pinned COLMAP vocabulary trees used for image pairing.

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
    COLMAP_VOCAB_TREE_ASSETS,
    colmap_vocab_tree_root,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Download and verify the vocabulary trees. Default is dry-run.",
    )
    args = parser.parse_args()

    root = colmap_vocab_tree_root()
    print("COLMAP vocabulary tree setup")
    print(f"  destination: {root}")
    for profile_id, asset in COLMAP_VOCAB_TREE_ASSETS.items():
        print(f"  profile: {profile_id}")
        print(f"    asset: {asset.filename}")
        print(f"    url: {asset.url}")
        print(f"    size_bytes: {asset.size_bytes}")
        print(f"    sha256: {asset.sha256}")
    print()

    if not args.install:
        print("dry_run=true")
        print("Add --install to download the vocabulary trees.")
        return

    root.mkdir(parents=True, exist_ok=True)
    for asset in COLMAP_VOCAB_TREE_ASSETS.values():
        destination = root / asset.filename
        if destination.is_file():
            verify_asset(destination, asset.size_bytes, asset.sha256)
            print(f"vocab_tree_exists={destination}")
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
        print(f"vocab_tree={destination}")
    print("COLMAP vocabulary tree setup complete.")


def verify_asset(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"vocabulary tree size mismatch for {path}: "
            f"expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"vocabulary tree SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
