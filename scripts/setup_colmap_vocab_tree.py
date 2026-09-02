"""Install the pinned COLMAP vocabulary trees used for image pairing.

Dry-run by default. Jobs never download these assets at runtime.
"""

from __future__ import annotations

import argparse
import sys

from image3d_scenegraph.file_integrity import (
    FileIntegrityError,
    install_verified_file,
)
from image3d_scenegraph.geometry.colmap import (
    COLMAP_VOCAB_TREE_ASSETS,
    colmap_vocab_tree_root,
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

    for asset in COLMAP_VOCAB_TREE_ASSETS.values():
        destination = root / asset.filename
        downloaded = install_verified_file(
            asset.url,
            destination,
            expected_size=asset.size_bytes,
            expected_sha256=asset.sha256,
            label="vocabulary tree",
        )
        print(f"{'vocab_tree' if downloaded else 'vocab_tree_exists'}={destination}")
    print("COLMAP vocabulary tree setup complete.")


if __name__ == "__main__":
    try:
        main()
    except FileIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        sys.exit(130)
