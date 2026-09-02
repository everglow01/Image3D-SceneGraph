"""Install the pinned COLMAP models used by learned features and local matchers.

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
    COLMAP_FEATURE_ASSETS,
    colmap_feature_asset_root,
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

    for asset in COLMAP_FEATURE_ASSETS.values():
        destination = root / asset.filename
        downloaded = install_verified_file(
            asset.url,
            destination,
            expected_size=asset.size_bytes,
            expected_sha256=asset.sha256,
            label="model",
        )
        print(f"{'model' if downloaded else 'model_exists'}={destination}")
    print("COLMAP learned feature setup complete.")


if __name__ == "__main__":
    try:
        main()
    except FileIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        sys.exit(130)
