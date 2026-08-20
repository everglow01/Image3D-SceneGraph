"""Download the COLMAP vocab tree used for sequential matching loop detection.

Dry-run by default; pass --install to download. The file lives under the
git-ignored external/ directory and is never fetched at job runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# COLMAP switched its visual index from flann to faiss in May 2025; the legacy
# demuc.de flann trees abort with "Failed to read faiss index" on COLMAP >= 3.11.
# This is the official faiss tree COLMAP itself auto-downloads.
VOCAB_TREE_URL = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "vocab_tree_faiss_flickr100K_words256K.bin"
)
VOCAB_TREE_SHA256 = "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"
VOCAB_TREE_BYTES = 72_412_636


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the COLMAP vocab tree.")
    parser.add_argument("--install", action="store_true", help="Actually download. Default is dry-run.")
    args = parser.parse_args()

    project_root = Path.cwd()
    destination = project_root / "external" / "colmap-vocab" / "vocab_tree_faiss_flickr100K_words256K.bin"

    print("COLMAP vocab tree setup")
    print(f"  url: {VOCAB_TREE_URL}")
    print(f"  destination: {destination}")
    print(f"  sha256: {VOCAB_TREE_SHA256}")
    print(f"  size_bytes: {VOCAB_TREE_BYTES}")
    print()

    if not args.install:
        print("dry_run=true")
        print("Add --install to download the vocab tree.")
        return

    if destination.is_file():
        print(f"vocab_tree_exists={destination}")
        verify_sha256(destination, VOCAB_TREE_SHA256)
        print("COLMAP vocab tree setup complete.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(VOCAB_TREE_URL, temporary)
        verify_sha256(temporary, VOCAB_TREE_SHA256)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"vocab_tree={destination}")
    print("COLMAP vocab tree setup complete.")


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(f"vocab tree SHA-256 mismatch for {path}: expected {expected}, got {actual}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
