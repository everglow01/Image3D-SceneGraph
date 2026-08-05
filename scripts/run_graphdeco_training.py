#!/usr/bin/env python3
"""Run pinned Graphdeco training with the project's explicit Validation list."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--graphdeco-root", required=True, type=Path)
    known, upstream = parser.parse_known_args()
    root = known.graphdeco_root.resolve()
    train = root / "train.py"
    if not train.is_file():
        raise SystemExit(f"Graphdeco train.py not found: {train}")
    sys.path.insert(0, str(root))

    from scene import dataset_readers

    original = dataset_readers.readColmapSceneInfo

    def read_explicit_validation(
        path, images, depths, eval, train_test_exp, *args, **kwargs
    ):
        return original(
            path,
            images,
            depths,
            eval,
            train_test_exp,
            llffhold=0,
        )

    dataset_readers.sceneLoadTypeCallbacks["Colmap"] = read_explicit_validation
    sys.argv = [str(train), *upstream]
    runpy.run_path(str(train), run_name="__main__")


if __name__ == "__main__":
    main()
