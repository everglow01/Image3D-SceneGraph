#!/usr/bin/env python3
"""Run pinned Graphdeco training with the project's explicit Validation list."""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--graphdeco-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    known, upstream = parser.parse_known_args()
    root = known.graphdeco_root.resolve()
    train = root / "train.py"
    if not train.is_file():
        raise SystemExit(f"Graphdeco train.py not found: {train}")
    sys.path.insert(0, str(root))

    from scene import dataset_readers
    from utils import general_utils

    original = dataset_readers.readColmapSceneInfo
    original_safe_state = general_utils.safe_state

    def seeded_safe_state(silent):
        original_safe_state(silent)
        random.seed(known.seed)
        np.random.seed(known.seed % (2**32))
        torch.manual_seed(known.seed)
        torch.cuda.manual_seed_all(known.seed)

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
    general_utils.safe_state = seeded_safe_state
    random.seed(known.seed)
    np.random.seed(known.seed % (2**32))
    torch.manual_seed(known.seed)
    torch.cuda.manual_seed_all(known.seed)
    sys.argv = [str(train), *upstream]
    runpy.run_path(str(train), run_name="__main__")


if __name__ == "__main__":
    main()
