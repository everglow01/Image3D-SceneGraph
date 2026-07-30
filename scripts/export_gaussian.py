#!/usr/bin/env python3
"""Export an immutable project Gaussian model to canonical and browser assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image3d_scenegraph.gaussian.export import export_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--resolved-config-json", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-hash")
    args = parser.parse_args()

    metadata = export_gaussians(
        model_path=args.model,
        contract=json.loads(args.dataset_contract.read_text(encoding="utf-8")),
        config_record=json.loads(args.resolved_config_json.read_text(encoding="utf-8")),
        evaluation_path=args.evaluation,
        output_dir=args.output_dir,
        checkpoint_hash=args.checkpoint_hash,
    )
    print(json.dumps(metadata, allow_nan=False))


if __name__ == "__main__":
    main()
