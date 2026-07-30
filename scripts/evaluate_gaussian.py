#!/usr/bin/env python3
"""Evaluate an immutable project Gaussian model on validation or frozen held-out test views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig, resolved_config_record
from image3d_scenegraph.gaussian.evaluation import run_evaluation, write_frozen_candidate
from image3d_scenegraph.gaussian.dataset import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--resolved-config-json", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--frozen-candidate", type=Path)
    parser.add_argument("--freeze-candidate-id")
    args = parser.parse_args()

    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    record = json.loads(args.resolved_config_json.read_text(encoding="utf-8"))
    resolved = ResolvedGaussianConfig(
        requested_profile=record["requested_profile"],
        effective_config=record["effective_config"],
        effective_config_hash=record["effective_config_hash"],
    )
    resolved_config_record(resolved)
    if args.freeze_candidate_id:
        if args.split != "test" or args.frozen_candidate is None:
            raise SystemExit("--freeze-candidate-id requires test split and --frozen-candidate")
        write_frozen_candidate(
            args.frozen_candidate,
            candidate_id=args.freeze_candidate_id,
            dataset_hash=str(contract["dataset_hash"]),
            effective_config_hash=resolved.effective_config_hash,
            model_sha256=sha256_file(args.model),
        )
    result = run_evaluation(
        contract=contract,
        dataset_root=args.dataset_root,
        model_path=args.model,
        resolved_config=resolved,
        split=args.split,
        output_dir=args.output_dir,
        frozen_candidate_path=args.frozen_candidate,
        progress_path=args.progress,
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
