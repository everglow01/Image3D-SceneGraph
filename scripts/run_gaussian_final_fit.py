#!/usr/bin/env python3
"""Run bounded Train+Validation final-fit on a selected native Gaussian snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image3d_scenegraph.gaussian.config import (
    ResolvedGaussianConfig,
    resolved_config_record,
)
from image3d_scenegraph.gaussian.dataset import validate_contract
from image3d_scenegraph.gaussian.trainer import final_fit_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--selection-evaluation", required=True, type=Path)
    parser.add_argument("--resolved-config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    validate_contract(contract, args.dataset_root)
    config_record = json.loads(args.resolved_config_json.read_text(encoding="utf-8"))
    resolved = ResolvedGaussianConfig(
        requested_profile=config_record["requested_profile"],
        effective_config=config_record["effective_config"],
        effective_config_hash=config_record["effective_config_hash"],
    )
    resolved_config_record(resolved)
    payload = {
        "contract": contract,
        "dataset_root": args.dataset_root,
        "source_model_path": args.source_model,
        "selection_evaluation_path": args.selection_evaluation,
        "resolved_config": resolved,
        "output_dir": args.output_dir,
    }
    cancel_requested = (
        (lambda: args.cancel_file.exists()) if args.cancel_file is not None else None
    )
    if args.distributed:
        from gsplat.distributed import cli

        cli(
            _distributed_final_fit,
            {"payload": payload, "cancel_file": args.cancel_file},
            verbose=True,
        )
        result = json.loads(
            (args.output_dir / "record.json").read_text(encoding="utf-8")
        )
    else:
        result = final_fit_gaussians(
            **payload,
            cancel_requested=cancel_requested,
        )
    print(json.dumps(result, allow_nan=False))


def _distributed_final_fit(
    local_rank: int, world_rank: int, world_size: int, payload: dict
) -> None:
    arguments = dict(payload["payload"])
    cancel_file = payload["cancel_file"]
    arguments.update(
        local_rank=local_rank,
        world_rank=world_rank,
        world_size=world_size,
        cancel_requested=(
            (lambda: cancel_file.exists()) if cancel_file is not None else None
        ),
    )
    final_fit_gaussians(**arguments)


if __name__ == "__main__":
    main()
