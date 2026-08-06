#!/usr/bin/env python3
"""Train the project-owned 3D Gaussian model from a frozen dataset contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from image3d_scenegraph.gaussian.config import (
    ResolvedGaussianConfig,
    resolve_internal_config,
    resolved_config_record,
)
from image3d_scenegraph.gaussian.dataset import sha256_file, validate_contract, with_initialization
from image3d_scenegraph.gaussian.initialization import (
    dense_initialization,
    sparse_initialization,
    write_initialization,
)
from image3d_scenegraph.gaussian.external_trainer import train_external_gaussians
from image3d_scenegraph.gaussian.trainers import TRAINER_IDS
from image3d_scenegraph.gaussian.trainer import train_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trainer", choices=TRAINER_IDS, default="project")
    parser.add_argument("--initialization", choices=["sparse", "dense"], required=True)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--support-diagnostics", type=Path)
    parser.add_argument("--max-initial-points", type=int, default=1_000_000)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--overrides-json", type=Path)
    parser.add_argument("--resolved-config-json", type=Path)
    parser.add_argument("--attempt-id", default="train-001")
    parser.add_argument("--attempt-kind", choices=["fresh", "retry", "resume"], default="fresh")
    parser.add_argument("--parent-attempt-id")
    parser.add_argument("--resume-iteration", type=int)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    validate_contract(contract, args.dataset_root)
    if args.overrides_json is not None and args.resolved_config_json is not None:
        raise SystemExit("use either --overrides-json or --resolved-config-json")
    if args.resolved_config_json is not None:
        record = json.loads(args.resolved_config_json.read_text(encoding="utf-8"))
        resolved = ResolvedGaussianConfig(
            requested_profile=record["requested_profile"],
            effective_config=record["effective_config"],
            effective_config_hash=record["effective_config_hash"],
        )
        resolved_config_record(resolved)
    else:
        overrides = None
        if args.overrides_json is not None:
            overrides = json.loads(args.overrides_json.read_text(encoding="utf-8"))
        resolved = resolve_internal_config(overrides=overrides)
    normalized_from_world = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    if args.initialization == "sparse":
        initialized = sparse_initialization(
            args.points,
            normalized_from_world,
            max_points=args.max_initial_points,
        )
    else:
        initialized = dense_initialization(
            args.points,
            normalized_from_world,
            max_points=args.max_initial_points,
            voxel_size=args.voxel_size,
            diagnostics_path=args.support_diagnostics,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
        )

    initialization_dir = args.run_dir / "preparation" / args.attempt_id / "initialization"
    asset_path = initialization_dir / f"{args.initialization}.npz"
    diagnostics_path = initialization_dir / f"{args.initialization}.json"
    write_initialization(asset_path, diagnostics_path, initialized)
    if args.attempt_kind == "resume":
        if args.parent_attempt_id is None:
            raise SystemExit("resume requires --parent-attempt-id")
        parent_contract_path = (
            args.run_dir / "preparation" / args.parent_attempt_id / "dataset.json"
        )
        effective_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
        validate_contract(effective_contract)
    else:
        effective_contract = with_initialization(
            contract,
            asset=asset_path.relative_to(args.run_dir / "preparation" / args.attempt_id).as_posix(),
            asset_sha256=sha256_file(asset_path),
        )
    contract_path = args.run_dir / "preparation" / args.attempt_id / "dataset.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(effective_contract, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_dir / "preparation" / args.attempt_id / "effective_config.json").write_text(
        json.dumps(
            {
                "requested_profile": resolved.requested_profile,
                "effective_config_hash": resolved.effective_config_hash,
                "effective_config": resolved.effective_config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    trainer = train_gaussians if args.trainer == "project" else train_external_gaussians
    trainer_args = {
        "contract": effective_contract,
        "dataset_root": args.dataset_root,
        "initialization": initialized,
        "resolved_config": resolved,
        "run_dir": args.run_dir,
        "attempt_id": args.attempt_id,
        "cancel_requested": (
            (lambda: args.cancel_file.exists()) if args.cancel_file is not None else None
        ),
    }
    if args.trainer == "project":
        trainer_args.update(
            attempt_kind=args.attempt_kind,
            parent_attempt_id=args.parent_attempt_id,
            resume_iteration=args.resume_iteration,
        )
    else:
        if args.attempt_kind != "fresh":
            raise SystemExit("external Gaussian trainers currently support fresh attempts only")
        trainer_args["trainer_id"] = args.trainer
    result = trainer(**trainer_args)
    print(json.dumps({**result.__dict__, "trainer_id": args.trainer}, allow_nan=False))


if __name__ == "__main__":
    main()
