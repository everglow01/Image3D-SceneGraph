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
    load_frozen_initialization,
    sparse_initialization,
    write_initialization,
)
from image3d_scenegraph.gaussian.readiness import (
    GeometryReadinessError,
    build_geometry_readiness,
    require_geometry_readiness,
    write_geometry_readiness,
)
from image3d_scenegraph.gaussian.replay import (
    build_replay_bundle,
    validate_replay_bundle,
)
from image3d_scenegraph.gaussian.external_trainer import train_external_gaussians
from image3d_scenegraph.gaussian.trainers import (
    TRAINER_IDS,
    validate_trainer_strategy,
)
from image3d_scenegraph.gaussian.trainer import train_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trainer", choices=TRAINER_IDS, default="graphdeco")
    parser.add_argument(
        "--initialization", choices=["sparse", "dense", "frozen"], required=True
    )
    parser.add_argument("--points", type=Path)
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
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--readiness-only", action="store_true")
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
        resolved = resolve_internal_config(
            "mcmc_v1" if args.trainer == "mcmc" else "standard_v1",
            overrides=overrides,
        )
    validate_trainer_strategy(args.trainer, resolved.effective_config)
    if args.initialization == "frozen":
        if args.trainer not in {"project", "mcmc"}:
            raise SystemExit("frozen initialization is supported only by project and mcmc trainers")
        validate_replay_bundle(args.dataset_root)
        asset_relative = Path(str(contract["initialization"]["asset"]))
        asset_path = args.dataset_root / asset_relative
        diagnostics_path = asset_path.with_suffix(".json")
        initialized = load_frozen_initialization(
            asset_path,
            diagnostics_path,
            expected_sha256=str(contract["initialization"]["sha256"]),
        )
    else:
        if args.points is None:
            raise SystemExit("--points is required for sparse or dense initialization")
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

    readiness = build_geometry_readiness(
        contract,
        initialized,
        resolved.effective_config,
        trainer_id=args.trainer,
    )
    write_geometry_readiness(
        args.run_dir
        / "preparation"
        / args.attempt_id
        / "geometry_readiness.json",
        readiness,
    )
    if args.readiness_only:
        _print_readiness_summary(readiness)
    try:
        require_geometry_readiness(readiness)
    except GeometryReadinessError as exc:
        raise SystemExit(str(exc)) from exc
    if args.readiness_only:
        return
    if args.attempt_kind == "resume":
        if args.parent_attempt_id is None:
            raise SystemExit("resume requires --parent-attempt-id")
        parent_contract_path = (
            args.run_dir / "preparation" / args.parent_attempt_id / "dataset.json"
        )
        effective_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
        validate_contract(effective_contract)
    elif args.initialization == "frozen":
        effective_contract = contract
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
    if args.initialization != "frozen":
        build_replay_bundle(
            contract=effective_contract,
            dataset_path=contract_path,
            dataset_root=args.dataset_root,
            initialization_path=asset_path,
            diagnostics_path=diagnostics_path,
            replay_root=args.run_dir / "replay",
        )

    native_trainer = args.trainer in {"project", "mcmc"}
    trainer = train_gaussians if native_trainer else train_external_gaussians
    trainer_args = {
        "contract": effective_contract,
        "dataset_root": args.dataset_root,
        "initialization": initialized,
        "resolved_config": resolved,
        "run_dir": args.run_dir,
        "attempt_id": args.attempt_id,
    }
    if native_trainer:
        trainer_args.update(
            attempt_kind=args.attempt_kind,
            parent_attempt_id=args.parent_attempt_id,
            resume_iteration=args.resume_iteration,
        )
        if args.distributed:
            from gsplat.distributed import cli

            cli(
                _distributed_native_train,
                {"trainer_args": trainer_args, "cancel_file": args.cancel_file},
                verbose=True,
            )
            result_path = (
                args.run_dir / "attempts" / args.attempt_id / "artifacts" / "result.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(json.dumps({**result, "trainer_id": args.trainer}, allow_nan=False))
            return
    else:
        if args.distributed:
            raise SystemExit("distributed mode is supported only by project and mcmc trainers")
        if args.attempt_kind != "fresh":
            raise SystemExit("external Gaussian trainers currently support fresh attempts only")
        trainer_args["trainer_id"] = args.trainer
    trainer_args["cancel_requested"] = (
        (lambda: args.cancel_file.exists()) if args.cancel_file is not None else None
    )
    result = trainer(**trainer_args)
    print(json.dumps({**result.__dict__, "trainer_id": args.trainer}, allow_nan=False))


def _print_readiness_summary(record: dict) -> None:
    reasons = ",".join(record["reason_codes"]) or "none"
    print(f"readiness_status={record['status']}")
    print(f"reason_codes={reasons}")
    print(
        "scale_floor_fraction="
        f"{record['initialization']['scale_floor_fraction']:.9f}"
    )
    for rank, camera in enumerate(
        record["camera_centers"]["largest_distances"], start=1
    ):
        median_ratio = camera["distance_to_median_ratio"]
        p99_ratio = camera["distance_to_p99_ratio"]
        print(
            f"camera_center_rank={rank:02d} image_id={camera['image_id']} "
            f"split={camera['split']} "
            f"median_ratio={median_ratio if median_ratio is not None else 'undefined'} "
            f"p99_ratio={p99_ratio if p99_ratio is not None else 'undefined'}"
        )


def _distributed_native_train(
    local_rank: int, world_rank: int, world_size: int, payload: dict
) -> None:
    trainer_args = dict(payload["trainer_args"])
    cancel_file = payload["cancel_file"]
    trainer_args.update(
        local_rank=local_rank,
        world_rank=world_rank,
        world_size=world_size,
        cancel_requested=(
            (lambda: cancel_file.exists()) if cancel_file is not None else None
        ),
    )
    train_gaussians(**trainer_args)


if __name__ == "__main__":
    main()
