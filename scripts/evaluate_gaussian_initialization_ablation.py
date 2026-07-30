#!/usr/bin/env python3
"""Run/check the R2.10 sparse-vs-dense initialization validation ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from image3d_scenegraph.gaussian.config import resolve_internal_config
from image3d_scenegraph.gaussian.initialization import dense_initialization, sparse_initialization
from image3d_scenegraph.gaussian.trainer import train_gaussians


SCHEMA_VERSION = 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_record(name: str, result: Any, initialization: Any, run_dir: Path) -> dict[str, Any]:
    model_path = run_dir / result.model_path
    result_path = run_dir / result.result_path
    progress_path = run_dir / result.progress_path
    return {
        "name": name,
        "initialization": initialization.diagnostics,
        "training": result.__dict__,
        "artifacts": {
            "model_bytes": model_path.stat().st_size,
            "model_sha256": file_hash(model_path),
            "result_sha256": file_hash(result_path),
            "progress_sha256": file_hash(progress_path),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing ablation output: {args.output_dir}")
    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    normalized_from_world = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    overrides = {
        "iterations": args.iterations,
        "resolution": {"longest_edge": args.longest_edge},
        "sh_schedule": {
            "initial_degree": 0,
            "max_degree": min(3, args.iterations),
            "increase_every_iterations": max(1, args.iterations // 3),
        },
        "densification": {
            "start_iteration": max(1, args.iterations // 4),
            "end_iteration": max(2, args.iterations - 1),
            "every_iterations": max(1, args.iterations // 4),
        },
        "opacity_reset": {"every_iterations": args.iterations},
        "evaluation": {"validation_every_iterations": args.iterations},
        "checkpoint": {"every_iterations": args.iterations},
        "gaussian_budget": {"max_count": args.gaussian_budget},
    }
    resolved = resolve_internal_config(overrides=overrides)
    sparse = sparse_initialization(
        args.sparse_points,
        normalized_from_world,
        max_points=args.max_initial_points,
    )
    dense = dense_initialization(
        args.dense_points,
        normalized_from_world,
        max_points=args.max_initial_points,
        voxel_size=args.voxel_size,
        diagnostics_path=args.support_diagnostics,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )
    args.output_dir.mkdir(parents=True)
    arms = []
    for name, initialization in (("sparse", sparse), ("dense", dense)):
        run_dir = args.output_dir / name
        result = train_gaussians(
            contract=contract,
            dataset_root=args.dataset_root,
            initialization=initialization,
            resolved_config=resolved,
            run_dir=run_dir,
        )
        arms.append(arm_record(name, result, initialization, run_dir))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "r2_10_sparse_vs_dense_initialization",
        "protocol": {
            "dataset_hash": contract["dataset_hash"],
            "effective_config_hash": resolved.effective_config_hash,
            "seed": resolved.effective_config["seed"],
            "split": "validation",
            "test_views_loaded": False,
            "arms_differ_only_in_initialization": True,
            "lpips": {"status": "not_run", "reason": "dependency_not_audited_in_r2_10"},
            "automatic_promotion": False,
        },
        "sources": {
            "dataset_contract_sha256": file_hash(args.dataset_contract),
            "sparse_points_sha256": file_hash(args.sparse_points),
            "dense_points_sha256": file_hash(args.dense_points),
            "support_diagnostics_sha256": file_hash(args.support_diagnostics),
        },
        "arms": arms,
        "comparison": {
            "validation_psnr_delta_dense_minus_sparse": (
                arms[1]["training"]["validation"]["mean_psnr"]
                - arms[0]["training"]["validation"]["mean_psnr"]
            ),
            "validation_ssim_delta_dense_minus_sparse": (
                arms[1]["training"]["validation"]["mean_ssim"]
                - arms[0]["training"]["validation"]["mean_ssim"]
            ),
            "training_seconds_delta_dense_minus_sparse": (
                arms[1]["training"]["elapsed_seconds"] - arms[0]["training"]["elapsed_seconds"]
            ),
            "peak_reserved_bytes_delta_dense_minus_sparse": (
                arms[1]["training"]["peak_reserved_bytes"]
                - arms[0]["training"]["peak_reserved_bytes"]
            ),
            "gaussian_count_delta_dense_minus_sparse": (
                arms[1]["training"]["gaussian_count"] - arms[0]["training"]["gaussian_count"]
            ),
        },
        "conclusion": "evidence_recorded_no_automatic_promotion",
    }
    report = args.output_dir / "ablation.json"
    report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def check(args: argparse.Namespace) -> dict[str, Any]:
    report = args.output_dir / "ablation.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("unsupported ablation schema")
    expected = {
        "dataset_contract_sha256": file_hash(args.dataset_contract),
        "sparse_points_sha256": file_hash(args.sparse_points),
        "dense_points_sha256": file_hash(args.dense_points),
        "support_diagnostics_sha256": file_hash(args.support_diagnostics),
    }
    if payload.get("sources") != expected:
        raise SystemExit("ablation source hash mismatch")
    for arm in payload["arms"]:
        root = args.output_dir / arm["name"]
        training = arm["training"]
        if file_hash(root / training["model_path"]) != arm["artifacts"]["model_sha256"]:
            raise SystemExit(f"model hash mismatch: {arm['name']}")
        if file_hash(root / training["result_path"]) != arm["artifacts"]["result_sha256"]:
            raise SystemExit(f"result hash mismatch: {arm['name']}")
        if file_hash(root / training["progress_path"]) != arm["artifacts"]["progress_sha256"]:
            raise SystemExit(f"progress hash mismatch: {arm['name']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sparse-points", required=True, type=Path)
    parser.add_argument("--dense-points", required=True, type=Path)
    parser.add_argument("--support-diagnostics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--longest-edge", type=int, default=320)
    parser.add_argument("--max-initial-points", type=int, default=20_000)
    parser.add_argument("--gaussian-budget", type=int, default=50_000)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = check(args) if args.check else run(args)
    print(json.dumps(payload["comparison"], indent=2))


if __name__ == "__main__":
    main()
