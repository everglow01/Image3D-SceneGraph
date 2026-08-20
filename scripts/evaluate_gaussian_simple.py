"""Render-evaluate a Gaussian snapshot on contract views without a resolved config.

Experiment path for derivative snapshots (e.g. SOR-filtered imports, or older
experiments whose resolved config predates the current config schema): calls
the same validate_contract + load_evaluation_views + evaluate_model chain as
run_evaluation, but takes longest_edge and sh_degree as explicit CLI args
instead of requiring a current-schema resolved config record. Validation
split only; test-split authorization stays in evaluate_gaussian.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from image3d_scenegraph.gaussian.dataset import sha256_file, validate_contract
from image3d_scenegraph.gaussian.evaluation import (
    GaussianEvaluationError,
    evaluate_model,
    load_model_snapshot,
)
from image3d_scenegraph.gaussian.runtime import load_evaluation_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not exist.")
    parser.add_argument("--longest-edge", type=int, default=1280)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()

    try:
        contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
        validate_contract(contract, args.dataset_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_snapshot(args.model, device)
        views = load_evaluation_views(
            contract,
            args.dataset_root,
            split="validation",
            longest_edge=args.longest_edge,
            device=torch.device("cpu"),
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        result = evaluate_model(
            model,
            views,
            split="validation",
            sh_degree=args.sh_degree,
            preview_dir=args.output_dir / "previews",
        )
        result["provenance"] = {
            "dataset_hash": contract["dataset_hash"],
            "model_sha256": sha256_file(args.model),
            "longest_edge": args.longest_edge,
            "sh_degree": args.sh_degree,
        }
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (GaussianEvaluationError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc))
    print(f"psnr_mean={result['psnr']['mean']:.4f}")
    print(f"ssim_mean={result['ssim']['mean']:.4f}")
    print(f"gaussian_count={result['gaussian_count']}")


if __name__ == "__main__":
    main()
