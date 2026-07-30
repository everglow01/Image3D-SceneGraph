#!/usr/bin/env python3
"""Collect an auditable RTX development profile from completed Gaussian artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from image3d_scenegraph.gaussian.dataset import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--initialization", required=True, type=Path)
    parser.add_argument("--effective-config", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--export-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"profile output already exists: {args.output}")

    training = _read(args.training_result)
    initialization = _read(args.initialization)
    config = _read(args.effective_config)
    evaluation = _read(args.evaluation)
    exported = _read(args.export_metadata)
    progress = [json.loads(line) for line in args.progress.read_text(encoding="utf-8").splitlines()]
    topology = {
        "densified": sum(int(event.get("densified", 0)) for event in progress),
        "pruned": sum(int(event.get("pruned", 0)) for event in progress),
        "opacity_resets": sum(int(event.get("opacity_reset") is True) for event in progress),
    }
    payload = {
        "schema_version": 1,
        "profile": "rtx4060_8gb_development_v1",
        "status": "measured",
        "scope": "single_gpu_development_not_company_server",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available()
                else None
            ),
        },
        "effective_config_hash": config["effective_config_hash"],
        "settings": config["effective_config"],
        "initialization_selected": initialization["counts"]["accepted"],
        "final_gaussian_count": training["gaussian_count"],
        "topology": topology,
        "training_seconds": training["elapsed_seconds"],
        "peak_allocated_bytes": training["peak_allocated_bytes"],
        "peak_reserved_bytes": training["peak_reserved_bytes"],
        "validation_psnr": evaluation["psnr"],
        "validation_ssim": evaluation["ssim"],
        "validation_render_fps": evaluation["render_fps"],
        "lpips": evaluation["lpips"],
        "asset_bytes": {
            "model": args.training_result.parent.joinpath("model.pt").stat().st_size,
            "canonical": args.export_metadata.parent.joinpath("canonical.ply").stat().st_size,
            "browser": args.export_metadata.parent.joinpath("scene.ply").stat().st_size,
            "bundle": args.export_metadata.parent.joinpath("result.zip").stat().st_size,
        },
        "source_hashes": {
            "training_result": sha256_file(args.training_result),
            "progress": sha256_file(args.progress),
            "initialization": sha256_file(args.initialization),
            "effective_config": sha256_file(args.effective_config),
            "evaluation": sha256_file(args.evaluation),
            "export_metadata": sha256_file(args.export_metadata),
        },
        "limitations": [
            "short Stage 2 development profile; not a final quality optimum",
            "world units remain arbitrary",
            "first-view CUDA warm-up is included in evaluation FPS",
            "LPIPS is not run until pretrained weight provenance is approved",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False))


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
