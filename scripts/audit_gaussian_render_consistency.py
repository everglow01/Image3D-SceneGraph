#!/usr/bin/env python3
"""Frozen Validation-only native/PLY rendering audit; never trains or edits sources."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.evaluation import load_model_snapshot
from image3d_scenegraph.gaussian.importer import import_inria_ply
from image3d_scenegraph.gaussian.render import (
    FAR_PLANE_NORMALIZED, NEAR_PLANE_NORMALIZED, render_gaussians,
)
from image3d_scenegraph.gaussian.runtime import load_evaluation_views
from image3d_scenegraph.gaussian.training_math import psnr, structural_similarity


def image_difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("comparison requires same-shape finite images")
    delta = (left - right).abs()
    return {"mae": float(delta.mean()), "max_absolute": float(delta.max()),
            "rmse": float(delta.square().mean().sqrt()), "psnr": psnr(left, right),
            "ssim": float(structural_similarity(left, right))}


def save_image(path: Path, image: torch.Tensor) -> None:
    Image.fromarray(image.detach().clamp(0, 1).mul(255).byte().cpu().numpy()).save(path)


def attribute_difference(left, right) -> dict:
    if left.count != right.count or left.max_sh_degree != right.max_sh_degree:
        raise ValueError("PLY round-trip changed Gaussian count or SH degree")
    result = {}
    for name in ("means", "log_scales", "opacity_logits", "sh_coeffs", "quats"):
        a, b = (getattr(model, name).detach() for model in (left, right))
        if name == "quats":
            a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"nonfinite model attribute: {name}")
        delta = (a - b).abs()
        result[name] = {"mae": float(delta.mean()), "max_absolute": float(delta.max()),
                        "comparison": "normalized_wxyz" if name == "quats" else "stored_values"}
    return result


def run(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = json.loads(args.dataset_contract.read_text())
    metadata = json.loads(args.export_metadata.read_text())
    sources = {"model": args.model, "ply": args.ply, "dataset": args.dataset_contract,
               "export_metadata": args.export_metadata}
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    if hashes["model"] != metadata["model_sha256"] or hashes["ply"] != metadata["browser_sha256"]:
        raise ValueError("source model/PLY does not match export metadata")
    if contract["dataset_hash"] != metadata["dataset_hash"] or metadata["coordinate_frame"] != "normalized":
        raise ValueError("export coordinate/dataset identity mismatch")
    views = load_evaluation_views(contract, args.dataset_root, split="validation", longest_edge=1280,
                                  device=torch.device("cpu"), image_ids=args.image_ids)
    if args.sh_probe not in args.image_ids:
        raise ValueError("SH probe must be one of the selected Validation views")
    entries = {str(row["image_id"]): row for row in contract["images"]}
    for view in views:
        sources["reference_" + view.camera.image_id] = args.dataset_root / entries[view.camera.image_id]["path"]
    hashes.update({key: sha256_file(path) for key, path in sources.items() if key not in hashes})
    if not torch.cuda.is_available():
        raise RuntimeError("this audit requires the authorized remote CUDA device")
    args.output_dir.mkdir(parents=True)
    record = {"schema_version": 1, "status": "running", "profile": "frozen_render_consistency_v1",
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": hashes, "dataset_hash": contract["dataset_hash"],
              "image_ids": args.image_ids, "split": "validation", "test_rgb": "not_loaded",
              "model_count": metadata["gaussian_count"], "sh_degree": 3,
              "background": [0, 0, 0], "near": NEAR_PLANE_NORMALIZED, "far": FAR_PLANE_NORMALIZED,
              "native_opacity_filter": "none", "native_rasterizer": "gsplat",
              "image_values": "RGB [0,1] as used by existing evaluator; no additional color conversion",
              "png_encoding": "clamp_0_1_then_floor_uint8", "views": [], "renders": {}}
    try:
        imported = args.output_dir / "ply-roundtrip.pt"
        import_inria_ply(args.ply, imported)
        original = load_model_snapshot(args.model, torch.device("cpu"))
        restored = load_model_snapshot(imported, torch.device("cpu"))
        record["attribute_difference"] = attribute_difference(original, restored)
        del original, restored
        gc.collect()
        for view in views:
            camera = view.camera
            record["views"].append({"image_id": camera.image_id, "width": camera.width, "height": camera.height,
                                    "intrinsic": camera.intrinsic.tolist(),
                                    "camera_from_normalized": camera.camera_from_normalized.tolist(),
                                    "reference_sha256": hashes["reference_" + camera.image_id]})
            save_image(args.output_dir / f"reference-{camera.image_id}.png", view.to(torch.device("cpu")).image)
        native_images = {}
        with torch.no_grad():
            for label, path in (("native", args.model), ("roundtrip", imported)):
                model = load_model_snapshot(path, torch.device("cuda:0"))
                for stored in views:
                    view = stored.to(torch.device("cuda:0"))
                    key = view.camera.image_id
                    degrees = [3, 0, 2] if label == "native" and key == args.sh_probe else [3]
                    for degree in degrees:
                        image = render_gaussians(model, view.camera, sh_degree=degree, background=None).image
                        name = f"{label}-sh{degree}-{key}"
                        save_image(args.output_dir / f"{name}.png", image)
                        item = {"reference": image_difference(image, view.image)}
                        cpu_image = image.cpu()
                        if degree == 3:
                            if label == "native":
                                native_images[key] = cpu_image
                                if args.existing_previews:
                                    old = torch.from_numpy(np.array(Image.open(args.existing_previews / f"{key}.png"))) / 255.0
                                    new_png = cpu_image.clamp(0, 1).mul(255).byte().float() / 255.0
                                    item["historical_png"] = image_difference(new_png, old)
                            else:
                                item["native_float"] = image_difference(cpu_image, native_images[key])
                        record["renders"][name] = item
                del model, image, view
                torch.cuda.empty_cache()
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["source_after_sha256"] = {key: sha256_file(path) for key, path in sources.items()}
        record["source_unchanged"] = record["source_after_sha256"] == hashes
        if not record["source_unchanged"]:
            record["status"] = "integrity_failed"
        (args.output_dir / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if record["status"] != "completed":
        raise RuntimeError(record["status"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset-contract", "dataset-root", "model", "ply", "export-metadata", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", required=True)
    parser.add_argument("--sh-probe", required=True)
    parser.add_argument("--existing-previews", type=Path)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "image_ids", "source_unchanged", "attribute_difference")}))


if __name__ == "__main__":
    main()
