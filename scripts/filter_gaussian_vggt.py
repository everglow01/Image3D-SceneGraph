from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from image3d_scenegraph.gaussian.dataset import spatial_order, validate_contract
from image3d_scenegraph.gaussian.evaluation import load_model_snapshot
from image3d_scenegraph.gaussian.vggt_filter import (
    DepthEvidence,
    GaussianVggtFilterError,
    classify_gaussians,
)


MAX_TRAIN_DEPTH_VIEWS = 64
BATCH_SIZE = 8
OVERLAP = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a conservative Train-depth-filtered Gaussian model with VGGT."
    )
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--colmap-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", type=Path, default=Path("external/vggt"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/vggt/facebook--VGGT-1B"),
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max-train-views", type=int, default=MAX_TRAIN_DEPTH_VIEWS)
    args = parser.parse_args()
    if args.max_train_views < 2:
        parser.error("--max-train-views must be at least 2")
    started = time.perf_counter()
    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    validate_contract(contract, args.dataset_root)
    selected = select_train_images(contract, args.max_train_views)
    if len(selected) < 2:
        raise GaussianVggtFilterError("dataset has fewer than two usable Train views")

    sys.path.insert(0, str(args.repo_dir.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    from run_colmap_vggt_dense import (
        build_vggt_image_transform,
        estimate_depth_scale,
        run_vggt_depth_batches,
    )
    from run_vggt_pointcloud import (
        load_vggt_model,
        select_device,
        select_dtype,
        validate_local_vggt,
    )
    from image3d_scenegraph.geometry.grouping import (
        parse_colmap_images_with_points,
        parse_colmap_points3d,
    )

    validate_local_vggt(args.repo_dir, args.checkpoint_dir)
    device = select_device(args.device)
    dtype = select_dtype(device, args.precision)
    model = load_vggt_model(
        model_cls=VGGT,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        dtype=dtype,
        enable_point=False,
    ).eval()

    colmap_images = parse_colmap_images_with_points(
        args.colmap_model_dir / "images.txt"
    )
    points3d = parse_colmap_points3d(args.colmap_model_dir / "points3D.txt")
    registered_by_name = {image.name: image for image in colmap_images}
    selected_paths = [args.dataset_root / str(image["path"]) for image in selected]
    if any(path.name not in registered_by_name for path in selected_paths):
        missing = [path.name for path in selected_paths if path.name not in registered_by_name]
        raise GaussianVggtFilterError(
            f"Train depth views are missing COLMAP observations: {missing}"
        )
    groups = overlapping_groups(selected_paths, BATCH_SIZE, OVERLAP)
    predictions, _capture = run_vggt_depth_batches(
        model=model,
        groups=groups,
        load_and_preprocess_images=load_and_preprocess_images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        device=device,
        dtype=dtype,
        frames_chunk_size=2,
        registered_by_name=registered_by_name,
        points3d=points3d,
        min_scale_observations=20,
        retain_point_diagnostics=True,
    )

    world_from_normalized = np.asarray(
        contract["normalization"]["world_from_normalized"], dtype=np.float64
    )
    entry_by_name = {Path(str(image["path"])).name: image for image in selected}
    evidence = []
    skipped = []
    for path in selected_paths:
        item = predictions.get(path)
        if item is None:
            skipped.append({"image": path.name, "reason": "missing_vggt_prediction"})
            continue
        estimate = item.get("source_scale_estimate")
        if estimate is None:
            estimate = estimate_depth_scale(
                colmap_image=registered_by_name[path.name],
                points3d=points3d,
                depth=item["depth"],
                image_shape=item["image_shape"],
                original_size=item["original_size"],
                min_observations=20,
            )
        if estimate is None:
            skipped.append({"image": path.name, "reason": "insufficient_sparse_scale_observations"})
            continue
        entry = entry_by_name[path.name]
        transform = build_vggt_image_transform(
            item["original_size"], item["image_shape"]
        )
        depth = np.asarray(item["depth"], dtype=np.float32) * float(estimate.scale)
        confidence = np.asarray(item["confidence"], dtype=np.float32)
        valid = (
            np.isfinite(depth)
            & np.isfinite(confidence)
            & (depth > 0)
            & (confidence > 0)
        )
        if not valid.any():
            skipped.append({"image": path.name, "reason": "no_valid_depth"})
            continue
        confidence_threshold = float(np.quantile(confidence[valid], 0.5))
        trusted = valid & (confidence >= confidence_threshold)
        far_depth = float(np.quantile(depth[trusted], 0.99))
        camera_from_world = np.asarray(entry["camera_from_world"], dtype=np.float64)
        # VGGT depth stays in raw COLMAP units; unlike RenderCamera, this
        # evidence transform intentionally retains the similarity scale.
        evidence.append(
            DepthEvidence(
                image_id=str(entry["image_id"]),
                image_name=path.name,
                camera_from_normalized=camera_from_world @ world_from_normalized,
                intrinsic=np.asarray(entry["intrinsic"], dtype=np.float64),
                depth=depth,
                confidence=confidence,
                confidence_threshold=confidence_threshold,
                scale_x=float(transform.scale_x),
                scale_y=float(transform.scale_y),
                pad_left=float(transform.pad_left),
                pad_top=float(transform.pad_top),
                far_depth=far_depth,
                scale_observations=int(estimate.observation_count),
                scale_log_mad=float(estimate.log_mad),
            )
        )
    if len(evidence) < 2:
        raise GaussianVggtFilterError(
            f"only {len(evidence)} Train views have scale-aligned VGGT depth"
        )

    import torch

    source_model = load_model_snapshot(args.model, torch.device("cpu"))
    means, quaternions, scales, opacities, _sh = source_model.activated()
    result = classify_gaussians(
        means=means.detach().cpu().numpy(),
        scales=scales.detach().cpu().numpy(),
        quaternions=quaternions.detach().cpu().numpy(),
        opacities=opacities.detach().cpu().numpy(),
        evidence=evidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    filtered_model_path = args.output_dir / "filtered-model.pt"
    indices = torch.from_numpy(np.flatnonzero(result.keep)).long()
    payload = {
        "max_sh_degree": source_model.max_sh_degree,
        "state_dict": {
            "means": source_model.means.detach()[indices],
            "log_scales": source_model.log_scales.detach()[indices],
            "quats": source_model.quats.detach()[indices],
            "opacity_logits": source_model.opacity_logits.detach()[indices],
            "sh_coeffs": source_model.sh_coeffs.detach()[indices],
        },
        "postprocess": {
            "profile": "vggt_visibility_v1",
            "source_model_sha256": sha256_file(args.model),
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    temporary = filtered_model_path.with_suffix(".pt.tmp")
    temporary.write_bytes(buffer.getvalue())
    os.replace(temporary, filtered_model_path)
    mask_path = args.output_dir / "filter-mask.npz"
    np.savez_compressed(
        mask_path,
        keep=result.keep,
        reasons=result.reasons,
        support_counts=result.support_counts,
        contradiction_counts=result.contradiction_counts,
        envelope_counts=result.envelope_counts,
        oversized=result.oversized,
    )
    diagnostics = {
        **result.diagnostics,
        "source_model": str(args.model),
        "source_model_sha256": sha256_file(args.model),
        "filtered_model": str(filtered_model_path),
        "filtered_model_sha256": sha256_file(filtered_model_path),
        "mask": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "selected_train_image_count": len(selected),
        "usable_train_depth_count": len(evidence),
        "skipped_train_depth_views": skipped,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_host_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "vggt_checkpoint_sha256": sha256_file(args.checkpoint_dir / "model.safetensors"),
        "runtime_network_access": False,
    }
    diagnostics_path = args.output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result_record = {
        "profile": "vggt_visibility_v1",
        "status": "available",
        "source_model": str(args.model),
        "source_model_sha256": diagnostics["source_model_sha256"],
        "filtered_model": str(filtered_model_path),
        "filtered_model_sha256": diagnostics["filtered_model_sha256"],
        "mask": str(mask_path),
        "mask_sha256": diagnostics["mask_sha256"],
        "diagnostics": str(diagnostics_path),
        "input_count": int(result.diagnostics["counts"]["input"]),
        "kept_count": int(result.diagnostics["counts"]["kept"]),
        "removed_count": int(result.diagnostics["counts"]["removed"]),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result_record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"filtered_model={filtered_model_path}")
    print(f"kept_gaussians={result_record['kept_count']}")
    print(f"removed_gaussians={result_record['removed_count']}")


def select_train_images(contract: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    train_ids = {str(image_id) for image_id in contract["splits"]["train"]}
    images = [image for image in contract["images"] if str(image["image_id"]) in train_ids]
    if len(images) <= maximum:
        return images
    centers = np.stack(
        [np.asarray(image["world_from_camera"], dtype=np.float64)[:3, 3] for image in images]
    )
    order = spatial_order([str(image["image_id"]) for image in images], centers)
    selected = set(order[:maximum])
    return [image for index, image in enumerate(images) if index in selected]


def overlapping_groups(paths: list[Path], size: int, overlap: int) -> list[list[Path]]:
    groups = []
    start = 0
    while start < len(paths):
        end = min(start + size, len(paths))
        groups.append(paths[start:end])
        if end == len(paths):
            break
        start = end - overlap
    return groups


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except GaussianVggtFilterError as exc:
        raise SystemExit(str(exc)) from exc
