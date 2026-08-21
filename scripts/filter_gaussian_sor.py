"""Post-hoc statistical outlier removal (SOR) cleanup for exported Gaussian splats.

Zero-retraining floater experiment (codex.md decision log, Stage 1): removes
gaussians whose mean distance to their k nearest neighbors exceeds the global
mean plus std_ratio standard deviations (Open3D statistical outlier removal).
The statistical test is invariant to uniform scale, so positions are used as
stored in the canonical normalized frame; no coordinate transform is needed.

Without --band-opacity every flagged gaussian is removed (full variant).
With it, SOR still runs over all gaussians but only flagged gaussians whose
activated opacity is below the band threshold are removed, protecting
isolated legitimate content (thin structures, distant detail).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from image3d_scenegraph.gaussian.dataset import sha256_file
from image3d_scenegraph.gaussian.export import PLY_FIELDS, read_gaussian_ply, write_binary_ply

PROFILE = "sor_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter isolated gaussians out of a Gaussian splat with SOR."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gaussian-ply", type=Path, help="Canonical exported scene.ply or canonical.ply.")
    source.add_argument("--model-snapshot", type=Path, help="Training model snapshot .pt; writes a row-filtered filtered-model.pt.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory; must not exist.")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="SOR neighbor count k.")
    parser.add_argument("--std-ratio", type=float, default=1.5, help="SOR standard-deviation ratio.")
    parser.add_argument("--band-opacity", type=float, help="When set, only remove flagged gaussians below this activated opacity (band variant).")
    parser.add_argument("--max-removal-fraction", type=float, default=0.5, help="Safety limit; abort when removal exceeds this fraction.")
    args = parser.parse_args()
    if args.nb_neighbors < 1:
        raise SystemExit("--nb-neighbors must be at least 1")
    if args.std_ratio <= 0:
        raise SystemExit("--std-ratio must be positive")
    if args.band_opacity is not None and not 0.0 < args.band_opacity < 1.0:
        raise SystemExit("--band-opacity must be between 0 and 1")
    if not 0.0 < args.max_removal_fraction <= 1.0:
        raise SystemExit("--max-removal-fraction must be between 0 and 1")

    started = time.perf_counter()
    try:
        if args.model_snapshot is not None:
            filter_model_snapshot(args, started)
        else:
            fields = read_gaussian_ply(args.gaussian_ply)
            rows = np.column_stack([fields[name] for name in PLY_FIELDS])
            kept_rows, keep = apply_sor_filter(
                rows,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio,
                band_opacity=args.band_opacity,
            )
            check_removal_limit(keep, args.max_removal_fraction)
            write_outputs(args, keep, kept_rows, started)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))


def apply_sor_filter(
    rows: np.ndarray,
    *,
    nb_neighbors: int,
    std_ratio: float,
    band_opacity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = rows[:, :3].astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-rows[:, 54].astype(np.float64)))
    keep = compute_band_keep_mask(
        positions, opacity, nb_neighbors=nb_neighbors, std_ratio=std_ratio, band_opacity=band_opacity
    )
    return rows[keep], keep


def compute_band_keep_mask(
    positions: np.ndarray,
    opacity: np.ndarray,
    *,
    nb_neighbors: int,
    std_ratio: float,
    band_opacity: float | None,
) -> np.ndarray:
    keep = compute_sor_keep_mask(positions, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if band_opacity is not None:
        keep = keep | (opacity >= band_opacity)
    return keep


def compute_sor_keep_mask(
    positions: np.ndarray, *, nb_neighbors: int, std_ratio: float
) -> np.ndarray:
    if len(positions) < max(nb_neighbors, 32):
        raise ValueError(
            f"SOR needs at least max(nb_neighbors, 32) points, got {len(positions)}"
        )
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(positions))
    _, indices = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    keep = np.zeros(len(positions), dtype=bool)
    keep[np.asarray(indices, dtype=np.int64)] = True
    return keep


def check_removal_limit(keep: np.ndarray, max_removal_fraction: float) -> None:
    removed_fraction = float((~keep).mean())
    if removed_fraction > max_removal_fraction:
        raise ValueError(
            f"safety limit: SOR would remove {removed_fraction:.3f} of gaussians "
            f"(limit {max_removal_fraction:.3f})"
        )


def filter_model_snapshot(args: argparse.Namespace, started: float) -> None:
    """Row-filter a training snapshot .pt; mirrors filter_gaussian_vggt.py output."""
    import io
    import os

    import torch

    from image3d_scenegraph.gaussian.evaluation import load_model_snapshot

    source_model = load_model_snapshot(args.model_snapshot, torch.device("cpu"))
    positions = source_model.means.detach().to(torch.float64).numpy()
    opacity = source_model.opacity_logits.detach().sigmoid().numpy().astype(np.float64)
    keep = compute_band_keep_mask(
        positions,
        opacity,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        band_opacity=args.band_opacity,
    )
    check_removal_limit(keep, args.max_removal_fraction)
    indices = torch.from_numpy(np.flatnonzero(keep)).long()
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
            "profile": PROFILE,
            "source_model_sha256": sha256_file(args.model_snapshot),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    filtered_path = args.output_dir / "filtered-model.pt"
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    temporary = filtered_path.with_suffix(".pt.tmp")
    temporary.write_bytes(buffer.getvalue())
    os.replace(temporary, filtered_path)
    mask_path = args.output_dir / "filter-mask.npz"
    np.savez_compressed(
        mask_path,
        keep=keep,
        nb_neighbors=np.int64(args.nb_neighbors),
        std_ratio=np.float64(args.std_ratio),
        band_opacity=np.float64(args.band_opacity) if args.band_opacity is not None else np.float64(np.nan),
    )
    record: dict[str, Any] = {
        "profile": PROFILE,
        "variant": "band" if args.band_opacity is not None else "full",
        "params": {
            "nb_neighbors": args.nb_neighbors,
            "std_ratio": args.std_ratio,
            "band_opacity": args.band_opacity,
            "max_removal_fraction": args.max_removal_fraction,
        },
        "source_model": str(args.model_snapshot),
        "source_model_sha256": sha256_file(args.model_snapshot),
        "filtered_model": str(filtered_path),
        "filtered_model_sha256": sha256_file(filtered_path),
        "mask": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "input_count": int(len(keep)),
        "kept_count": int(keep.sum()),
        "removed_count": int((~keep).sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "filter-record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"filtered_model={filtered_path}")
    print(f"variant={record['variant']}")
    print(f"kept_gaussians={record['kept_count']}")
    print(f"removed_gaussians={record['removed_count']}")
    print(f"removed_fraction={record['removed_count'] / record['input_count']:.4f}")


def write_outputs(args: argparse.Namespace, keep: np.ndarray, kept_rows: np.ndarray, started: float) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    filtered_path = args.output_dir / "filtered.ply"
    write_binary_ply(filtered_path, kept_rows)
    mask_path = args.output_dir / "filter-mask.npz"
    np.savez_compressed(
        mask_path,
        keep=keep,
        nb_neighbors=np.int64(args.nb_neighbors),
        std_ratio=np.float64(args.std_ratio),
        band_opacity=np.float64(args.band_opacity) if args.band_opacity is not None else np.float64(np.nan),
    )
    source_export_path = args.gaussian_ply.parent / "export.json"
    world_transform = None
    if source_export_path.is_file():
        source_export = json.loads(source_export_path.read_text(encoding="utf-8"))
        world_transform = source_export.get("world_from_normalized")
    record: dict[str, Any] = {
        "profile": PROFILE,
        "variant": "band" if args.band_opacity is not None else "full",
        "params": {
            "nb_neighbors": args.nb_neighbors,
            "std_ratio": args.std_ratio,
            "band_opacity": args.band_opacity,
            "max_removal_fraction": args.max_removal_fraction,
        },
        "source_ply": str(args.gaussian_ply),
        "source_ply_sha256": sha256_file(args.gaussian_ply),
        "filtered_ply": str(filtered_path),
        "filtered_ply_sha256": sha256_file(filtered_path),
        "mask": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "input_count": int(len(keep)),
        "kept_count": int(keep.sum()),
        "removed_count": int((~keep).sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "filter-record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if world_transform is not None:
        sidecar = {
            "schema_version": 1,
            "format": "sor_experiment_derivative",
            "coordinate_frame": "normalized",
            "gaussian_count": record["kept_count"],
            "world_from_normalized": world_transform,
            "source_export_sha256": sha256_file(source_export_path),
            "note": "minimal sidecar for analyze_gaussian_floaters; opacity statistics of the source export.json do not apply",
        }
        (args.output_dir / "export.json").write_text(
            json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"filtered_ply={filtered_path}")
    print(f"variant={record['variant']}")
    print(f"kept_gaussians={record['kept_count']}")
    print(f"removed_gaussians={record['removed_count']}")
    print(f"removed_fraction={record['removed_count'] / record['input_count']:.4f}")


if __name__ == "__main__":
    main()
