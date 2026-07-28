#!/usr/bin/env python3
"""Export G1.15 category point clouds from G1.14 final-point diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pointcloud import read_ply_points_and_colors
from run_vggt_pointcloud import write_ply


HIGH_DISAGREEMENT_THRESHOLD = 0.02
POINT_ORDER = "exactly matches geometry/points.ply vertex order"
CATEGORY_RULES = {
    "supported": {
        "expression": "support_counts > 0",
        "color": (0, 170, 255),
    },
    "unverified": {
        "expression": "visible_counts == 0",
        "color": (255, 170, 0),
    },
    "high_disagreement": {
        "expression": "isfinite(overlap_disagreement) & (overlap_disagreement >= 0.02)",
        "color": (220, 40, 180),
    },
}


class DiagnosticPlyError(RuntimeError):
    """Raised when retained diagnostics cannot produce trustworthy exports."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticPlyError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticPlyError(f"expected a JSON object: {path}")
    return payload


def category_masks(
    *,
    support_counts: np.ndarray,
    visible_counts: np.ndarray,
    overlap_disagreement: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "supported": support_counts > 0,
        "unverified": visible_counts == 0,
        "high_disagreement": np.isfinite(overlap_disagreement)
        & (overlap_disagreement >= HIGH_DISAGREEMENT_THRESHOLD),
    }


def load_diagnostics(
    path: Path,
    *,
    expected_point_count: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    required = {"support_counts", "visible_counts", "overlap_disagreement"}
    try:
        with np.load(path) as payload:
            names = list(payload.files)
            missing = required - set(names)
            if missing:
                raise DiagnosticPlyError(f"diagnostic sidecar is missing arrays: {sorted(missing)}")
            arrays: dict[str, np.ndarray] = {}
            for name in names:
                values = payload[name]
                if values.ndim != 1 or len(values) != expected_point_count:
                    raise DiagnosticPlyError(
                        f"diagnostic array {name} has shape {values.shape}; "
                        f"expected ({expected_point_count},)"
                    )
                if name in required:
                    arrays[name] = values
    except (OSError, ValueError) as exc:
        raise DiagnosticPlyError(f"cannot read diagnostic sidecar {path}: {exc}") from exc
    return arrays, names


def validate_sources(
    *,
    points_path: Path,
    diagnostics_path: Path,
    diagnostics_index_path: Path,
    consistency_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], dict[str, Any], dict[str, Any]]:
    for path in (points_path, diagnostics_path, diagnostics_index_path, consistency_path):
        if not path.is_file():
            raise DiagnosticPlyError(f"missing input file: {path}")

    index = read_json(diagnostics_index_path)
    consistency = read_json(consistency_path)
    if index.get("schema_version") != 1:
        raise DiagnosticPlyError("unsupported support diagnostics schema")
    if index.get("point_order") != POINT_ORDER:
        raise DiagnosticPlyError("support diagnostics do not declare exact final PLY row order")
    if Path(str(index.get("sidecar", ""))).name != diagnostics_path.name:
        raise DiagnosticPlyError("support diagnostics index names a different sidecar")
    if index.get("sidecar_sha256") != sha256_file(diagnostics_path):
        raise DiagnosticPlyError("support diagnostics sidecar SHA-256 mismatch")

    points, _ = read_ply_points_and_colors(points_path)
    point_count = len(points)
    if index.get("point_count") != point_count:
        raise DiagnosticPlyError(
            f"point count differs between PLY ({point_count}) and diagnostics index "
            f"({index.get('point_count')})"
        )
    if consistency.get("accepted_points") != point_count:
        raise DiagnosticPlyError(
            "final PLY point count differs from consistency accepted_points"
        )
    if consistency.get("relative_threshold") != HIGH_DISAGREEMENT_THRESHOLD:
        raise DiagnosticPlyError(
            "consistency relative_threshold differs from the frozen high-disagreement threshold"
        )
    arrays, array_names = load_diagnostics(
        diagnostics_path,
        expected_point_count=point_count,
    )
    return points, arrays, array_names, index, consistency


def build_payload(
    *,
    points_path: Path,
    diagnostics_path: Path,
    diagnostics_index_path: Path,
    consistency_path: Path,
    point_count: int,
    array_names: list[str],
    masks: dict[str, np.ndarray],
    overlap_disagreement: np.ndarray,
    consistency: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    supported_count = int(masks["supported"].sum())
    unverified_count = int(masks["unverified"].sum())
    expected_supported = consistency.get("supported_points")
    expected_unverified = consistency.get("unverified_points")
    if supported_count != expected_supported:
        raise DiagnosticPlyError(
            f"supported count {supported_count} differs from consistency {expected_supported}"
        )
    if unverified_count != expected_unverified:
        raise DiagnosticPlyError(
            f"unverified count {unverified_count} differs from consistency {expected_unverified}"
        )

    exports: dict[str, Any] = {}
    for name, rule in CATEGORY_RULES.items():
        path = output_dir / f"{name}.ply"
        exports[name] = {
            "path": path.name,
            "count": int(masks[name].sum()),
            "fraction_of_final_points": float(masks[name].mean()),
            "mask": rule["expression"],
            "color_rgb": list(rule["color"]),
            "file_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return {
        "schema_version": 1,
        "evaluation": "g1_15_support_diagnostic_point_clouds",
        "protocol": {
            "source_row_order": POINT_ORDER,
            "reconstruction_rerun": False,
            "categories_are_independent": True,
            "high_disagreement_threshold_absolute_log_depth": HIGH_DISAGREEMENT_THRESHOLD,
            "high_disagreement_threshold_rationale": (
                "frozen private-225 cross-view relative threshold; not tuned from ROI or GT"
            ),
            "ground_truth_used": False,
            "production_default_changed": False,
        },
        "sources": {
            "points": points_path.resolve().as_posix(),
            "points_sha256": sha256_file(points_path),
            "support_diagnostics": diagnostics_path.resolve().as_posix(),
            "support_diagnostics_sha256": sha256_file(diagnostics_path),
            "support_diagnostics_index": diagnostics_index_path.resolve().as_posix(),
            "support_diagnostics_index_sha256": sha256_file(diagnostics_index_path),
            "consistency": consistency_path.resolve().as_posix(),
            "consistency_sha256": sha256_file(consistency_path),
        },
        "point_count": point_count,
        "diagnostic_arrays": sorted(array_names),
        "finite_overlap_disagreement_count": int(np.isfinite(overlap_disagreement).sum()),
        "consistency_count_check": {
            "supported": {
                "recomputed": supported_count,
                "recorded": expected_supported,
                "matches": True,
            },
            "unverified": {
                "recomputed": unverified_count,
                "recorded": expected_unverified,
                "matches": True,
            },
        },
        "exports": exports,
    }


def export_support_diagnostic_plys(
    *,
    points_path: Path,
    diagnostics_path: Path,
    diagnostics_index_path: Path,
    consistency_path: Path,
    output_dir: Path,
    check: bool = False,
) -> dict[str, Any]:
    points, arrays, array_names, _, consistency = validate_sources(
        points_path=points_path,
        diagnostics_path=diagnostics_path,
        diagnostics_index_path=diagnostics_index_path,
        consistency_path=consistency_path,
    )
    masks = category_masks(
        support_counts=arrays["support_counts"],
        visible_counts=arrays["visible_counts"],
        overlap_disagreement=arrays["overlap_disagreement"],
    )

    metadata_path = output_dir / "diagnostic_plys.json"
    if check:
        if not metadata_path.is_file():
            raise DiagnosticPlyError(f"missing diagnostic export metadata: {metadata_path}")
        for name, rule in CATEGORY_RULES.items():
            path = output_dir / f"{name}.ply"
            if not path.is_file():
                raise DiagnosticPlyError(f"missing diagnostic export: {path}")
            exported_points, exported_colors = read_ply_points_and_colors(path)
            expected_points = points[masks[name]]
            expected_color = np.asarray(rule["color"], dtype=np.uint8)
            if not np.array_equal(exported_points, expected_points):
                raise DiagnosticPlyError(f"diagnostic export row mismatch: {path}")
            if len(expected_points) and (
                exported_colors is None or not np.all(exported_colors == expected_color)
            ):
                raise DiagnosticPlyError(f"diagnostic export color mismatch: {path}")
    else:
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise DiagnosticPlyError(f"output directory must be absent or empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, rule in CATEGORY_RULES.items():
            selected_points = points[masks[name]]
            colors = np.broadcast_to(
                np.asarray(rule["color"], dtype=np.uint8),
                (len(selected_points), 3),
            )
            write_ply(output_dir / f"{name}.ply", selected_points, colors)

    payload = build_payload(
        points_path=points_path,
        diagnostics_path=diagnostics_path,
        diagnostics_index_path=diagnostics_index_path,
        consistency_path=consistency_path,
        point_count=len(points),
        array_names=array_names,
        masks=masks,
        overlap_disagreement=arrays["overlap_disagreement"],
        consistency=consistency,
        output_dir=output_dir,
    )
    content = json.dumps(payload, indent=2) + "\n"
    if check:
        if metadata_path.read_text(encoding="utf-8") != content:
            raise DiagnosticPlyError(f"diagnostic export metadata differs: {metadata_path}")
    else:
        metadata_path.write_text(content, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--diagnostics-index", required=True, type=Path)
    parser.add_argument("--consistency", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        payload = export_support_diagnostic_plys(
            points_path=args.points,
            diagnostics_path=args.diagnostics,
            diagnostics_index_path=args.diagnostics_index,
            consistency_path=args.consistency,
            output_dir=args.output_dir,
            check=args.check,
        )
    except DiagnosticPlyError as exc:
        parser.error(str(exc))
    action = "verified" if args.check else "wrote"
    print(f"{action} {args.output_dir / 'diagnostic_plys.json'}")
    for name, record in payload["exports"].items():
        print(f"{name}={record['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
