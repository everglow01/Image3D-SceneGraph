#!/usr/bin/env python3
"""Summarize frozen multi-window pressure and G1.17 point-removal attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pointcloud import read_ply_points
from evaluate_fixed_roi_quality import read_definition, transform_points


POINT_ORDER = "exactly matches geometry/points.ply vertex order"
REQUIRED_ARRAYS = {
    "support_counts",
    "contradicted_counts",
    "overlap_disagreement",
    "source_image_index",
    "source_group_index",
    "source_window_role",
}


class PressureAnalysisError(RuntimeError):
    """Raised when frozen evidence cannot support a trustworthy analysis."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PressureAnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PressureAnalysisError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def distribution(values: list[int] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values)
    if not len(array):
        return {"p50": 0.0, "p90": 0.0, "max": 0}
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
    }


def support_strata(values: np.ndarray) -> dict[str, int]:
    return {
        "0": int(np.count_nonzero(values == 0)),
        "1": int(np.count_nonzero(values == 1)),
        "2_plus": int(np.count_nonzero(values >= 2)),
    }


def concentration_records(
    values: np.ndarray,
    *,
    names: dict[int, str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    counts = Counter(int(value) for value in values)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(values)
    return {
        "unique_count": len(counts),
        "top_1_fraction": fraction(sum(count for _, count in ordered[:1]), total),
        "top_5_fraction": fraction(sum(count for _, count in ordered[:5]), total),
        "top_10_fraction": fraction(sum(count for _, count in ordered[:10]), total),
        "top": [
            {
                "id": key,
                **({"name": names[key]} if names is not None else {}),
                "count": count,
                "fraction": fraction(count, total),
            }
            for key, count in ordered[:limit]
        ],
    }


def load_sidecar(path: Path, index: dict[str, Any]) -> dict[str, np.ndarray]:
    if index.get("schema_version") != 1 or index.get("point_order") != POINT_ORDER:
        raise PressureAnalysisError("unsupported support sidecar schema or point order")
    if Path(str(index.get("sidecar", ""))).name != path.name:
        raise PressureAnalysisError("support sidecar index names a different file")
    if index.get("sidecar_sha256") != sha256_file(path):
        raise PressureAnalysisError("support sidecar SHA-256 mismatch")
    point_count = int(index.get("point_count", -1))
    try:
        with np.load(path) as payload:
            missing = REQUIRED_ARRAYS - set(payload.files)
            if missing:
                raise PressureAnalysisError(f"support sidecar missing arrays: {sorted(missing)}")
            arrays = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
            for name in payload.files:
                if np.asarray(payload[name]).shape != (point_count,):
                    raise PressureAnalysisError(
                        f"support array {name} does not match point count {point_count}"
                    )
    except (OSError, ValueError) as exc:
        raise PressureAnalysisError(f"cannot read support sidecar {path}: {exc}") from exc
    return arrays


def scene_paths(record: dict[str, Any]) -> dict[str, Path]:
    capture = Path(record["capture_dir"])
    baseline = Path(record["baseline_dir"])
    candidate = Path(record["candidate_dir"])
    return {
        "predictions": capture / "diagnostics/vggt_window_predictions.json",
        "groups": capture / "diagnostics/vggt_groups.json",
        "visibility": capture / "diagnostics/visibility_graph.json",
        "overlap": Path(record["overlap_diagnostics"]),
        "support_index": baseline / "diagnostics/support_points.json",
        "support_sidecar": baseline / "diagnostics/support_points.npz",
        "consistency": baseline / "diagnostics/consistency.json",
        "candidate": candidate / "diagnostics/g1_17_support_policy.json",
        "points": baseline / "geometry/points.ply",
    }


def analyze_scene(
    name: str,
    record: dict[str, Any],
    g1_17_scene: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Path]]:
    paths = scene_paths(record)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise PressureAnalysisError(f"{name} missing frozen inputs: {missing}")
    predictions = read_json(paths["predictions"])
    groups = read_json(paths["groups"])
    visibility = read_json(paths["visibility"])
    overlap = read_json(paths["overlap"])
    support_index = read_json(paths["support_index"])
    consistency = read_json(paths["consistency"])
    candidate = read_json(paths["candidate"])
    arrays = load_sidecar(paths["support_sidecar"], support_index)
    if (
        predictions.get("schema_version") != 1
        or groups.get("schema_version") != 1
        or overlap.get("schema_version") != 1
        or candidate.get("schema_version") != 1
    ):
        raise PressureAnalysisError(f"{name} has an unsupported frozen schema")

    image_names = [str(item["image"]) for item in support_index.get("images", [])]
    prediction_names = [str(item["image"]) for item in predictions.get("predictions", [])]
    unique_prediction_names = set(prediction_names)
    visibility_names = {str(item["image"]) for item in visibility.get("images", [])}
    if set(image_names) != unique_prediction_names or visibility_names != unique_prediction_names:
        raise PressureAnalysisError(f"{name} image inventories differ")
    image_count = len(unique_prediction_names)
    if (
        predictions.get("schema_version") != 1
        or predictions.get("unique_image_count") != image_count
        or groups.get("image_count") != image_count
        or visibility.get("image_count") != image_count
    ):
        raise PressureAnalysisError(f"{name} frozen image counts differ")
    source_index = Path(str(support_index.get("source_prediction_index", ""))).resolve()
    if source_index != paths["predictions"].resolve():
        raise PressureAnalysisError(f"{name} support sidecar references another prediction index")

    prediction_counts = Counter(prediction_names)
    multi_image_count = sum(count > 1 for count in prediction_counts.values())
    prediction_histogram = {
        str(key): value
        for key, value in sorted(Counter(prediction_counts.values()).items())
    }
    degrees = [len(item.get("neighbors", [])) for item in visibility["images"]]
    finite_overlap = arrays["overlap_disagreement"][
        np.isfinite(arrays["overlap_disagreement"])
    ]
    final_count = len(arrays["support_counts"])
    candidate_points = int(consistency["candidate_points"])
    candidate_accepted = int(candidate["filter"]["accepted_points"])
    baseline_accepted = int(consistency["accepted_points"])
    if (
        final_count != int(support_index["point_count"])
        or final_count != int(g1_17_scene["baseline"]["output_points"])
        or candidate_accepted != int(g1_17_scene["contradiction_free"]["accepted_points"])
    ):
        raise PressureAnalysisError(f"{name} G1.17 point counts differ")

    aggregate = overlap.get("aggregate", {})
    pair_metrics = aggregate.get("pair_metric_medians", {})
    result = {
        "image_count": image_count,
        "group_count": int(predictions["group_count"]),
        "prediction_count": len(prediction_names),
        "predictions_per_image": {
            "histogram": prediction_histogram,
            **distribution(list(prediction_counts.values())),
        },
        "multi_prediction_image_count": multi_image_count,
        "multi_prediction_image_fraction": fraction(multi_image_count, image_count),
        "overlap": {
            "evaluated_pair_count": int(aggregate.get("evaluated_pair_count", 0)),
            "anchored_absolute_p50": pair_metrics.get("anchored_absolute_p50"),
            "anchored_absolute_p90": pair_metrics.get("anchored_absolute_p90"),
        },
        "covisibility_degree": distribution(degrees),
        "final_points": {
            "point_count": final_count,
            "finite_overlap_count": len(finite_overlap),
            "finite_overlap_fraction": fraction(len(finite_overlap), final_count),
            "finite_overlap_p50": (
                float(np.percentile(finite_overlap, 50)) if len(finite_overlap) else None
            ),
            "finite_overlap_p90": (
                float(np.percentile(finite_overlap, 90)) if len(finite_overlap) else None
            ),
            "support_strata": support_strata(arrays["support_counts"]),
        },
        "candidate_visibility": {
            key: {
                "count": int(consistency[key]),
                "fraction": fraction(int(consistency[key]), candidate_points),
            }
            for key in (
                "occluded_only_points",
                "not_observed_only_points",
                "contradicted_only_points",
                "supported_and_contradicted_points",
            )
        },
        "g1_17": {
            "baseline_accepted_points": baseline_accepted,
            "candidate_accepted_points": candidate_accepted,
            "removed_points": baseline_accepted - candidate_accepted,
            "removed_from_baseline_fraction": fraction(
                baseline_accepted - candidate_accepted,
                baseline_accepted,
            ),
            "effect": (
                {
                    "private_rois": {
                        roi: {
                            "coverage_delta": (
                                g1_17_scene["contradiction_free"]["rois"][roi]["coverage"]
                                - g1_17_scene["baseline"]["rois"][roi]["coverage"]
                            ),
                            "layer_delta": (
                                g1_17_scene["contradiction_free"]["rois"][roi]["layers"]
                                - g1_17_scene["baseline"]["rois"][roi]["layers"]
                            ),
                            "thickness_delta": (
                                g1_17_scene["contradiction_free"]["rois"][roi]["thickness"]
                                - g1_17_scene["baseline"]["rois"][roi]["thickness"]
                            ),
                        }
                        for roi in g1_17_scene["baseline"]["rois"]
                    }
                }
                if "rois" in g1_17_scene["baseline"]
                else {
                    "eth3d_f1_delta": {
                        str(item["tolerance"]): item["f1"]
                        for item in g1_17_scene["deltas"]
                    }
                }
            ),
        },
        "source_hashes": {
            key: sha256_file(path)
            for key, path in paths.items()
        },
    }
    return result, arrays, support_index, paths


def overlap_summary(values: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, Any]:
    selected = values[mask]
    finite = np.isfinite(selected)
    high = finite & (selected >= threshold)
    return {
        "finite_count": int(finite.sum()),
        "finite_fraction": fraction(int(finite.sum()), len(selected)),
        "high_count": int(high.sum()),
        "high_fraction": fraction(int(high.sum()), len(selected)),
    }


def private_attribution(
    arrays: dict[str, np.ndarray],
    support_index: dict[str, Any],
    paths: dict[str, Path],
    *,
    roi_definition_path: Path,
    alignment_path: Path,
    expected_removed: int,
) -> dict[str, Any]:
    support = arrays["support_counts"]
    contradicted = arrays["contradicted_counts"]
    removed = (support > 0) & (contradicted > 0)
    removed_count = int(removed.sum())
    if removed_count != expected_removed:
        raise PressureAnalysisError(
            f"private removal count {removed_count} differs from G1.17 {expected_removed}"
        )
    image_records = support_index["images"]
    image_names = {int(item["source_image_index"]): str(item["image"]) for item in image_records}
    role_names = {
        int(code): str(role) for role, code in support_index["window_role_codes"].items()
    }
    threshold = float(read_json(paths["consistency"])["relative_threshold"])
    removed_overlap = overlap_summary(arrays["overlap_disagreement"], removed, threshold)

    points = read_ply_points(paths["points"])
    if len(points) != len(support):
        raise PressureAnalysisError("private point cloud and support sidecar lengths differ")
    selection_points = transform_points(points, alignment_path)
    roi_definition = read_definition(roi_definition_path)
    rois: dict[str, Any] = {}
    for roi in roi_definition["rois"]:
        lower = np.asarray(roi["min"], dtype=np.float64)
        upper = np.asarray(roi["max"], dtype=np.float64)
        roi_mask = np.all((selection_points >= lower) & (selection_points <= upper), axis=1)
        roi_removed = roi_mask & removed
        roi_count = int(roi_mask.sum())
        roi_removed_count = int(roi_removed.sum())
        rois[str(roi["name"])] = {
            "baseline_points": roi_count,
            "removed_points": roi_removed_count,
            "removed_fraction": fraction(roi_removed_count, roi_count),
            "retained_points": roi_count - roi_removed_count,
            "baseline_support_strata": support_strata(support[roi_mask]),
            "removed_support_strata": support_strata(support[roi_removed]),
            "baseline_overlap": overlap_summary(
                arrays["overlap_disagreement"], roi_mask, threshold
            ),
            "removed_overlap": overlap_summary(
                arrays["overlap_disagreement"], roi_removed, threshold
            ),
        }

    role_counts = Counter(int(value) for value in arrays["source_window_role"][removed])
    return {
        "mask": "support_counts > 0 & contradicted_counts > 0",
        "removed_points": removed_count,
        "removed_fraction": fraction(removed_count, len(removed)),
        "source_images": concentration_records(
            arrays["source_image_index"][removed], names=image_names
        ),
        "source_groups": concentration_records(arrays["source_group_index"][removed]),
        "window_roles": {
            role_names[key]: {"count": count, "fraction": fraction(count, removed_count)}
            for key, count in sorted(role_counts.items())
        },
        "support_strata": support_strata(support[removed]),
        "overlap_disagreement": {
            "threshold": threshold,
            **removed_overlap,
        },
        "fixed_rois": rois,
    }


def ranking(scenes: dict[str, Any], path: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    for name, scene in scenes.items():
        value: Any = scene
        for key in path:
            value = value[key]
        records.append({"scene": name, "value": value})
    return sorted(records, key=lambda item: (-item["value"], item["scene"]))


def analyze(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != 1:
        raise PressureAnalysisError("unsupported G1.17a-lite config schema")
    g1_path = Path(config["g1_17_summary"])
    g1 = read_json(g1_path)
    scene_configs = config.get("scenes")
    if not isinstance(scene_configs, dict) or not scene_configs:
        raise PressureAnalysisError("config must contain scenes")
    if set(scene_configs) != set(g1.get("scenes", {})):
        raise PressureAnalysisError("config and G1.17 scene inventories differ")

    scenes: dict[str, Any] = {}
    retained: dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Path]]] = {}
    for name, record in scene_configs.items():
        result, arrays, index, paths = analyze_scene(name, record, g1["scenes"][name])
        scenes[name] = result
        retained[name] = (arrays, index, paths)

    private_name = str(config["private_scene"])
    if private_name not in retained:
        raise PressureAnalysisError("private scene is not in scene inventory")
    arrays, index, paths = retained[private_name]
    attribution = private_attribution(
        arrays,
        index,
        paths,
        roi_definition_path=Path(config["private_roi_definition"]),
        alignment_path=Path(config["private_alignment"]),
        expected_removed=int(scenes[private_name]["g1_17"]["removed_points"]),
    )

    return {
        "schema_version": 1,
        "evaluation": "g1_17a_lite_scene_pressure",
        "protocol": {
            "inference_rerun": False,
            "colmap_rerun": False,
            "point_cloud_written": False,
            "eth3d_evaluator_rerun": False,
            "ground_truth_used": False,
            "threshold_fitted": False,
            "production_policy_changed": False,
        },
        "sources": {
            "config": config_path.as_posix(),
            "config_sha256": sha256_file(config_path),
            "g1_17_summary": g1_path.as_posix(),
            "g1_17_summary_sha256": sha256_file(g1_path),
        },
        "scenes": scenes,
        "pressure_rankings": {
            "image_count": ranking(scenes, ("image_count",)),
            "group_count": ranking(scenes, ("group_count",)),
            "multi_prediction_image_fraction": ranking(
                scenes, ("multi_prediction_image_fraction",)
            ),
            "finite_overlap_fraction": ranking(
                scenes, ("final_points", "finite_overlap_fraction")
            ),
            "candidate_mixed_conflict_fraction": ranking(
                scenes,
                ("candidate_visibility", "supported_and_contradicted_points", "fraction"),
            ),
            "g1_17_removal_fraction": ranking(
                scenes, ("g1_17", "removed_from_baseline_fraction")
            ),
        },
        "private_removal_attribution": attribution,
        "interpretation": {
            "observations": [
                "private-225 is the largest scene by images, groups, and absolute overlap pairs",
                "G1.17 effects remain mixed across smaller scenes, including delivery_area",
                "private-225 removals are attributable to mixed support-and-contradiction evidence",
            ],
            "limits": [
                "only four frozen scenes are compared",
                "only one large target-domain scene is available",
                "rank association does not establish causality",
                "no automatic activation threshold can be selected from this report",
            ],
            "decision": (
                "retain G1.7 and G1.17 as limited-evidence target-domain conditional research "
                "candidates; preserve existing production defaults"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        payload = analyze(args.config)
        content = json.dumps(payload, indent=2) + "\n"
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
                raise PressureAnalysisError(f"analysis output differs: {args.output}")
        else:
            if args.output.exists():
                raise PressureAnalysisError(f"output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
    except PressureAnalysisError as exc:
        parser.error(str(exc))
    print(f"{'verified' if args.check else 'wrote'} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
