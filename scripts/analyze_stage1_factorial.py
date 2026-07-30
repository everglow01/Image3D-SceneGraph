from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
SCENES = ("pipes", "terrains", "delivery_area")
METRICS = ("accuracy", "completeness", "f1")
ARM_CONFIGS = {
    "baseline": (0, 0, 0),
    "phase1": (1, 0, 0),
    "phase2": (0, 1, 0),
    "phase1_phase2": (1, 1, 0),
    "phase3": (0, 0, 1),
    "phase1_phase3": (1, 0, 1),
    "phase2_phase3": (0, 1, 1),
    "phase1_phase2_phase3": (1, 1, 1),
}
LEVELS = {
    "confidence_threshold_scope": ("global", "per_frame"),
    "support_policy": ("any_support", "adaptive_two"),
    "point_budget_policy": ("random", "spatial_balanced"),
}
CONTRAST_FACTORS = {
    "A": (0,),
    "B": (1,),
    "C": (2,),
    "AB": (0, 1),
    "AC": (0, 2),
    "BC": (1, 2),
    "ABC": (0, 1, 2),
}


class FactorialAnalysisError(RuntimeError):
    """Raised when frozen Stage 1 evidence fails integrity validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FactorialAnalysisError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactorialAnalysisError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FactorialAnalysisError(f"{label} must contain a JSON object: {path}")
    return value


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_recorded_path(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FactorialAnalysisError(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    project_path = (PROJECT_ROOT / path).resolve()
    if project_path.exists():
        return project_path
    return (relative_to / path).resolve()


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorialAnalysisError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FactorialAnalysisError(f"{label} must be finite")
    return number


def find_fusion_report(factorial_dir: Path) -> Path:
    matches: list[Path] = []
    for path in sorted(factorial_dir.glob("runner_reconstruction*/diagnostics/fusion.json")):
        report = load_json(path, "factorial fusion diagnostics")
        arms = report.get("factorial_outputs")
        if isinstance(arms, list) and len(arms) == len(ARM_CONFIGS):
            matches.append(path)
    if len(matches) != 1:
        raise FactorialAnalysisError(
            f"Expected one eight-arm fusion report under {factorial_dir}, found {len(matches)}"
        )
    return matches[0]


def validate_metrics(result: dict[str, Any], label: str) -> list[dict[str, float]] | None:
    status = result.get("status")
    if status != "succeeded":
        metrics = result.get("metrics")
        if metrics not in (None, []):
            raise FactorialAnalysisError(f"{label} failed but contains non-empty metrics")
        return None
    evaluator = result.get("evaluator")
    if not isinstance(evaluator, dict) or evaluator.get("return_code") != 0:
        raise FactorialAnalysisError(f"{label} succeeded without evaluator return_code=0")
    rows = result.get("metrics")
    if not isinstance(rows, list) or not rows:
        raise FactorialAnalysisError(f"{label} succeeded without metrics")
    normalized: list[dict[str, float]] = []
    tolerances: set[float] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise FactorialAnalysisError(f"{label} metric {index} must be an object")
        tolerance = finite_number(row.get("tolerance"), f"{label} tolerance")
        if tolerance in tolerances:
            raise FactorialAnalysisError(f"{label} repeats tolerance {tolerance}")
        tolerances.add(tolerance)
        normalized.append(
            {
                "tolerance": tolerance,
                **{
                    metric: finite_number(row.get(metric), f"{label} {metric}")
                    for metric in METRICS
                },
            }
        )
    return sorted(normalized, key=lambda row: row["tolerance"])


def analyze_scene(scene: str, benchmark_root: Path) -> dict[str, Any]:
    factorial_dir = benchmark_root / scene / "phase3_factorial"
    fusion_path = find_fusion_report(factorial_dir)
    fusion = load_json(fusion_path, f"{scene} fusion diagnostics")
    summaries = fusion.get("factorial_outputs")
    if not isinstance(summaries, list):
        raise FactorialAnalysisError(f"{scene} factorial_outputs must be a list")

    summary_by_arm: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, dict) or not isinstance(summary.get("arm"), str):
            raise FactorialAnalysisError(f"{scene} has malformed factorial arm summary")
        arm = summary["arm"]
        if arm in summary_by_arm:
            raise FactorialAnalysisError(f"{scene} repeats factorial arm {arm}")
        summary_by_arm[arm] = summary
    if set(summary_by_arm) != set(ARM_CONFIGS):
        raise FactorialAnalysisError(
            f"{scene} factorial arms differ from contract: {sorted(summary_by_arm)}"
        )

    cells: list[dict[str, Any]] = []
    tolerance_grid: list[float] | None = None
    camera_hashes: set[str] = set()
    for arm, bits in ARM_CONFIGS.items():
        summary = summary_by_arm[arm]
        expected_levels = {
            name: levels[bits[index]]
            for index, (name, levels) in enumerate(LEVELS.items())
        }
        for name, expected in expected_levels.items():
            if summary.get(name) != expected:
                raise FactorialAnalysisError(
                    f"{scene}/{arm} expected {name}={expected}, got {summary.get(name)!r}"
                )

        ply_path = resolve_recorded_path(
            summary.get("path"), relative_to=factorial_dir, label=f"{scene}/{arm} PLY"
        )
        cameras_path = ply_path.with_name("cameras.json")
        if not ply_path.is_file() or not cameras_path.is_file():
            raise FactorialAnalysisError(f"{scene}/{arm} is missing PLY or cameras")
        evaluation_path = factorial_dir / "evaluations" / "arms" / arm / "result.json"
        result = load_json(evaluation_path, f"{scene}/{arm} result")
        inputs = result.get("inputs")
        if not isinstance(inputs, dict):
            raise FactorialAnalysisError(f"{scene}/{arm} result has no inputs")
        recorded_ply = resolve_recorded_path(
            inputs.get("reconstruction_ply"),
            relative_to=evaluation_path.parent,
            label=f"{scene}/{arm} result reconstruction",
        )
        if recorded_ply != ply_path:
            raise FactorialAnalysisError(
                f"{scene}/{arm} result references {recorded_ply}, expected {ply_path}"
            )
        actual_ply_hash = sha256_file(ply_path)
        if inputs.get("reconstruction_ply_sha256") != actual_ply_hash:
            raise FactorialAnalysisError(f"{scene}/{arm} PLY SHA-256 mismatch")
        metrics = validate_metrics(result, f"{scene}/{arm}")
        if metrics is not None:
            grid = [row["tolerance"] for row in metrics]
            if tolerance_grid is None:
                tolerance_grid = grid
            elif grid != tolerance_grid:
                raise FactorialAnalysisError(f"{scene}/{arm} tolerance grid mismatch")

        point_budget = {
            "input_points": int(summary["input_points"]),
            "output_points": int(summary["output_points"]),
            "applied": bool(summary["applied"]),
        }
        if point_budget["input_points"] < point_budget["output_points"]:
            raise FactorialAnalysisError(f"{scene}/{arm} point budget increases point count")
        camera_hash = sha256_file(cameras_path)
        camera_hashes.add(camera_hash)
        cells.append(
            {
                "arm": arm,
                "factor_bits": list(bits),
                "factors": expected_levels,
                "status": result.get("status"),
                "result": {
                    "path": display_path(evaluation_path),
                    "sha256": sha256_file(evaluation_path),
                },
                "point_cloud": {
                    "path": display_path(ply_path),
                    "sha256": actual_ply_hash,
                },
                "cameras": {
                    "path": display_path(cameras_path),
                    "sha256": camera_hash,
                },
                "point_budget": point_budget,
                "metrics": metrics,
                "failure": None
                if result.get("status") == "succeeded"
                else {
                    "kind": result.get("status", "unknown"),
                    "message": result.get("error"),
                },
            }
        )
    if len(camera_hashes) != 1:
        raise FactorialAnalysisError(f"{scene} factorial arms do not share identical cameras")
    if tolerance_grid is None:
        raise FactorialAnalysisError(f"{scene} has no successful metric cells")

    consistency_path = fusion_path.with_name("consistency.json")
    consistency = load_json(consistency_path, f"{scene} consistency diagnostics")
    budget_active = all(cell["point_budget"]["applied"] for cell in cells)
    return {
        "scene": scene,
        "source": {
            "factorial_directory": display_path(factorial_dir),
            "fusion": {"path": display_path(fusion_path), "sha256": sha256_file(fusion_path)},
            "consistency": {
                "path": display_path(consistency_path),
                "sha256": sha256_file(consistency_path),
            },
            "shared_camera_sha256": next(iter(camera_hashes)),
            "candidate_points": sorted({int(item["candidate_points"]) for item in summaries}),
        },
        "tolerances": tolerance_grid,
        "point_budget_factor": "active" if budget_active else "inactive",
        "cells": cells,
        "contrasts": calculate_contrasts(cells, tolerance_grid, budget_active),
        "consistency_status": consistency.get("status", "available"),
    }


def cell_metric_map(cells: Iterable[dict[str, Any]]) -> dict[tuple[int, int, int], dict[float, dict[str, float]]]:
    values: dict[tuple[int, int, int], dict[float, dict[str, float]]] = {}
    for cell in cells:
        bits = tuple(cell["factor_bits"])
        metrics = cell["metrics"]
        if metrics is not None:
            values[bits] = {row["tolerance"]: row for row in metrics}
    return values


def contrast_terms(factors: tuple[int, ...]) -> list[tuple[tuple[int, int, int], int]]:
    remaining = [index for index in range(3) if index not in factors]
    terms: list[tuple[tuple[int, int, int], int]] = []
    for fixed in itertools.product((0, 1), repeat=len(remaining)):
        for changed in itertools.product((0, 1), repeat=len(factors)):
            bits = [0, 0, 0]
            for index, value in zip(remaining, fixed, strict=True):
                bits[index] = value
            for index, value in zip(factors, changed, strict=True):
                bits[index] = value
            sign = -1 if (len(factors) - sum(changed)) % 2 else 1
            terms.append((tuple(bits), sign))
    return terms


def calculate_contrasts(
    cells: list[dict[str, Any]], tolerances: list[float], budget_active: bool
) -> list[dict[str, Any]]:
    values = cell_metric_map(cells)
    arm_by_bits = {bits: arm for arm, bits in ARM_CONFIGS.items()}
    records: list[dict[str, Any]] = []
    for tolerance in tolerances:
        for metric in METRICS:
            for name, factors in CONTRAST_FACTORS.items():
                terms = contrast_terms(factors)
                arms = [arm_by_bits[bits] for bits, _ in terms]
                if 2 in factors and not budget_active:
                    records.append(
                        {
                            "tolerance": tolerance,
                            "metric": metric,
                            "contrast": name,
                            "estimability": "inactive_point_budget",
                            "value": None,
                            "contributing_arms": arms,
                        }
                    )
                    continue
                missing = [arm_by_bits[bits] for bits, _ in terms if bits not in values]
                if missing:
                    records.append(
                        {
                            "tolerance": tolerance,
                            "metric": metric,
                            "contrast": name,
                            "estimability": "missing_or_failed_cells",
                            "value": None,
                            "contributing_arms": arms,
                            "missing_arms": missing,
                        }
                    )
                    continue
                total = sum(sign * values[bits][tolerance][metric] for bits, sign in terms)
                replicate_count = 2 ** (3 - len(factors))
                records.append(
                    {
                        "tolerance": tolerance,
                        "metric": metric,
                        "contrast": name,
                        "estimability": "estimable",
                        "value": total / replicate_count,
                        "contributing_arms": arms,
                    }
                )
    return records


def evidence_record(path: Path, status: str, interaction_status: str) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "status": status,
        "interaction_status": interaction_status,
    }


def build_evidence_inventory(
    *,
    g1_7_root: Path,
    g1_17_report: Path,
    g1_20_report: Path,
    g1_24_report: Path,
    private_fusion: Path,
) -> dict[str, Any]:
    g1_7_paths = [
        g1_7_root / scene / "evaluations" / "covisibility" / "result.json"
        for scene in SCENES
    ] + [
        g1_7_root / "private_225" / "evaluations" / name
        for name in ("sequential_g1_4_raw.json", "covisibility_g1_4_raw.json")
    ]
    for path in g1_7_paths:
        load_json(path, "G1.7 evidence")
    g1_17 = load_json(g1_17_report, "G1.17 report")
    g1_20 = load_json(g1_20_report, "G1.20 report")
    g1_24 = load_json(g1_24_report, "G1.24 report")
    private = load_json(private_fusion, "private retained fusion diagnostics")

    cross_view = private.get("cross_view_filter")
    budget = private.get("point_budget")
    if not isinstance(cross_view, dict) or not isinstance(budget, dict):
        raise FactorialAnalysisError("Private fusion diagnostics lack factor configuration")
    return {
        "g1_7_covisibility": {
            "status": "conditional_research_candidate",
            "interaction_status": "not_estimable_unrun_with_phase_factorial",
            "evidence": [
                {"path": display_path(path), "sha256": sha256_file(path)}
                for path in g1_7_paths
            ],
        },
        "g1_17_contradiction_free": {
            **evidence_record(
                g1_17_report, "rejected_default_promotion", "not_estimable_unrun_with_other_factors"
            ),
            "gate": g1_17.get("decision_gate"),
        },
        "g1_20_pixel_center_intrinsics": {
            **evidence_record(g1_20_report, "rejected_default_promotion", "not_eligible"),
            "evaluation": g1_20.get("evaluation"),
        },
        "g1_24_manhattan_alignment": {
            **evidence_record(
                g1_24_report, "private_orientation_candidate", "not_estimable_private_only"
            ),
            "selected_strategy": (g1_24.get("selection") or {}).get("selected_strategy"),
            "rigid_transform_invariants": (g1_24.get("metrics") or {}).get(
                "rigid_transform_invariants"
            ),
        },
        "private_retained_configuration": {
            "source": {"path": display_path(private_fusion), "sha256": sha256_file(private_fusion)},
            "confidence_threshold_scope": cross_view.get("confidence_threshold_scope"),
            "support_policy": cross_view.get("support_policy"),
            "point_budget_policy": budget.get("policy"),
            "point_budget_applied": budget.get("applied"),
            "status": "observed_configuration_not_factorial_interaction",
        },
    }


def descriptive_summary(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tolerance in (0.02, 0.05):
        for contrast in CONTRAST_FACTORS:
            values = []
            inactive = []
            for scene in scenes:
                match = next(
                    item
                    for item in scene["contrasts"]
                    if item["tolerance"] == tolerance
                    and item["metric"] == "f1"
                    and item["contrast"] == contrast
                )
                if match["estimability"] == "estimable":
                    values.append({"scene": scene["scene"], "value": match["value"]})
                else:
                    inactive.append(
                        {"scene": scene["scene"], "reason": match["estimability"]}
                    )
            records.append(
                {
                    "tolerance": tolerance,
                    "metric": "f1",
                    "contrast": contrast,
                    "positive_scenes": sum(item["value"] > 0 for item in values),
                    "negative_scenes": sum(item["value"] < 0 for item in values),
                    "zero_scenes": sum(item["value"] == 0 for item in values),
                    "values": values,
                    "not_estimable": inactive,
                }
            )
    return records


def analyze_stage1_factorial(
    *,
    benchmark_root: Path,
    scenes: Iterable[str],
    g1_7_root: Path,
    g1_17_report: Path,
    g1_20_report: Path,
    g1_24_report: Path,
    private_fusion: Path,
) -> dict[str, Any]:
    scene_reports = [analyze_scene(scene, benchmark_root) for scene in scenes]
    inventory = build_evidence_inventory(
        g1_7_root=g1_7_root,
        g1_17_report=g1_17_report,
        g1_20_report=g1_20_report,
        g1_24_report=g1_24_report,
        private_fusion=private_fusion,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "g1_25_stage1_combination_factorial",
        "protocol": {
            "frozen_artifacts_only": True,
            "reconstruction_rerun": False,
            "official_evaluator_rerun": False,
            "ground_truth_used_for_reconstruction_or_tuning": False,
            "eth3d_metric_label": "GT-camera-Sim(3)-aligned geometry quality",
            "metric_scale_recovered": False,
            "production_defaults_changed": False,
            "failed_or_mixed_results_preserved": True,
        },
        "factors": {
            "A": {"name": "confidence_threshold_scope", "levels": list(LEVELS["confidence_threshold_scope"])},
            "B": {"name": "support_policy", "levels": list(LEVELS["support_policy"])},
            "C": {"name": "point_budget_policy", "levels": list(LEVELS["point_budget_policy"])},
            "arm_contract": [
                {"arm": arm, "bits": list(bits)} for arm, bits in ARM_CONFIGS.items()
            ],
        },
        "scenes": scene_reports,
        "descriptive_cross_scene_f1": descriptive_summary(scene_reports),
        "factor_conclusions": {
            "A_per_frame_confidence": {
                "single_factor_status": "mixed_transfer",
                "combination_status": "not_robust_across_scenes",
                "default_eligible": False,
            },
            "B_adaptive_two_support": {
                "single_factor_status": "constructive_at_2cm_and_5cm",
                "combination_status": "balanced_effect_changes_sign_across_scenes",
                "default_eligible": False,
            },
            "C_spatial_balanced_budget": {
                "single_factor_status": "constructive_where_budget_active",
                "combination_status": "constructive_on_two_capped_eth3d_scenes",
                "default_eligible": False,
                "limitation": "inactive on pipes and retained private-225; no active non-ETH3D validation",
            },
        },
        "stage1_evidence_inventory": inventory,
        "unrun_interactions": [
            {
                "factors": ["covisibility_grouping", "A", "B", "C"],
                "status": "not_estimable",
                "reason": "not jointly generated from the frozen Phase 1/2/3 intermediates",
            },
            {
                "factors": ["contradiction_free", "A", "B", "C"],
                "status": "not_estimable",
                "reason": "candidate failed its universal transfer gate and joint arms were not run",
            },
            {
                "factors": ["manhattan_alignment", "A", "B", "C"],
                "status": "not_estimable",
                "reason": "private-only rigid orientation evidence; no same-source joint factorial",
            },
        ],
        "decision": {
            "status": "stage1_combination_evidence_frozen",
            "production_default_selection": "deferred_to_g1_26",
            "automatic_selector_fitted": False,
            "g1_26_started": False,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen Stage 1 geometry factorial.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--g1-7-root", type=Path, default=Path("outputs/experiments/g1_7"))
    parser.add_argument("--g1-17-report", type=Path, required=True)
    parser.add_argument("--g1-20-report", type=Path, required=True)
    parser.add_argument("--g1-24-report", type=Path, required=True)
    parser.add_argument(
        "--private-fusion",
        type=Path,
        default=Path("outputs/jobs/20260723_070028_024e9f25/diagnostics/fusion.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    try:
        payload = analyze_stage1_factorial(
            benchmark_root=args.benchmark_root,
            scenes=args.scenes,
            g1_7_root=args.g1_7_root,
            g1_17_report=args.g1_17_report,
            g1_20_report=args.g1_20_report,
            g1_24_report=args.g1_24_report,
            private_fusion=args.private_fusion,
        )
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not args.output.is_file():
                raise FactorialAnalysisError(f"Check output does not exist: {args.output}")
            if args.output.read_text(encoding="utf-8") != text:
                raise FactorialAnalysisError(f"Check output differs: {args.output}")
            print(f"check passed: {args.output}")
        else:
            write_json(args.output, payload)
            print(f"wrote {args.output}")
    except (OSError, ValueError, FactorialAnalysisError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
