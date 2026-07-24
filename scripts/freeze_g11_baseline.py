#!/usr/bin/env python3
"""Freeze reproducible evidence for the G1.1 private-scene baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import struct
from pathlib import Path
from typing import Any, Iterable


REQUIRED_VIEW_NAMES = (
    "bookshelf_raw",
    "bookshelf_aligned",
    "monitor_raw",
    "monitor_aligned",
)

SOURCE_PATHS = {
    "manifest": "manifest.json",
    "run_log": "logs/run.log",
    "fusion": "diagnostics/fusion.json",
    "consistency": "diagnostics/consistency.json",
    "visibility_graph": "diagnostics/visibility_graph.json",
    "alignment": "diagnostics/alignment.json",
    "point_cloud_raw": "geometry/points.ply",
    "point_cloud_aligned": "geometry/points_aligned.ply",
}


class BaselineError(RuntimeError):
    """Raised when retained job evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError(f"expected a JSON object: {path}")
    return value


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BaselineError("cannot summarize an empty distribution")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float]:
    collected = list(values)
    return {
        "min": percentile(collected, 0.0),
        "p05": percentile(collected, 0.05),
        "p25": percentile(collected, 0.25),
        "p50": percentile(collected, 0.50),
        "p75": percentile(collected, 0.75),
        "p95": percentile(collected, 0.95),
        "max": percentile(collected, 1.0),
    }


def parse_run_log(path: Path) -> tuple[dict[str, str], str]:
    values: dict[str, str] = {}
    runner = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if not key or not all(character.isalnum() or character == "_" for character in key):
            continue
        values.setdefault(key, value)
        if key == "runner":
            runner = value
    if not runner:
        raise BaselineError(f"missing runner= line in {path}")
    return values, runner


def as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{label} is not an integer: {value!r}") from exc


def as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{label} is not numeric: {value!r}") from exc


def expect_equal(label: str, *values: Any) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise BaselineError(f"inconsistent {label}: {values!r}")


def safe_job_path(job_dir: Path, relative_path: str) -> Path:
    path = (job_dir / relative_path).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise BaselineError(f"path escapes job directory: {relative_path}") from exc
    if not path.is_file():
        raise BaselineError(f"missing retained artifact: {relative_path}")
    return path


def inventory(job_dir: Path, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    aggregate = hashlib.sha256()
    for item in inputs:
        relative_path = str(item.get("path", ""))
        path = safe_job_path(job_dir, relative_path)
        size = path.stat().st_size
        expected_size = as_int(item.get("size_bytes"), f"size for {relative_path}")
        expect_equal(f"size for {relative_path}", size, expected_size)
        filename = str(item.get("filename", ""))
        expect_equal(f"filename for {relative_path}", filename, path.name)
        record = {
            "filename": filename,
            "path": relative_path,
            "size_bytes": size,
            "sha256": sha256_file(path),
        }
        records.append(record)
        total_bytes += size
        aggregate.update(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return {
        "count": len(records),
        "total_bytes": total_bytes,
        "inventory_sha256": aggregate.hexdigest(),
        "files": records,
    }


def artifact_inventory(job_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, relative_path in SOURCE_PATHS.items():
        path = safe_job_path(job_dir, relative_path)
        result[name] = {
            "path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def png_dimensions(path: Path) -> list[int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise BaselineError(f"screenshot is not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return [width, height]


def view_evidence(job_dir: Path, views_path: Path | None) -> dict[str, Any]:
    if views_path is None:
        return {
            "definition_path": None,
            "definition_sha256": None,
            "complete": False,
            "views": [],
        }
    definitions = read_json(views_path)
    views = definitions.get("views")
    if not isinstance(views, list):
        raise BaselineError(f"views must be a list: {views_path}")
    by_name = {str(view.get("name")): view for view in views if isinstance(view, dict)}
    expect_equal("view names", set(by_name), set(REQUIRED_VIEW_NAMES))
    records: list[dict[str, Any]] = []
    complete = True
    for name in REQUIRED_VIEW_NAMES:
        view = by_name[name]
        screenshot_path = str(view.get("screenshot", ""))
        expected_filename = f"20260723_070028_024e9f25_g1_1_{name}.png"
        expect_equal(f"screenshot filename for {name}", Path(screenshot_path).name, expected_filename)
        state = view.get("state")
        state_complete = isinstance(state, dict) and bool(state)
        screenshot = job_dir / screenshot_path
        screenshot_present = screenshot.is_file()
        screenshot_record: dict[str, Any] = {
            "path": screenshot_path,
            "present": screenshot_present,
            "sha256": None,
            "dimensions": None,
        }
        dimensions = None
        if screenshot_present:
            dimensions = png_dimensions(screenshot)
            screenshot_record.update(
                sha256=sha256_file(screenshot), dimensions=dimensions
            )
        viewport_matches = (
            state_complete
            and isinstance(state.get("captureViewport"), list)
            and state["captureViewport"] == dimensions
        )
        complete = complete and state_complete and screenshot_present and viewport_matches
        records.append(
            {
                "name": name,
                "variant": view.get("variant"),
                "state": state,
                "state_complete": state_complete,
                "screenshot": screenshot_record,
                "screenshot_matches_capture_viewport": viewport_matches,
            }
        )
    return {
        "definition_path": views_path.as_posix(),
        "definition_sha256": sha256_file(views_path),
        "complete": complete,
        "views": records,
    }


def build_baseline(job_dir: Path, views_path: Path | None = None) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    manifest = read_json(safe_job_path(job_dir, SOURCE_PATHS["manifest"]))
    fusion = read_json(safe_job_path(job_dir, SOURCE_PATHS["fusion"]))
    consistency = read_json(safe_job_path(job_dir, SOURCE_PATHS["consistency"]))
    visibility = read_json(safe_job_path(job_dir, SOURCE_PATHS["visibility_graph"]))
    alignment = read_json(safe_job_path(job_dir, SOURCE_PATHS["alignment"]))
    log_values, runner = parse_run_log(safe_job_path(job_dir, SOURCE_PATHS["run_log"]))

    inputs = manifest.get("inputs")
    metrics = manifest.get("metrics")
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise BaselineError("manifest inputs must be a list of objects")
    if not isinstance(metrics, dict):
        raise BaselineError("manifest metrics must be an object")

    job_id = str(manifest.get("job_id"))
    expect_equal("job id", job_id, log_values.get("job_id"))
    expect_equal(
        "input count",
        len(inputs),
        as_int(metrics.get("num_inputs"), "manifest num_inputs"),
        as_int(log_values.get("num_inputs"), "log num_inputs"),
    )
    registered = as_int(metrics.get("registered_images"), "registered_images")
    scaled = as_int(metrics.get("scaled_images"), "scaled_images")
    points = as_int(metrics.get("num_points"), "num_points")
    expect_equal(
        "registered images",
        registered,
        as_int(log_values.get("registered_images"), "log registered_images"),
        as_int(fusion.get("registered_images"), "fusion registered_images"),
        as_int(visibility.get("image_count"), "visibility image_count"),
    )
    expect_equal("scaled images", scaled, as_int(log_values.get("scaled_images"), "log scaled_images"))
    expect_equal(
        "output points",
        points,
        as_int(log_values.get("num_points"), "log num_points"),
        as_int(consistency.get("accepted_points"), "accepted_points"),
        as_int(alignment.get("num_points"), "alignment num_points"),
    )

    fusion_images = fusion.get("images")
    consistency_images = consistency.get("images")
    if not isinstance(fusion_images, list) or len(fusion_images) != scaled:
        raise BaselineError("fusion image records do not match scaled image count")
    if not isinstance(consistency_images, list) or len(consistency_images) != registered:
        raise BaselineError("consistency image records do not match registered image count")

    support_fields = (
        "candidate_points",
        "accepted_points",
        "rejected_points",
        "unverified_points",
        "supported_points",
        "multi_visible_points",
        "policy_rejected_supported_points",
    )
    for field in support_fields:
        expect_equal(
            f"summed {field}",
            as_int(consistency.get(field), field),
            sum(as_int(image.get(field), f"{field} for image") for image in consistency_images),
        )

    point_budget = fusion.get("point_budget")
    if not isinstance(point_budget, dict):
        raise BaselineError("fusion point_budget must be an object")
    budget_input = as_int(point_budget.get("input_points"), "budget input")
    budget_output = as_int(point_budget.get("output_points"), "budget output")
    budget_applied = point_budget.get("applied")
    expect_equal("budget output and cloud points", budget_output, points)
    expect_equal("budget policy", point_budget.get("policy"), metrics.get("point_budget_policy"), log_values.get("point_budget_policy"))
    expect_equal("budget input", budget_input, as_int(metrics.get("point_budget_input_points"), "manifest budget input"))
    expect_equal("budget output", budget_output, as_int(metrics.get("point_budget_output_points"), "manifest budget output"))
    expect_equal("budget applied", budget_applied, metrics.get("point_budget_applied"), log_values.get("point_budget_applied") == "true")

    configuration = {
        "matcher": metrics.get("matcher"),
        "vggt_batch_size": as_int(metrics.get("vggt_batch_size"), "vggt_batch_size"),
        "requested_overlap_size": as_int(metrics.get("vggt_overlap_size"), "vggt_overlap_size"),
        "effective_overlap_size": as_int(metrics.get("overlap_size"), "overlap_size"),
        "vggt_grouping": metrics.get("vggt_grouping"),
        "fusion_mode": metrics.get("fusion_mode"),
        "fusion_intrinsics": metrics.get("fusion_intrinsics"),
        "confidence_percentile": as_float(metrics.get("conf_percentile"), "conf_percentile"),
        "confidence_threshold_scope": metrics.get("confidence_threshold_scope"),
        "consistency_support_policy": metrics.get("consistency_support_policy"),
        "max_points": as_int(metrics.get("max_points"), "max_points"),
        "point_budget_policy": metrics.get("point_budget_policy"),
    }
    for key in (
        "matcher",
        "vggt_batch_size",
        "vggt_grouping",
        "fusion_mode",
        "confidence_threshold_scope",
        "point_budget_policy",
    ):
        expect_equal(f"configuration {key}", str(configuration[key]), log_values.get(key))
    expect_equal("requested overlap", str(configuration["requested_overlap_size"]), log_values.get("vggt_overlap_size"))
    expect_equal("effective overlap", str(configuration["effective_overlap_size"]), log_values.get("overlap_size"))
    expect_equal("support policy", configuration["consistency_support_policy"], consistency.get("support_policy"), fusion.get("cross_view_filter", {}).get("support_policy"), log_values.get("consistency_support_policy"))
    expect_equal("confidence scope", configuration["confidence_threshold_scope"], consistency.get("confidence_threshold_scope"), fusion.get("cross_view_filter", {}).get("confidence_threshold_scope"), log_values.get("confidence_threshold_scope"))
    expect_equal("fusion intrinsics", configuration["fusion_intrinsics"], fusion.get("intrinsics_source"), log_values.get("fusion_intrinsics"))
    expect_equal("point cap", str(configuration["max_points"]), log_values.get("max_points"))
    expect_equal("confidence percentile", configuration["confidence_percentile"], as_float(log_values.get("conf_percentile"), "log conf_percentile"))

    scale_summary = {
        "count": len(fusion_images),
        "depth_scale": distribution(image["depth_scale"] for image in fusion_images),
        "observations": distribution(image["scale_observations"] for image in fusion_images),
        "log_mad": distribution(image["scale_log_mad"] for image in fusion_images),
    }
    support_totals = {field: as_int(consistency.get(field), field) for field in support_fields}
    support_summary = {
        "totals": support_totals,
        "accepted_fraction": support_totals["accepted_points"] / support_totals["candidate_points"],
        "accepted_unverified_fraction": support_totals["unverified_points"] / support_totals["accepted_points"],
        "per_image": {
            field: distribution(image[field] for image in consistency_images)
            for field in support_fields
        },
        "residual_p50": consistency.get("residual_p50"),
        "residual_p90": consistency.get("residual_p90"),
        "note": "These are aggregate/per-image categories, not per-final-point support provenance.",
    }

    num_groups = as_int(metrics.get("num_groups"), "num_groups")
    expect_equal("num groups", num_groups, as_int(log_values.get("num_groups"), "log num_groups"))
    phase3_inactive = budget_applied is False and budget_input == budget_output == points

    return {
        "schema_version": 1,
        "task": "G1.1",
        "job_id": job_id,
        "baseline_kind": "private_scene_no_gt_all_on",
        "inputs": inventory(job_dir, inputs),
        "source_artifacts": artifact_inventory(job_dir),
        "runner": {
            "command": runner,
            "argv": shlex.split(runner),
            "source": "logs/run.log runner= line",
            "manifest_command_note": "manifest metrics.command is a COLMAP subcommand, not the full reconstruction invocation",
        },
        "historical_provenance": {
            "environment_variables": "not_recorded",
            "checkpoint_sha256": "not_recorded",
            "source_commit": "not_recorded",
        },
        "configuration": configuration,
        "reconstruction": {
            "input_images": len(inputs),
            "registered_images": registered,
            "scaled_images": scaled,
            "colmap_sparse_points": as_int(metrics.get("colmap_points"), "colmap_points"),
            "vggt_groups": num_groups,
            "output_points": points,
            "visibility_directed_edges": as_int(visibility.get("directed_edge_count"), "directed_edge_count"),
            "runtime_seconds": {
                "colmap": as_float(metrics.get("colmap_seconds"), "colmap_seconds"),
                "vggt": as_float(metrics.get("vggt_seconds"), "vggt_seconds"),
                "elapsed": as_float(metrics.get("elapsed_seconds"), "elapsed_seconds"),
            },
        },
        "scale_distribution": scale_summary,
        "support_distribution": support_summary,
        "alignment": {
            "status": alignment.get("status"),
            "target_axis": alignment.get("target_axis"),
            "translate_plane_to_zero": alignment.get("translate_plane_to_zero"),
            "source_plane_inlier_ratio": alignment.get("source_plane", {}).get("inlier_ratio"),
            "source_plane_rms_distance": alignment.get("source_plane", {}).get("rms_distance"),
            "transform": alignment.get("transform"),
            "interpretation": "Aligned applies one global rigid transform; it does not repair local repeated geometry.",
        },
        "phase3_point_budget": {
            "requested_policy": point_budget.get("policy"),
            "max_points": configuration["max_points"],
            "input_points": budget_input,
            "output_points": budget_output,
            "applied": budget_applied,
            "inactive": phase3_inactive,
            "conclusion": "Policy requested but not activated; no point was budget-selected." if phase3_inactive else "Point budget changed the retained cloud.",
        },
        "qualitative_observation": {
            "user_confirmed": True,
            "variants": ["raw", "aligned"],
            "finding": "Bookshelf/monitor displacement is visible in both variants.",
            "scope": "Qualitative baseline only; quantitative ROI geometry metrics belong to G1.4.",
        },
        "views": view_evidence(job_dir, views_path),
        "limitations": [
            "Private scene with no geometric ground truth.",
            "World scale is arbitrary; this is not metric-scale recovery.",
            "No GT data was used for reconstruction, filtering, alignment, or baseline selection.",
            "Screenshots establish repeatable qualitative observations, not quantitative geometry scores.",
        ],
    }


def serialized(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--views", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output = args.job_dir / "diagnostics" / "g1_1_baseline.json"
    try:
        baseline = build_baseline(args.job_dir, args.views)
        content = serialized(baseline)
        if args.write:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            print(f"wrote {output}")
        else:
            if not output.is_file():
                raise BaselineError(f"missing frozen baseline: {output}")
            if output.read_text(encoding="utf-8") != content:
                raise BaselineError(f"frozen baseline differs from retained evidence: {output}")
            print(f"verified {output}")
        print(f"views_complete={str(baseline['views']['complete']).lower()}")
    except BaselineError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
