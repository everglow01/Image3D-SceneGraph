from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image3d_scenegraph.video.keyframes import (
    V2_PROFILE_ID,
    materialize_video_candidates,
)
from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT,
    MIN_VIDEO_REGISTRATION_RATE,
    MIN_VIDEO_TEMPORAL_COVERAGE,
    analyze_registration_timeline,
)

MAX_RECOVERY_ROUNDS = 2
ROUND_BUDGET_FRACTION = 0.25
TOTAL_BUDGET_FRACTION = 0.50
PAIR_WINDOW_SECONDS = 4.0
GAP_BRIDGE_SECONDS = 2.0
MIN_POINT_RETENTION = 0.90


def sequential_overlap(selection: dict[str, Any]) -> int:
    selected = selection.get("selected")
    if selection.get("profile") != V2_PROFILE_ID or not isinstance(selected, list):
        raise ValueError("dynamic sequential overlap requires standard_v2 selection metadata")
    times = sorted(float(item["time_seconds"]) for item in selected)
    if len(times) < 2 or times[-1] <= times[0]:
        raise ValueError("video selection does not define an effective frame rate")
    effective_fps = len(times) / (times[-1] - times[0])
    return min(24, max(16, math.ceil(effective_fps * 4.0)))


def recover_video_registration(
    *,
    colmap: str,
    database_path: Path,
    image_dir: Path,
    initial_model: Path,
    selection_path: Path,
    video_source: Path,
    diagnostics_path: Path,
    use_gpu: bool,
    gpu_index: str | None,
    num_threads: int | None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any], list[str]]:
    command_logs: list[str] = []
    current_model = initial_model
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "profile": "video_registration_recovery_v1",
        "method": "incremental_colmap",
        "policy": {
            "maximum_rounds": MAX_RECOVERY_ROUNDS,
            "round_budget_fraction": ROUND_BUDGET_FRACTION,
            "total_budget_fraction": TOTAL_BUDGET_FRACTION,
            "pair_window_seconds": PAIR_WINDOW_SECONDS,
            "gap_bridge_seconds": GAP_BRIDGE_SECONDS,
            "minimum_point_retention": MIN_POINT_RETENTION,
        },
        "status": "unavailable",
        "reason": None,
        "rounds": [],
    }
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("profile") != V2_PROFILE_ID:
            diagnostics.update(status="not_applicable", reason="profile_is_not_standard_v2")
            _write_json(diagnostics_path, diagnostics)
            return current_model, diagnostics, command_logs
        selected = selection.get("selected")
        if not isinstance(selected, list) or not selected:
            raise ValueError("video selection metadata has no selected frames")
        initial_selected_count = len(selected)
        round_limit = math.ceil(initial_selected_count * ROUND_BUDGET_FRACTION)
        total_limit = math.ceil(initial_selected_count * TOTAL_BUDGET_FRACTION)
        diagnostics["initial_selected_count"] = initial_selected_count
        diagnostics["round_frame_limit"] = round_limit
        diagnostics["total_frame_limit"] = total_limit

        recovery_root = diagnostics_path.parent.parent / "colmap" / "video_recovery"
        initial_state = inspect_sparse_model(
            colmap,
            current_model,
            recovery_root / "initial_txt",
            command_logs,
        )
        current_state = initial_state
        initial_timeline = _timeline(selection, current_state["registered_names"])
        diagnostics["initial"] = _timeline_payload(initial_timeline, current_state)
        if not initial_timeline["gap_violations"]:
            diagnostics.update(status="not_needed", reason="no_registration_gaps")
            diagnostics["final"] = diagnostics["initial"]
            diagnostics["final_selected_count"] = initial_selected_count
            _write_json(diagnostics_path, diagnostics)
            return current_model, diagnostics, command_logs

        camera_id = read_unique_camera_id(database_path)
        cumulative_added = 0
        for round_index in range(1, MAX_RECOVERY_ROUNDS + 1):
            budget = min(round_limit, total_limit - cumulative_added)
            if budget <= 0:
                diagnostics["reason"] = "recovery_budget_exhausted"
                break
            before_timeline = _timeline(selection, current_state["registered_names"])
            candidates = plan_recovery_candidates(
                selection,
                before_timeline["gap_violations"],
                budget,
            )
            round_record: dict[str, Any] = {
                "round": round_index,
                "budget": budget,
                "before": _timeline_payload(before_timeline, current_state),
                "candidate_count": len(candidates),
                "accepted": False,
            }
            diagnostics["rounds"].append(round_record)
            if not candidates:
                round_record["reason"] = "no_viable_recovery_candidates"
                diagnostics["reason"] = round_record["reason"]
                break

            if progress is not None:
                progress(f"video_registration_recovery_round_{round_index}")
            paths = materialize_video_candidates(
                video_source,
                image_dir,
                candidates,
                selection,
            )
            proposed_selection = copy.deepcopy(selection)
            _append_materialized_selection(
                proposed_selection,
                candidates,
                paths,
                round_index,
            )
            round_record["materialized_count"] = len(paths)

            round_dir = recovery_root / f"round-{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            image_list_path = round_dir / "new-images.txt"
            image_list_path.write_text(
                "\n".join(path.name for path in paths) + "\n",
                encoding="utf-8",
            )
            selected_timestamps = _selected_timestamps(proposed_selection)
            pair_list = build_local_pairs(
                [path.name for path in paths],
                current_state["registered_names"],
                selected_timestamps,
            )
            pair_list_path = round_dir / "pairs.txt"
            pair_list_path.write_text(
                "\n".join(f"{left} {right}" for left, right in pair_list) + "\n",
                encoding="utf-8",
            )
            round_record["pair_count"] = len(pair_list)
            if not pair_list:
                round_record["reason"] = "no_local_match_pairs"
                diagnostics["reason"] = round_record["reason"]
                break

            feature_command = [
                colmap,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--image_list_path",
                str(image_list_path),
                "--ImageReader.existing_camera_id",
                str(camera_id),
                "--FeatureExtraction.use_gpu",
                "1" if use_gpu else "0",
            ]
            if gpu_index is not None:
                feature_command.extend(("--FeatureExtraction.gpu_index", gpu_index))
            if num_threads is not None:
                feature_command.extend(("--FeatureExtraction.num_threads", str(num_threads)))
            command_logs.append(run_command(feature_command))

            match_command = [
                colmap,
                "matches_importer",
                "--database_path",
                str(database_path),
                "--match_list_path",
                str(pair_list_path),
                "--match_type",
                "pairs",
                "--FeatureMatching.use_gpu",
                "1" if use_gpu else "0",
            ]
            if gpu_index is not None:
                match_command.extend(("--FeatureMatching.gpu_index", gpu_index))
            if num_threads is not None:
                match_command.extend(("--FeatureMatching.num_threads", str(num_threads)))
            command_logs.append(run_command(match_command))

            registered_dir = round_dir / "registered"
            triangulated_dir = round_dir / "triangulated"
            adjusted_dir = round_dir / "adjusted"
            for directory in (registered_dir, triangulated_dir, adjusted_dir):
                directory.mkdir()
            command_logs.append(
                run_command(
                    [
                        colmap,
                        "image_registrator",
                        "--database_path",
                        str(database_path),
                        "--input_path",
                        str(current_model),
                        "--output_path",
                        str(registered_dir),
                    ]
                )
            )
            triangulation_command = [
                colmap,
                "point_triangulator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--input_path",
                str(registered_dir),
                "--output_path",
                str(triangulated_dir),
                "--clear_points",
                "0",
            ]
            if num_threads is not None:
                triangulation_command.extend(("--Mapper.num_threads", str(num_threads)))
            command_logs.append(run_command(triangulation_command))
            command_logs.append(
                run_command(
                    [
                        colmap,
                        "bundle_adjuster",
                        "--input_path",
                        str(triangulated_dir),
                        "--output_path",
                        str(adjusted_dir),
                        "--BundleAdjustment.refine_focal_length",
                        "1",
                        "--BundleAdjustment.refine_principal_point",
                        "0",
                        "--BundleAdjustment.refine_extra_params",
                        "1",
                        "--BundleAdjustmentCeres.function_tolerance",
                        "0.000001",
                    ]
                )
            )
            after_state = inspect_sparse_model(
                colmap,
                adjusted_dir,
                round_dir / "adjusted_txt",
                command_logs,
            )
            after_timeline = _timeline(
                proposed_selection,
                after_state["registered_names"],
            )
            accepted, reason = accept_recovered_model(
                current_state,
                before_timeline,
                after_state,
                after_timeline,
            )
            round_record["after"] = _timeline_payload(after_timeline, after_state)
            round_record["accepted"] = accepted
            round_record["reason"] = reason
            if not accepted:
                diagnostics["reason"] = reason
                break
            _write_json(selection_path, proposed_selection)
            selection = proposed_selection
            cumulative_added += len(candidates)
            current_model = adjusted_dir
            current_state = after_state
            if not after_timeline["gap_violations"]:
                diagnostics["reason"] = "registration_gaps_closed"
                break

        final_timeline = _timeline(selection, current_state["registered_names"])
        diagnostics["final"] = _timeline_payload(final_timeline, current_state)
        diagnostics["final_selected_count"] = len(selection["selected"])
        diagnostics["recovery_selected_count"] = cumulative_added
        diagnostics["status"] = (
            "recovered" if not final_timeline["gap_violations"] else "partial"
        )
    except Exception as exc:
        diagnostics.update(status="unavailable", reason=str(exc))
    _write_json(diagnostics_path, diagnostics)
    return current_model, diagnostics, command_logs


def plan_recovery_candidates(
    selection: dict[str, Any],
    gaps: list[dict[str, float]],
    budget: int,
) -> list[dict[str, Any]]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or budget <= 0:
        return []
    eligible = [
        item
        for item in candidates
        if isinstance(item, dict)
        and not item.get("selected", False)
        and not item.get("rejection_reason")
        and any(
            float(gap["start_seconds"]) - GAP_BRIDGE_SECONDS
            <= float(item["time_seconds"])
            <= float(gap["end_seconds"]) + GAP_BRIDGE_SECONDS
            for gap in gaps
        )
    ]
    eligible.sort(key=lambda item: (float(item["time_seconds"]), int(item["pts"])))
    target = min(budget, len(eligible))
    if target == len(eligible):
        return eligible
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(target)]
    start = float(eligible[0]["time_seconds"])
    span = max(float(eligible[-1]["time_seconds"]) - start, 1e-9)
    for item in eligible:
        index = min(
            target - 1,
            int((float(item["time_seconds"]) - start) / span * target),
        )
        buckets[index].append(item)
    selected = [max(bucket, key=_candidate_key) for bucket in buckets if bucket]
    if len(selected) < target:
        selected_ids = {int(item["candidate_index"]) for item in selected}
        remaining = [
            item
            for item in eligible
            if int(item["candidate_index"]) not in selected_ids
        ]
        remaining.sort(key=_candidate_key, reverse=True)
        selected.extend(remaining[: target - len(selected)])
    selected.sort(key=lambda item: (float(item["time_seconds"]), int(item["pts"])))
    return selected


def build_local_pairs(
    new_names: list[str],
    registered_names: list[str],
    timestamps: dict[str, float],
) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    new_set = set(new_names)
    available = sorted(new_set | set(registered_names))
    for index, left in enumerate(available):
        if left not in timestamps:
            continue
        for right in available[index + 1 :]:
            if right not in timestamps or (left not in new_set and right not in new_set):
                continue
            if abs(timestamps[right] - timestamps[left]) <= PAIR_WINDOW_SECONDS:
                pairs.add((left, right))
    return sorted(pairs, key=lambda pair: (timestamps[pair[0]], timestamps[pair[1]], pair))


def accept_recovered_model(
    before_state: dict[str, Any],
    before_timeline: dict[str, Any],
    after_state: dict[str, Any],
    after_timeline: dict[str, Any],
) -> tuple[bool, str]:
    before_names = set(before_state["registered_names"])
    after_names = set(after_state["registered_names"])
    if not before_names <= after_names:
        return False, "rejected_registered_camera_loss"
    if int(after_state["point_count"]) < math.ceil(
        int(before_state["point_count"]) * MIN_POINT_RETENTION
    ):
        return False, "rejected_sparse_point_regression"
    if (
        int(after_timeline["registered_count"]) < MIN_VIDEO_REGISTERED_COUNT
        or float(after_timeline["registration_rate"])
        < MIN_VIDEO_REGISTRATION_RATE
        or float(after_timeline["temporal_coverage"])
        < MIN_VIDEO_TEMPORAL_COVERAGE
    ):
        return False, "rejected_registration_quality_gate"
    improved = (
        float(after_timeline["maximum_registered_gap_seconds"])
        < float(before_timeline["maximum_registered_gap_seconds"])
        or float(after_timeline["gap_violation_excess_seconds"])
        < float(before_timeline["gap_violation_excess_seconds"])
        or int(after_timeline["registered_count"])
        > int(before_timeline["registered_count"])
    )
    return (True, "accepted") if improved else (False, "rejected_no_strict_improvement")


def inspect_sparse_model(
    colmap: str,
    model_dir: Path,
    text_dir: Path,
    command_logs: list[str],
) -> dict[str, Any]:
    text_dir.mkdir(parents=True, exist_ok=True)
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )
    image_lines = [
        line
        for line in (text_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    point_lines = [
        line
        for line in (text_dir / "points3D.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    names: list[str] = []
    for line in image_lines:
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            int(parts[0])
            [float(value) for value in parts[1:8]]
            int(parts[8])
        except ValueError:
            continue
        names.append(parts[9])
    return {
        "registered_names": sorted(names),
        "registered_count": len(names),
        "point_count": len(point_lines),
    }


def read_unique_camera_id(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT camera_id FROM cameras ORDER BY camera_id").fetchall()
    if len(rows) != 1:
        raise ValueError("incremental recovery requires exactly one existing COLMAP camera")
    return int(rows[0][0])


def _timeline(selection: dict[str, Any], registered_names: list[str]) -> dict[str, Any]:
    return analyze_registration_timeline(_selected_timestamps(selection), registered_names)


def _selected_timestamps(selection: dict[str, Any]) -> dict[str, float]:
    selected = selection.get("selected")
    if not isinstance(selected, list):
        raise ValueError("video selection metadata has no selected frames")
    return {
        Path(str(item["path"])).name: float(item["time_seconds"])
        for item in selected
    }


def _timeline_payload(
    timeline: dict[str, Any], state: dict[str, Any]
) -> dict[str, int | float]:
    return {
        "selected_count": int(timeline["selected_count"]),
        "registered_count": int(timeline["registered_count"]),
        "registration_rate": float(timeline["registration_rate"]),
        "temporal_coverage": float(timeline["temporal_coverage"]),
        "maximum_registered_gap_seconds": float(
            timeline["maximum_registered_gap_seconds"]
        ),
        "gap_violation_count": int(timeline["gap_violation_count"]),
        "gap_violation_total_seconds": float(timeline["gap_violation_total_seconds"]),
        "gap_violation_excess_seconds": float(timeline["gap_violation_excess_seconds"]),
        "sparse_point_count": int(state["point_count"]),
    }


def _append_materialized_selection(
    selection: dict[str, Any],
    candidates: list[dict[str, Any]],
    paths: list[Path],
    round_index: int,
) -> None:
    selected = selection["selected"]
    reference = selected[0]
    relative_parent = Path(str(reference["path"])).parent
    candidate_records = {
        int(item["candidate_index"]): item
        for item in selection.get("candidates", [])
        if isinstance(item, dict) and "candidate_index" in item
    }
    for candidate, path in zip(candidates, paths):
        stored_candidate = candidate_records[int(candidate["candidate_index"])]
        stored_candidate["selected"] = True
        stored_candidate["selection_reason"] = f"recovery_round_{round_index}"
        selected.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "pts": int(candidate["pts"]),
                "time_seconds": float(candidate["time_seconds"]),
                "path": (relative_parent / path.name).as_posix(),
                "width": int(reference["width"]),
                "height": int(reference["height"]),
                "sha256": _sha256_file(path),
                "exif": {
                    "orientation": 1,
                    "software": f"Image3D-SceneGraph {V2_PROFILE_ID}",
                },
                "selection_reason": f"recovery_round_{round_index}",
            }
        )
    selected.sort(key=lambda item: (float(item["time_seconds"]), int(item["pts"])))
    selection["selected_count"] = len(selected)
    selection["recovery_selected_count"] = sum(
        str(item.get("selection_reason", "")).startswith("recovery_round_")
        for item in selected
    )


def _candidate_key(item: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(item.get("motion_score", item.get("novelty", 0.0))),
        float(item.get("quality_score", 0.0)),
        -int(item["pts"]),
    )


def run_command(command: list[str]) -> str:
    started_at = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return (
        "command="
        + " ".join(command)
        + "\nstdout="
        + completed.stdout.strip()
        + "\nstderr="
        + completed.stderr.strip()
        + f"\nelapsed_seconds={time.perf_counter() - started_at:.3f}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
