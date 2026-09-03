from __future__ import annotations

import copy
import json
import math
import sqlite3
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image3d_scenegraph.file_integrity import sha256_file
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
V2_MAPPER_SEED_MAX_IMAGES = 1_000
V2_INITIAL_EXPANSION_PASSES = 2
V2_MAPPER_OPTIONS = (
    ("--Mapper.ba_global_frames_ratio", "1.5"),
    ("--Mapper.ba_global_points_ratio", "1.5"),
    ("--Mapper.ba_global_frames_freq", "1000"),
    ("--Mapper.ba_global_points_freq", "1000000"),
    ("--Mapper.ba_global_max_refinements", "1"),
)


def v2_mapper_options(selection: dict[str, Any] | None) -> list[str]:
    if selection is None or selection.get("profile") != V2_PROFILE_ID:
        return []
    return [value for option in V2_MAPPER_OPTIONS for value in option]


def v2_mapper_seed_image_names(selection: dict[str, Any]) -> list[str]:
    if selection.get("profile") != V2_PROFILE_ID:
        raise ValueError("v2 Mapper seeding requires standard_v2 selection metadata")
    selected = selection.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("video selection metadata has no selected frames")
    ordered = sorted(
        selected,
        key=lambda item: (float(item["time_seconds"]), int(item["pts"])),
    )
    base = [item for item in ordered if item.get("selection_reason") == "base"]
    pool = base if len(base) >= V2_MAPPER_SEED_MAX_IMAGES else ordered
    target = min(V2_MAPPER_SEED_MAX_IMAGES, len(pool))
    if target == len(pool):
        seed = pool
    elif target == 1:
        seed = [pool[0]]
    else:
        indices = [
            round(index * (len(pool) - 1) / (target - 1))
            for index in range(target)
        ]
        seed = [pool[index] for index in indices]
    names = [Path(str(item["path"])).name for item in seed]
    if len(names) != len(set(names)):
        raise ValueError("v2 Mapper seed selection contains duplicate image names")
    return names


def sequential_overlap(selection: dict[str, Any]) -> int:
    selected = selection.get("selected")
    if selection.get("profile") != V2_PROFILE_ID or not isinstance(selected, list):
        raise ValueError("dynamic sequential overlap requires standard_v2 selection metadata")
    times = sorted(float(item["time_seconds"]) for item in selected)
    if len(times) < 2 or times[-1] <= times[0]:
        raise ValueError("video selection does not define an effective frame rate")
    effective_fps = len(times) / (times[-1] - times[0])
    return min(24, max(16, math.ceil(effective_fps * 4.0)))


def expand_v2_initial_registration(
    *,
    colmap: str,
    database_path: Path,
    image_dir: Path,
    initial_model: Path,
    selection: dict[str, Any],
    work_dir: Path,
    diagnostics_path: Path,
    num_threads: int | None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any], list[str]]:
    if selection.get("profile") != V2_PROFILE_ID:
        raise ValueError("initial registration expansion requires standard_v2")
    started_at = time.perf_counter()
    command_logs: list[str] = []
    current_model = initial_model
    current_state = inspect_sparse_model(
        colmap,
        current_model,
        work_dir / "initial_txt",
        command_logs,
    )
    initial_state = current_state
    initial_timeline = _timeline(selection, current_state["registered_names"])
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "profile": "video_initial_registration_expansion_v1",
        "maximum_passes": V2_INITIAL_EXPANSION_PASSES,
        "initial": _timeline_payload(initial_timeline, current_state),
        "passes": [],
    }
    for pass_index in range(1, V2_INITIAL_EXPANSION_PASSES + 1):
        if progress is not None:
            progress(f"video_initial_registration_expansion_pass_{pass_index}")
        pass_started_at = time.perf_counter()
        pass_dir = work_dir / f"pass-{pass_index:02d}"
        registered_dir = pass_dir / "registered"
        triangulated_dir = pass_dir / "triangulated"
        registered_dir.mkdir(parents=True)
        triangulated_dir.mkdir()
        record: dict[str, Any] = {
            "pass": pass_index,
            "before": _timeline_payload(
                _timeline(selection, current_state["registered_names"]),
                current_state,
            ),
            "accepted": False,
            "stage_elapsed_seconds": {},
        }
        diagnostics["passes"].append(record)
        record["stage_elapsed_seconds"]["image_registration"] = (
            _run_timed_command(
                [
                    colmap,
                    "image_registrator",
                    "--database_path",
                    str(database_path),
                    "--input_path",
                    str(current_model),
                    "--output_path",
                    str(registered_dir),
                ],
                command_logs,
            )
        )
        registered_state = inspect_sparse_model(
            colmap,
            registered_dir,
            pass_dir / "registered_txt",
            command_logs,
        )
        registered_timeline = _timeline(
            selection,
            registered_state["registered_names"],
        )
        record["post_registration"] = _timeline_payload(
            registered_timeline,
            registered_state,
        )
        record["registered_gain"] = int(registered_state["registered_count"]) - int(
            current_state["registered_count"]
        )
        if not set(current_state["registered_names"]) <= set(
            registered_state["registered_names"]
        ):
            record["reason"] = "rejected_registered_camera_loss"
            record["elapsed_seconds"] = time.perf_counter() - pass_started_at
            diagnostics["reason"] = record["reason"]
            break
        if record["registered_gain"] <= 0:
            record["reason"] = "no_registration_progress"
            record["elapsed_seconds"] = time.perf_counter() - pass_started_at
            diagnostics["reason"] = record["reason"]
            break

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
        record["stage_elapsed_seconds"]["triangulation"] = _run_timed_command(
            triangulation_command,
            command_logs,
        )
        after_state = inspect_sparse_model(
            colmap,
            triangulated_dir,
            pass_dir / "triangulated_txt",
            command_logs,
        )
        after_timeline = _timeline(selection, after_state["registered_names"])
        record["after"] = _timeline_payload(after_timeline, after_state)
        record["sparse_point_delta"] = int(after_state["point_count"]) - int(
            current_state["point_count"]
        )
        if not set(current_state["registered_names"]) <= set(
            after_state["registered_names"]
        ):
            record["reason"] = "rejected_registered_camera_loss"
        elif int(after_state["point_count"]) < math.ceil(
            int(current_state["point_count"]) * MIN_POINT_RETENTION
        ):
            record["reason"] = "rejected_sparse_point_regression"
        else:
            record["accepted"] = True
            record["reason"] = "accepted"
            current_model = triangulated_dir
            current_state = after_state
        record["elapsed_seconds"] = time.perf_counter() - pass_started_at
        if not record["accepted"]:
            diagnostics["reason"] = record["reason"]
            break
    else:
        diagnostics["reason"] = "maximum_passes_reached"

    final_timeline = _timeline(selection, current_state["registered_names"])
    initial_registered_names = set(initial_state["registered_names"])
    final_registered_names = set(current_state["registered_names"])
    retained_count = len(initial_registered_names & final_registered_names)
    diagnostics.update(
        status=(
            "expanded"
            if any(bool(record.get("accepted")) for record in diagnostics["passes"])
            else "no_progress"
        ),
        accepted_pass_count=sum(
            bool(record.get("accepted")) for record in diagnostics["passes"]
        ),
        final=_timeline_payload(final_timeline, current_state),
        registered_camera_retention={
            "initial_count": len(initial_registered_names),
            "retained_count": retained_count,
            "lost_count": len(initial_registered_names) - retained_count,
            "passed": initial_registered_names <= final_registered_names,
        },
        elapsed_seconds=time.perf_counter() - started_at,
    )
    _write_json(diagnostics_path, diagnostics)
    return current_model, diagnostics, command_logs


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
    feature_extraction_options: tuple[str, ...] = (),
    local_matching_options: tuple[str, ...] = (),
    geometric_verification_options: tuple[str, ...] = (),
    sfm_feature_profile: str = "sift_v1",
    sfm_local_matcher: str = "SIFT_BRUTEFORCE",
    sfm_geometric_verification: str = "default_v1",
    initial_sfm_pairing: str = "exhaustive",
    progress: Callable[[str], None] | None = None,
    force_final_bundle_adjustment: bool = False,
) -> tuple[Path, dict[str, Any], list[str]]:
    command_logs: list[str] = []
    current_model = initial_model
    recovery_started_at = time.perf_counter()
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "profile": "video_registration_recovery_v1",
        "method": "incremental_colmap",
        "sfm_feature_profile": sfm_feature_profile,
        "sfm_local_matcher": sfm_local_matcher,
        "sfm_geometric_verification": sfm_geometric_verification,
        "initial_sfm_pairing": initial_sfm_pairing,
        "recovery_pairing": "bounded_temporal_pair_list",
        "policy": {
            "maximum_rounds": MAX_RECOVERY_ROUNDS,
            "round_budget_fraction": ROUND_BUDGET_FRACTION,
            "total_budget_fraction": TOTAL_BUDGET_FRACTION,
            "pair_window_seconds": PAIR_WINDOW_SECONDS,
            "gap_bridge_seconds": GAP_BRIDGE_SECONDS,
            "minimum_point_retention": MIN_POINT_RETENTION,
            "registration_rounds_before_final_bundle_adjustment": MAX_RECOVERY_ROUNDS,
        },
        "status": "unavailable",
        "reason": None,
        "attempted_round_count": 0,
        "accepted_round_count": 0,
        "pair_count": 0,
        "elapsed_seconds": 0.0,
        "rounds": [],
        "final_bundle_adjustment": None,
    }
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("profile") != V2_PROFILE_ID:
            diagnostics.update(
                status="not_applicable",
                reason="profile_is_not_standard_v2",
                elapsed_seconds=time.perf_counter() - recovery_started_at,
            )
            _write_json(diagnostics_path, diagnostics)
            return current_model, diagnostics, command_logs
        selected = selection.get("selected")
        if not isinstance(selected, list) or not selected:
            raise ValueError("video selection metadata has no selected frames")
        initial_selected_count = len(selected)
        round_limit = math.ceil(initial_selected_count * ROUND_BUDGET_FRACTION)
        total_limit = math.ceil(initial_selected_count * TOTAL_BUDGET_FRACTION)
        diagnostics["initial_selected_count"] = initial_selected_count
        diagnostics["final_selected_count"] = initial_selected_count
        diagnostics["recovery_selected_count"] = 0
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
            if force_final_bundle_adjustment:
                current_model, current_state, adjustment = (
                    _run_final_bundle_adjustment(
                        colmap=colmap,
                        current_model=current_model,
                        current_state=current_state,
                        selection=selection,
                        recovery_root=recovery_root,
                        command_logs=command_logs,
                        use_gpu=use_gpu,
                        gpu_index=gpu_index,
                    )
                )
                diagnostics["final_bundle_adjustment"] = adjustment
            final_timeline = _timeline(
                selection,
                current_state["registered_names"],
            )
            diagnostics.update(
                status="not_needed",
                reason="no_registration_gaps",
                elapsed_seconds=time.perf_counter() - recovery_started_at,
            )
            diagnostics["final"] = _timeline_payload(final_timeline, current_state)
            diagnostics["registered_camera_retention"] = {
                "initial_count": len(initial_state["registered_names"]),
                "retained_count": len(initial_state["registered_names"]),
                "lost_count": 0,
                "passed": True,
            }
            diagnostics["final_selected_count"] = initial_selected_count
            _write_json(diagnostics_path, diagnostics)
            return current_model, diagnostics, command_logs

        camera_id = read_unique_camera_id(database_path)
        cumulative_added = 0
        previous_round_accepted = False
        for round_index in range(1, MAX_RECOVERY_ROUNDS + 1):
            round_started_at = time.perf_counter()
            budget = min(round_limit, total_limit - cumulative_added)
            before_timeline = _timeline(selection, current_state["registered_names"])
            candidates = (
                plan_recovery_candidates(
                    selection,
                    before_timeline["gap_violations"],
                    budget,
                )
                if budget > 0
                else []
            )
            propagation_only = (
                not candidates and round_index > 1 and previous_round_accepted
            )
            round_record: dict[str, Any] = {
                "round": round_index,
                "mode": "propagation" if propagation_only else "augmentation",
                "budget": budget,
                "before": _timeline_payload(before_timeline, current_state),
                "candidate_count": len(candidates),
                "materialized_count": 0,
                "pair_count": 0,
                "accepted": False,
                "stage_elapsed_seconds": {},
            }
            diagnostics["rounds"].append(round_record)
            if not candidates and not propagation_only:
                round_record["reason"] = (
                    "recovery_budget_exhausted"
                    if budget <= 0
                    else "no_viable_recovery_candidates"
                )
                round_record["elapsed_seconds"] = time.perf_counter() - round_started_at
                diagnostics["reason"] = round_record["reason"]
                break

            if progress is not None:
                progress(f"video_registration_recovery_round_{round_index}")
            proposed_selection = copy.deepcopy(selection)
            round_dir = recovery_root / f"round-{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            if candidates:
                materialize_started_at = time.perf_counter()
                paths = materialize_video_candidates(
                    video_source,
                    image_dir,
                    candidates,
                    selection,
                )
                round_record["stage_elapsed_seconds"]["materialization"] = (
                    time.perf_counter() - materialize_started_at
                )
                _append_materialized_selection(
                    proposed_selection,
                    candidates,
                    paths,
                    round_index,
                )
                round_record["materialized_count"] = len(paths)

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
                    round_record["elapsed_seconds"] = (
                        time.perf_counter() - round_started_at
                    )
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
                    *feature_extraction_options,
                ]
                if gpu_index is not None:
                    feature_command.extend(
                        ("--FeatureExtraction.gpu_index", gpu_index)
                    )
                if num_threads is not None:
                    feature_command.extend(
                        ("--FeatureExtraction.num_threads", str(num_threads))
                    )
                round_record["stage_elapsed_seconds"]["feature_extraction"] = (
                    _run_timed_command(feature_command, command_logs)
                )

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
                    *local_matching_options,
                    *geometric_verification_options,
                ]
                if gpu_index is not None:
                    match_command.extend(("--FeatureMatching.gpu_index", gpu_index))
                if num_threads is not None:
                    match_command.extend(
                        ("--FeatureMatching.num_threads", str(num_threads))
                    )
                round_record["stage_elapsed_seconds"]["matching"] = (
                    _run_timed_command(match_command, command_logs)
                )

            registered_dir = round_dir / "registered"
            registered_dir.mkdir()
            round_record["stage_elapsed_seconds"]["image_registration"] = (
                _run_timed_command(
                    [
                        colmap,
                        "image_registrator",
                        "--database_path",
                        str(database_path),
                        "--input_path",
                        str(current_model),
                        "--output_path",
                        str(registered_dir),
                    ],
                    command_logs,
                )
            )
            registered_state = inspect_sparse_model(
                colmap,
                registered_dir,
                round_dir / "registered_txt",
                command_logs,
            )
            registered_timeline = _timeline(
                proposed_selection,
                registered_state["registered_names"],
            )
            round_record["post_registration"] = _timeline_payload(
                registered_timeline,
                registered_state,
            )
            round_record["registered_gain"] = int(
                registered_timeline["registered_count"]
            ) - int(before_timeline["registered_count"])
            registration_accepted, registration_reason = accept_recovered_model(
                current_state,
                before_timeline,
                registered_state,
                registered_timeline,
            )
            if not registration_accepted:
                round_record["reason"] = registration_reason
                round_record["elapsed_seconds"] = (
                    time.perf_counter() - round_started_at
                )
                diagnostics["reason"] = registration_reason
                previous_round_accepted = False
                break

            triangulated_dir = round_dir / "triangulated"
            triangulated_dir.mkdir()
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
            round_record["stage_elapsed_seconds"]["triangulation"] = (
                _run_timed_command(triangulation_command, command_logs)
            )
            after_state = inspect_sparse_model(
                colmap,
                triangulated_dir,
                round_dir / "triangulated_txt",
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
            round_record["sparse_point_delta"] = int(after_state["point_count"]) - int(
                current_state["point_count"]
            )
            round_record["accepted"] = accepted
            round_record["reason"] = reason
            round_record["elapsed_seconds"] = time.perf_counter() - round_started_at
            if not accepted:
                diagnostics["reason"] = reason
                previous_round_accepted = False
                break
            if candidates:
                _write_json(selection_path, proposed_selection)
                selection = proposed_selection
                cumulative_added += len(candidates)
            current_model = triangulated_dir
            current_state = after_state
            previous_round_accepted = True
            if not after_timeline["gap_violations"]:
                diagnostics["reason"] = "registration_gaps_closed"
                break

        if (
            any(bool(record.get("accepted")) for record in diagnostics["rounds"])
            or force_final_bundle_adjustment
        ):
            current_model, current_state, adjustment = _run_final_bundle_adjustment(
                colmap=colmap,
                current_model=current_model,
                current_state=current_state,
                selection=selection,
                recovery_root=recovery_root,
                command_logs=command_logs,
                use_gpu=use_gpu,
                gpu_index=gpu_index,
            )
            diagnostics["final_bundle_adjustment"] = adjustment

        final_timeline = _timeline(selection, current_state["registered_names"])
        initial_registered_names = set(initial_state["registered_names"])
        final_registered_names = set(current_state["registered_names"])
        retained_count = len(initial_registered_names & final_registered_names)
        diagnostics["registered_camera_retention"] = {
            "initial_count": len(initial_registered_names),
            "retained_count": retained_count,
            "lost_count": len(initial_registered_names) - retained_count,
            "passed": initial_registered_names <= final_registered_names,
        }
        diagnostics["final"] = _timeline_payload(final_timeline, current_state)
        diagnostics["final_selected_count"] = len(selection["selected"])
        diagnostics["recovery_selected_count"] = cumulative_added
        diagnostics["attempted_round_count"] = len(diagnostics["rounds"])
        diagnostics["accepted_round_count"] = sum(
            bool(round_record.get("accepted"))
            for round_record in diagnostics["rounds"]
        )
        diagnostics["pair_count"] = sum(
            int(round_record.get("pair_count", 0))
            for round_record in diagnostics["rounds"]
        )
        diagnostics["elapsed_seconds"] = time.perf_counter() - recovery_started_at
        diagnostics["status"] = (
            "recovered" if not final_timeline["gap_violations"] else "partial"
        )
    except Exception as exc:
        diagnostics.update(
            status="unavailable",
            reason=str(exc),
            attempted_round_count=len(diagnostics["rounds"]),
            accepted_round_count=sum(
                bool(round_record.get("accepted"))
                for round_record in diagnostics["rounds"]
            ),
            pair_count=sum(
                int(round_record.get("pair_count", 0))
                for round_record in diagnostics["rounds"]
            ),
            elapsed_seconds=time.perf_counter() - recovery_started_at,
        )
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
    safe, reason = _validate_model_safety(
        before_state,
        after_state,
        after_timeline,
    )
    if not safe:
        return False, reason
    improved = (
        int(after_timeline["gap_violation_count"])
        < int(before_timeline["gap_violation_count"])
        or float(after_timeline["maximum_registered_gap_seconds"])
        < float(before_timeline["maximum_registered_gap_seconds"])
        or float(after_timeline["gap_violation_excess_seconds"])
        < float(before_timeline["gap_violation_excess_seconds"])
    )
    return (
        (True, "accepted")
        if improved
        else (False, "rejected_no_registration_gap_improvement")
    )


def _validate_model_safety(
    before_state: dict[str, Any],
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
    return True, "accepted"


def _run_final_bundle_adjustment(
    *,
    colmap: str,
    current_model: Path,
    current_state: dict[str, Any],
    selection: dict[str, Any],
    recovery_root: Path,
    command_logs: list[str],
    use_gpu: bool,
    gpu_index: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    record: dict[str, Any] = {
        "attempted": True,
        "accepted": False,
        "fallback_to_cpu": False,
        "attempts": [],
    }
    backends = ["cuda", "cpu"] if use_gpu else ["cpu"]
    adjusted_dir: Path | None = None
    for backend in backends:
        candidate_dir = recovery_root / f"final-adjusted-{backend}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        command = [
            colmap,
            "bundle_adjuster",
            "--input_path",
            str(current_model),
            "--output_path",
            str(candidate_dir),
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--BundleAdjustmentCeres.function_tolerance",
            "0.000001",
        ]
        if backend == "cuda":
            command.extend(("--BundleAdjustmentCeres.use_gpu", "1"))
            if gpu_index is not None:
                command.extend(
                    ("--BundleAdjustmentCeres.gpu_index", gpu_index.split(",")[0])
                )
        attempt_started_at = time.perf_counter()
        try:
            elapsed = _run_timed_command(command, command_logs)
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - attempt_started_at
            command_logs.append(_failed_command_log(command, exc, elapsed))
            record["attempts"].append(
                {
                    "backend": backend,
                    "status": "failed",
                    "elapsed_seconds": elapsed,
                    "reason": str(exc),
                }
            )
            if backend == "cuda":
                record["fallback_to_cpu"] = True
                continue
            record["reason"] = "bundle_adjustment_failed"
            record["elapsed_seconds"] = sum(
                float(item["elapsed_seconds"]) for item in record["attempts"]
            )
            return current_model, current_state, record
        record["attempts"].append(
            {
                "backend": backend,
                "status": "complete",
                "elapsed_seconds": elapsed,
            }
        )
        adjusted_dir = candidate_dir
        break

    if adjusted_dir is None:
        record["reason"] = "bundle_adjustment_failed"
        record["elapsed_seconds"] = sum(
            float(item["elapsed_seconds"]) for item in record["attempts"]
        )
        return current_model, current_state, record
    try:
        adjusted_state = inspect_sparse_model(
            colmap,
            adjusted_dir,
            recovery_root / f"{adjusted_dir.name}-txt",
            command_logs,
        )
        adjusted_timeline = _timeline(
            selection,
            adjusted_state["registered_names"],
        )
        accepted, reason = _validate_model_safety(
            current_state,
            adjusted_state,
            adjusted_timeline,
        )
    except Exception as exc:
        record["reason"] = f"bundle_adjustment_model_invalid: {exc}"
        record["elapsed_seconds"] = sum(
            float(item["elapsed_seconds"]) for item in record["attempts"]
        )
        return current_model, current_state, record
    record.update(
        accepted=accepted,
        reason=reason,
        after=_timeline_payload(adjusted_timeline, adjusted_state),
        elapsed_seconds=sum(
            float(item["elapsed_seconds"]) for item in record["attempts"]
        ),
    )
    if not accepted:
        return current_model, current_state, record
    return adjusted_dir, adjusted_state, record


def _failed_command_log(
    command: list[str],
    error: subprocess.CalledProcessError,
    elapsed_seconds: float,
) -> str:
    return (
        "command="
        + " ".join(command)
        + "\nstatus=failed"
        + "\nstdout="
        + str(error.stdout or "").strip()
        + "\nstderr="
        + str(error.stderr or "").strip()
        + f"\nelapsed_seconds={elapsed_seconds:.3f}"
    )


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
                "sha256": sha256_file(path),
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


def _run_timed_command(command: list[str], command_logs: list[str]) -> float:
    started_at = time.perf_counter()
    command_logs.append(run_command(command))
    return time.perf_counter() - started_at


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
