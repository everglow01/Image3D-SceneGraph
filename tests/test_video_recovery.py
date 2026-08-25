from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from image3d_scenegraph.geometry import video_recovery
from image3d_scenegraph.geometry.video_recovery import (
    accept_recovered_model,
    build_local_pairs,
    plan_recovery_candidates,
    recover_video_registration,
    sequential_overlap,
)
from image3d_scenegraph.video.keyframes import V2_PROFILE_ID, candidate_frame_filename


def _selection() -> dict:
    selected_times = [*map(float, range(6)), *map(float, range(10, 16))]
    candidates = []
    selected = []
    for index, timestamp in enumerate(map(float, range(16))):
        candidate = {
            "candidate_index": index,
            "pts": index * 10,
            "time_seconds": timestamp,
            "selected": timestamp in selected_times,
            "rejection_reason": None,
            "motion_score": timestamp / 16,
            "quality_score": 0.8,
        }
        candidates.append(candidate)
        if candidate["selected"]:
            selected.append(
                {
                    "candidate_index": index,
                    "pts": candidate["pts"],
                    "time_seconds": timestamp,
                    "path": f"frames/selected/{candidate_frame_filename(candidate)}",
                    "width": 1280,
                    "height": 720,
                    "sha256": "0" * 64,
                    "exif": {"orientation": 1, "software": V2_PROFILE_ID},
                    "selection_reason": "base",
                }
            )
    return {
        "schema_version": 2,
        "profile": V2_PROFILE_ID,
        "rotation": {"applied_degrees": 0},
        "candidates": candidates,
        "selected": selected,
        "selected_count": len(selected),
    }


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO cameras(camera_id) VALUES (7)")


def _registered_names(selection: dict) -> list[str]:
    return [Path(item["path"]).name for item in selection["selected"]]


def test_dynamic_sequential_overlap_covers_four_seconds() -> None:
    selection = {"profile": V2_PROFILE_ID, "selected": []}
    selection["selected"] = [{"time_seconds": index / 5} for index in range(21)]
    assert sequential_overlap(selection) == 21
    selection["selected"] = [{"time_seconds": index / 2} for index in range(21)]
    assert sequential_overlap(selection) == 16
    selection["selected"] = [{"time_seconds": index / 8} for index in range(21)]
    assert sequential_overlap(selection) == 24


def test_recovery_candidate_budget_fills_across_separated_gaps() -> None:
    selection = {
        "candidates": [
            {
                "candidate_index": index,
                "pts": index,
                "time_seconds": timestamp,
                "selected": False,
                "rejection_reason": None,
                "motion_score": index / 10,
                "quality_score": 0.5,
            }
            for index, timestamp in enumerate([1.0, 1.2, 1.4, 98.6, 98.8, 99.0])
        ]
    }
    selected = plan_recovery_candidates(
        selection,
        [
            {"start_seconds": 1.0, "end_seconds": 1.5, "seconds": 0.5},
            {"start_seconds": 98.5, "end_seconds": 99.0, "seconds": 0.5},
        ],
        4,
    )
    assert len(selected) == 4
    assert min(float(item["time_seconds"]) for item in selected) < 2.0
    assert max(float(item["time_seconds"]) for item in selected) > 98.0


def test_local_pairs_include_only_new_frames_within_four_seconds() -> None:
    timestamps = {"old-a.jpg": 0.0, "old-b.jpg": 10.0, "new-a.jpg": 4.0, "new-b.jpg": 8.0}
    pairs = build_local_pairs(
        ["new-a.jpg", "new-b.jpg"],
        ["old-a.jpg", "old-b.jpg"],
        timestamps,
    )
    assert pairs == [
        ("new-a.jpg", "old-a.jpg"),
        ("new-a.jpg", "new-b.jpg"),
        ("new-b.jpg", "old-b.jpg"),
    ]


def test_model_acceptance_rejects_camera_and_point_regressions() -> None:
    before_state = {"registered_names": ["a", "b"], "point_count": 100}
    before_timeline = {
        "maximum_registered_gap_seconds": 5.0,
        "gap_violation_excess_seconds": 3.0,
        "registered_count": 2,
    }
    improved = {
        "maximum_registered_gap_seconds": 2.0,
        "gap_violation_excess_seconds": 0.0,
        "registered_count": 3,
    }
    assert accept_recovered_model(
        before_state,
        before_timeline,
        {"registered_names": ["b", "c"], "point_count": 110},
        improved,
    ) == (False, "rejected_registered_camera_loss")
    assert accept_recovered_model(
        before_state,
        before_timeline,
        {"registered_names": ["a", "b", "c"], "point_count": 89},
        improved,
    ) == (False, "rejected_sparse_point_regression")
    weak_registration = {
        **improved,
        "registered_count": 12,
        "registration_rate": 0.69,
        "temporal_coverage": 1.0,
    }
    assert accept_recovered_model(
        before_state,
        before_timeline,
        {
            "registered_names": ["a", "b", *[f"new-{index}" for index in range(10)]],
            "point_count": 110,
        },
        weak_registration,
    ) == (False, "rejected_registration_quality_gate")


def test_sparse_model_inspection_handles_empty_observation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command):
        text_dir = Path(command[command.index("--output_path") + 1])
        (text_dir / "images.txt").write_text(
            "# images\n"
            "1 1 0 0 0 0 0 0 1 first.jpg\n\n"
            "2 1 0 0 0 0 0 0 1 second.jpg\n"
            "10 20 1 30 40 2\n",
            encoding="utf-8",
        )
        (text_dir / "points3D.txt").write_text("# points\n1 point\n", encoding="utf-8")
        return "ok"

    monkeypatch.setattr(video_recovery, "run_command", fake_run)
    state = video_recovery.inspect_sparse_model(
        "colmap", tmp_path / "model", tmp_path / "text", []
    )

    assert state == {
        "registered_names": ["first.jpg", "second.jpg"],
        "registered_count": 2,
        "point_count": 1,
    }


def test_incremental_recovery_reuses_database_and_accepts_improved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection()
    selection_path = tmp_path / "frames" / "selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    image_dir = tmp_path / "frames" / "selected"
    image_dir.mkdir()
    database_path = tmp_path / "colmap" / "database.db"
    database_path.parent.mkdir()
    _database(database_path)
    initial_model = tmp_path / "colmap" / "sparse" / "0"
    initial_model.mkdir(parents=True)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    initial_names = _registered_names(selection)
    commands: list[list[str]] = []

    def fake_materialize(_source, output_dir, candidates, _selection_payload):
        paths = []
        for candidate in candidates:
            path = output_dir / candidate_frame_filename(candidate)
            path.write_bytes(f"frame-{candidate['pts']}".encode())
            paths.append(path)
        return paths

    def fake_inspect(_colmap, model_dir, _text_dir, _logs):
        if model_dir == initial_model:
            return {
                "registered_names": initial_names,
                "registered_count": len(initial_names),
                "point_count": 100,
            }
        names = sorted(
            set(initial_names)
            | {path.name for path in image_dir.glob("*.jpg")}
        )
        return {
            "registered_names": names,
            "registered_count": len(names),
            "point_count": 110,
        }

    def fake_run(command):
        commands.append(command)
        return "command=" + " ".join(command)

    monkeypatch.setattr(video_recovery, "materialize_video_candidates", fake_materialize)
    monkeypatch.setattr(video_recovery, "inspect_sparse_model", fake_inspect)
    monkeypatch.setattr(video_recovery, "run_command", fake_run)

    model, diagnostics, _logs = recover_video_registration(
        colmap="colmap",
        database_path=database_path,
        image_dir=image_dir,
        initial_model=initial_model,
        selection_path=selection_path,
        video_source=source,
        diagnostics_path=tmp_path / "diagnostics" / "recovery.json",
        use_gpu=True,
        gpu_index="0,1",
        num_threads=12,
    )

    assert diagnostics["status"] == "recovered"
    assert diagnostics["recovery_selected_count"] == 3
    assert diagnostics["attempted_round_count"] == 1
    assert diagnostics["accepted_round_count"] == 1
    assert diagnostics["pair_count"] == diagnostics["rounds"][0]["pair_count"]
    assert diagnostics["elapsed_seconds"] >= 0
    assert diagnostics["rounds"][0]["accepted"] is True
    assert diagnostics["rounds"][0]["elapsed_seconds"] >= 0
    assert set(diagnostics["rounds"][0]["stage_elapsed_seconds"]) == {
        "materialization",
        "feature_extraction",
        "matching",
        "image_registration",
        "triangulation",
    }
    assert diagnostics["final_bundle_adjustment"]["accepted"] is True
    assert diagnostics["final_bundle_adjustment"]["attempts"][0]["backend"] == "cuda"
    assert model.name == "final-adjusted-cuda"
    assert [command[1] for command in commands] == [
        "feature_extractor",
        "matches_importer",
        "image_registrator",
        "point_triangulator",
        "bundle_adjuster",
    ]
    assert "--image_list_path" in commands[0]
    assert commands[0][commands[0].index("--ImageReader.existing_camera_id") + 1] == "7"
    assert commands[3][commands[3].index("--clear_points") + 1] == "0"
    updated = json.loads(selection_path.read_text(encoding="utf-8"))
    assert updated["selected_count"] == 15
    assert updated["recovery_selected_count"] == 3
    assert sum(bool(item["selected"]) for item in updated["candidates"]) == 15


def test_incremental_recovery_command_failure_keeps_initial_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection()
    selection_path = tmp_path / "frames" / "selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    image_dir = tmp_path / "frames" / "selected"
    image_dir.mkdir()
    database_path = tmp_path / "colmap" / "database.db"
    database_path.parent.mkdir()
    _database(database_path)
    initial_model = tmp_path / "colmap" / "sparse" / "0"
    initial_model.mkdir(parents=True)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    def fake_materialize(_source, output_dir, candidates, _selection_payload):
        paths = []
        for candidate in candidates:
            path = output_dir / candidate_frame_filename(candidate)
            path.write_bytes(b"frame")
            paths.append(path)
        return paths

    monkeypatch.setattr(video_recovery, "materialize_video_candidates", fake_materialize)
    monkeypatch.setattr(
        video_recovery,
        "inspect_sparse_model",
        lambda *_args: {
            "registered_names": _registered_names(selection),
            "registered_count": 12,
            "point_count": 100,
        },
    )
    monkeypatch.setattr(
        video_recovery,
        "run_command",
        lambda command: (_ for _ in ()).throw(RuntimeError("matching failed"))
        if command[1] == "matches_importer"
        else "ok",
    )

    model, diagnostics, _logs = recover_video_registration(
        colmap="colmap",
        database_path=database_path,
        image_dir=image_dir,
        initial_model=initial_model,
        selection_path=selection_path,
        video_source=source,
        diagnostics_path=tmp_path / "diagnostics" / "recovery.json",
        use_gpu=False,
        gpu_index=None,
        num_threads=None,
    )

    assert model == initial_model
    assert diagnostics["status"] == "unavailable"
    assert diagnostics["reason"] == "matching failed"
    assert json.loads(selection_path.read_text())["selected_count"] == 12
    assert json.loads((tmp_path / "diagnostics" / "recovery.json").read_text())["status"] == "unavailable"


def _propagation_selection() -> dict:
    selection = _selection()
    by_time = {float(item["time_seconds"]): item for item in selection["candidates"]}
    for timestamp in (6.0, 9.0):
        by_time[timestamp]["rejection_reason"] = "severe_blur"
    recovered_later = by_time[8.0]
    recovered_later["selected"] = True
    recovered_later["selection_reason"] = "base"
    selection["selected"].append(
        {
            "candidate_index": recovered_later["candidate_index"],
            "pts": recovered_later["pts"],
            "time_seconds": recovered_later["time_seconds"],
            "path": f"frames/selected/{candidate_frame_filename(recovered_later)}",
            "width": 1280,
            "height": 720,
            "sha256": "0" * 64,
            "exif": {"orientation": 1, "software": V2_PROFILE_ID},
            "selection_reason": "base",
        }
    )
    selection["selected"].sort(key=lambda item: float(item["time_seconds"]))
    selection["selected_count"] = len(selection["selected"])
    return selection


def test_second_round_propagates_without_new_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _propagation_selection()
    selection_path = tmp_path / "frames" / "selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    image_dir = tmp_path / "frames" / "selected"
    image_dir.mkdir()
    database_path = tmp_path / "colmap" / "database.db"
    database_path.parent.mkdir()
    _database(database_path)
    initial_model = tmp_path / "colmap" / "sparse" / "0"
    initial_model.mkdir(parents=True)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    initial_names = [
        name
        for name in _registered_names(selection)
        if "pts80.jpg" not in name
    ]
    candidate_name = candidate_frame_filename(selection["candidates"][7])
    commands: list[list[str]] = []

    def fake_materialize(_source, output_dir, candidates, _selection_payload):
        paths = []
        for candidate in candidates:
            path = output_dir / candidate_frame_filename(candidate)
            path.write_bytes(b"frame")
            paths.append(path)
        return paths

    def fake_inspect(_colmap, model_dir, _text_dir, _logs):
        model = str(model_dir)
        if model_dir == initial_model:
            names, points = initial_names, 100
        elif "round-01" in model:
            names, points = sorted([*initial_names, candidate_name]), 110
        else:
            names = sorted({*_registered_names(selection), candidate_name})
            points = 120
        return {
            "registered_names": names,
            "registered_count": len(names),
            "point_count": points,
        }

    def fake_run(command):
        commands.append(command)
        return "command=" + " ".join(command)

    monkeypatch.setattr(video_recovery, "materialize_video_candidates", fake_materialize)
    monkeypatch.setattr(video_recovery, "inspect_sparse_model", fake_inspect)
    monkeypatch.setattr(video_recovery, "run_command", fake_run)

    model, diagnostics, _logs = recover_video_registration(
        colmap="colmap",
        database_path=database_path,
        image_dir=image_dir,
        initial_model=initial_model,
        selection_path=selection_path,
        video_source=source,
        diagnostics_path=tmp_path / "diagnostics" / "recovery.json",
        use_gpu=False,
        gpu_index=None,
        num_threads=4,
    )

    assert diagnostics["status"] == "recovered"
    assert diagnostics["accepted_round_count"] == 2
    assert diagnostics["rounds"][1]["mode"] == "propagation"
    assert diagnostics["rounds"][1]["candidate_count"] == 0
    assert diagnostics["rounds"][1]["pair_count"] == 0
    assert model.name == "final-adjusted-cpu"
    command_names = [command[1] for command in commands]
    assert command_names.count("feature_extractor") == 1
    assert command_names.count("matches_importer") == 1
    assert command_names.count("image_registrator") == 2
    assert command_names.count("point_triangulator") == 2
    assert command_names.count("bundle_adjuster") == 1


def test_registration_without_gap_improvement_skips_expensive_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _propagation_selection()
    selection_path = tmp_path / "frames" / "selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    image_dir = tmp_path / "frames" / "selected"
    image_dir.mkdir()
    database_path = tmp_path / "colmap" / "database.db"
    database_path.parent.mkdir()
    _database(database_path)
    initial_model = tmp_path / "colmap" / "sparse" / "0"
    initial_model.mkdir(parents=True)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    initial_names = [
        name
        for name in _registered_names(selection)
        if "pts80.jpg" not in name
    ]
    commands: list[list[str]] = []

    def fake_materialize(_source, output_dir, candidates, _selection_payload):
        paths = []
        for candidate in candidates:
            path = output_dir / candidate_frame_filename(candidate)
            path.write_bytes(b"frame")
            paths.append(path)
        return paths

    monkeypatch.setattr(video_recovery, "materialize_video_candidates", fake_materialize)
    monkeypatch.setattr(
        video_recovery,
        "inspect_sparse_model",
        lambda *_args: {
            "registered_names": initial_names,
            "registered_count": len(initial_names),
            "point_count": 100,
        },
    )
    monkeypatch.setattr(
        video_recovery,
        "run_command",
        lambda command: commands.append(command) or "ok",
    )

    model, diagnostics, _logs = recover_video_registration(
        colmap="colmap",
        database_path=database_path,
        image_dir=image_dir,
        initial_model=initial_model,
        selection_path=selection_path,
        video_source=source,
        diagnostics_path=tmp_path / "diagnostics" / "recovery.json",
        use_gpu=False,
        gpu_index=None,
        num_threads=None,
    )

    assert model == initial_model
    assert diagnostics["status"] == "partial"
    assert diagnostics["reason"] == "rejected_no_registration_gap_improvement"
    assert diagnostics["accepted_round_count"] == 0
    assert diagnostics["final_bundle_adjustment"] is None
    assert [command[1] for command in commands] == [
        "feature_extractor",
        "matches_importer",
        "image_registrator",
    ]
    assert json.loads(selection_path.read_text())["selected_count"] == 13


def test_final_bundle_adjustment_falls_back_from_cuda_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection()
    names = _registered_names(selection)
    state = {
        "registered_names": names,
        "registered_count": len(names),
        "point_count": 100,
    }
    model = tmp_path / "model"
    model.mkdir()
    commands: list[list[str]] = []

    def fake_run(command):
        commands.append(command)
        if "--BundleAdjustmentCeres.use_gpu" in command:
            raise subprocess.CalledProcessError(
                1,
                command,
                output="",
                stderr="CUDA sparse solver unavailable",
            )
        return "ok"

    monkeypatch.setattr(video_recovery, "run_command", fake_run)
    monkeypatch.setattr(
        video_recovery,
        "inspect_sparse_model",
        lambda *_args: state,
    )

    adjusted, adjusted_state, record = video_recovery._run_final_bundle_adjustment(
        colmap="colmap",
        current_model=model,
        current_state=state,
        selection=selection,
        recovery_root=tmp_path / "recovery",
        command_logs=[],
        use_gpu=True,
        gpu_index="1,0",
    )

    assert adjusted.name == "final-adjusted-cpu"
    assert adjusted_state == state
    assert record["accepted"] is True
    assert record["fallback_to_cpu"] is True
    assert [attempt["status"] for attempt in record["attempts"]] == [
        "failed",
        "complete",
    ]
    assert [command[1] for command in commands] == [
        "bundle_adjuster",
        "bundle_adjuster",
    ]
    assert commands[0][commands[0].index("--BundleAdjustmentCeres.gpu_index") + 1] == "1"
    assert "--BundleAdjustmentCeres.use_gpu" not in commands[1]
