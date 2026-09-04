from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from image3d_scenegraph.geometry.colmap import (
    ResolvedColmapFeatureProfile,
    ResolvedColmapLocalMatcher,
)
from scripts import run_colmap_sparse
from scripts.run_colmap_sparse import colmap_version, find_largest_sparse_model, read_sparse_model_counts


def _write_binary_count(path, count: int) -> None:
    path.write_bytes(struct.pack("<Q", count))


def _accept_pose_health(monkeypatch) -> None:
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_sfm_pose_health_from_text",
        lambda **_kwargs: {
            "status": "passed",
            "reason_codes": [],
            "temporal": {
                "registration_timeline": {
                    "registration_rate": 1.0,
                    "temporal_coverage": 1.0,
                }
            },
            "automatic_repair": {
                "eligible": False,
                "reason": "pose_health_passed",
            },
        },
    )


def test_sparse_model_selection_prefers_registered_images_then_points(tmp_path):
    first = tmp_path / "0"
    second = tmp_path / "1"
    third = tmp_path / "2"
    for path in (first, second, third):
        path.mkdir()

    _write_binary_count(first / "images.bin", 12)
    _write_binary_count(first / "points3D.bin", 5_000)
    _write_binary_count(second / "images.bin", 15)
    _write_binary_count(second / "points3D.bin", 1_000)
    _write_binary_count(third / "images.bin", 15)
    _write_binary_count(third / "points3D.bin", 2_000)

    assert read_sparse_model_counts(first) == (12, 5_000)
    assert find_largest_sparse_model(tmp_path) == (third, 15, 2_000)


def test_pose_selection_uses_healthy_primary_without_recovery(
    tmp_path, monkeypatch
):
    sparse = tmp_path / "sparse"
    model = sparse / "0"
    model.mkdir(parents=True)
    _write_binary_count(model / "images.bin", 20)
    _write_binary_count(model / "points3D.bin", 100)
    database = tmp_path / "database.db"
    sqlite3.connect(database).close()
    monkeypatch.setattr(
        run_colmap_sparse,
        "_evaluate_pose_candidate",
        lambda **kwargs: {
            "kind": kwargs["kind"],
            "status": "accepted",
            "accepted": True,
            "model_path": str(kwargs["model_dir"]),
            "database_path": str(kwargs["database_path"]),
            "registered_count": 20,
            "point_count": 100,
            "model_files_sha256": {
                "images.bin": hashlib.sha256(
                    (kwargs["model_dir"] / "images.bin").read_bytes()
                ).hexdigest(),
                "points3D.bin": hashlib.sha256(
                    (kwargs["model_dir"] / "points3D.bin").read_bytes()
                ).hexdigest(),
            },
            "pose_health": {"status": "passed", "reason_codes": []},
        },
    )
    monkeypatch.setattr(
        run_colmap_sparse,
        "_try_global_pose_recovery",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected Global")),
    )

    selected, registered, points, effective_database, record = (
        run_colmap_sparse.select_or_recover_sparse_model(
            colmap="colmap",
            sparse_dir=sparse,
            database_path=database,
            image_dir=tmp_path,
            work_dir=tmp_path / "work",
            output_dir=tmp_path,
            selected_timestamps=None,
            mapper_seed_path=None,
            use_gpu=False,
            gpu_index=None,
            num_threads=None,
            command_logs=[],
        )
    )

    assert (selected, registered, points, effective_database) == (
        model,
        20,
        100,
        database,
    )
    assert record["status"] == "not_needed"
    assert record["effective_mapper"] == "incremental"
    published = json.loads(
        (tmp_path / "diagnostics" / "sfm_pose_recovery.json").read_text()
    )
    assert published["selected"]["model_path"] == "sparse/0"
    assert published["selected"]["database_path"] == "database.db"
    assert published["source_database_sha256"] == hashlib.sha256(b"").hexdigest()
    assert published["effective_database_sha256"] == hashlib.sha256(b"").hexdigest()
    assert published["primary_candidates"][0]["model_files_sha256"] == {
        "images.bin": hashlib.sha256((model / "images.bin").read_bytes()).hexdigest(),
        "points3D.bin": hashlib.sha256(
            (model / "points3D.bin").read_bytes()
        ).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(published)


def test_pose_recovery_report_rejects_paths_outside_output(tmp_path):
    outside = tmp_path.parent / "outside-model"
    record = {
        "primary_candidates": [{"model_path": str(outside)}],
        "recovery_candidates": [],
    }

    with pytest.raises(RuntimeError, match="escapes the output directory"):
        run_colmap_sparse._relativize_pose_recovery_paths(record, tmp_path)


def test_pose_candidate_requires_video_registration_and_coverage_gates(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    _write_binary_count(model / "images.bin", 12)
    _write_binary_count(model / "points3D.bin", 100)

    def fake_run(command):
        output = Path(command[command.index("--output_path") + 1])
        output.joinpath("images.txt").write_text(
            "".join(
                f"{index + 1} 1 0 0 0 0 0 0 1 frame-{index:04d}.jpg\n\n"
                for index in range(12)
            ),
            encoding="utf-8",
        )
        return "ok"

    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_sfm_pose_health_from_text",
        lambda **_kwargs: {
            "status": "passed",
            "reason_codes": [],
            "temporal": {
                "registration_timeline": {
                    "registration_rate": 0.6,
                    "temporal_coverage": 11 / 19,
                }
            },
        },
    )
    candidate = run_colmap_sparse._evaluate_pose_candidate(
        colmap="colmap",
        model_dir=model,
        text_dir=tmp_path / "text",
        database_path=tmp_path / "database.db",
        selected_timestamps={
            f"frame-{index:04d}.jpg": float(index) for index in range(20)
        },
        command_logs=[],
        kind="global_recovery_v1",
    )

    assert candidate["accepted"] is False
    assert candidate["gate_reason_codes"] == [
        "registration_rate_below_gate",
        "temporal_coverage_below_gate",
    ]


def test_pose_candidate_failure_is_portable_and_fail_soft(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    _write_binary_count(model / "images.bin", 20)
    _write_binary_count(model / "points3D.bin", 100)
    monkeypatch.setattr(
        run_colmap_sparse,
        "_evaluate_pose_candidate_unchecked",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(f"failed at {tmp_path}")
        ),
    )

    candidate = run_colmap_sparse._evaluate_pose_candidate(
        colmap="colmap",
        model_dir=model,
        text_dir=tmp_path / "text",
        database_path=tmp_path / "database.db",
        selected_timestamps=None,
        command_logs=[],
        kind="incremental",
    )

    assert candidate["status"] == "failed"
    assert candidate["registered_count"] == 20
    assert candidate["reason"] == "candidate_evaluation_failed"
    assert candidate["error_type"] == "RuntimeError"
    record = {"primary_candidates": [candidate], "recovery_candidates": []}
    run_colmap_sparse._relativize_pose_recovery_paths(record, tmp_path)
    assert str(tmp_path) not in json.dumps(record)


def test_pose_selection_stops_recovery_after_healthy_global(
    tmp_path, monkeypatch
):
    sparse = tmp_path / "sparse"
    model = sparse / "0"
    model.mkdir(parents=True)
    _write_binary_count(model / "images.bin", 20)
    _write_binary_count(model / "points3D.bin", 100)
    database = tmp_path / "database.db"
    sqlite3.connect(database).close()
    failed = {
        "kind": "incremental",
        "status": "rejected",
        "accepted": False,
        "model_path": str(model),
        "database_path": str(database),
        "registered_count": 20,
        "point_count": 100,
        "pose_health": {
            "status": "failed",
            "reason_codes": ["multiscale_camera_pose_branch"],
            "automatic_repair": {"eligible": True, "excluded_image_ids": [7]},
        },
    }
    global_model = tmp_path / "global"
    global_database = tmp_path / "global.db"
    monkeypatch.setattr(
        run_colmap_sparse, "_evaluate_pose_candidate", lambda **_kwargs: failed
    )
    monkeypatch.setattr(
        run_colmap_sparse,
        "_try_global_pose_recovery",
        lambda **_kwargs: {
            **failed,
            "kind": "global_recovery_v1",
            "accepted": True,
            "status": "accepted",
            "model_path": str(global_model),
            "database_path": str(global_database),
            "registered_count": 18,
        },
    )
    monkeypatch.setattr(
        run_colmap_sparse,
        "_try_core_pose_repair",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected core repair")),
    )

    selected, registered, _, effective_database, record = (
        run_colmap_sparse.select_or_recover_sparse_model(
            colmap="colmap",
            sparse_dir=sparse,
            database_path=database,
            image_dir=tmp_path,
            work_dir=tmp_path / "work",
            output_dir=tmp_path,
            selected_timestamps=None,
            mapper_seed_path=None,
            use_gpu=False,
            gpu_index=None,
            num_threads=None,
            command_logs=[],
        )
    )

    assert selected == global_model
    assert registered == 18
    assert effective_database == global_database
    assert record["status"] == "recovered"
    assert [item["kind"] for item in record["recovery_candidates"]] == [
        "global_recovery_v1"
    ]


def test_pose_selection_repairs_eligible_primary_after_global_failure(
    tmp_path, monkeypatch
):
    sparse = tmp_path / "sparse"
    larger = sparse / "0"
    repairable = sparse / "1"
    larger.mkdir(parents=True)
    repairable.mkdir()
    for model, count in ((larger, 25), (repairable, 20)):
        _write_binary_count(model / "images.bin", count)
        _write_binary_count(model / "points3D.bin", 100)
    database = tmp_path / "database.db"
    sqlite3.connect(database).close()

    def evaluate(**kwargs):
        eligible = kwargs["model_dir"] == repairable
        return {
            "kind": "incremental",
            "status": "rejected",
            "accepted": False,
            "model_path": str(kwargs["model_dir"]),
            "database_path": str(database),
            "registered_count": 20 if eligible else 25,
            "point_count": 100,
            "pose_health": {
                "status": "failed",
                "reason_codes": ["multiscale_camera_pose_branch"],
                "automatic_repair": {"eligible": eligible},
            },
        }

    selected_primary = {}
    monkeypatch.setattr(run_colmap_sparse, "_evaluate_pose_candidate", evaluate)
    monkeypatch.setattr(
        run_colmap_sparse,
        "_try_global_pose_recovery",
        lambda **_kwargs: {
            "kind": "global_recovery_v1",
            "status": "failed",
            "accepted": False,
            "reason": "global_recovery_failed",
        },
    )

    def repair(**kwargs):
        selected_primary.update(kwargs["primary"])
        return {
            "kind": "incremental_core_repair_v1",
            "status": "accepted",
            "accepted": True,
            "model_path": str(tmp_path / "core"),
            "database_path": str(database),
            "registered_count": 19,
            "point_count": 90,
            "excluded_image_ids": [7],
        }

    monkeypatch.setattr(run_colmap_sparse, "_try_core_pose_repair", repair)
    selected, _, _, _, record = run_colmap_sparse.select_or_recover_sparse_model(
        colmap="colmap",
        sparse_dir=sparse,
        database_path=database,
        image_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
        selected_timestamps=None,
        mapper_seed_path=None,
        use_gpu=False,
        gpu_index=None,
        num_threads=None,
        command_logs=[],
    )

    assert selected_primary["model_path"] == str(repairable)
    assert selected == tmp_path / "core"
    assert record["effective_mapper"] == "incremental_core_repair_v1"


def test_global_pose_recovery_uses_database_copy_and_calibration(
    tmp_path, monkeypatch
):
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "global_mapper":
            output = Path(command[command.index("--output_path") + 1])
            _write_binary_count(output / "images.bin", 15)
            _write_binary_count(output / "points3D.bin", 100)
        return "ok"

    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "_evaluate_pose_candidate",
        lambda **kwargs: {
            "kind": kwargs["kind"],
            "status": "accepted",
            "accepted": True,
            "model_path": str(kwargs["model_dir"]),
            "database_path": str(kwargs["database_path"]),
            "registered_count": 15,
            "point_count": 100,
            "pose_health": {"status": "passed", "reason_codes": []},
        },
    )

    candidate = run_colmap_sparse._try_global_pose_recovery(
        colmap="colmap",
        database_path=database,
        image_dir=tmp_path / "images",
        recovery_dir=tmp_path / "recovery",
        selected_timestamps=None,
        mapper_seed_path=None,
        use_gpu=False,
        gpu_index=None,
        num_threads=4,
        command_logs=[],
    )

    assert candidate["accepted"] is True
    assert [command[1] for command in commands] == [
        "view_graph_calibrator",
        "global_mapper",
    ]
    assert all(
        command[command.index("--default_random_seed") + 1] == "0"
        for command in commands
    )
    copied = Path(commands[0][commands[0].index("--database_path") + 1])
    assert copied != database
    assert copied.is_file()


def test_core_pose_repair_runs_delete_filter_and_bundle_adjustment(
    tmp_path, monkeypatch
):
    commands = []
    monkeypatch.setattr(
        run_colmap_sparse,
        "run_command",
        lambda command: commands.append(command) or "ok",
    )
    monkeypatch.setattr(
        run_colmap_sparse,
        "_evaluate_pose_candidate",
        lambda **kwargs: {
            "kind": kwargs["kind"],
            "status": "accepted",
            "accepted": True,
            "model_path": str(kwargs["model_dir"]),
            "database_path": str(kwargs["database_path"]),
            "registered_count": 19,
            "point_count": 90,
            "pose_health": {"status": "passed", "reason_codes": []},
        },
    )
    primary = {
        "model_path": str(tmp_path / "primary"),
        "pose_health": {
            "automatic_repair": {
                "eligible": True,
                "reason": "eligible",
                "excluded_image_ids": [7, 9],
            }
        },
    }

    candidate = run_colmap_sparse._try_core_pose_repair(
        colmap="colmap",
        primary=primary,
        database_path=tmp_path / "database.db",
        recovery_dir=tmp_path / "repair",
        selected_timestamps=None,
        use_gpu=False,
        gpu_index=None,
        command_logs=[],
    )

    assert candidate["accepted"] is True
    assert candidate["excluded_image_ids"] == [7, 9]
    assert [command[1] for command in commands] == [
        "image_deleter",
        "point_filtering",
        "bundle_adjuster",
    ]
    bundle_adjuster = commands[-1]
    assert bundle_adjuster[
        bundle_adjuster.index("--default_random_seed") + 1
    ] == "0"
    assert "--random_seed" not in bundle_adjuster
    assert (tmp_path / "repair" / "excluded-image-ids.txt").read_text() == "7\n9\n"


def test_sparse_model_counts_support_text_models(tmp_path):
    model = tmp_path / "0"
    model.mkdir()
    (model / "images.txt").write_text(
        "# images\n1 pose.jpg\n1 2 3\n2 pose.jpg\n4 5 6\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "# points\n1 0 0 0 1 2 3 0.1 1 0\n2 0 0 0 1 2 3 0.1 1 0\n",
        encoding="utf-8",
    )

    assert read_sparse_model_counts(model) == (2, 2)


def test_colmap_version_includes_cuda_build_line(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess(
            [],
            0,
            "COLMAP 4.0.0 -- Structure-from-Motion and Multi-View Stereo\n"
            "(Commit 8bac7b9 on 2026-03-15 with CUDA)\n",
            "",
        ),
    )

    assert colmap_version("colmap") == (
        "COLMAP 4.0.0 -- Structure-from-Motion and Multi-View Stereo "
        "(Commit 8bac7b9 on 2026-03-15 with CUDA)"
    )


@pytest.mark.parametrize(
    ("help_output", "expected"),
    [
        ("  --Mapper.image_list_path arg\n", "--Mapper.image_list_path"),
        ("  --image_list_path arg\n", "--image_list_path"),
    ],
)
def test_mapper_image_list_option_supports_both_colmap_clis(
    monkeypatch, help_output, expected
):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], 0, help_output, ""),
    )

    assert run_colmap_sparse.mapper_image_list_option("colmap") == expected


def test_run_command_preserves_colmap_stderr(monkeypatch):
    command = ["colmap", "mapper"]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: (_ for _ in ()).throw(
            subprocess.CalledProcessError(
                1,
                command,
                output="mapper stdout",
                stderr="unrecognised option",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="unrecognised option"):
        run_colmap_sparse.run_command(command)


def test_gpu_indices_accept_multiple_visible_devices():
    assert run_colmap_sparse.parse_gpu_indices("0,1") == "0,1"
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        run_colmap_sparse.parse_gpu_indices("0,-1")


def test_runner_applies_thread_limit_and_writes_progress(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    progress_path = tmp_path / "progress.json"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 1)
            _write_binary_count(model / "points3D.bin", 1)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            path = output_dir / "geometry" / "points.ply"
            path.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
        elif command[1] == "model_converter" and command[-1] == "TXT":
            path = output_dir / "colmap" / "sparse_txt"
            (path / "cameras.txt").write_text(
                "1 PINHOLE 64 64 50 50 32 32\n", encoding="utf-8"
            )
            (path / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 frame.jpg\n\n", encoding="utf-8"
            )
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(
        run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0 with CUDA"
    )
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--no-use-gpu",
            "--gpu-index",
            "0",
            "--num-threads",
            "4",
            "--progress-file",
            str(progress_path),
        ],
    )

    run_colmap_sparse.main()

    feature, matcher, mapper = commands[:3]
    assert feature[feature.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert feature[feature.index("--FeatureExtraction.gpu_index") + 1] == "0"
    assert feature[feature.index("--FeatureExtraction.num_threads") + 1] == "4"
    assert feature[feature.index("--FeatureExtraction.type") + 1] == "SIFT"
    assert feature[feature.index("--SiftExtraction.max_num_features") + 1] == "8192"
    assert matcher[matcher.index("--FeatureMatching.use_gpu") + 1] == "0"
    assert matcher[matcher.index("--FeatureMatching.gpu_index") + 1] == "0"
    assert matcher[matcher.index("--FeatureMatching.num_threads") + 1] == "4"
    assert matcher[matcher.index("--FeatureMatching.type") + 1] == "SIFT_BRUTEFORCE"
    assert matcher[matcher.index("--default_random_seed") + 1] == "0"
    assert "--random_seed" not in matcher
    assert mapper[mapper.index("--Mapper.num_threads") + 1] == "4"
    assert mapper[mapper.index("--default_random_seed") + 1] == "0"
    assert "--random_seed" not in mapper
    assert "--Mapper.ba_global_frames_ratio" not in mapper
    assert "--Mapper.image_list_path" not in mapper
    assert "--image_list_path" not in mapper
    assert json.loads(progress_path.read_text()) == {"stage": "colmap_mapping"}
    log = (output_dir / "logs" / "run.log").read_text()
    assert "colmap_executable=" in log
    assert "colmap_build=COLMAP 4.0.0 with CUDA\n" in log
    assert "use_gpu=False\n" in log
    assert "gpu_index=0\n" in log
    assert "num_threads=4\n" in log


def test_runner_batches_auto_grouped_camera_extraction(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    exif = Image.Exif()
    exif[271] = "Maker"
    exif[272] = "Body"
    exif[37386] = 24.0
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (64, 48)).save(image_dir / name, exif=exif)
    Image.new("RGB", (64, 48)).save(image_dir / "c.jpg")
    commands = []
    diagnostics_call = {}

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 3)
            _write_binary_count(model / "points3D.bin", 1)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            (output_dir / "geometry" / "points.ply").write_text(
                "ply\nelement vertex 1\nend_header\n", encoding="utf-8"
            )
        return "ok"

    camera_payload = {
        "cameras": [
            {
                "camera_id": 1,
                "model": "SIMPLE_RADIAL",
                "width": 64,
                "height": 48,
                "params": [50.0, 32.0, 24.0, 0.0],
            },
            {
                "camera_id": 2,
                "model": "SIMPLE_RADIAL",
                "width": 64,
                "height": 48,
                "params": [50.0, 32.0, 24.0, 0.0],
            },
        ],
        "images": [
            {"image_id": 1, "name": "a.jpg", "camera_id": 1},
            {"image_id": 2, "name": "b.jpg", "camera_id": 1},
            {"image_id": 3, "name": "c.jpg", "camera_id": 2},
        ],
    }

    def fake_diagnostics(**kwargs):
        diagnostics_call.update(kwargs)
        plan = kwargs["plan"]
        return {
            "schema_version": 1,
            "profile": "sfm_camera_calibration_diagnostics_v1",
            "calibration": plan.calibration.provenance(),
            "grouping": {"planned_camera_count": 2},
            "initial": {"camera_count": 2, "prior_focal_camera_count": 1},
            "final": {
                "camera_count": 2,
                "median_focal_length_ratio": 0.8,
            },
            "sparse": {
                "median_reprojection_error_pixels": 0.5,
                "median_track_length": 2.0,
            },
            "plausibility": {"warning_count": 0},
        }

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse, "build_camera_payload", lambda _: camera_payload
    )
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_calibration_diagnostics",
        fake_diagnostics,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--camera-calibration",
            "auto_grouped_simple_radial_v1",
        ],
    )

    run_colmap_sparse.main()

    feature_commands = [command for command in commands if command[1] == "feature_extractor"]
    assert len(feature_commands) == 2
    assert [
        Path(command[command.index("--image_list_path") + 1]).read_text().splitlines()
        for command in feature_commands
    ] == [["a.jpg", "b.jpg"], ["c.jpg"]]
    assert feature_commands[0][
        feature_commands[0].index("--ImageReader.single_camera") + 1
    ] == "1"
    assert feature_commands[1][
        feature_commands[1].index("--ImageReader.single_camera_per_image") + 1
    ] == "1"
    assert diagnostics_call["plan"].calibration.profile_id == (
        "auto_grouped_simple_radial_v1"
    )
    assert (
        output_dir / "diagnostics" / "sfm_camera_calibration.json"
    ).is_file()


def test_gaussian_runner_caps_undistorted_images_and_uses_all_visible_gpus(
    tmp_path, monkeypatch
):
    _accept_pose_health(monkeypatch)
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 12)
            _write_binary_count(model / "points3D.bin", 100)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            path = output_dir / "geometry" / "points.ply"
            path.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_payload",
        lambda _: {
            "cameras": [{"model": "PINHOLE"}],
            "images": [{"image_id": index} for index in range(12)],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--matcher",
            "exhaustive",
            "--gaussian-baseline",
            "--max-image-size",
            "3072",
        ],
    )

    run_colmap_sparse.main()

    feature, matcher = commands[:2]
    undistorter = next(command for command in commands if command[1] == "image_undistorter")
    assert "--FeatureExtraction.gpu_index" not in feature
    assert "--FeatureMatching.gpu_index" not in matcher
    assert undistorter[undistorter.index("--max_image_size") + 1] == "3072"
    log = (output_dir / "logs" / "run.log").read_text()
    assert "gpu_index=all_visible\n" in log
    assert "max_image_size=3072\n" in log


def test_runner_applies_aliked_feature_profile(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []
    profile = ResolvedColmapFeatureProfile(
        profile_id="aliked_n16rot_v1",
        extractor="ALIKED_N16ROT",
        descriptor="ALIKED",
        max_features=8_192,
        extraction_options=(
            "--FeatureExtraction.type",
            "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features",
            "8192",
            "--AlikedExtraction.min_score",
            "0.2",
            "--AlikedExtraction.n16rot_model_path",
            "/models/aliked.onnx",
        ),
        extractor_model_sha256="a" * 64,
    )
    local_matcher = ResolvedColmapLocalMatcher(
        profile_id="lightglue",
        name="ALIKED_LIGHTGLUE",
        matching_options=(
            "--FeatureMatching.type",
            "ALIKED_LIGHTGLUE",
            "--AlikedMatching.lightglue_min_score",
            "0.1",
            "--AlikedMatching.lightglue_model_path",
            "/models/aliked-lightglue.onnx",
        ),
        model_sha256="b" * 64,
    )

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 1)
            _write_binary_count(model / "points3D.bin", 1)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            (output_dir / "geometry" / "points.ply").write_text(
                "ply\nelement vertex 1\nend_header\n", encoding="utf-8"
            )
        elif command[1] == "model_converter" and command[-1] == "TXT":
            text = output_dir / "colmap" / "sparse_txt"
            (text / "cameras.txt").write_text(
                "1 PINHOLE 64 64 50 50 32 32\n", encoding="utf-8"
            )
            (text / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 frame.jpg\n\n", encoding="utf-8"
            )
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_feature_profile", lambda _: profile
    )
    monkeypatch.setattr(
        run_colmap_sparse,
        "resolve_colmap_local_matcher",
        lambda _feature, _profile: local_matcher,
    )
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--feature-profile",
            "aliked_n16rot_v1",
            "--local-matcher",
            "lightglue",
            "--geometric-verification",
            "guided_v1",
        ],
    )

    run_colmap_sparse.main()

    feature, matcher = commands[:2]
    assert feature[feature.index("--FeatureExtraction.type") + 1] == "ALIKED_N16ROT"
    assert feature[feature.index("--AlikedExtraction.n16rot_model_path") + 1] == (
        "/models/aliked.onnx"
    )
    assert matcher[matcher.index("--FeatureMatching.type") + 1] == "ALIKED_LIGHTGLUE"
    assert matcher[matcher.index("--AlikedMatching.lightglue_min_score") + 1] == "0.1"
    assert matcher[matcher.index("--AlikedMatching.lightglue_model_path") + 1] == (
        "/models/aliked-lightglue.onnx"
    )
    assert matcher[matcher.index("--FeatureMatching.guided_matching") + 1] == "1"
    assert (
        matcher[matcher.index("--FeatureMatching.skip_geometric_verification") + 1]
        == "0"
    )
    log = (output_dir / "logs" / "run.log").read_text()
    assert "sfm_feature_profile=aliked_n16rot_v1\n" in log
    assert "sfm_local_matcher_profile=lightglue\n" in log
    assert "sfm_local_matcher=ALIKED_LIGHTGLUE\n" in log
    assert "sfm_geometric_verification_profile=guided_v1\n" in log


def test_aliked_profile_rejects_sift_vocab_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--feature-profile",
            "aliked_n16rot_v1",
            "--vocab-tree-path",
            str(tmp_path / "sift-tree.bin"),
        ],
    )

    with pytest.raises(SystemExit, match="SIFT-only"):
        run_colmap_sparse.main()


def test_gaussian_sequential_matcher_enables_vocab_tree_loop_detection(
    tmp_path, monkeypatch
):
    _accept_pose_health(monkeypatch)
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    vocab_tree = tmp_path / "vocab_tree.bin"
    vocab_tree.write_bytes(b"tree")
    monkeypatch.setenv("IMAGE3D_COLMAP_VOCAB_TREE", str(vocab_tree))
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 12)
            _write_binary_count(model / "points3D.bin", 100)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            path = output_dir / "geometry" / "points.ply"
            path.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_payload",
        lambda _: {
            "cameras": [{"model": "PINHOLE"}],
            "images": [{"image_id": index} for index in range(12)],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--pairing",
            "sequential_loop",
            "--gaussian-baseline",
        ],
    )

    run_colmap_sparse.main()

    matcher = commands[1]
    assert matcher[1] == "sequential_matcher"
    assert matcher[matcher.index("--SequentialMatching.loop_detection") + 1] == "1"
    assert (
        matcher[matcher.index("--SequentialMatching.vocab_tree_path") + 1]
        == str(vocab_tree)
    )
    log = (output_dir / "logs" / "run.log").read_text()
    assert f"vocab_tree={vocab_tree}\n" in log
    assert "sfm_pairing=sequential_loop\n" in log
    assert f"sfm_pairing_vocab_tree_sha256={hashlib.sha256(b'tree').hexdigest()}\n" in log


def test_v2_runner_sets_dynamic_overlap_and_recovers_before_undistortion(
    tmp_path, monkeypatch
):
    _accept_pose_health(monkeypatch)
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    vocab_tree = tmp_path / "vocab_tree.bin"
    vocab_tree.write_bytes(b"tree")
    monkeypatch.setenv("IMAGE3D_COLMAP_VOCAB_TREE", str(vocab_tree))
    video_source = tmp_path / "video.mp4"
    selection_path = tmp_path / "selection.json"
    image_dir.mkdir()
    for index in range(21):
        (image_dir / f"frame_{index}.jpg").write_bytes(b"image")
    video_source.write_bytes(b"video")
    selection_path.write_text(
        json.dumps(
            {
                "profile": "video_keyframes_standard_v2",
                "selected": [
                    {
                        "path": f"frames/frame_{index}.jpg",
                        "time_seconds": index / 5,
                        "pts": index,
                    }
                    for index in range(21)
                ],
            }
        ),
        encoding="utf-8",
    )
    commands = []
    events = []
    recovery_call = {}

    def fake_run(command):
        commands.append(command)
        events.append(command[1])
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 12)
            _write_binary_count(model / "points3D.bin", 100)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            (output_dir / "geometry" / "points.ply").write_text(
                "ply\nelement vertex 1\nend_header\n", encoding="utf-8"
            )
        return "ok"

    def fake_expansion(**kwargs):
        events.append("video_initial_registration_expansion")
        return (
            kwargs["initial_model"],
            {
                "status": "no_progress",
                "accepted_pass_count": 0,
            },
            ["expansion"],
        )

    def fake_recovery(**kwargs):
        recovery_call.update(kwargs)
        events.append("video_registration_recovery")
        return kwargs["initial_model"], {"status": "recovered", "rounds": [{}]}, ["recovery"]

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(
        run_colmap_sparse,
        "mapper_image_list_option",
        lambda _: "--Mapper.image_list_path",
    )
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "expand_v2_initial_registration",
        fake_expansion,
    )
    monkeypatch.setattr(run_colmap_sparse, "recover_video_registration", fake_recovery)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_payload",
        lambda _: {
            "cameras": [{"model": "PINHOLE"}],
            "images": [{"image_id": index} for index in range(12)],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--pairing",
            "sequential_loop",
            "--geometric-verification",
            "guided_v1",
            "--gaussian-baseline",
            "--video-source",
            str(video_source),
            "--video-selection",
            str(selection_path),
        ],
    )

    run_colmap_sparse.main()

    matcher = commands[1]
    mapper = commands[2]
    assert matcher[matcher.index("--SequentialMatching.overlap") + 1] == "21"
    assert mapper[mapper.index("--Mapper.ba_global_frames_ratio") + 1] == "1.5"
    assert mapper[mapper.index("--Mapper.ba_global_points_ratio") + 1] == "1.5"
    assert mapper[mapper.index("--Mapper.ba_global_frames_freq") + 1] == "1000"
    assert mapper[mapper.index("--Mapper.ba_global_points_freq") + 1] == "1000000"
    assert mapper[mapper.index("--Mapper.ba_global_max_refinements") + 1] == "1"
    seed_path = Path(mapper[mapper.index("--Mapper.image_list_path") + 1])
    assert seed_path.read_text().splitlines() == [
        f"frame_{index}.jpg" for index in range(21)
    ]
    assert events.index("video_initial_registration_expansion") < events.index(
        "video_registration_recovery"
    )
    assert events.index("video_registration_recovery") < events.index("image_undistorter")
    assert recovery_call["database_path"] == output_dir / "colmap" / "database.db"
    assert recovery_call["selection_path"] == selection_path
    assert recovery_call["initial_sfm_pairing"] == "sequential_loop"
    assert recovery_call["sfm_geometric_verification"] == "guided_v1"
    assert recovery_call["geometric_verification_options"] == (
        "--FeatureMatching.guided_matching",
        "1",
        "--FeatureMatching.skip_geometric_verification",
        "0",
    )
    log = (output_dir / "logs" / "run.log").read_text()
    assert "sequential_overlap=21\n" in log
    assert "num_images=21\n" in log
    assert "initial_input_count=21\n" in log
    assert "registration_ratio=0.571429\n" in log
    assert "video_registration_recovery_status=recovered\n" in log
    timing = json.loads(
        (output_dir / "diagnostics" / "colmap_timing.json").read_text()
    )
    assert timing["video_profile"] == "video_keyframes_standard_v2"
    assert timing["colmap_build"] == "COLMAP 4.0.0"
    assert timing["matcher"] == "sequential"
    assert timing["pairing"] == "sequential_loop"
    assert timing["geometric_verification"]["profile"] == "guided_v1"
    assert timing["geometric_verification"]["guided_matching"] is True
    assert timing["vocab_tree_sha256"] == hashlib.sha256(b"tree").hexdigest()
    assert set(timing["stage_elapsed_seconds"]) == {
        "feature_extraction",
        "feature_matching",
        "mapping",
        "initial_registration_expansion",
        "registration_recovery",
        "raw_model_conversion",
        "undistortion",
        "point_cloud_conversion",
        "text_conversion",
    }
    assert timing["v2_mapper_options"] == run_colmap_sparse.v2_mapper_options(
        json.loads(selection_path.read_text())
    )
    frontend_contract = json.loads(
        (output_dir / "diagnostics" / "sfm_frontend_contract.json").read_text()
    )
    assert frontend_contract["profile"] == "sfm_frontend_contract_v1"
    assert frontend_contract["feature"] == timing["feature"]
    assert frontend_contract["pairing"] == timing["pairing"]
    assert frontend_contract["initial_video_selection_sha256"] == hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    assert timing["sfm_frontend_contract_path"] == (
        "diagnostics/sfm_frontend_contract.json"
    )
    assert timing["total_elapsed_seconds"] >= sum(
        timing["stage_elapsed_seconds"].values()
    )


def test_gaussian_baseline_sequential_requires_vocab_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--matcher",
            "sequential",
            "--gaussian-baseline",
        ],
    )

    with pytest.raises(SystemExit, match="vocab-tree-path"):
        run_colmap_sparse.main()


def test_runner_rejects_nonpositive_thread_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--num-threads",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        run_colmap_sparse.main()
