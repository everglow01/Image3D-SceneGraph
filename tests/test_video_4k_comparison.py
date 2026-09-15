from __future__ import annotations

import json

import pytest

from scripts import run_video_4k_comparison as experiment


def test_configs_freeze_uhd_method_budgets_and_defaults():
    records = experiment.configs()
    for arm, record in records.items():
        config = record["effective_config"]
        assert config["resolution"]["longest_edge"] == 3840
        assert config["iterations"] == 30000
        assert config["seed"] == 20260729
        assert config["opacity_reset"]["recovery_prune"]["enabled"] == (arm == "project")
    assert records["mcmc"]["effective_config"]["strategy"]["gaussian_cap"] == 3000000
    assert records["project"]["effective_config"]["strategy"]["gaussian_cap"] is None


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_uhd_request_threads_resolution_to_job_config(tmp_path, arm):
    from image3d_scenegraph.jobs import JobStore, UploadedInput
    from backend.main import create_app

    store = JobStore(tmp_path / "jobs")
    manifest = store.enqueue_job(
        "video", [UploadedInput(filename="new.MOV", content=b"video")],
        geometry_backend="project_3dgs", output_type="gaussian_splat",
        options={"gaussian_trainer": arm, "gaussian_longest_edge": 3840},
    )
    assert manifest["gaussian_config"]["effective_config"]["resolution"]["longest_edge"] == 3840
    schemas = create_app(tmp_path / "api", start_worker=False).openapi()["components"]["schemas"]
    fields = next(value["properties"] for value in schemas.values() if "gaussian_longest_edge" in value.get("properties", {}))
    bounds = fields["gaussian_longest_edge"]["anyOf"]
    assert next(value for value in bounds if value["type"] == "integer")["maximum"] == 3840


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_arm_dispatch_preserves_no_test_and_train_only_boundary(tmp_path, monkeypatch, arm):
    replay = tmp_path / "replay"
    replay.mkdir()
    monkeypatch.setattr(experiment, "load_protocol", lambda *_args: {"replay": str(replay)})
    monkeypatch.setattr(experiment, "require_resources", lambda *_args, **_kwargs: None)
    commands = {}

    def run(output, name, arguments):
        commands[name] = arguments
        if name == "train":
            path = output / "training" / "attempts" / "train-001" / "artifacts"
            path.mkdir(parents=True)
            (path / "result.json").write_text(json.dumps({
                "iteration": 30000, "world_size": 2, "model_path": "model.pt",
                "progress_path": "progress.jsonl", "final_checkpoint_hash": "a" * 64,
            }))
        if name in {"selection", "train-only"}:
            (output / name).mkdir()
            (output / name / "evaluation.json").write_text("{}")
        if name == "train-only":
            (output / name / "record.json").write_text(json.dumps({
                "optimizer_updates": 2000, "camera_samples": 4000,
                "input_splits": ["train"], "topology_changed": False,
            }))

    monkeypatch.setattr(experiment, "run_command", run)
    experiment.run_arm(tmp_path, arm, "b" * 64)
    assert list(commands) == ["train", "sor", "selection", "train-only", "export"]
    assert "--no-intermediate-previews" in commands["train"]
    assert "--distributed" in commands["train"]
    assert "--train-only-control" in commands["train-only"]
    assert "--distributed" in commands["train-only"]
    assert commands["selection"][commands["selection"].index("--split") + 1] == "validation"
    assert "--progress" in commands["selection"]
    assert all("test" not in arguments for arguments in commands.values())
    assert "--final-fit-record" in commands["export"]
    assert json.loads((tmp_path / arm / "complete.json").read_text())["test"] == "not_run"
    with pytest.raises(FileExistsError):
        experiment.run_arm(tmp_path, arm, "b" * 64)


def test_reused_video_hardlinks_frames_but_copies_mutable_selection(tmp_path):
    from image3d_scenegraph.file_integrity import sha256_file

    parent = tmp_path / "old"
    shared = parent / "shared"
    for directory in ("frames/selected", "diagnostics", "colmap"):
        (shared / directory).mkdir(parents=True)
    image = shared / "frames/selected/frame.jpg"
    image.write_bytes(b"immutable-frame")
    database = shared / "colmap/database.db"
    database.write_bytes(b"database-fixture")
    selection = shared / "frames/selection.json"
    selection.write_text(json.dumps({
        "profile": "video_keyframes_standard_v2", "source_sha256": "a" * 64,
        "duration_seconds": 601.3, "selected_count": 1,
        "selected": [{"path": "frames/selected/frame.jpg", "sha256": sha256_file(image)}],
    }))
    (shared / "diagnostics/video_probe.json").write_text(json.dumps({
        "source": {"sha256": "a" * 64}, "duration_seconds": 601.3,
    }))
    (shared / "diagnostics/sfm_frontend_contract.json").write_text(json.dumps({
        "initial_video_selection_sha256": sha256_file(selection), "v2_mapper_seed_count": 1000,
    }))
    (shared / "diagnostics/sfm_pose_recovery.json").write_text(json.dumps({
        "source_database_sha256": sha256_file(database),
    }))
    (shared / "diagnostics/video_keyframe_timing.json").write_text("{}")
    (shared / "diagnostics/video_keyframes.jpg").write_bytes(b"contact")
    target = tmp_path / "new/shared"
    record = experiment.reuse_video_preparation(parent, target, "a" * 64)
    assert (target / "frames/selected/frame.jpg").stat().st_ino == image.stat().st_ino
    assert (target / "frames/selection.json").stat().st_ino != selection.stat().st_ino
    assert (target / "frames/selection.json").read_bytes() == selection.read_bytes()
    assert record["mapper_seed_limit"] == 2500
    assert record["source_database_sha256"] == sha256_file(database)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frame hash mismatch"):
        experiment.reuse_video_preparation(parent, tmp_path / "bad/shared", "a" * 64)


def test_protocol_hash_failure_precedes_gpu_or_output_creation(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        experiment.load_protocol(tmp_path, "a" * 64)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["protocol.json"]
