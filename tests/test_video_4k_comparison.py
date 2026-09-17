from __future__ import annotations

import json

import pytest

from scripts import run_video_4k_comparison as experiment


@pytest.mark.parametrize("longest_edge", [1920, 3840])
def test_configs_freeze_native_method_budgets_and_defaults(longest_edge):
    records = experiment.configs(longest_edge)
    assert experiment.configs()["project"]["effective_config"]["resolution"]["longest_edge"] == 3840
    for arm, record in records.items():
        config = record["effective_config"]
        assert config["resolution"]["longest_edge"] == longest_edge
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


@pytest.mark.parametrize("edge,seeds", [(3840, 1000), (1920, 2500), (1920, 3500)])
def test_fresh_prepare_freezes_native_resolution_and_seed_budget(tmp_path, monkeypatch, edge, seeds):
    from image3d_scenegraph.geometry.adapters import ReconstructionResult

    source = tmp_path / "retake.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "experiment"
    probe = {
        "source_width": edge, "source_height": edge * 9 // 16,
        "display_width": edge * 9 // 16, "display_height": edge,
        "source": {"sha256": experiment.sha256_file(source)},
        "rotation": {"applied_degrees": 90, "quarter_turn": True},
    }
    resources = []
    contexts = []
    monkeypatch.setattr(experiment, "require_resources", lambda path, **kw: resources.append((path, kw)))
    monkeypatch.setattr(experiment, "revision", lambda: "frozen-code")
    monkeypatch.setattr(experiment, "probe_video", lambda _: probe)
    monkeypatch.setattr(experiment, "validate_replay_bundle", lambda _: None)

    def prepare_only(self, context):
        contexts.append(context)
        replay = context.job_dir / "gaussian/replay"
        replay.mkdir(parents=True)
        experiment.write_json(replay / "dataset.json", {
            "dataset_hash": "dataset", "images": [{"width": edge * 9 // 16, "height": edge}],
            "splits": {"train": ["1"], "validation": ["2"], "test": ["3"]},
        })
        experiment.write_json(replay / "replay.json", {})
        return ReconstructionResult("gaussian_prepared", {}, {}, [])

    monkeypatch.setattr(experiment.ProjectGaussianAdapter, "run", prepare_only)
    experiment.prepare(source, root, longest_edge=edge, mapper_seed_limit=seeds)
    protocol = experiment.load_protocol(root, experiment.sha256_file(root / "protocol.json"))
    assert protocol["profile"] == (experiment.PROFILE if edge == 3840 else experiment.NATIVE_1080_PROFILE)
    assert protocol["geometry_variant"] == f"fresh_seed{seeds}_v1"
    assert protocol["mapper_seed_limit"] == seeds
    assert protocol["longest_edge"] == edge
    assert protocol["image_dimensions"] == [[edge * 9 // 16, edge]]
    assert protocol["test"] == "not_authorized"
    assert (protocol["world_size"], protocol["main_updates"], protocol["final_fit_updates"]) == (2, 30000, 2000)
    assert resources == [(tmp_path, {"minimum_free_gib": 40})]
    options = contexts[0].options
    assert options["v2_mapper_seed_limit"] == seeds
    assert options["gaussian_longest_edge"] == edge
    assert options["video_rotation"] == "auto"
    assert options["video_keyframe_profile"] == "standard_v2"
    assert options["gaussian_prepare_only"] is True
    assert not any("reuse" in key for key in options)
    for arm in experiment.ARMS:
        config = experiment.read_json(root / f"{arm}.config.json")
        assert config["effective_config"]["resolution"]["longest_edge"] == edge
    started = experiment.read_json(root / "prepare-started.json")
    assert started["mapper_seed_limit"] == seeds
    assert started["probe"]["rotation"] == probe["rotation"]
    assert (root / "shared/input/retake.mp4").stat().st_ino == source.stat().st_ino
    with pytest.raises(ValueError, match="already exists"):
        experiment.prepare(source, root, longest_edge=edge, mapper_seed_limit=seeds)
    with pytest.raises(ValueError, match="expects native"):
        experiment.prepare(source, tmp_path / "wrong-size", longest_edge=1920 if edge == 3840 else 3840)
    assert not (tmp_path / "wrong-size").exists()


def test_prepare_cli_preserves_defaults_and_accepts_retake(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(experiment, "prepare", lambda *args, **kw: calls.append((args, kw)))
    base = ["experiment", "prepare", "--source", str(tmp_path / "source.mp4"), "--output-dir", str(tmp_path / "out")]
    for extra in ([], ["--longest-edge", "1920", "--mapper-seed-limit", "3500"]):
        monkeypatch.setattr(experiment.sys, "argv", base + extra)
        experiment.main()
    assert calls[0][1] == {"longest_edge": 3840, "mapper_seed_limit": 1000}
    assert calls[1][1] == {"longest_edge": 1920, "mapper_seed_limit": 3500}


@pytest.mark.parametrize("count", [3200, 4000])
def test_retake_seed_budget_is_bounded_by_available_frames(count):
    from image3d_scenegraph.geometry.video_recovery import v2_mapper_seed_image_names

    selection = {
        "profile": "video_keyframes_standard_v2",
        "selected": [
            {"path": f"frames/{index:06d}.jpg", "pts": index, "time_seconds": index / 5,
             "selection_reason": "base" if index < 3394 else "adaptive_motion"}
            for index in range(count)
        ],
    }
    names = v2_mapper_seed_image_names(selection, max_images=3500)
    assert len(names) == len(set(names)) == min(3500, count)
    assert names[0] == "000000.jpg"
    assert names[-1] == f"{count - 1:06d}.jpg"


def test_protocol_hash_failure_precedes_gpu_or_output_creation(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        experiment.load_protocol(tmp_path, "a" * 64)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["protocol.json"]
