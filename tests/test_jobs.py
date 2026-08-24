from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from image3d_scenegraph.gaussian.config import effective_config_hash, resolve_public_config
from image3d_scenegraph.geometry.adapters import (
    ProjectGaussianAdapter,
    ReconstructionContext,
    ReconstructionError,
    _automatic_test_evaluation_enabled,
    _write_video_registration_diagnostics,
)
from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput


def test_project_gaussian_colmap_progress_callback_reports_new_substages(tmp_path):
    progress_path = tmp_path / "progress.json"
    updates = []
    context = ReconstructionContext(
        job_id="job",
        job_dir=tmp_path,
        mode="multi_image",
        input_assets=[],
        options={},
        progress_callback=lambda stage, progress: updates.append((stage, progress)),
    )
    poll = ProjectGaussianAdapter._colmap_progress_callback(context, progress_path)

    poll()
    progress_path.write_text('{"stage":"colmap_feature_matching"}\n', encoding="utf-8")
    poll()
    poll()
    progress_path.write_text('{"stage":"colmap_mapping"}\n', encoding="utf-8")
    poll()

    assert updates == [
        ("colmap_feature_matching", 0.20),
        ("colmap_mapping", 0.26),
    ]


def test_project_gaussian_vggt_ba_progress_callback_reports_recovery_and_fallback(
    tmp_path,
):
    progress_path = tmp_path / "progress.json"
    updates = []
    context = ReconstructionContext(
        job_id="job",
        job_dir=tmp_path,
        mode="video",
        input_assets=[],
        options={},
        progress_callback=lambda stage, progress: updates.append((stage, progress)),
    )
    poll = ProjectGaussianAdapter._vggt_ba_progress_callback(context, progress_path)

    for stage in (
        "vggt_ba_recovery",
        "vggt_ba_image_registration",
        "colmap_fallback_mapping",
    ):
        progress_path.write_text(json.dumps({"stage": stage}), encoding="utf-8")
        poll()

    assert updates == [
        ("vggt_ba_recovery", 0.20),
        ("vggt_ba_image_registration", 0.295),
        ("colmap_fallback_mapping", 0.30),
    ]


def test_project_gaussian_colmap_uses_gpu_and_bounded_cpu_resources(
    tmp_path, monkeypatch
):
    captured = []

    def fake_run(command, *args, **kwargs):
        captured.append(command)
        raise ReconstructionError("stop after COLMAP command capture")

    monkeypatch.setattr(
        "image3d_scenegraph.geometry.adapters._run_adapter_command", fake_run
    )
    monkeypatch.setattr(
        "image3d_scenegraph.geometry.adapters.os.cpu_count", lambda: 20
    )
    monkeypatch.delenv("IMAGE3D_COLMAP_NUM_THREADS", raising=False)
    context = ReconstructionContext(
        job_id="job",
        job_dir=tmp_path,
        mode="multi_image",
        input_assets=[],
        options={},
    )

    with pytest.raises(ReconstructionError, match="stop after COLMAP"):
        ProjectGaussianAdapter().run(context)

    command = captured[0]
    assert "--use-gpu" in command
    assert "--no-use-gpu" not in command
    assert "--gpu-index" not in command
    assert command[command.index("--max-image-size") + 1] == "1280"
    assert command[command.index("--num-threads") + 1] == "8"
    assert command[command.index("--matcher") + 1] == "exhaustive"
    assert "--vocab-tree-path" not in command
    assert "--gaussian-baseline" in command


def test_project_gaussian_sequential_matcher_threads_vocab_tree(
    tmp_path, monkeypatch
):
    vocab_tree = tmp_path / "vocab_tree.bin"
    vocab_tree.write_bytes(b"tree")
    captured = []

    def fake_run(command, *args, **kwargs):
        captured.append(command)
        raise ReconstructionError("stop after COLMAP command capture")

    monkeypatch.setattr(
        "image3d_scenegraph.geometry.adapters._run_adapter_command", fake_run
    )
    monkeypatch.delenv("IMAGE3D_GAUSSIAN_COLMAP_MATCHER", raising=False)
    monkeypatch.setenv("IMAGE3D_COLMAP_VOCAB_TREE", str(vocab_tree))
    context = ReconstructionContext(
        job_id="job",
        job_dir=tmp_path,
        mode="multi_image",
        input_assets=[],
        options={"colmap_matcher": "sequential"},
    )

    with pytest.raises(ReconstructionError, match="stop after COLMAP"):
        ProjectGaussianAdapter().run(context)

    command = captured[0]
    assert command[command.index("--matcher") + 1] == "sequential"
    assert command[command.index("--vocab-tree-path") + 1] == str(vocab_tree)


def test_project_gaussian_vggt_ba_threads_sequential_matcher(
    tmp_path, monkeypatch
):
    vocab_tree = tmp_path / "vocab_tree.bin"
    vocab_tree.write_bytes(b"tree")
    captured = []

    def fake_run(command, context, *args, **kwargs):
        if any("extract_video_keyframes.py" in item for item in command):
            frames = context.job_dir / "frames"
            diagnostics = context.job_dir / "diagnostics"
            frames.mkdir(parents=True)
            diagnostics.mkdir(parents=True)
            (frames / "selection.json").write_text(
                json.dumps(
                    {
                        "profile": "video_keyframes_standard_v1",
                        "duration_seconds": 60.0,
                        "candidate_count": 24,
                        "selected_count": 24,
                        "selected": [],
                    }
                ),
                encoding="utf-8",
            )
            (diagnostics / "video_probe.json").write_text(
                json.dumps(
                    {
                        "orientation": "landscape",
                        "rotation": {"applied_degrees": 0},
                        "source_width": 1280,
                        "source_height": 720,
                        "display_width": 1280,
                        "display_height": 720,
                    }
                ),
                encoding="utf-8",
            )
            (diagnostics / "video_keyframes.jpg").write_bytes(b"sheet")
            return subprocess.CompletedProcess(command, 0, "", "")
        captured.append(command)
        raise ReconstructionError("stop after VGGT-BA command capture")

    monkeypatch.setattr(
        "image3d_scenegraph.geometry.adapters._run_adapter_command", fake_run
    )
    monkeypatch.setenv("IMAGE3D_COLMAP_VOCAB_TREE", str(vocab_tree))
    context = ReconstructionContext(
        job_id="job",
        job_dir=tmp_path,
        mode="video",
        input_assets=[{"path": "input/video.mp4"}],
        options={
            "gaussian_geometry_source": "vggt_ba",
            "colmap_matcher": "sequential",
        },
    )

    with pytest.raises(ReconstructionError, match="stop after VGGT-BA"):
        ProjectGaussianAdapter().run(context)

    command = captured[0]
    assert command[1].endswith("run_vggt_ba_sparse.py")
    assert command[command.index("--matcher") + 1] == "sequential"
    assert command[command.index("--vocab-tree-path") + 1] == str(vocab_tree)


def test_project_gaussian_sequential_matcher_without_vocab_tree_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("IMAGE3D_COLMAP_VOCAB_TREE", str(tmp_path / "missing.bin"))
    context = ReconstructionContext(
        job_id="job",
        job_dir=tmp_path,
        mode="multi_image",
        input_assets=[],
        options={"colmap_matcher": "sequential"},
    )

    with pytest.raises(ReconstructionError, match="vocab tree"):
        ProjectGaussianAdapter().run(context)


def test_frontend_gaussian_jobs_do_not_automatically_consume_test():
    assert _automatic_test_evaluation_enabled("project") is False
    assert _automatic_test_evaluation_enabled("graphdeco") is False


def test_video_registration_gate_writes_temporal_diagnostics(tmp_path):
    selected = [
        {
            "path": f"frames/selected/frame_{index:03d}.jpg",
            "time_seconds": float(index),
        }
        for index in range(20)
    ]
    cameras = {
        "cameras": [
            {
                "camera_id": 1,
                "model": "PINHOLE",
            }
        ],
        "images": [
            {
                "image_id": index + 1,
                "camera_id": 1,
                "name": f"frame_{index:03d}.jpg",
            }
            for index in range(18)
        ],
    }
    cameras_path = tmp_path / "cameras.json"
    output_path = tmp_path / "video_registration.json"
    cameras_path.write_text(json.dumps(cameras), encoding="utf-8")

    timestamps, metrics, gap_violations = _write_video_registration_diagnostics(
        {"profile": "video_keyframes_standard_v1", "selected": selected},
        cameras_path,
        output_path,
    )

    assert len(timestamps) == 18
    assert metrics["video_registration_rate"] == 0.9
    assert metrics["video_registration_temporal_coverage"] > 0.8
    assert gap_violations == []
    assert "video_registration_gap_violation_count" not in metrics
    assert json.loads(output_path.read_text())["registered_count"] == 18


def test_video_registration_gap_violation_is_soft_warning(tmp_path):
    selected = [
        {
            "path": f"frames/selected/frame_{index:03d}.jpg",
            "time_seconds": float(index),
        }
        for index in range(9)
    ] + [
        {
            "path": f"frames/selected/frame_{index:03d}.jpg",
            "time_seconds": float(index) + 4.0,
        }
        for index in range(9, 18)
    ]
    cameras = {
        "cameras": [{"camera_id": 1, "model": "PINHOLE"}],
        "images": [
            {
                "image_id": index + 1,
                "camera_id": 1,
                "name": f"frame_{index:03d}.jpg",
            }
            for index in range(18)
        ],
    }
    cameras_path = tmp_path / "cameras.json"
    output_path = tmp_path / "video_registration.json"
    cameras_path.write_text(json.dumps(cameras), encoding="utf-8")

    timestamps, metrics, gap_violations = _write_video_registration_diagnostics(
        {"profile": "video_keyframes_standard_v1", "selected": selected},
        cameras_path,
        output_path,
    )

    payload = json.loads(output_path.read_text())
    assert payload["maximum_registered_gap_threshold_seconds"] == 2.0
    assert payload["gap_violations"] == [
        {"start_seconds": 8.0, "end_seconds": 13.0, "seconds": 5.0}
    ]
    assert gap_violations == payload["gap_violations"]
    assert metrics["video_registration_gap_violation_count"] == 1
    assert len(timestamps) == 18


def test_video_registration_gate_failure_raises_and_writes_diagnostics(tmp_path):
    selected = [
        {
            "path": f"frames/selected/frame_{index:03d}.jpg",
            "time_seconds": float(index),
        }
        for index in range(20)
    ]
    cameras = {
        "cameras": [{"camera_id": 1, "model": "PINHOLE"}],
        "images": [
            {
                "image_id": index + 1,
                "camera_id": 1,
                "name": f"frame_{index:03d}.jpg",
            }
            for index in range(12)
        ],
    }
    cameras_path = tmp_path / "cameras.json"
    output_path = tmp_path / "video_registration.json"
    cameras_path.write_text(json.dumps(cameras), encoding="utf-8")

    with pytest.raises(
        ReconstructionError, match="video_registration_quality_gate_failed"
    ):
        _write_video_registration_diagnostics(
            {"profile": "video_keyframes_standard_v1", "selected": selected},
            cameras_path,
            output_path,
        )

    payload = json.loads(output_path.read_text())
    assert payload["gate"]["passed"] is False
    assert any("registration_rate" in failure for failure in payload["gate"]["failures"])


def test_list_jobs_returns_valid_manifests_newest_first(tmp_path):
    root = tmp_path / "jobs"
    store = JobStore(output_root=root)

    def write_manifest(job_id: str, updated_at: str) -> None:
        directory = root / job_id
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": "done",
                    "stage": "done",
                    "mode": "image",
                    "geometry_backend": "mock",
                    "output_type": "point_cloud",
                    "metrics": {},
                    "updated_at": updated_at,
                }
            ),
            encoding="utf-8",
        )

    write_manifest("older", "2026-08-12T00:00:00Z")
    write_manifest("newer", "2026-08-13T00:00:00Z")
    (root / "missing-manifest").mkdir()
    malformed = root / "malformed"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("{", encoding="utf-8")
    mismatched = root / "mismatched"
    mismatched.mkdir()
    (mismatched / "manifest.json").write_text(
        json.dumps(
            {
                "job_id": "other",
                "status": "done",
                "stage": "done",
                "mode": "image",
                "geometry_backend": "mock",
                "output_type": "point_cloud",
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    assert [job["job_id"] for job in store.list_jobs()] == ["newer", "older"]


def test_create_image_job_and_read_outputs(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_mock_job(
        "image",
        [
            UploadedInput(
                filename="room.jpg",
                content=b"fake-image",
                content_type="image/jpeg",
            )
        ],
    )

    job_id = manifest["job_id"]
    assert manifest["status"] == "done"
    assert manifest["mode"] == "image"
    assert manifest["geometry_backend"] == "mock"
    assert manifest["output_type"] == "point_cloud"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["point_cloud_aligned"] == "geometry/points_aligned.ply"
    assert manifest["assets"]["alignment_diagnostics"] == "diagnostics/alignment.json"
    assert manifest["metrics"]["alignment_status"] == "aligned"

    assert "gaussian_config" not in manifest
    loaded_manifest = store.get_manifest(job_id)
    assert loaded_manifest["metrics"]["num_points"] == 5

    scene = store.get_scene(job_id)
    assert scene["objects"][0]["label"] == "scene_proxy"

    asset_path = store.get_asset_path(job_id, "geometry/points.ply")
    assert asset_path.read_text(encoding="utf-8").startswith("ply\n")
    aligned_asset_path = store.get_asset_path(job_id, "geometry/points_aligned.ply")
    assert aligned_asset_path.read_bytes().startswith(b"ply\n")

    bundle_path = store.build_zip(job_id)
    assert bundle_path == tmp_path / "jobs" / f"{job_id}.zip"
    assert bundle_path.exists()
    with zipfile.ZipFile(bundle_path) as archive:
        assert "manifest.json" in archive.namelist()


def test_persist_internal_gaussian_config_in_manifest_and_log(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    resolved = resolve_public_config("standard_v1")

    manifest = store.create_job(
        "image",
        [UploadedInput(filename="room.jpg", content=b"fake-image")],
        gaussian_config=resolved,
    )

    record = manifest["gaussian_config"]
    assert record["schema_version"] == 9
    assert record["requested_profile"] == "standard_v1"
    assert record["effective_config_hash"] == effective_config_hash(record["effective_config"])
    log = store.get_asset_path(manifest["job_id"], "logs/run.log").read_text(encoding="utf-8")
    assert "gaussian_config_schema_version=9\n" in log
    assert "gaussian_requested_profile=standard_v1\n" in log
    assert f"gaussian_effective_config_hash={record['effective_config_hash']}\n" in log
    assert 'gaussian_effective_config={"densification":' in log


def test_project_gaussian_job_persists_selected_trainer_before_execution(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    files = [
        UploadedInput(filename=f"{index}.jpg", content=b"image")
        for index in range(12)
    ]

    manifest = store.enqueue_job(
        "multi_image",
        files,
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
        options={"gaussian_trainer": "graphdeco", "gaussian_longest_edge": 3072},
    )
    request = json.loads((store.job_dir(manifest["job_id"]) / "request.json").read_text())

    assert manifest["gaussian_trainer"]["id"] == "graphdeco"
    assert request["options"]["gaussian_trainer"] == "graphdeco"
    assert request["options"]["gaussian_longest_edge"] == 3072
    assert request["gaussian_trainer"] == manifest["gaussian_trainer"]
    assert manifest["gaussian_config"]["schema_version"] == 9
    assert manifest["gaussian_config"]["effective_config"]["resolution"]["longest_edge"] == 3072


def test_project_video_job_stages_source_and_persists_profile(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    staging = tmp_path / "source.upload"
    staging.write_bytes(b"video-bytes")

    manifest = store.enqueue_job(
        "video",
        [
            UploadedInput(
                filename="portrait.mp4",
                staged_path=staging,
                size_bytes=11,
                sha256="a" * 64,
                content_type="video/mp4",
            )
        ],
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
        options={"video_rotation": "clockwise_90"},
    )
    job_dir = store.job_dir(manifest["job_id"])
    request = json.loads((job_dir / "request.json").read_text())

    assert not staging.exists()
    assert (job_dir / "input" / "portrait.mp4").read_bytes() == b"video-bytes"
    assert manifest["inputs"][0]["sha256"] == "a" * 64
    assert request["video_source"] == manifest["inputs"][0]
    assert request["options"]["video_keyframe_profile"] == "standard_v1"
    assert request["options"]["video_rotation"] == "clockwise_90"


def test_video_job_rejects_non_project_pipeline(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    with pytest.raises(JobError, match="project_3dgs"):
        store.enqueue_job(
            "video",
            [UploadedInput(filename="portrait.mp4", content=b"video")],
            geometry_backend="mock",
            output_type="point_cloud",
        )
    assert (
        store._execution_error_code("runner: insufficient_video_keyframes: 7 accepted")
        == "insufficient_video_keyframes"
    )


def test_project_gaussian_job_defaults_to_graphdeco(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    files = [
        UploadedInput(filename=f"{index}.jpg", content=b"image")
        for index in range(12)
    ]

    manifest = store.enqueue_job(
        "multi_image",
        files,
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
    )
    request = json.loads((store.job_dir(manifest["job_id"]) / "request.json").read_text())

    assert manifest["gaussian_trainer"]["id"] == "graphdeco"
    assert request["options"]["gaussian_trainer"] == "graphdeco"
    assert request["options"]["gaussian_geometry_source"] == "colmap"
    assert request["options"]["gaussian_postprocess"] == "none"
    assert manifest["gaussian_geometry_source"] == "colmap"
    assert manifest["gaussian_geometry_effective_source"] is None
    assert manifest["gaussian_geometry_fallback_applied"] is False
    assert manifest["gaussian_geometry_fallback_reason"] is None
    assert manifest["gaussian_postprocess"] == "none"
    assert manifest["gaussian_postprocess_status"] == "not_requested"
    assert request["options"]["gaussian_sor_filter"] == "on"
    assert manifest["gaussian_sor_filter"] == "on"
    assert manifest["gaussian_sor_filter_status"] == "pending"
    assert request["options"]["gaussian_recovery_prune"] == "off"
    assert manifest["gaussian_recovery_prune"] == "off"
    recovery_prune_leaf = manifest["gaussian_config"]["effective_config"][
        "opacity_reset"
    ]["recovery_prune"]
    assert recovery_prune_leaf["enabled"] is False


def test_project_gaussian_job_persists_experimental_options(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.enqueue_job(
        "video",
        [UploadedInput(filename="portrait.mp4", content=b"video")],
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
        options={
            "gaussian_geometry_source": "vggt_ba",
            "gaussian_postprocess": "vggt_visibility_v1",
            "gaussian_sor_filter": "off",
            "gaussian_recovery_prune": "on",
        },
    )
    request = json.loads((store.job_dir(manifest["job_id"]) / "request.json").read_text())

    assert request["options"]["gaussian_geometry_source"] == "vggt_ba"
    assert request["options"]["gaussian_postprocess"] == "vggt_visibility_v1"
    assert request["options"]["gaussian_sor_filter"] == "off"
    assert manifest["gaussian_geometry_source"] == "vggt_ba"
    assert manifest["gaussian_geometry_effective_source"] is None
    assert manifest["gaussian_geometry_fallback_applied"] is False
    assert manifest["gaussian_geometry_fallback_reason"] is None
    assert manifest["gaussian_postprocess"] == "vggt_visibility_v1"
    assert manifest["gaussian_postprocess_status"] == "pending"
    assert manifest["gaussian_sor_filter"] == "off"
    assert manifest["gaussian_sor_filter_status"] == "disabled"
    assert request["options"]["gaussian_recovery_prune"] == "on"
    assert manifest["gaussian_recovery_prune"] == "on"
    recovery_prune_leaf = manifest["gaussian_config"]["effective_config"][
        "opacity_reset"
    ]["recovery_prune"]
    assert recovery_prune_leaf["enabled"] is True


def test_project_gaussian_job_env_enables_recovery_prune(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE3D_GAUSSIAN_RECOVERY_PRUNE", "on")
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.enqueue_job(
        "multi_image",
        [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
        options={},
    )

    assert manifest["gaussian_recovery_prune"] == "on"
    recovery_prune_leaf = manifest["gaussian_config"]["effective_config"][
        "opacity_reset"
    ]["recovery_prune"]
    assert recovery_prune_leaf["enabled"] is True


def test_project_gaussian_job_env_disables_sor_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE3D_GAUSSIAN_SOR_FILTER", "off")
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.enqueue_job(
        "multi_image",
        [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
        options={},
    )

    assert manifest["gaussian_sor_filter"] == "off"
    assert manifest["gaussian_sor_filter_status"] == "disabled"


def test_project_gaussian_job_rejects_unknown_sor_filter_setting(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="SOR filter"):
        store.enqueue_job(
            "multi_image",
            [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"gaussian_sor_filter": "maybe"},
        )


def test_project_gaussian_job_rejects_unknown_recovery_prune_setting(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="recovery prune"):
        store.enqueue_job(
            "multi_image",
            [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"gaussian_recovery_prune": "maybe"},
        )


def test_vggt_ba_gaussian_geometry_rejects_non_video_input(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="requires video mode"):
        store.enqueue_job(
            "multi_image",
            [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"gaussian_geometry_source": "vggt_ba"},
        )


def test_sequential_colmap_matcher_rejects_non_video_input(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="requires video mode"):
        store.enqueue_job(
            "multi_image",
            [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"colmap_matcher": "sequential"},
        )


def test_project_gaussian_job_rejects_unknown_trainer(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    with pytest.raises(JobError, match="unsupported Gaussian trainer"):
        store.enqueue_job(
            "multi_image",
            [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)],
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"gaussian_trainer": "unknown"},
        )


def test_project_gaussian_job_rejects_unknown_experimental_options(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    files = [
        UploadedInput(filename=f"{index}.jpg", content=b"image")
        for index in range(12)
    ]

    with pytest.raises(JobError, match="unsupported Gaussian geometry source"):
        store.enqueue_job(
            "multi_image",
            files,
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"gaussian_geometry_source": "unknown"},
        )
    with pytest.raises(JobError, match="unsupported Gaussian postprocess"):
        store.enqueue_job(
            "multi_image",
            files,
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options={"gaussian_postprocess": "unknown"},
        )


def test_create_panorama_job(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_mock_job(
        "panorama",
        [
            UploadedInput(
                filename="office_360.jpg",
                content=b"fake-panorama",
                content_type="image/jpeg",
            )
        ],
    )

    assert manifest["mode"] == "panorama"
    assert manifest["input_type"] == "equirectangular_panorama"
    assert manifest["inputs"][0]["path"] == "input/images/office_360.jpg"


def test_load_legacy_manifest_without_policy_metrics(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    job_dir = store.job_dir("legacy-job")
    job_dir.mkdir(parents=True)
    manifest = {
        "job_id": "legacy-job",
        "status": "done",
        "stage": "colmap_vggt_dense_reconstruction",
        "progress": 1.0,
        "mode": "multi_image",
        "geometry_backend": "colmap_vggt",
        "output_type": "point_cloud",
        "assets": {"point_cloud": "geometry/points.ply"},
        "metrics": {"num_points": 99},
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = store.get_manifest("legacy-job")

    assert loaded["job_id"] == manifest["job_id"]
    assert loaded["assets"] == manifest["assets"]
    assert loaded["metrics"] == manifest["metrics"]
    assert loaded["mesh_variants"] == []
    assert "vggt_grouping" not in loaded["metrics"]
    assert "confidence_threshold_scope" not in loaded["metrics"]
    assert "consistency_support_policy" not in loaded["metrics"]
    assert "point_budget_policy" not in loaded["metrics"]
    assert "gaussian_config" not in loaded


def test_preserve_multi_image_folder_paths(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_mock_job(
        "multi_image",
        [
            UploadedInput(filename="scan/front/frame_001.jpg", content=b"front"),
            UploadedInput(filename="scan/back/frame_001.jpg", content=b"back"),
        ],
    )

    input_paths = [item["path"] for item in manifest["inputs"]]
    assert input_paths == [
        "input/images/scan/front/frame_001.jpg",
        "input/images/scan/back/frame_001.jpg",
    ]


def test_reject_multiple_panorama_inputs(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="panorama mode requires exactly one file"):
        store.create_mock_job(
            "panorama",
            [
                UploadedInput(filename="front.jpg", content=b"fake-image"),
                UploadedInput(filename="back.jpg", content=b"fake-image"),
            ],
        )


def test_reject_invalid_mode(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="unsupported mode"):
        store.create_mock_job(
            "rgbd",
            [UploadedInput(filename="room.jpg", content=b"fake-image")],
        )


def test_reject_unimplemented_reconstruction_option(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="not implemented"):
        store.create_job(
            "image",
            [UploadedInput(filename="room.jpg", content=b"fake-image")],
            geometry_backend="dust3r",
            output_type="point_cloud",
        )


def test_create_vggt_point_cloud_job_uses_adapter_contract(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
        (job_dir / "geometry" / "cameras.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=vggt",
                    "num_images=1",
                    "num_groups=1",
                    "batch_size=8",
                    "overlap_size=4",
                    "num_points=123",
                    "inference_seconds=0.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "image",
        [UploadedInput(filename="room.jpg", content=b"fake-image", content_type="image/jpeg")],
        geometry_backend="vggt",
        output_type="point_cloud",
        options={"vggt_max_images": 225, "vggt_batch_size": 8, "vggt_overlap_size": 4},
    )

    assert manifest["stage"] == "vggt_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["cameras"] == "geometry/cameras.json"
    assert manifest["metrics"]["num_points"] == 123
    assert manifest["metrics"]["num_groups"] == 1
    assert manifest["metrics"]["batch_size"] == 8
    assert manifest["metrics"]["overlap_size"] == 4
    assert manifest["metrics"]["inference_seconds"] == 0.5
    assert captured_command[captured_command.index("--max-images") + 1] == "225"
    assert captured_command[captured_command.index("--batch-size") + 1] == "8"
    assert captured_command[captured_command.index("--overlap-size") + 1] == "4"


def test_create_colmap_point_cloud_job_uses_adapter_contract(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 9",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (job_dir / "geometry" / "cameras.json").write_text('{"images": []}\n', encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=colmap",
                    "num_images=2",
                    "registered_images=2",
                    "num_points=9",
                    "matcher=sequential",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "multi_image",
        [
            UploadedInput(filename="frame_001.jpg", content=b"fake-image", content_type="image/jpeg"),
            UploadedInput(filename="frame_002.jpg", content=b"fake-image", content_type="image/jpeg"),
        ],
        geometry_backend="colmap",
        output_type="point_cloud",
    )

    assert manifest["stage"] == "colmap_sparse_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["cameras"] == "geometry/cameras.json"
    assert manifest["metrics"]["registered_images"] == 2
    assert manifest["metrics"]["num_points"] == 9
    assert captured_command[captured_command.index("--matcher") + 1] == "sequential"


def test_create_colmap_vggt_point_cloud_job_uses_adapter_contract(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
        (job_dir / "geometry" / "cameras.json").write_text('{"images": []}\n', encoding="utf-8")
        (job_dir / "diagnostics" / "fusion.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "diagnostics" / "visibility_graph.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "diagnostics" / "scale_disagreement.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "diagnostics" / "consistency.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=colmap_vggt",
                    "num_images=2",
                    "registered_images=2",
                    "scaled_images=2",
                    "num_points=99",
                    "scale_median=0.25",
                    "vggt_batch_size=4",
                    "vggt_overlap_size=1",
                    "overlap_size=1",
                    "vggt_grouping=covisibility",
                    "conf_percentile=45.0",
                    "confidence_threshold_scope=per_frame",
                    "consistency_support_policy=adaptive_two",
                    "max_points=1500000",
                    "point_budget_policy=spatial_balanced",
                    "point_budget_input_points=2000000",
                    "point_budget_output_points=1500000",
                    "point_budget_applied=true",
                    "point_budget_quantization_bits=0",
                    "point_budget_occupied_codes=0",
                    "factorial_output_count=0",
                    "point_budget_sensitivity_output_count=0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "multi_image",
        [
            UploadedInput(filename="frame_001.jpg", content=b"fake-image", content_type="image/jpeg"),
            UploadedInput(filename="frame_002.jpg", content=b"fake-image", content_type="image/jpeg"),
        ],
        geometry_backend="colmap_vggt",
        output_type="point_cloud",
        options={
            "vggt_batch_size": 4,
            "colmap_vggt_grouping": "covisibility",
            "colmap_vggt_overlap_size": 1,
            "colmap_vggt_max_points": 1_500_000,
            "colmap_vggt_conf_percentile": 45.0,
            "colmap_vggt_confidence_threshold_scope": "per_frame",
            "colmap_vggt_consistency_support_policy": "adaptive_two",
            "colmap_vggt_point_budget_policy": "spatial_balanced",
        },
    )

    assert manifest["stage"] == "colmap_vggt_dense_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["cameras"] == "geometry/cameras.json"
    assert manifest["assets"]["fusion_diagnostics"] == "diagnostics/fusion.json"
    assert manifest["assets"]["visibility_graph"] == "diagnostics/visibility_graph.json"
    assert manifest["assets"]["scale_disagreement_diagnostics"] == "diagnostics/scale_disagreement.json"
    assert manifest["assets"]["consistency_diagnostics"] == "diagnostics/consistency.json"
    assert manifest["metrics"]["registered_images"] == 2
    assert manifest["metrics"]["scaled_images"] == 2
    assert manifest["metrics"]["num_points"] == 99
    assert manifest["metrics"]["scale_median"] == 0.25
    assert manifest["metrics"]["vggt_batch_size"] == 4
    assert manifest["metrics"]["vggt_grouping"] == "covisibility"
    assert manifest["metrics"]["vggt_overlap_size"] == 1
    assert manifest["metrics"]["overlap_size"] == 1
    assert manifest["metrics"]["max_points"] == 1500000
    assert manifest["metrics"]["confidence_threshold_scope"] == "per_frame"
    assert manifest["metrics"]["consistency_support_policy"] == "adaptive_two"
    assert manifest["metrics"]["point_budget_policy"] == "spatial_balanced"
    assert manifest["metrics"]["point_budget_input_points"] == 2000000
    assert manifest["metrics"]["point_budget_output_points"] == 1500000
    assert manifest["metrics"]["point_budget_applied"] is True
    assert manifest["metrics"]["factorial_output_count"] == 0
    assert manifest["metrics"]["conf_percentile"] == 45.0
    assert captured_command[captured_command.index("--matcher") + 1] == "exhaustive"
    assert captured_command[captured_command.index("--vggt-batch-size") + 1] == "4"
    assert captured_command[captured_command.index("--vggt-overlap-size") + 1] == "1"
    assert captured_command[captured_command.index("--vggt-grouping") + 1] == "covisibility"
    assert captured_command[captured_command.index("--fusion-mode") + 1] == "points"
    assert captured_command[captured_command.index("--max-points") + 1] == "1500000"
    assert captured_command[captured_command.index("--conf-percentile") + 1] == "45.0"
    assert captured_command[captured_command.index("--confidence-threshold-scope") + 1] == "per_frame"
    assert captured_command[captured_command.index("--consistency-support-policy") + 1] == "adaptive_two"
    assert captured_command[captured_command.index("--point-budget-policy") + 1] == "spatial_balanced"


def test_reject_colmap_vggt_overlap_not_smaller_than_batch(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="colmap_vggt_overlap_size must be smaller"):
        store.create_job(
            "multi_image",
            [
                UploadedInput(filename="first.jpg", content=b"first"),
                UploadedInput(filename="second.jpg", content=b"second"),
            ],
            geometry_backend="colmap_vggt",
            output_type="point_cloud",
            options={"vggt_batch_size": 4, "colmap_vggt_overlap_size": 4},
        )


def test_create_colmap_vggt_mesh_job_runs_mesh_postprocess(tmp_path, monkeypatch):
    captured_commands = []

    def fake_run(command, **kwargs):
        captured_commands.append(command)
        if str(command[1]).endswith("mesh_from_pointcloud.py"):
            output_path = Path(command[3])
            diagnostics_path = Path(command[command.index("--diagnostics-output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-glb")
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "method": "poisson",
                        "vertices": 42,
                        "triangles": 80,
                        "processed_points": 120,
                        "cleanup": {
                            "component_count": 3,
                            "long_edge_removed_triangles": 7,
                            "small_component_removed_triangles": 11,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="mesh ok\n", stderr="")

        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
        (job_dir / "geometry" / "cameras.json").write_text('{"images": []}\n', encoding="utf-8")
        (job_dir / "diagnostics" / "fusion.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "diagnostics" / "visibility_graph.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "diagnostics" / "scale_disagreement.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "diagnostics" / "consistency.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=colmap_vggt",
                    "num_images=2",
                    "num_points=99",
                    "fusion_mode=points",
                    "vggt_batch_size=4",
                    "vggt_overlap_size=2",
                    "overlap_size=0",
                    "vggt_grouping=sequential",
                    "conf_percentile=50.0",
                    "confidence_threshold_scope=global",
                    "consistency_support_policy=any_support",
                    "max_points=2000000",
                    "point_budget_policy=random",
                    "point_budget_input_points=99",
                    "point_budget_output_points=99",
                    "point_budget_applied=false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="points ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "multi_image",
        [
            UploadedInput(filename="frame_001.jpg", content=b"fake-image", content_type="image/jpeg"),
            UploadedInput(filename="frame_002.jpg", content=b"fake-image", content_type="image/jpeg"),
        ],
        geometry_backend="colmap_vggt",
        output_type="mesh",
    )

    assert manifest["output_type"] == "mesh"
    assert manifest["stage"] == "mesh_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["mesh"] == "geometry/mesh.glb"
    assert manifest["assets"]["mesh_diagnostics"] == "diagnostics/mesh.json"
    assert manifest["metrics"]["mesh_status"] == "built"
    assert manifest["metrics"]["mesh_method"] == "poisson"
    assert manifest["metrics"]["mesh_vertices"] == 42
    assert manifest["metrics"]["mesh_triangles"] == 80
    assert manifest["metrics"]["mesh_component_count"] == 3
    assert manifest["metrics"]["mesh_long_edge_removed_triangles"] == 7
    assert manifest["metrics"]["mesh_small_component_removed_triangles"] == 11
    assert manifest["metrics"]["fusion_mode"] == "points"
    assert manifest["metrics"]["vggt_grouping"] == "sequential"
    assert manifest["metrics"]["vggt_overlap_size"] == 2
    assert manifest["metrics"]["overlap_size"] == 0
    assert manifest["metrics"]["conf_percentile"] == 50.0
    assert manifest["metrics"]["confidence_threshold_scope"] == "global"
    assert manifest["metrics"]["consistency_support_policy"] == "any_support"
    assert manifest["metrics"]["max_points"] == 2_000_000
    assert manifest["metrics"]["point_budget_policy"] == "random"
    assert manifest["metrics"]["point_budget_applied"] is False
    assert manifest["mesh_variants"][0]["id"] == "baseline"
    assert manifest["mesh_variants"][0]["mesh_asset"] == "geometry/mesh.glb"
    assert any(str(command[1]).endswith("run_colmap_vggt_dense.py") for command in captured_commands)
    assert any(str(command[1]).endswith("mesh_from_pointcloud.py") for command in captured_commands)
    points_command = next(command for command in captured_commands if str(command[1]).endswith("run_colmap_vggt_dense.py"))
    assert points_command[points_command.index("--vggt-grouping") + 1] == "sequential"
    assert points_command[points_command.index("--vggt-overlap-size") + 1] == "2"
    assert points_command[points_command.index("--fusion-mode") + 1] == "points"
    assert points_command[points_command.index("--max-points") + 1] == "2000000"
    assert points_command[points_command.index("--conf-percentile") + 1] == "50.0"
    assert points_command[points_command.index("--confidence-threshold-scope") + 1] == "global"
    assert points_command[points_command.index("--consistency-support-policy") + 1] == "any_support"
    assert points_command[points_command.index("--point-budget-policy") + 1] == "random"
    mesh_command = next(command for command in captured_commands if str(command[1]).endswith("mesh_from_pointcloud.py"))
    assert "--edge-trim-factor" in mesh_command
    assert "--radius-outlier-radius" in mesh_command


def test_build_mesh_variant_reuses_existing_point_cloud(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_path = Path(command[3])
        diagnostics_path = Path(command[command.index("--diagnostics-output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"variant-glb")
        diagnostics_path.write_text(
            json.dumps(
                {
                    "method": "ball_pivoting",
                    "options": {"method": "ball_pivoting", "voxel_size": 0.07},
                    "vertices": 55,
                    "triangles": 77,
                    "processed_points": 110,
                    "cleanup": {
                        "component_count": 4,
                        "long_edge_removed_triangles": 8,
                        "small_component_removed_triangles": 9,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="variant ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")
    manifest = store.create_mock_job("image", [UploadedInput(filename="room.jpg", content=b"fake-image")])

    updated_manifest = store.build_mesh_variant(
        manifest["job_id"],
        {
            "method": "ball_pivoting",
            "voxel_size": 0.07,
            "normal_radius": 0.28,
            "edge_trim_factor": 1.6,
            "max_triangles": 120_000,
        },
    )

    variant = updated_manifest["mesh_variants"][-1]
    assert variant["method"] == "ball_pivoting"
    assert variant["mesh_asset"].startswith("geometry/mesh_ball_pivoting_")
    assert variant["metrics"]["mesh_triangles"] == 77
    assert (tmp_path / "jobs" / manifest["job_id"] / variant["mesh_asset"]).read_bytes() == b"variant-glb"
    assert captured_command[captured_command.index("--method") + 1] == "ball_pivoting"
    assert captured_command[captured_command.index("--voxel-size") + 1] == "0.07"


def test_reject_invalid_output_type(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="unsupported output_type"):
        store.create_job(
            "image",
            [UploadedInput(filename="room.jpg", content=b"fake-image")],
            geometry_backend="mock",
            output_type="rgbd",
        )


def test_asset_path_cannot_escape_job_dir(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    manifest = store.create_mock_job(
        "image",
        [UploadedInput(filename="room.jpg", content=b"fake-image")],
    )

    with pytest.raises(JobError, match="escapes job directory"):
        store.get_asset_path(manifest["job_id"], "../manifest.json")
