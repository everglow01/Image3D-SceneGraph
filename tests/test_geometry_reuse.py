from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from image3d_scenegraph.execution import JobCancelled
from image3d_scenegraph.geometry import reuse
from image3d_scenegraph.geometry.adapters import ReconstructionResult
from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput


def geometry(root):
    for name in reuse.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"status":"passed"}' if name.endswith(".json") else "geometry fixture"
        )
    path = root / "colmap/undistorted/images/a.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"undistorted fixture")


def inputs():
    return [
        UploadedInput(filename=f"10/{i}.jpg", content=f"image{i}".encode())
        for i in range(12)
    ]


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(
        store,
        "_try_generate_navigation",
        lambda *_a, **_k: ({}, {}, "unavailable", "test fixture", None),
    )
    monkeypatch.setattr(store, "_try_align_point_cloud", lambda *_a: ({}, {}, []))
    calls = []

    class Adapter:
        def run(self, context):
            root = context.job_dir
            reused = context.options.get("_geometry_reused", False)
            if not reused:
                geometry(root)
                calls.append(("geometry", context.job_id))
            else:
                assert (root / reuse.RECEIPT).is_file()
                assert context.options["_geometry_reuse_splits"] == {
                    "train": ["1"],
                    "validation": ["2"],
                    "test": ["3"],
                }
            contract = {"splits": {"train": ["1"], "validation": ["2"], "test": ["3"]}}
            reuse.freeze_geometry(
                root, context.job_id, context.options, context.input_assets, contract
            )
            trainer = context.options["gaussian_trainer"]
            for phase in ("training", "validation", "sor", "export"):
                calls.append((phase, context.job_id, trainer))
            (root / "gaussian/model.pt").write_bytes(trainer.encode())
            (root / "gaussian/scene.ply").write_bytes(b"ply\n" + trainer.encode())
            assets = {
                "gaussian_model": "gaussian/model.pt",
                "scene_splat": "gaussian/scene.ply",
                "gaussian_geometry_bundle": reuse.BUNDLE,
            }
            if reused:
                assets["gaussian_geometry_reuse"] = reuse.RECEIPT
            return ReconstructionResult(
                stage="gaussian_export",
                assets=assets,
                metrics={
                    "gaussian_geometry_origin": "reused" if reused else "computed",
                    "sfm_camera_calibration_profile": context.options[
                        "sfm_camera_calibration"
                    ],
                    "sfm_effective_mapper": context.options["sfm_mapper"],
                },
                log_lines=["fixture standard training and export completed"],
            )

    monkeypatch.setattr(
        "image3d_scenegraph.jobs.get_reconstruction_adapter", lambda *_: Adapter()
    )
    store.test_calls = calls
    return store


def enqueue(store, trainer="project", source=None, files=None, **options):
    if source is not None:
        options["gaussian_geometry_source_job_id"] = source
    return store.enqueue_job(
        "multi_image",
        files or inputs(),
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
        options={"gaussian_trainer": trainer, **options},
    )


def test_two_standard_jobs_publish_independently_and_survive_source_removal(
    store, monkeypatch
):
    from image3d_scenegraph.worker import LocalJobWorker

    monkeypatch.setattr("image3d_scenegraph.gpu_lease.cloud_enabled", lambda: False)
    worker = LocalJobWorker(store)
    first = enqueue(store)
    assert worker.run_once() == first["job_id"]
    a = store.get_manifest(first["job_id"])
    assert a["status"] == "done", a.get("error")
    second = enqueue(store, "mcmc", a["job_id"])
    assert second["job_id"] != a["job_id"] and second["status"] == "queued"
    assert worker.run_once() == second["job_id"]
    b = store.get_manifest(second["job_id"])
    assert b["status"] == "done", b.get("error")
    assert b["gaussian_geometry_origin"] == "reused"
    assert b["gaussian_geometry_source_job_id"] == a["job_id"]
    assert b["gaussian_trainer"]["id"] == "mcmc"
    assert [call[0] for call in store.test_calls].count("geometry") == 1
    for phase in ("training", "validation", "sor", "export"):
        assert (phase, a["job_id"], "project") in store.test_calls
        assert (phase, b["job_id"], "mcmc") in store.test_calls
    source = store.job_dir(a["job_id"])
    dest = store.job_dir(b["job_id"])
    original = source / "colmap/undistorted/images/a.jpg"
    copied = dest / "colmap/undistorted/images/a.jpg"
    assert original.stat().st_ino != copied.stat().st_ino
    assert not copied.is_symlink() and copied.read_bytes() == original.read_bytes()
    shutil.rmtree(source)
    assert (
        store.get_asset_path(b["job_id"], "gaussian/scene.ply").read_bytes()
        == b"ply\nmcmc"
    )
    assert store.build_zip(b["job_id"]).is_file()
    assert store.get_manifest(b["job_id"])["status"] == "done"
    assert not (dest / "lifecycle/attempts/attempt-001/workspace").exists()


@pytest.mark.parametrize(
    "change", ["input", "geometry", "bundle", "source_deleted", "source_input"]
)
def test_reuse_fails_before_training_on_source_or_input_drift(store, change):
    a = store.execute_job(enqueue(store)["job_id"])
    assert a["status"] == "done"
    files = inputs()
    if change == "input":
        files[0] = UploadedInput(filename="10/0.jpg", content=b"changed")
    b = enqueue(store, "mcmc", a["job_id"], files=files)
    source = store.job_dir(a["job_id"])
    if change == "geometry":
        (source / "geometry/points.ply").write_text("changed")
    elif change == "bundle":
        (source / reuse.BUNDLE).write_text("{}")
    elif change == "source_deleted":
        shutil.rmtree(source)
    elif change == "source_input":
        (source / "input/images/10/0.jpg").write_bytes(b"changed source input")
    result = store.execute_job(b["job_id"])
    assert result["status"] == "failed"
    assert not any(c[0] == "training" and c[1] == b["job_id"] for c in store.test_calls)
    assert not (store.job_dir(b["job_id"]) / "gaussian/scene.ply").exists()


def test_source_identity_and_geometry_settings_are_not_silently_changed(store):
    queued = enqueue(store)
    with pytest.raises(JobError, match="completed"):
        enqueue(store, "mcmc", queued["job_id"])
    a = store.execute_job(queued["job_id"])
    with pytest.raises(JobError, match="differs"):
        enqueue(store, "mcmc", a["job_id"], gaussian_longest_edge=1920)
    with pytest.raises(JobError, match="invalid geometry source"):
        enqueue(store, "mcmc", "../escape")
    with pytest.raises(JobError, match="internals"):
        enqueue(store, _geometry_reused=True)


def test_copy_cancellation_and_symlink_rejection(store, tmp_path):
    a = store.execute_job(enqueue(store)["job_id"])
    source = store.job_dir(a["job_id"])
    manifest = json.loads((source / "manifest.json").read_text())
    target = tmp_path / "copy"
    shutil.copytree(source / "input", target / "input")
    bundle = reuse.load_bundle(source)
    digest = reuse.sha256_file(source / reuse.BUNDLE)
    with pytest.raises(JobCancelled):
        reuse.copy_geometry(
            source,
            target,
            digest,
            bundle["geometry_options"],
            manifest["inputs"],
            lambda: True,
        )
    path = source / "geometry/points.ply"
    path.unlink()
    path.symlink_to(source / "geometry/cameras.json")
    with pytest.raises(ValueError, match="symlink"):
        reuse.load_bundle(source)


@pytest.mark.parametrize("trainer", ["project", "mcmc"])
def test_real_adapter_reuse_skips_sfm_but_reaches_native_training_command(
    tmp_path, monkeypatch, trainer
):
    from types import SimpleNamespace
    from image3d_scenegraph.geometry import adapters
    from image3d_scenegraph.gaussian.config import (
        resolve_mcmc_config,
        resolve_public_config,
        resolved_config_record,
    )

    geometry(tmp_path)
    (tmp_path / "colmap/database.db").write_bytes(b"database fixture")
    timing = {
        "schema_version": 1,
        "profile": "colmap_timing_v1",
        "requested_mapper": "incremental",
        "mapper": "incremental",
        "effective_database_sha256": "a" * 64,
        "effective_database_path": "colmap/database.db",
        "sfm_pose_health_path": "diagnostics/sfm_pose_health.json",
        "sfm_pose_recovery_path": "diagnostics/sfm_pose_recovery.json",
    }
    (tmp_path / "diagnostics/colmap_timing.json").write_text(json.dumps(timing))
    (tmp_path / "diagnostics/sfm_pose_recovery.json").write_text(
        json.dumps({"status": "not_needed", "recovery_applied": False})
    )
    monkeypatch.setattr(
        adapters, "_camera_calibration_diagnostics_metrics", lambda *_: {}
    )
    monkeypatch.setattr(adapters, "_validate_colmap_pose_evidence", lambda **_: 0)
    monkeypatch.setattr(
        adapters, "_try_export_sfm_diagnostics", lambda **_: ({}, {}, [])
    )
    contract = {"splits": {"train": ["1"], "validation": ["2"], "test": ["3"]}}
    monkeypatch.setattr(
        "image3d_scenegraph.gaussian.dataset.build_colmap_contract",
        lambda **_: contract,
    )
    monkeypatch.setattr(
        "image3d_scenegraph.gaussian.dataset.write_contract", lambda *_: None
    )
    monkeypatch.setattr(
        "image3d_scenegraph.gaussian.trainers.get_gaussian_trainer_specs",
        lambda *_: [SimpleNamespace(trainer_id=trainer, available=True, label=trainer)],
    )
    record = (
        resolve_mcmc_config()
        if trainer == "mcmc"
        else resolve_public_config("standard_v1")
    )
    options = {
        "_geometry_reused": True,
        "_geometry_reuse_splits": contract["splits"],
        "gaussian_geometry_source_job_id": "source",
        "gaussian_trainer": trainer,
        "gaussian_config_record": json.dumps(resolved_config_record(record)),
    }
    commands = []

    def stop_at_training(command, *_args, **_kwargs):
        commands.append(command)
        assert Path(command[1]).name == "run_gaussian_training.py"
        assert command[command.index("--trainer") + 1] == trainer
        raise RuntimeError("verified native training boundary; no training executed")

    monkeypatch.setattr(adapters, "_run_adapter_command", stop_at_training)
    context = adapters.ReconstructionContext(
        job_id="new-job",
        job_dir=tmp_path,
        mode="multi_image",
        input_assets=[],
        options=options,
    )
    with pytest.raises(RuntimeError, match="verified native training boundary"):
        adapters.ProjectGaussianAdapter().run(context)
    assert len(commands) == 1
    assert (tmp_path / reuse.BUNDLE).is_file()


def test_hybrid_source_metadata_is_inherited_by_second_standard_job(store, monkeypatch):
    from types import SimpleNamespace

    jobs = "image3d_scenegraph.jobs."
    monkeypatch.setattr(
        jobs + "resolve_colmap_executable", lambda *_: Path("/mock/colmap")
    )
    for name in (
        "colmap_pairing_support_reason",
        "colmap_camera_calibration_support_reason",
        "colmap_local_matcher_support_reason",
    ):
        monkeypatch.setattr(jobs + name, lambda *_: None)
    monkeypatch.setattr(jobs + "resolve_colmap_pairing", lambda *_: SimpleNamespace())
    files = [
        UploadedInput(
            filename=f"{10 + i // 6}/{i % 6}.jpg", content=f"image{i}".encode()
        )
        for i in range(12)
    ]
    metadata = {
        "schema_version": 1,
        "images": {
            f.filename: {
                "camera_id": f.filename.split("/")[0],
                "capture_index": int(Path(f.filename).stem),
            }
            for f in files
        },
    }
    a = store.execute_job(
        enqueue(
            store,
            files=files,
            sfm_pairing="rig_neighbors_vocab_v1",
            sfm_camera_calibration="folder_grouped_opencv_v1",
            sfm_capture_metadata=json.dumps(metadata),
        )["job_id"]
    )
    assert a["status"] == "done", a.get("error")
    b = enqueue(
        store,
        "mcmc",
        a["job_id"],
        files=files,
        sfm_capture_metadata=json.dumps(metadata, indent=2),
    )
    request = json.loads((store.job_dir(b["job_id"]) / "request.json").read_text())
    assert request["options"]["sfm_pairing"] == "rig_neighbors_vocab_v1"
    assert request["options"]["sfm_camera_calibration"] == "folder_grouped_opencv_v1"
    assert store.execute_job(b["job_id"])["status"] == "done"


def test_cancelled_reuse_job_keeps_its_normal_retry_lifecycle(store, monkeypatch):
    a = store.execute_job(enqueue(store)["job_id"])
    b = enqueue(store, "mcmc", a["job_id"])
    original = reuse.copy_geometry

    def cancel_during_copy(*args, **kwargs):
        store.cancel_job(b["job_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(reuse, "copy_geometry", cancel_during_copy)
    cancelled = store.execute_job(b["job_id"])
    assert cancelled["status"] == "cancelled"
    assert not any(c[0] == "training" and c[1] == b["job_id"] for c in store.test_calls)
    monkeypatch.setattr(reuse, "copy_geometry", original)
    retried = store.retry_job(b["job_id"])
    assert retried["active_attempt_id"] == "attempt-002"
    done = store.execute_job(b["job_id"])
    assert done["status"] == "done" and len(done["attempts"]) == 2
    assert done["gaussian_geometry_source_job_id"] == a["job_id"]


@pytest.mark.parametrize("path", ["../escape", "/absolute", "gaussian/model.pt"])
def test_geometry_bundle_cannot_import_arbitrary_paths_or_models(store, path):
    a = store.execute_job(enqueue(store)["job_id"])
    root = store.job_dir(a["job_id"])
    p = root / reuse.BUNDLE
    data = json.loads(p.read_text())
    data["files"][path] = "a" * 64
    p.write_text(json.dumps(data))
    with pytest.raises(
        ValueError, match="invalid geometry bundle file|unsafe geometry path"
    ):
        reuse.load_bundle(root)
