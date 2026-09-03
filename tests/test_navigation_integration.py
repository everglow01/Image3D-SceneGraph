from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from image3d_scenegraph.gaussian.dataset import sha256_file
from image3d_scenegraph.geometry.adapters import ReconstructionResult
from image3d_scenegraph.jobs import (
    NAVIGATION_ASSET_ROLES,
    JobError,
    JobStore,
    UploadedInput,
)
from image3d_scenegraph.worker import LocalJobWorker


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _completed_gaussian_job(store: JobStore, job_id: str = "gaussian-job") -> tuple[Path, dict]:
    job_dir = store.job_dir(job_id)
    model = job_dir / "gaussian" / "attempts" / "train-001" / "artifacts" / "model.pt"
    dataset = job_dir / "gaussian" / "preparation" / "train-001" / "dataset.json"
    config = dataset.with_name("effective_config.json")
    export = job_dir / "gaussian" / "export" / "train-001" / "export.json"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    contract = {
        "dataset_hash": "dataset-hash",
        "splits": {"train": ["train-1", "train-2"], "validation": ["val-1"], "test": ["test-1"]},
    }
    config_record = {"effective_config_hash": "config-hash"}
    _write_json(dataset, contract)
    _write_json(config, config_record)
    _write_json(export, {"model_sha256": sha256_file(model)})
    manifest = {
        "job_id": job_id,
        "status": "done",
        "stage": "gaussian_export",
        "progress": 1.0,
        "mode": "multi_image",
        "geometry_backend": "project_3dgs",
        "output_type": "gaussian_splat",
        "created_at": "2026-08-11T00:00:00Z",
        "assets": {
            "gaussian_model": model.relative_to(job_dir).as_posix(),
            "gaussian_dataset": dataset.relative_to(job_dir).as_posix(),
            "gaussian_effective_config": config.relative_to(job_dir).as_posix(),
            "gaussian_export_metadata": export.relative_to(job_dir).as_posix(),
            "scene_splat": "gaussian/export/train-001/scene.ply",
        },
        "metrics": {},
    }
    (export.parent / "scene.ply").write_bytes(b"splat")
    _write_json(job_dir / "manifest.json", manifest)
    return job_dir, manifest


def _write_navigation_output(output_dir: Path, job_dir: Path, manifest: dict) -> None:
    sources = JobStore(job_dir.parent)._navigation_sources(job_dir, manifest)
    output_dir.mkdir(parents=True)
    collision = output_dir / "collision.glb"
    collision.write_bytes(b"collision")
    contract = json.loads(sources["dataset"].read_text())
    config = json.loads(sources["config"].read_text())
    navigation = {
        "schema_version": 1,
        "status": "available",
        "coordinate_frame": "normalized",
        "world_units": "arbitrary",
        "provenance": {
            "dataset_hash": contract["dataset_hash"],
            "effective_config_hash": config["effective_config_hash"],
            "model_sha256": sha256_file(sources["model"]),
            "export_sha256": sha256_file(sources["export"]),
            "train_image_ids": contract["splits"]["train"],
            "selected_render_image_ids": ["train-1"],
            "validation_image_ids_used": [],
            "test_image_ids_used": [],
        },
        "collision": {
            "asset": "collision.glb",
            "sha256": sha256_file(collision),
            "bytes": collision.stat().st_size,
            "triangles": 1_234,
        },
        "generation": {"elapsed_seconds": 1.25},
        "quality": {
            "topology": {
                "self_intersecting": False,
                "vertex_manifold": True,
                "edge_manifold_allow_boundary": True,
                "orientable": True,
            }
        },
    }
    _write_json(output_dir / "navigation.json", navigation)
    _write_json(
        output_dir / "diagnostics.json",
        {"schema_version": 1, "status": "complete", "train_only": True},
    )


def test_old_gaussian_navigation_generation_is_queued_idempotent_and_atomic(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job_dir, _ = _completed_gaussian_job(store)

    first = store.request_navigation_assets("gaussian-job")
    second = store.request_navigation_assets("gaussian-job")

    assert first["navigation_status"] == "queued"
    assert second["navigation_attempt"] == 1
    assert not (job_dir / "navigation").exists()

    def fake_builder(job_dir, manifest, output_dir, *, cancel_requested):
        _write_navigation_output(output_dir, job_dir, manifest)

    monkeypatch.setattr(store, "_run_navigation_builder", fake_builder)
    done = store.execute_navigation_job("gaussian-job")

    assert done["status"] == "done"
    assert done["navigation_status"] == "available"
    assert done["assets"] | NAVIGATION_ASSET_ROLES == done["assets"]
    assert done["navigation_details"]["train_only"] is True
    assert (job_dir / done["assets"]["collision_mesh"]).read_bytes() == b"collision"
    assert not (job_dir / "lifecycle/navigation/attempt-001/workspace").exists()
    assert store.request_navigation_assets("gaussian-job")["navigation_attempt"] == 1

    bundle = store.build_zip("gaussian-job")
    with zipfile.ZipFile(bundle) as archive:
        assert not any(name.startswith("navigation/") for name in archive.namelist())
        assert not any(name.startswith("lifecycle/navigation/") for name in archive.namelist())
        assert "manifest.json" in archive.namelist()
        bundled_manifest = json.loads(archive.read("manifest.json"))
        assert "navigation_status" not in bundled_manifest
        assert not NAVIGATION_ASSET_ROLES.keys() & bundled_manifest["assets"].keys()


def test_invalid_published_navigation_is_quarantined_and_can_retry(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job_dir, _ = _completed_gaussian_job(store)
    store.request_navigation_assets("gaussian-job")

    monkeypatch.setattr(
        store,
        "_run_navigation_builder",
        lambda job_dir, manifest, output_dir, *, cancel_requested: _write_navigation_output(
            output_dir, job_dir, manifest
        ),
    )
    store.execute_navigation_job("gaussian-job")
    (job_dir / "navigation/collision.glb").write_bytes(b"tampered")

    queued = store.request_navigation_assets("gaussian-job")

    assert queued["navigation_status"] == "queued"
    assert queued["navigation_attempt"] == 2
    assert not (job_dir / "navigation").exists()
    assert (job_dir / "lifecycle/navigation/invalid_published/collision.glb").is_file()



def test_schema_valid_published_tampering_is_detected(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job_dir, _ = _completed_gaussian_job(store)
    store.request_navigation_assets("gaussian-job")
    monkeypatch.setattr(
        store,
        "_run_navigation_builder",
        lambda job_dir, manifest, output_dir, *, cancel_requested: _write_navigation_output(
            output_dir, job_dir, manifest
        ),
    )
    store.execute_navigation_job("gaussian-job")
    diagnostics = json.loads((job_dir / "navigation/diagnostics.json").read_text())
    diagnostics["extra"] = "tampered"
    _write_json(job_dir / "navigation/diagnostics.json", diagnostics)

    manifest = store.get_manifest("gaussian-job")

    assert manifest["navigation_status"] == "unavailable"
    assert manifest["navigation_reason"] == "published_assets_invalid"
    assert not NAVIGATION_ASSET_ROLES.keys() & manifest["assets"].keys()



def test_navigation_rejects_escaped_source_path(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job_dir, manifest = _completed_gaussian_job(store)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    manifest["assets"]["gaussian_model"] = "../../outside.pt"
    _write_json(job_dir / "manifest.json", manifest)

    with pytest.raises(JobError, match="escapes job directory"):
        store.request_navigation_assets("gaussian-job")


def test_navigation_rejects_symlinked_output(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job_dir, manifest = _completed_gaussian_job(store)
    output = tmp_path / "navigation-output"
    _write_navigation_output(output, job_dir, manifest)
    real_collision = output / "real-collision.glb"
    (output / "collision.glb").rename(real_collision)
    (output / "collision.glb").symlink_to(real_collision)

    with pytest.raises(JobError, match="must not be symbolic links"):
        store._validate_navigation_workspace(job_dir, manifest, output)



def test_navigation_validation_rejects_held_out_provenance(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job_dir, manifest = _completed_gaussian_job(store)
    output = tmp_path / "navigation-output"
    _write_navigation_output(output, job_dir, manifest)
    navigation_path = output / "navigation.json"
    navigation = json.loads(navigation_path.read_text())
    navigation["provenance"]["test_image_ids_used"] = ["test-1"]
    _write_json(navigation_path, navigation)

    with pytest.raises(JobError, match="provenance validation failed"):
        store._validate_navigation_workspace(job_dir, manifest, output)


def test_navigation_validation_rejects_bad_topology(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job_dir, manifest = _completed_gaussian_job(store)
    output = tmp_path / "navigation-output"
    _write_navigation_output(output, job_dir, manifest)
    navigation_path = output / "navigation.json"
    navigation = json.loads(navigation_path.read_text())
    navigation["quality"]["topology"]["self_intersecting"] = True
    _write_json(navigation_path, navigation)

    with pytest.raises(JobError, match="topology validation failed"):
        store._validate_navigation_workspace(job_dir, manifest, output)



def test_navigation_failure_is_fail_soft_for_new_gaussian_job(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    files = [UploadedInput(filename=f"{index}.jpg", content=b"image") for index in range(12)]

    class GaussianAdapter:
        def run(self, context):
            (context.job_dir / "gaussian/export/train-001").mkdir(parents=True)
            scene = context.job_dir / "gaussian/export/train-001/scene.ply"
            scene.write_bytes(b"splat")
            return ReconstructionResult(
                "gaussian_export",
                {"scene_splat": "gaussian/export/train-001/scene.ply"},
                {"sfm_camera_calibration_profile": "shared_opencv_v1"},
                [],
            )

    monkeypatch.setattr("image3d_scenegraph.jobs.get_reconstruction_adapter", lambda *_: GaussianAdapter())
    monkeypatch.setattr(
        store,
        "_try_generate_navigation",
        lambda *args, **kwargs: ({}, {"navigation_status": "unavailable"}, "unavailable", "navigation_generation_failed", None),
    )
    queued = store.enqueue_job(
        "multi_image",
        files,
        geometry_backend="project_3dgs",
        output_type="gaussian_splat",
    )
    done = store.execute_job(queued["job_id"], cancellable=False)

    assert done["status"] == "done"
    assert done["navigation_status"] == "unavailable"
    assert done["navigation_reason"] == "navigation_generation_failed"
    assert "scene_splat" in done["assets"]
    assert not NAVIGATION_ASSET_ROLES.keys() & done["assets"].keys()


def test_navigation_queue_uses_same_worker_after_reconstruction_queue(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    _completed_gaussian_job(store)
    store.request_navigation_assets("gaussian-job")
    calls = []

    monkeypatch.setattr(store, "execute_navigation_job", lambda job_id: calls.append(job_id))
    worker = LocalJobWorker(store)

    assert worker.run_once() == "gaussian-job"
    assert calls == ["gaussian-job"]


def test_cancel_and_restart_recovery_keep_gaussian_result(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job_dir, _ = _completed_gaussian_job(store)
    store.request_navigation_assets("gaussian-job")

    cancelled = store.cancel_job("gaussian-job")
    assert cancelled["status"] == "done"
    assert cancelled["navigation_status"] == "unavailable"
    assert cancelled["navigation_reason"] == "cancelled"
    assert cancelled["assets"]["scene_splat"].endswith("scene.ply")

    queued = store.request_navigation_assets("gaussian-job")
    queued["navigation_status"] = "generating"
    _write_json(job_dir / "manifest.json", queued)
    (job_dir / "navigation").mkdir()
    (job_dir / "navigation/collision.glb").write_bytes(b"partial")

    recovered = store.recover_interrupted_jobs()
    manifest = store.get_manifest("gaussian-job")

    assert recovered == ["gaussian-job"]
    assert manifest["status"] == "done"
    assert manifest["navigation_status"] == "unavailable"
    assert manifest["navigation_reason"] == "worker_interrupted"
    assert not (job_dir / "navigation").exists()
    assert (job_dir / "lifecycle/navigation/attempt-002/partial_published/collision.glb").is_file()
