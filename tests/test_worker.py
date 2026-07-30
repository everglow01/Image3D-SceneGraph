from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

from image3d_scenegraph.geometry.adapters import ReconstructionResult
from image3d_scenegraph.jobs import JobCancelled, JobError, JobStore, UploadedInput
from image3d_scenegraph.worker import LocalJobWorker, run_cancellable_command


def upload(name: str = "room.jpg") -> list[UploadedInput]:
    return [UploadedInput(filename=name, content=b"image")]


def test_enqueue_is_durable_and_execute_publishes_only_complete_assets(tmp_path):
    store = JobStore(tmp_path / "jobs")
    queued = store.enqueue_job("image", upload())

    assert queued["status"] == "queued"
    assert queued["assets"] == {}
    assert store.get_manifest(queued["job_id"])["active_attempt_id"] == "attempt-001"
    assert (store.job_dir(queued["job_id"]) / "input" / "images" / "room.jpg").read_bytes() == b"image"

    done = store.execute_job(queued["job_id"])

    assert done["status"] == "done"
    assert done["attempts"][0]["status"] == "done"
    assert (store.job_dir(queued["job_id"]) / done["assets"]["point_cloud"]).is_file()


def test_worker_runs_fifo_serially(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    queued = [store.enqueue_job("image", upload(f"{index}.jpg")) for index in range(2)]
    active = 0
    maximum = 0
    order: list[str] = []
    lock = threading.Lock()

    class BlockingAdapter:
        def run(self, context):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                order.append(context.job_id)
            time.sleep(0.03)
            (context.job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
            with lock:
                active -= 1
            return ReconstructionResult("fake", {"point_cloud": "geometry/points.ply"}, {}, [])

    monkeypatch.setattr("image3d_scenegraph.jobs.get_reconstruction_adapter", lambda *_: BlockingAdapter())
    worker = LocalJobWorker(store, poll_seconds=0.01)
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if all(store.get_manifest(item["job_id"])["status"] == "done" for item in queued):
                break
            time.sleep(0.01)
    finally:
        worker.stop()

    assert order == [item["job_id"] for item in queued]
    assert maximum == 1


def test_restart_recovers_stale_running_without_false_success(tmp_path):
    store = JobStore(tmp_path / "jobs")
    queued = store.enqueue_job("image", upload())
    manifest_path = store.job_dir(queued["job_id"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["stage"] = "geometry_reconstruction"
    manifest["attempts"][0]["status"] = "running"
    store._write_json(manifest_path, manifest)

    recovered = JobStore(tmp_path / "jobs").recover_interrupted_jobs()
    failed = store.get_manifest(queued["job_id"])

    assert recovered == [queued["job_id"]]
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "worker_interrupted"
    assert failed["assets"] == {}


def test_cancel_queued_preserves_inputs_and_retry_is_bounded(tmp_path):
    store = JobStore(tmp_path / "jobs")
    queued = store.enqueue_job("image", upload())
    job_id = queued["job_id"]

    cancelled = store.cancel_job(job_id)
    assert cancelled["status"] == "cancelled"
    assert (store.job_dir(job_id) / "input" / "images" / "room.jpg").is_file()

    second = store.retry_job(job_id)
    assert second["active_attempt_id"] == "attempt-002"
    assert second["attempts"][-1]["parent_attempt_id"] == "attempt-001"
    store.cancel_job(job_id)
    third = store.retry_job(job_id)
    assert third["active_attempt_id"] == "attempt-003"
    store.cancel_job(job_id)
    with pytest.raises(JobError, match="retry limit"):
        store.retry_job(job_id)


def test_running_cancellation_preserves_partial_workspace(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    queued = store.enqueue_job("image", upload())
    entered = threading.Event()

    class WaitingAdapter:
        def run(self, context):
            entered.set()
            partial = context.job_dir / "diagnostics" / "partial.txt"
            partial.write_text("kept", encoding="utf-8")
            while not context.cancel_requested():
                time.sleep(0.005)
            raise JobCancelled("cancelled")

    monkeypatch.setattr("image3d_scenegraph.jobs.get_reconstruction_adapter", lambda *_: WaitingAdapter())
    thread = threading.Thread(target=store.execute_job, args=(queued["job_id"],))
    thread.start()
    assert entered.wait(1)
    store.cancel_job(queued["job_id"])
    thread.join(1)

    cancelled = store.get_manifest(queued["job_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["assets"] == {}
    partial = store.job_dir(queued["job_id"]) / "lifecycle" / "attempts" / "attempt-001" / "partial"
    assert (partial / "diagnostics" / "partial.txt").read_text(encoding="utf-8") == "kept"


def test_cancellable_command_terminates_process_group(tmp_path):
    started = time.monotonic()

    with pytest.raises(JobCancelled):
        run_cancellable_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            cancel_requested=lambda: time.monotonic() - started > 0.05,
            poll_seconds=0.01,
            terminate_timeout=0.1,
        )

    assert time.monotonic() - started < 2
