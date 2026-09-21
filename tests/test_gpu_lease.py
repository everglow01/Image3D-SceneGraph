import subprocess
import sys

import pytest

from image3d_scenegraph.gpu_lease import FileLease, LeaseBusy


def test_lease_is_exclusive_and_released(tmp_path):
    path = tmp_path / "gpu.lock"
    with FileLease(path):
        with pytest.raises(LeaseBusy):
            FileLease(path).acquire()
    with FileLease(path) as lease:
        assert lease.fileno() >= 0
    with pytest.raises(RuntimeError):
        lease.fileno()


def test_child_keeps_lease_after_parent_closes(tmp_path):
    path = tmp_path / "gpu.lock"
    lease = FileLease(path).acquire()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.read(1)",
        ],
        pass_fds=(lease.fileno(),),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        lease.close()
        with pytest.raises(LeaseBusy):
            FileLease(path).acquire()
        process.communicate("x", timeout=5)
        with FileLease(path):
            pass
    finally:
        lease.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_symlink_lock_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("unchanged")
    path = tmp_path / "gpu.lock"
    path.symlink_to(target)
    with pytest.raises(OSError):
        FileLease(path).acquire()
    assert target.read_text() == "unchanged"


def test_worker_queues_behind_renderer_and_command_inherits_lease(
    tmp_path, monkeypatch
):
    from unittest.mock import Mock
    import image3d_scenegraph.gpu_lease as module
    from image3d_scenegraph.worker import LocalJobWorker
    from image3d_scenegraph.execution import run_cancellable_command

    monkeypatch.setenv("IMAGE3D_CLOUD_ENABLED", "1")
    probe = Mock()
    monkeypatch.setattr(module, "require_idle_gpu", probe)
    store = Mock(output_root=tmp_path / "jobs")
    store.list_queued_jobs.return_value = ["queued-job"]
    worker = LocalJobWorker(store)
    with FileLease(module.gpu_lease_path(store.output_root)):
        assert worker.run_once() is None
        store.execute_job.assert_not_called()
        probe.assert_not_called()
    worker._gpu_retry_at = 0

    def execute(_job):
        (fd,) = module.inherited_gpu_fds()
        result = run_cancellable_command(
            [sys.executable, "-c", f"import os; os.fstat({fd}); print('inherited')"],
            cwd=tmp_path,
            cancel_requested=lambda: False,
        )
        assert result.stdout.strip() == "inherited"
        with pytest.raises(LeaseBusy):
            FileLease(module.gpu_lease_path(store.output_root)).acquire()

    store.execute_job.side_effect = execute
    assert worker.run_once() == "queued-job"
    assert not module.inherited_gpu_fds()
    with FileLease(module.gpu_lease_path(store.output_root)):
        pass
    store.list_queued_jobs.return_value = []
    store.list_queued_navigation_jobs.return_value = ["nav-job"]
    store.execute_navigation_job.side_effect = execute
    assert worker.run_once() == "nav-job"


def test_gpu_probe_fails_closed_and_disabled_feature_does_not_probe(
    tmp_path, monkeypatch
):
    from unittest.mock import Mock
    import image3d_scenegraph.gpu_lease as module

    monkeypatch.setattr(
        module.subprocess, "run", Mock(return_value=Mock(stdout="12345\n"))
    )
    with pytest.raises(LeaseBusy):
        module.require_idle_gpu()
    monkeypatch.setattr(
        module, "require_idle_gpu", Mock(side_effect=AssertionError("must not probe"))
    )
    monkeypatch.delenv("IMAGE3D_CLOUD_ENABLED", raising=False)
    with module.product_gpu_lease(tmp_path / "jobs"):
        assert not module.inherited_gpu_fds()
