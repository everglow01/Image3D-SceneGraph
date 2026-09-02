from __future__ import annotations

import fcntl
import threading

from image3d_scenegraph.execution import (
    run_cancellable_command as run_cancellable_command,
)
from image3d_scenegraph.jobs import JobStore


class LocalJobWorker:
    """One filesystem-backed worker for serial local GPU-heavy execution."""

    def __init__(self, store: JobStore, *, poll_seconds: float = 0.1) -> None:
        self.store = store
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.store.output_root.mkdir(parents=True, exist_ok=True)
        lease_path = self.store.output_root / ".worker.lock"
        self._lease = lease_path.open("a+")
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lease.close()
            self._lease = None
            raise RuntimeError("another local job worker already owns this output root")
        self.store.recover_interrupted_jobs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="image3d-local-worker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._lease is not None:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
            self._lease.close()
            self._lease = None

    def notify(self) -> None:
        self._wake.set()

    def run_once(self) -> str | None:
        queued = self.store.list_queued_jobs()
        if queued:
            job_id = queued[0]
            self.store.execute_job(job_id)
            return job_id
        navigation_queued = self.store.list_queued_navigation_jobs()
        if not navigation_queued:
            return None
        job_id = navigation_queued[0]
        self.store.execute_navigation_job(job_id)
        return job_id

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.run_once() is None:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()
