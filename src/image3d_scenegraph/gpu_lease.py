"""POSIX leases shared by GPU processes and edit-document writers."""

from __future__ import annotations

import fcntl
import os
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


_gpu_fd: ContextVar[int | None] = ContextVar("gpu_lease_fd", default=None)


def cloud_enabled() -> bool:
    return os.environ.get("IMAGE3D_CLOUD_ENABLED") == "1"


def gpu_lease_path(output_root: Path) -> Path:
    return Path(output_root).resolve().parent / ".gpu.lock"


def inherited_gpu_fds() -> tuple[int, ...]:
    fd = _gpu_fd.get()
    return () if fd is None else (fd,)


def require_idle_gpu() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeaseBusy("无法确认 GPU 空闲；未启动新任务") from exc
    if result.stdout.strip():
        raise LeaseBusy("GPU 有其他计算进程；等待其自行结束")


@contextmanager
def product_gpu_lease(output_root: Path):
    if not cloud_enabled():
        yield
        return
    with FileLease(gpu_lease_path(output_root)) as lease:
        require_idle_gpu()
        token = _gpu_fd.set(lease.fileno())
        try:
            yield
        finally:
            _gpu_fd.reset(token)


class LeaseBusy(RuntimeError):
    pass


class FileLease:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> FileLease:
        if self._fd is not None:
            raise RuntimeError("lease is already acquired")
        fd = os.open(
            self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise LeaseBusy("resource is occupied") from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def fileno(self) -> int:
        if self._fd is None:
            raise RuntimeError("lease is not acquired")
        return self._fd

    def close(self) -> None:
        if self._fd is not None:
            # Closing, rather than LOCK_UN, preserves a lease inherited by a live child.
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> FileLease:
        return self.acquire()

    def __exit__(self, *_args) -> None:
        self.close()
