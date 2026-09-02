from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence


class JobCancelled(RuntimeError):
    """Raised when a queued or running local job is cancelled."""


def run_cancellable_command(
    command: Sequence[str],
    *,
    cwd: Path,
    cancel_requested: Callable[[], bool],
    env: Mapping[str, str] | None = None,
    poll_seconds: float = 0.1,
    terminate_timeout: float = 2.0,
    poll_callback: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess group and terminate it when cancellation is requested."""
    with (
        tempfile.TemporaryFile(mode="w+") as stdout_file,
        tempfile.TemporaryFile(mode="w+") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if poll_callback is not None:
                    poll_callback()
                if cancel_requested():
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=terminate_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise JobCancelled("job cancellation requested")
                time.sleep(poll_seconds)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed
