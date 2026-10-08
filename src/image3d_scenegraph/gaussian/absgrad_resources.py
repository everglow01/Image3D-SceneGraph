"""Opt-in process telemetry and stop gates for the frozen AbsGrad candidate."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time


POLL_SECONDS = 10
TELEMETRY_START_SECONDS = 600
TELEMETRY_STALE_SECONDS = 120
LIMITS = {
    "max_observed_global_gaussians": 2975056,
    "max_main_stage_wall_seconds": 10586.586387421936,
    "max_per_rank_peak_reserved_bytes": [8162115584, 7105150976],
}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


@contextmanager
def memory_telemetry(directory: Path | None, local_rank: int, world_rank: int, *, sample=None):
    if directory is None:
        yield
        return
    if sample is None:
        import torch

        def sample():
            return int(torch.cuda.max_memory_reserved(local_rank))

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{world_rank}.json"
    peak = 0
    errors = []
    stopped = threading.Event()

    def snapshot(*, finished=False):
        nonlocal peak
        peak = max(peak, sample())
        return {
            "rank": world_rank, "local_rank": local_rank, "pid": os.getpid(),
            "peak_reserved_bytes": peak, "monotonic_seconds": time.monotonic(),
            "finished": finished,
        }

    def publish(*, finished=False):
        temporary = path.with_suffix(".tmp")
        write_json(temporary, snapshot(finished=finished))
        os.replace(temporary, path)

    write_json(path, snapshot())

    def watch():
        while not stopped.wait(POLL_SECONDS):
            try:
                publish()
            except Exception as exc:
                errors.append(exc)
                return

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=POLL_SECONDS)
        if thread.is_alive():
            raise RuntimeError("resource telemetry thread did not stop")
        if errors:
            raise RuntimeError("resource telemetry failed") from errors[0]
        publish(finished=True)


class TrainingMonitor:
    def __init__(self, progress: Path, telemetry: Path, *, updates: int, main_stage: bool):
        self.progress = progress
        self.telemetry = telemetry
        self.updates = updates
        self.main_stage = main_stage
        self.offset = 0
        self.pending = b""
        self.iteration = 0
        self.max_gaussians = 0
        self.peak_reserved = [0, 0]
        self.camera_sequence = hashlib.sha256()

    def __call__(self, elapsed: float, *, final: bool = False) -> str | None:
        if self.main_stage and elapsed > LIMITS["max_main_stage_wall_seconds"]:
            return "main_wall_exceeded_2x_signed"
        if self.progress.exists():
            with self.progress.open("rb") as handle:
                if os.fstat(handle.fileno()).st_size < self.offset:
                    raise ValueError("progress log was truncated")
                handle.seek(self.offset)
                self.pending += handle.read()
                self.offset = handle.tell()
            lines = self.pending.split(b"\n")
            self.pending = lines.pop()
            for line in lines:
                event = json.loads(line)
                count = event["gaussian_count"]
                if type(count) is not int or count <= 0:
                    raise ValueError("invalid global Gaussian count")
                self.max_gaussians = max(self.max_gaussians, count)
                if event.get("event") != "validation":
                    iteration = event["iteration"]
                    if type(iteration) is not int or iteration != self.iteration + 1:
                        raise ValueError("missing or reordered training progress")
                    if event["world_size"] != 2:
                        raise ValueError("training world size changed")
                    batch = event["batch_view_ids"]
                    if not isinstance(batch, list) or len(batch) != 2 or any(not isinstance(v, str) for v in batch):
                        raise ValueError("expected two camera samples per update")
                    self.camera_sequence.update(("\0".join(batch) + "\n").encode())
                    self.iteration = iteration
        if self.max_gaussians > LIMITS["max_observed_global_gaussians"]:
            return "global_gaussians_exceeded_2x_signed"
        now = time.monotonic()
        for rank, limit in enumerate(LIMITS["max_per_rank_peak_reserved_bytes"]):
            path = self.telemetry / f"rank-{rank}.json"
            if not path.exists():
                if final or elapsed > TELEMETRY_START_SECONDS:
                    return f"rank_{rank}_telemetry_missing"
                continue
            record = json.loads(path.read_text())
            peak = record["peak_reserved_bytes"]
            stamp = record["monotonic_seconds"]
            if (record["rank"] != rank or record["local_rank"] != rank
                    or type(peak) is not int or peak < 0
                    or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                    or stamp > now + 1 or type(record["finished"]) is not bool):
                raise ValueError("invalid rank telemetry")
            if now - stamp > TELEMETRY_STALE_SECONDS:
                return f"rank_{rank}_telemetry_stale"
            self.peak_reserved[rank] = max(self.peak_reserved[rank], peak)
            if self.peak_reserved[rank] > limit:
                return f"rank_{rank}_reserved_exceeded_2x_signed"
            if final and not record["finished"]:
                return f"rank_{rank}_telemetry_unfinished"
        if final and (self.pending or self.iteration != self.updates):
            return "training_progress_incomplete"
        return None

    def record(self) -> dict:
        return {
            "optimizer_updates_observed": self.iteration,
            "camera_samples_observed": self.iteration * 2,
            "camera_sequence_sha256": self.camera_sequence.hexdigest(),
            "max_observed_global_gaussians": self.max_gaussians,
            "per_rank_peak_reserved_bytes": self.peak_reserved,
        }


def stop_process(process, cancel: Path) -> None:
    cancel.touch(exist_ok=True)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_stage(root: Path, name: str, command: list[str], *, cwd: Path,
              monitor: TrainingMonitor | None = None, pass_fds=()) -> None:
    write_json(root / f"{name}.command.json", {"argv": command})
    print(f"stage={name} started", flush=True)
    started = time.monotonic()
    failure = None
    process = None
    with (root / f"{name}.log").open("x") as log:
        try:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=pass_fds,
            )
            while True:
                try:
                    process.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                finished = process.poll() is not None
                elapsed = time.monotonic() - started
                if shutil.disk_usage(root).free < 4 * 1024**3:
                    failure = "free_disk_below_4_gib"
                elif elapsed > 6 * 3600:
                    failure = "stage_exceeded_6_hours"
                elif monitor is not None:
                    try:
                        failure = monitor(elapsed, final=finished)
                    except Exception as exc:
                        failure = f"monitor_failed: {type(exc).__name__}: {exc}"
                if failure and not finished:
                    stop_process(process, root / "cancel")
                if finished or failure:
                    break
        except BaseException as exc:
            failure = f"runner_failed: {type(exc).__name__}: {exc}"
            raise
        finally:
            if process is not None and process.poll() is None:
                stop_process(process, root / "cancel")
            write_json(root / f"{name}.exit.json", {
                "returncode": None if process is None else process.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "resource_failure": failure,
                "resources": None if monitor is None else monitor.record(),
            })
    if failure or process.returncode:
        raise RuntimeError(f"{name} failed: {failure or process.returncode}; retained {root}")
    print(f"stage={name} complete", flush=True)
