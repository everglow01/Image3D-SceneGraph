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
QUALITY_UNIT = "image3d-absgrad-quality-20261009-v1.service"
QUALITY_LIMITS = {
    "max_observed_global_gaussians": 6000000,
    "max_main_stage_wall_seconds": 21600,
    "max_per_rank_peak_reserved_bytes": [18 * 1024**3, 18 * 1024**3],
}
QUALITY_HOST_POLICY = {
    "startup_available_bytes": 22 * 1024**3,
    "minimum_available_bytes": 6 * 1024**3,
    "stop_current_bytes": 14 * 1024**3,
    "memory_max_bytes": 16 * 1024**3,
    "memory_swap_max_bytes": 0,
    "poll_seconds": 2,
    "term_grace_seconds": 5,
}
LIMITS = {
    "max_observed_global_gaussians": 2975056,
    "max_main_stage_wall_seconds": 10586.586387421936,
    "max_per_rank_peak_reserved_bytes": [8162115584, 7105150976],
}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def available_host_bytes() -> int:
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    value = int(memory["MemAvailable"].split()[0]) * 1024
    if value < 0:
        raise ValueError("invalid MemAvailable")
    return value


def quality_cgroup() -> Path:
    expected = f"/system.slice/{QUALITY_UNIT}"
    groups = Path("/proc/self/cgroup").read_text().splitlines()
    if f"0::{expected}" not in groups:
        raise ValueError("quality exploration requires its isolated systemd cgroup")
    group = Path("/sys/fs/cgroup") / expected.lstrip("/")
    for name, expected_value in (("memory.max", QUALITY_HOST_POLICY["memory_max_bytes"]),
                                 ("memory.swap.max", 0), ("memory.oom.group", 1)):
        if int((group / name).read_text()) != expected_value:
            raise ValueError(f"unexpected task cgroup {name}")
    return group


def host_snapshot(group: Path) -> dict:
    events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
    current = int((group / "memory.current").read_text())
    if current < 0:
        raise ValueError("invalid cgroup memory.current")
    return {"available_bytes": available_host_bytes(), "task_current_bytes": current,
            "memory_events": {key: int(value) for key, value in events.items()}}


def host_failure(snapshot: dict, *, admission=False) -> str | None:
    minimum = QUALITY_HOST_POLICY["startup_available_bytes" if admission else "minimum_available_bytes"]
    if snapshot["available_bytes"] < minimum:
        return "host_available_below_22_gib" if admission else "host_available_below_6_gib"
    if snapshot["task_current_bytes"] >= QUALITY_HOST_POLICY["stop_current_bytes"]:
        return "task_memory_at_14_gib"
    if snapshot["memory_events"].get("oom_kill", 0):
        return "task_cgroup_oom_kill"
    return None


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
    def __init__(self, progress: Path, telemetry: Path, *, updates: int, main_stage: bool, limits=None):
        self.limits = LIMITS if limits is None else limits
        self.limit_label = "2x_signed" if self.limits == LIMITS else "hardware_budget"
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
        if self.main_stage and elapsed > self.limits["max_main_stage_wall_seconds"]:
            return f"main_wall_exceeded_{self.limit_label}"
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
        if self.max_gaussians > self.limits["max_observed_global_gaussians"]:
            return f"global_gaussians_exceeded_{self.limit_label}"
        now = time.monotonic()
        for rank, limit in enumerate(self.limits["max_per_rank_peak_reserved_bytes"]):
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
                return f"rank_{rank}_reserved_exceeded_{self.limit_label}"
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


def stop_process(process, cancel: Path, *, emergency=False) -> None:
    if emergency:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=QUALITY_HOST_POLICY["term_grace_seconds"])
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return
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
              monitor: TrainingMonitor | None = None, pass_fds=(), host_group: Path | None = None) -> None:
    write_json(root / f"{name}.command.json", {"argv": command})
    print(f"stage={name} started", flush=True)
    started = time.monotonic()
    failure = None
    process = None
    host = None
    elapsed = 0.0
    failure_path = root / f"{name}.failure.json"

    def retain_failure():
        if not failure_path.exists():
            write_json(failure_path, {"reason": failure, "elapsed_seconds": elapsed,
                "host": host, "resources": None if monitor is None else monitor.record()})

    def stop():
        if host_group is None:
            stop_process(process, root / "cancel")
        else:
            stop_process(process, root / "cancel", emergency=True)

    with (root / f"{name}.log").open("x") as log:
        try:
            if host_group is not None:
                host = host_snapshot(host_group)
                failure = host_failure(host, admission=True)
                if failure:
                    raise RuntimeError(failure)
            process = subprocess.Popen(
                command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=pass_fds,
            )
            while True:
                try:
                    process.wait(timeout=POLL_SECONDS if host_group is None else QUALITY_HOST_POLICY["poll_seconds"])
                except subprocess.TimeoutExpired:
                    pass
                finished = process.poll() is not None
                elapsed = time.monotonic() - started
                if host_group is not None:
                    host = host_snapshot(host_group)
                    failure = host_failure(host)
                if failure:
                    pass
                elif shutil.disk_usage(root).free < 4 * 1024**3:
                    failure = "free_disk_below_4_gib"
                elif elapsed > 6 * 3600:
                    failure = "stage_exceeded_6_hours"
                elif monitor is not None:
                    try:
                        failure = monitor(elapsed, final=finished)
                    except Exception as exc:
                        failure = f"monitor_failed: {type(exc).__name__}: {exc}"
                if host_group is not None and finished and process.returncode and failure is None:
                    failure = f"child_exit_{process.returncode}"
                if failure:
                    retain_failure()
                    if not finished:
                        stop()
                if finished or failure:
                    break
        except BaseException as exc:
            failure = failure or f"runner_failed: {type(exc).__name__}: {exc}"
            retain_failure()
            raise
        finally:
            if process is not None and process.poll() is None:
                stop()
            write_json(root / f"{name}.exit.json", {
                "returncode": None if process is None else process.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "resource_failure": failure,
                "host": host,
                "resources": None if monitor is None else monitor.record(),
            })
    if failure or process.returncode:
        raise RuntimeError(f"{name} failed: {failure or process.returncode}; retained {root}")
    print(f"stage={name} complete", flush=True)
