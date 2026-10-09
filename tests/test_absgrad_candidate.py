from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest

from image3d_scenegraph.gaussian import absgrad_resources as resources
from image3d_scenegraph.gaussian.config import resolve_internal_config, resolved_config_record
from scripts import run_absgrad_candidate as candidate


def signed_config():
    return resolved_config_record(resolve_internal_config("absgrad_ablation_v1", {
        "resolution": {"longest_edge": 1920},
        "opacity_reset": {"recovery_prune": {"enabled": True}},
    }))


def test_candidate_preserves_single_leaf_and_rejects_config_drift():
    signed = signed_config()
    absolute = candidate.absolute_config(signed)
    assert absolute["effective_config_hash"] == candidate.ABSOLUTE_CONFIG_HASH
    restored = copy.deepcopy(absolute["effective_config"])
    assert restored["densification"]["absgrad"] is True
    restored["densification"]["absgrad"] = False
    assert restored == signed["effective_config"]
    changed = resolved_config_record(resolve_internal_config("absgrad_ablation_v1", {
        "resolution": {"longest_edge": 1920},
        "opacity_reset": {"recovery_prune": {"enabled": True}}, "seed": 123,
    }))
    with pytest.raises(ValueError):
        candidate.absolute_config(changed)


def approved_gate():
    gate = candidate.gate_template()
    gate.update(status="APPROVED_FOR_CANDIDATE_EXECUTION", absolute_training_authorized=True)
    return gate


def test_gate_requires_explicit_approval_and_exact_frozen_limits():
    with pytest.raises(ValueError, match="approved"):
        candidate.validate_gate(candidate.gate_template())
    candidate.validate_gate(approved_gate())
    for key, value in (("absolute_training_authorized", 1), ("test_authorized", True),
                       ("quality_proposal_sha256", "changed"), ("train_roi_sha256", "changed")):
        gate = approved_gate()
        gate[key] = value
        with pytest.raises(ValueError):
            candidate.validate_gate(gate)
    gate = approved_gate()
    gate["resource_limits"]["max_observed_global_gaussians"] += 1
    with pytest.raises(ValueError):
        candidate.validate_gate(gate)


def test_existing_or_symlink_output_rejected_before_reading_sources(tmp_path):
    for path in (tmp_path, tmp_path / "dangling"):
        if path != tmp_path:
            path.symlink_to(tmp_path / "missing")
        with pytest.raises(ValueError, match="already exists"):
            candidate.preflight(tmp_path, tmp_path / "unused", "unused", path)


def test_checked_json_and_template_never_overwrite(tmp_path):
    path = tmp_path / "gate.json"
    resources.write_json(path, candidate.gate_template())
    with pytest.raises(FileExistsError):
        resources.write_json(path, {})
    with pytest.raises(ValueError, match="identity mismatch"):
        candidate.checked_json(path, "0" * 64)


def monitor_fixture(tmp_path, *, updates=2, main=True):
    progress = tmp_path / "progress.jsonl"
    telemetry = tmp_path / "memory"
    telemetry.mkdir()
    monitor = resources.TrainingMonitor(progress, telemetry, updates=updates, main_stage=main)
    for rank in (0, 1):
        write_rank(telemetry, rank)
    return monitor


def write_rank(directory, rank, *, peak=100, finished=True, stamp=None):
    (directory / f"rank-{rank}.json").write_text(json.dumps({
        "rank": rank, "local_rank": rank, "pid": os.getpid(), "finished": finished,
        "peak_reserved_bytes": peak, "monotonic_seconds": time.monotonic() if stamp is None else stamp,
    }))


def event(iteration, count=100):
    return json.dumps({"iteration": iteration, "world_size": 2, "gaussian_count": count, "batch_view_ids": ["1", "2"]}) + "\n"


def test_progress_partial_lines_and_final_checks(tmp_path):
    monitor = monitor_fixture(tmp_path)
    second = event(2)
    monitor.progress.write_text(event(1) + second[:12])
    assert monitor(10) is None
    assert monitor.iteration == 1
    assert monitor(11, final=True) == "training_progress_incomplete"
    with monitor.progress.open("a") as stream:
        stream.write(second[12:])
    assert monitor(12, final=True) is None
    assert monitor.record()["optimizer_updates_observed"] == 2
    monitor.progress.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        monitor(13)


@pytest.mark.parametrize("kind", ["gaussians", "rank0", "rank1", "wall"])
def test_each_resource_limit_is_enforced(tmp_path, kind):
    monitor = monitor_fixture(tmp_path, updates=1)
    monitor.progress.write_text(event(1, resources.LIMITS["max_observed_global_gaussians"]))
    for rank, cap in enumerate(resources.LIMITS["max_per_rank_peak_reserved_bytes"]):
        write_rank(monitor.telemetry, rank, peak=cap)
    wall = resources.LIMITS["max_main_stage_wall_seconds"]
    assert monitor(wall, final=True) is None
    if kind == "gaussians":
        with monitor.progress.open("a") as stream:
            stream.write(event(2, resources.LIMITS["max_observed_global_gaussians"] + 1))
        expected = "global_gaussians_exceeded"
    elif kind.startswith("rank"):
        rank = int(kind[-1])
        write_rank(monitor.telemetry, rank, peak=resources.LIMITS["max_per_rank_peak_reserved_bytes"][rank] + 1)
        expected = f"rank_{rank}_reserved_exceeded"
    else:
        wall += 1
        expected = "main_wall_exceeded"
    assert expected in monitor(wall)


def test_missing_stale_malformed_and_unfinished_telemetry_fail_closed(tmp_path):
    monitor = monitor_fixture(tmp_path, updates=1)
    monitor.progress.write_text(event(1))
    path = monitor.telemetry / "rank-1.json"
    path.unlink()
    assert monitor(10) is None
    assert monitor(601) == "rank_1_telemetry_missing"
    assert monitor(10, final=True) == "rank_1_telemetry_missing"
    write_rank(monitor.telemetry, 1, stamp=time.monotonic() - 121)
    assert monitor(10) == "rank_1_telemetry_stale"
    write_rank(monitor.telemetry, 1, finished=False)
    assert monitor(10, final=True) == "rank_1_telemetry_unfinished"
    path.write_text("{")
    with pytest.raises(ValueError):
        monitor(10)


def test_progress_missing_steps_are_not_success(tmp_path):
    monitor = monitor_fixture(tmp_path)
    monitor.progress.write_text(event(2))
    with pytest.raises(ValueError, match="missing or reordered"):
        monitor(1)


def test_memory_telemetry_opt_in_and_final_snapshot(tmp_path):
    samples = iter([12, 34])
    with resources.memory_telemetry(tmp_path / "memory", 1, 1, sample=lambda: next(samples)):
        record = json.loads((tmp_path / "memory/rank-1.json").read_text())
        assert record["peak_reserved_bytes"] == 12
        assert record["finished"] is False
    record = json.loads((tmp_path / "memory/rank-1.json").read_text())
    assert record["peak_reserved_bytes"] == 34
    assert record["finished"] is True
    with pytest.raises(FileExistsError):
        with resources.memory_telemetry(tmp_path / "memory", 1, 1, sample=lambda: 1):
            pass
    with resources.memory_telemetry(None, 0, 0, sample=lambda: pytest.fail("unexpected sample")):
        pass


def test_stop_escalates_only_own_process_group(tmp_path, monkeypatch):
    calls = []
    class Process:
        pid = 12345
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake", timeout)
    monkeypatch.setattr(resources.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    resources.stop_process(Process(), tmp_path / "cancel")
    assert (tmp_path / "cancel").exists()
    assert calls == [("wait", 120), (12345, signal.SIGTERM), ("wait", 30),
                     (12345, signal.SIGKILL), ("wait", None)]


@pytest.mark.parametrize("failure", ["monitor", "exit", "disk", "wall", "ok"])
def test_stage_fake_child_records_all_terminal_states(tmp_path, monkeypatch, failure):
    stopped = []
    class Process:
        pid = 12345
        returncode = None
        def wait(self, timeout=None):
            if failure in {"disk", "wall", "monitor"}:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = 7 if failure == "exit" else 0
        def poll(self):
            return self.returncode
    process = Process()
    monkeypatch.setattr(resources.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=(0 if failure == "disk" else 30 * 1024**3)))
    times = iter([0, 21601, 21601])
    if failure == "wall":
        monkeypatch.setattr(resources.time, "monotonic", lambda: next(times))
    def stop(child, cancel):
        stopped.append(child)
        child.returncode = -15
    monkeypatch.setattr(resources, "stop_process", stop)
    monitor = None
    if failure == "monitor":
        class Broken:
            def __call__(self, elapsed, *, final):
                raise ValueError("bad telemetry")
            def record(self):
                return {}
        monitor = Broken()
    if failure == "ok":
        resources.run_stage(tmp_path, "train", ["fake"], cwd=tmp_path)
    else:
        with pytest.raises(RuntimeError):
            resources.run_stage(tmp_path, "train", ["fake"], cwd=tmp_path, monitor=monitor)
    record = json.loads((tmp_path / "train.exit.json").read_text())
    assert record["returncode"] == process.returncode
    assert bool(stopped) == (failure in {"monitor", "disk", "wall"})
    assert (record["resource_failure"] is not None) == (failure in {"monitor", "disk", "wall"})


def test_real_cpu_child_exit_without_model_loading(tmp_path):
    resources.run_stage(tmp_path, "success", [sys.executable, "-I", "-c", "print('ok')"], cwd=tmp_path)
    assert (tmp_path / "success.log").read_text().strip() == "ok"
    with pytest.raises(RuntimeError):
        resources.run_stage(tmp_path, "failed", [sys.executable, "-I", "-c", "raise SystemExit(7)"], cwd=tmp_path)
    assert json.loads((tmp_path / "failed.exit.json").read_text())["returncode"] == 7


def test_pipeline_order_and_no_test_no_resume(tmp_path, monkeypatch):
    calls = []
    admissions = []
    def fake_stage(root, name, command, **kwargs):
        calls.append((name, command, kwargs["monitor"]))
        if kwargs["monitor"] is not None:
            digest = candidate.MAIN_CAMERA_SHA256 if name == "train" else candidate.TRAIN_ONLY_CAMERA_SHA256
            kwargs["monitor"].camera_sequence = types.SimpleNamespace(hexdigest=lambda: digest)
        if name == "train":
            path = root / "training/attempts/train-001/artifacts"
            path.mkdir(parents=True)
            resources.write_json(path / "result.json", {
                "iteration": 30000, "world_size": 2, "per_rank_peak_reserved_bytes": [1, 2],
                "model_path": "attempts/train-001/artifacts/model.pt",
            })
        elif name in {"selection", "train-only"}:
            final = name == "train-only"
            path = root / name
            path.mkdir()
            resources.write_json(path / "evaluation.json", {
                "status": "complete", "num_views": 377, "successful_views": 377, "failed_views": [],
                "quality_role": "held_out_after_train_only_control" if final else "held_out_model_selection",
                "selection_eligible": not final,
            })
            if final:
                resources.write_json(path / "record.json", {
                    "optimizer_updates": 2000, "camera_samples": 4000, "world_size": 2,
                    "input_splits": ["train"], "topology_changed": False, "source_model_unchanged": True,
                })
                (path / "model.pt").write_bytes(b"test placeholder, not a model")
    monkeypatch.setattr(candidate, "run_stage", fake_stage)
    candidate.run_pipeline(tmp_path, tmp_path / "replay", lease_fd=123,
                           require_resources=lambda *a, **kw: admissions.append(kw))
    assert [name for name, _, _ in calls] == ["train", "sor", "selection", "train-only"]
    assert admissions == [{"minimum_free_gib": 8}] * 4
    for _, command, _ in calls:
        assert "test" not in command and "--resume-iteration" not in command
    assert "--initialization" in calls[0][1] and "frozen" in calls[0][1]
    assert "--train-only-control" in calls[-1][1]
    assert calls[0][2].updates == 30000 and calls[-1][2].updates == 2000
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["status"] == "absolute_training_complete_quality_pending"
    assert complete["promotion_eligible"] is False


def test_distributed_entrypoints_opt_in_without_importing_trainer(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for script, function, trainer, arguments in (
        ("run_gaussian_training.py", "_distributed_native_train", "train_gaussians", "trainer_args"),
        ("run_gaussian_final_fit.py", "_distributed_final_fit", "final_fit_gaussians", "payload"),
    ):
        tree = ast.parse((scripts / script).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        calls = []
        scope = {trainer: lambda **kw: calls.append(kw), "memory_telemetry": resources.memory_telemetry}
        exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), scope)
        scope[function](0, 0, 2, {arguments: {}, "cancel_file": None})
        assert calls[0]["world_size"] == 2
        assert calls[0]["cancel_requested"] is None


def test_preflight_binds_evidence_and_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "PROJECT_ROOT", tmp_path)
    template = candidate.gate_template()
    for key, constant in (("quality_proposal", "QUALITY_SHA256"), ("train_roi", "ROI_SHA256")):
        path = tmp_path / template[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        resources.write_json(path, {"fixture": key})
        monkeypatch.setattr(candidate, constant, candidate.sha256_file(path))
    source = tmp_path / "source"
    source.mkdir()
    resources.write_json(source / "source.json", {"fixture": "source"})
    resources.write_json(source / "protocol.json", {"files": {
        str(source / "source.json"): candidate.sha256_file(source / "source.json"),
    }})
    signed = tmp_path / "outputs/experiments/signed"
    (signed / "signed/selection").mkdir(parents=True)
    (signed / "signed/train-only").mkdir()
    final_record = {}
    for relative, key in (("selection/evaluation.json", "selection_evaluation_sha256"),
                          ("train-only/evaluation.json", "evaluation_sha256"),
                          ("train-only/model.pt", "final_model_sha256")):
        path = signed / "signed" / relative
        path.write_bytes(b"test placeholder")
        final_record[key] = candidate.sha256_file(path)
    records = [
        ("protocol.json", "SIGNED_PROTOCOL_SHA256", {
            "source": str(source), "source_protocol_sha256": candidate.sha256_file(source / "protocol.json"),
        }),
        ("signed.config.json", "SIGNED_CONFIG_SHA256", signed_config()),
        ("signed/train-only/record.json", "SIGNED_FINAL_RECORD_SHA256", final_record),
    ]
    for relative, constant, record in records:
        resources.write_json(signed / relative, record)
        monkeypatch.setattr(candidate, constant, candidate.sha256_file(signed / relative))
    resources.write_json(signed / "complete.json", {"status": "signed_control_complete"})
    gate_path = tmp_path / "gate.json"
    resources.write_json(gate_path, approved_gate())
    output = signed.parent / "candidate"
    prepared = candidate.preflight(signed, gate_path, candidate.sha256_file(gate_path), output)
    assert prepared["absolute_config"]["effective_config_hash"] == candidate.ABSOLUTE_CONFIG_HASH
    assert not output.exists()
    (source / "source.json").write_text("changed")
    with pytest.raises(ValueError, match="source identity changed"):
        candidate.preflight(signed, gate_path, candidate.sha256_file(gate_path), output)
    assert not output.exists()


def test_telemetry_thread_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    import threading

    failed = threading.Event()
    samples = 0
    def sample():
        nonlocal samples
        samples += 1
        if samples > 1:
            failed.set()
            raise OSError("injected sampler failure")
        return 10
    monkeypatch.setattr(resources, "POLL_SECONDS", 0.01)
    with pytest.raises(RuntimeError, match="telemetry failed"):
        with resources.memory_telemetry(tmp_path / "memory", 0, 0, sample=sample):
            assert failed.wait(timeout=1)
    record = json.loads((tmp_path / "memory/rank-0.json").read_text())
    assert record["finished"] is False


def test_stage_launch_failure_retains_exit_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected launch failure")
    monkeypatch.setattr(resources.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="launch failure"):
        resources.run_stage(tmp_path, "train", ["fake"], cwd=tmp_path)
    record = json.loads((tmp_path / "train.exit.json").read_text())
    assert record["returncode"] is None
    assert "runner_failed" in record["resource_failure"]


def test_short_process_still_gets_final_monitor_check(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=30 * 1024**3))
    checked = []
    class Monitor:
        def __call__(self, elapsed, *, final):
            checked.append(final)
            return "rank_1_telemetry_missing"
        def record(self):
            return {}
    with pytest.raises(RuntimeError, match="telemetry_missing"):
        resources.run_stage(tmp_path, "short", [sys.executable, "-I", "-c", "pass"],
                            cwd=tmp_path, monitor=Monitor())
    assert checked == [True]


def test_execution_is_rejected_on_local_host(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate.socket, "gethostname", lambda: "local-test-host")
    monkeypatch.setattr(sys, "argv", ["candidate", "execute", "--signed-experiment", str(tmp_path),
        "--gate-contract", str(tmp_path / "missing"), "--gate-sha256", "unused",
        "--output-dir", str(tmp_path / "new")])
    with pytest.raises(ValueError, match="remote workspace"):
        candidate.main()
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("final", [False, True])
def test_incomplete_or_wrong_role_evaluation_cannot_complete(tmp_path, final):
    record = {
        "status": "complete", "num_views": 377, "successful_views": 377, "failed_views": [],
        "quality_role": "held_out_after_train_only_control" if final else "held_out_model_selection",
        "selection_eligible": not final,
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(record))
    candidate.validate_evaluation(path, final=final)
    for key, value in (("successful_views", 376), ("failed_views", ["failed"]),
                       ("selection_eligible", final), ("quality_role", "wrong_role")):
        invalid = {**record, key: value}
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="Validation"):
            candidate.validate_evaluation(path, final=final)


def test_enabled_telemetry_reaches_both_distributed_entrypoints(tmp_path):
    from contextlib import contextmanager

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for script, function, trainer, arguments in (
        ("run_gaussian_training.py", "_distributed_native_train", "train_gaussians", "trainer_args"),
        ("run_gaussian_final_fit.py", "_distributed_final_fit", "final_fit_gaussians", "payload"),
    ):
        tree = ast.parse((scripts / script).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        calls = []
        @contextmanager
        def telemetry(directory, local_rank, world_rank):
            calls.append(("enter", directory, local_rank, world_rank))
            yield
            calls.append(("exit",))
        scope = {trainer: lambda **kw: calls.append(("train", kw)), "memory_telemetry": telemetry}
        exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), scope)
        scope[function](1, 1, 2, {arguments: {}, "cancel_file": tmp_path / "cancel",
                               "resource_telemetry_dir": tmp_path / "memory"})
        assert calls[0] == ("enter", tmp_path / "memory", 1, 1)
        assert calls[1][0] == "train" and calls[1][1]["world_rank"] == 1
        assert calls[1][1]["cancel_requested"]() is False
        assert calls[2] == ("exit",)


def test_camera_samples_are_counted_and_sequence_is_hashed(tmp_path):
    import hashlib

    monitor = monitor_fixture(tmp_path)
    monitor.progress.write_text(event(1) + event(2))
    assert monitor(1, final=True) is None
    record = monitor.record()
    assert record["camera_samples_observed"] == 4
    assert record["camera_sequence_sha256"] == hashlib.sha256(b"1\x002\n1\x002\n").hexdigest()
    with monitor.progress.open("a") as stream:
        stream.write(json.dumps({"iteration": 3, "world_size": 2, "gaussian_count": 100,
                                 "batch_view_ids": ["1"]}) + "\n")
    with pytest.raises(ValueError, match="two camera samples"):
        monitor(2)


def test_quality_profile_is_separately_approved_and_old_gate_unchanged():
    from image3d_scenegraph.file_integrity import sha256_file

    path = Path(__file__).resolve().parents[1] / "outputs/analysis/absgrad-candidate-launch-20261008-v1/gate.approved.json"
    if path.exists():
        assert sha256_file(path) == candidate.ORIGINAL_GATE_SHA256
        candidate.validate_gate(json.loads(path.read_text()))
    gate = candidate.gate_template(quality_exploration=True)
    assert gate["resource_limits"] == resources.QUALITY_LIMITS
    with pytest.raises(ValueError):
        candidate.validate_gate(gate, quality_exploration=True)
    gate.update(status="APPROVED_FOR_CANDIDATE_EXECUTION", absolute_training_authorized=True)
    candidate.validate_gate(gate, quality_exploration=True)
    with pytest.raises(ValueError):
        candidate.validate_gate(gate)
    with pytest.raises(ValueError):
        candidate.validate_gate(approved_gate(), quality_exploration=True)
    changed = copy.deepcopy(gate)
    changed["host_policy"]["memory_max_bytes"] += 1
    with pytest.raises(ValueError):
        candidate.validate_gate(changed, quality_exploration=True)
    changed = copy.deepcopy(gate)
    changed["absolute_training_authorized"] = 1
    with pytest.raises(ValueError):
        candidate.validate_gate(changed, quality_exploration=True)


def test_quality_monitor_does_not_reuse_old_stop_threshold(tmp_path):
    monitor = monitor_fixture(tmp_path, updates=1)
    monitor.limits = resources.QUALITY_LIMITS
    monitor.limit_label = "hardware_budget"
    monitor.progress.write_text(event(1, resources.LIMITS["max_observed_global_gaussians"] + 1))
    write_rank(monitor.telemetry, 0, peak=resources.LIMITS["max_per_rank_peak_reserved_bytes"][0] + 1)
    assert monitor(18000, final=True) is None
    write_rank(monitor.telemetry, 0, peak=resources.QUALITY_LIMITS["max_per_rank_peak_reserved_bytes"][0] + 1)
    assert monitor(18000, final=True) == "rank_0_reserved_exceeded_hardware_budget"


def test_host_boundaries_and_cgroup_oom_fail_closed():
    policy = resources.QUALITY_HOST_POLICY
    snapshot = {"available_bytes": policy["startup_available_bytes"], "task_current_bytes": 0,
                "memory_events": {"oom_kill": 0}}
    assert resources.host_failure(snapshot, admission=True) is None
    snapshot["available_bytes"] -= 1
    assert resources.host_failure(snapshot, admission=True) == "host_available_below_22_gib"
    snapshot["available_bytes"] = policy["minimum_available_bytes"]
    assert resources.host_failure(snapshot) is None
    snapshot["available_bytes"] -= 1
    assert resources.host_failure(snapshot) == "host_available_below_6_gib"
    snapshot["available_bytes"] = policy["startup_available_bytes"]
    snapshot["task_current_bytes"] = policy["stop_current_bytes"]
    assert resources.host_failure(snapshot) == "task_memory_at_14_gib"
    snapshot["task_current_bytes"] = 0
    snapshot["memory_events"]["oom_kill"] = 1
    assert resources.host_failure(snapshot) == "task_cgroup_oom_kill"


def test_quality_cgroup_rejects_panel_and_wrong_hard_limits(monkeypatch):
    values = {"/proc/self/cgroup": "0::/system.slice/gpu-panel.service\n"}
    monkeypatch.setattr(Path, "read_text", lambda p, **kw: values[str(p)])
    with pytest.raises(ValueError, match="isolated"):
        resources.quality_cgroup()
    expected = f"/sys/fs/cgroup/system.slice/{resources.QUALITY_UNIT}"
    values["/proc/self/cgroup"] = f"0::/system.slice/{resources.QUALITY_UNIT}\n"
    values.update({expected + "/memory.max": str(16 * 1024**3), expected + "/memory.swap.max": "0",
                   expected + "/memory.oom.group": "1"})
    assert str(resources.quality_cgroup()) == expected
    values[expected + "/memory.swap.max"] = "max"
    with pytest.raises(ValueError):
        resources.quality_cgroup()


def test_emergency_stop_never_requests_checkpoint_and_handles_exit_race(tmp_path, monkeypatch):
    calls = []
    class Child:
        pid = 1234
        def poll(self):
            return None
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout:
                raise subprocess.TimeoutExpired("fake", timeout)
    monkeypatch.setattr(resources.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    resources.stop_process(Child(), tmp_path / "cancel", emergency=True)
    assert not (tmp_path / "cancel").exists()
    assert calls == [(1234, signal.SIGTERM), ("wait", 5), (1234, signal.SIGKILL), ("wait", None)]
    monkeypatch.setattr(resources.os, "killpg", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    resources.stop_process(Child(), tmp_path / "cancel", emergency=True)


def test_quality_host_failure_is_recorded_before_emergency_stop(tmp_path, monkeypatch):
    samples = iter([
        {"available_bytes": 23 * 1024**3, "task_current_bytes": 0, "memory_events": {}},
        {"available_bytes": 5 * 1024**3, "task_current_bytes": 0, "memory_events": {}},
    ])
    monkeypatch.setattr(resources, "host_snapshot", lambda p: next(samples))
    class Child:
        pid = 1234
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("fake", timeout)
    child = Child()
    monkeypatch.setattr(resources.subprocess, "Popen", lambda *a, **kw: child)
    def stop(process, cancel, *, emergency):
        assert emergency and not cancel.exists()
        assert json.loads((tmp_path / "train.failure.json").read_text())["reason"] == "host_available_below_6_gib"
        child.returncode = -15
    monkeypatch.setattr(resources, "stop_process", stop)
    with pytest.raises(RuntimeError, match="host_available"):
        resources.run_stage(tmp_path, "train", ["fake"], cwd=tmp_path, host_group=tmp_path)
    assert json.loads((tmp_path / "train.exit.json").read_text())["returncode"] == -15


def test_quality_admission_failure_does_not_launch_child(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "host_snapshot", lambda _: {
        "available_bytes": 20 * 1024**3, "task_current_bytes": 0, "memory_events": {}})
    monkeypatch.setattr(resources.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="22_gib"):
        resources.run_stage(tmp_path, "train", ["fake"], cwd=tmp_path, host_group=tmp_path)
    assert json.loads((tmp_path / "train.exit.json").read_text())["returncode"] is None
