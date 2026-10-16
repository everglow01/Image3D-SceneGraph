from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mcmc_ablation", ROOT / "scripts/run_mcmc_stability_ablation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_memory_v2_budget_preserves_host_headroom_without_soft_throttling():
    assert runner.PROFILE == "apartment_mcmc_single_factor_memory_v2"
    policy = runner.HOST_POLICY
    assert policy["memory_high_bytes"] is None
    assert policy["memory_max_bytes"] == 12 * 1024**3
    assert policy["stop_current_bytes"] == 11.5 * 1024**3
    assert policy["startup_available_bytes"] >= policy["memory_max_bytes"] + policy["minimum_available_bytes"]
    assert policy["memory_swap_max_bytes"] == 0
    assert runner.LIMITS["validation_progress_timeout_seconds"] == 300


def test_apartment_configs_are_frozen_single_factor():
    records = runner.configs()
    assert set(records) == set(runner.ARMS)
    assert records["baseline"]["effective_config_hash"] == runner.SOURCE_CONFIG_HASH
    for record in records.values():
        config = record["effective_config"]
        assert config["iterations"] == 30000
        assert config["seed"] == 20260729
        assert config["strategy"]["gaussian_cap"] == 3000000
        assert config["resolution"]["longest_edge"] == 2048
        assert config["evaluation"]["validation_iterations"][-1] == 30000
    assert records["raw_loss"]["effective_config"]["loss"]["clamp_render"] is False
    assert records["baseline"]["effective_config"]["loss"]["clamp_render"] is True
    assert records["position_lr"]["effective_config"]["learning_rate"]["position"]["delay_multiplier"] == 1.


@pytest.mark.parametrize("trainer,profile,distributed,prepare,expected", [
    ("mcmc", "mcmc_v1", True, False, False),
    ("project", "absgrad_ablation_v1", True, False, False),
    ("project", "standard_v1", True, False, True),
    ("mcmc", "mcmc_v1", False, False, True),
    ("mcmc", "mcmc_v1", True, True, True),
    ("graphdeco", "mcmc_v1", True, False, True),
])
def test_telemetry_cli_guard(trainer, profile, distributed, prepare, expected):
    tree = ast.parse((ROOT / "scripts/run_gaussian_training.py").read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and "args.resource_telemetry_dir is not None" in ast.unparse(node.test))
    args = SimpleNamespace(resource_telemetry_dir=Path("memory"), trainer=trainer,
                           distributed=distributed, prepare_only=prepare, readiness_only=False)
    assert eval(compile(ast.Expression(guard.test), "guard", "eval"),
                {"args": args, "resolved": SimpleNamespace(requested_profile=profile)}) is expected


def test_unlimited_host_policy_removes_only_host_memory_gates():
    assert runner.UNLIMITED_PROFILE != runner.PROFILE
    for key in ("startup_available_bytes", "minimum_available_bytes", "stop_current_bytes",
                "memory_max_bytes", "memory_high_bytes", "memory_swap_max_bytes"):
        assert runner.UNLIMITED_HOST_POLICY[key] is None
    assert runner.HOST_POLICY["memory_max_bytes"] == 12 * 1024**3
    assert runner.LIMITS["max_per_rank_peak_reserved_bytes"] == [18 * 1024**3] * 2
    assert runner.LIMITS["validation_progress_timeout_seconds"] == 300


def test_unlimited_cgroup_checks_actual_and_inherited_limits(monkeypatch):
    unit = "image3d-mcmc-baseline-20261016-v3.service"
    group = Path("/sys/fs/cgroup/system.slice") / unit
    values = {"/proc/self/cgroup": f"0::/system.slice/{unit}\n"}
    for directory in (group, group.parent):
        for name in ("memory.max", "memory.high", "memory.swap.max"):
            values[str(directory / name)] = "max\n"
    monkeypatch.setattr(Path, "read_text", lambda p, **kw: values[str(p)])
    monkeypatch.setattr(Path, "exists", lambda p: str(p) in values)
    assert runner.unlimited_host_cgroup(unit) == group
    for directory in (group, group.parent, group.parent.parent):
        for name in ("memory.max", "memory.high", "memory.swap.max"):
            path = str(directory / name)
            old = values.get(path)
            values[path] = "0" if name == "memory.swap.max" else str(12 * 1024**3)
            with pytest.raises(ValueError, match="still limited"):
                runner.unlimited_host_cgroup(unit)
            if old is None:
                del values[path]
            else:
                values[path] = old
    values["/proc/self/cgroup"] = "0::/system.slice/gpu-panel.service\n"
    with pytest.raises(ValueError, match="own systemd cgroup"):
        runner.unlimited_host_cgroup(unit)


@pytest.mark.parametrize("unlimited", [False, True])
def test_host_memory_guard_wiring_is_bound_to_protocol(tmp_path, monkeypatch, unlimited):
    policy = runner.UNLIMITED_HOST_POLICY if unlimited else runner.HOST_POLICY
    protocol = {"profile": runner.UNLIMITED_PROFILE if unlimited else runner.PROFILE,
                "host_policy": policy, "limits": runner.LIMITS,
                "training_code_hash": "core", "files": {}, "configs": {},
                "overlays": {}, "replay": str(tmp_path / "replay")}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "code_hash", lambda: "core")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=30 * 1024**3))
    group = tmp_path / "group"
    group.mkdir()
    (group / "memory.high").write_text("max")
    checked = []
    def capped_group(**kwargs):
        checked.append("capped")
        assert kwargs["policy"] == runner.HOST_POLICY
        return group
    def unlimited_group(unit):
        checked.append("unlimited")
        assert unit.endswith("20261016-v3.service")
        return group
    monkeypatch.setattr(runner, "quality_cgroup", capped_group)
    monkeypatch.setattr(runner, "unlimited_host_cgroup", unlimited_group)
    class Lease:
        def __init__(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def fileno(self):
            return 123
    monkeypatch.setattr(runner, "FileLease", Lease)
    monkeypatch.setattr(runner, "require_idle_gpu", lambda: None)
    def intercept(*args, **kwargs):
        assert kwargs["host_group"] == (None if unlimited else group)
        assert kwargs["host_policy"] == policy
        assert kwargs["monitor"].limits == runner.LIMITS
        raise RuntimeError("stage intercepted")
    monkeypatch.setattr(runner, "run_stage", intercept)
    with pytest.raises(RuntimeError, match="stage intercepted"):
        runner.run(SimpleNamespace(output_root=tmp_path, arm="baseline"))
    assert checked == ["unlimited" if unlimited else "capped"]
