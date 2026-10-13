from __future__ import annotations

import ast
import importlib.util
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
