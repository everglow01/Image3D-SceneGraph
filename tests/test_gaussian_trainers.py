from __future__ import annotations

from image3d_scenegraph.gaussian.trainers import (
    GRAPHDECO_COMMIT,
    get_gaussian_trainer_specs,
    validate_trainer_id,
    validate_trainer_strategy,
)
from image3d_scenegraph.gaussian.config import resolve_mcmc_config, resolve_public_config


def test_trainer_registry_reports_pinned_external_environments(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))

    specs = {spec.trainer_id: spec for spec in get_gaussian_trainer_specs(tmp_path)}

    assert tuple(specs) == ("project", "graphdeco", "mcmc")
    assert specs["project"].label == "Project v7 (gsplat)"
    assert specs["project"].license == "Apache-2.0"
    assert specs["mcmc"].label == "MCMC v1 (gsplat)"
    assert specs["mcmc"].revision == specs["project"].revision
    assert specs["mcmc"].available == specs["project"].available
    assert specs["mcmc"].license == "Apache-2.0"
    assert specs["graphdeco"].revision == GRAPHDECO_COMMIT
    assert specs["graphdeco"].available is False
    assert "repo missing" in (specs["graphdeco"].reason or "")
    assert "--accept-research-license" in (specs["graphdeco"].setup_command or "")


def test_trainer_strategy_contract_rejects_cross_dispatch():
    validate_trainer_strategy("project", resolve_public_config("standard_v1").effective_config)
    validate_trainer_strategy("graphdeco", resolve_public_config("standard_v1").effective_config)
    validate_trainer_strategy("mcmc", resolve_mcmc_config().effective_config)

    try:
        validate_trainer_strategy("mcmc", resolve_public_config("standard_v1").effective_config)
    except ValueError as exc:
        assert "requires strategy 'mcmc_v1'" in str(exc)
    else:
        raise AssertionError("MCMC trainer accepted Default strategy")


def test_validate_trainer_id_rejects_unknown():
    assert validate_trainer_id("project") == "project"
    assert validate_trainer_id("mcmc") == "mcmc"
    try:
        validate_trainer_id("unknown")
    except ValueError as exc:
        assert "unsupported Gaussian trainer" in str(exc)
    else:
        raise AssertionError("unknown trainer was accepted")
