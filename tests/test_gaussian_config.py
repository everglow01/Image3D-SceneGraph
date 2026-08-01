from __future__ import annotations

import pytest

from image3d_scenegraph.gaussian.config import (
    CONFIG_SCHEMA_VERSION,
    GaussianConfigError,
    ResolvedGaussianConfig,
    assert_single_field_ablation,
    effective_config_hash,
    resolve_internal_config,
    resolve_public_config,
    resolved_config_record,
    validate_effective_config,
)


def test_public_profile_resolves_deterministically_and_returns_fresh_data():
    first = resolve_public_config("standard_v1")
    second = resolve_public_config("standard_v1")

    assert first.requested_profile == "standard_v1"
    assert first.effective_config["schema_version"] == CONFIG_SCHEMA_VERSION
    assert first.effective_config["resolution"] == {
        "policy": "explicit_only",
        "longest_edge": 960,
    }
    assert first.effective_config["iterations"] == 8_000
    assert first.effective_config["gaussian_budget"]["max_count"] == 350_000
    assert first.effective_config["densification"] == {
        "enabled": True,
        "start_iteration": 500,
        "end_iteration": 4_000,
        "every_iterations": 100,
        "gradient_threshold": 0.0002,
        "duplicate_scale_threshold": 0.01,
        "split_children": 2,
    }
    assert first.effective_config == second.effective_config
    assert first.effective_config is not second.effective_config
    assert first.effective_config_hash == second.effective_config_hash
    assert first.effective_config_hash == effective_config_hash(first.effective_config)

    first.effective_config["iterations"] = 1
    assert resolve_public_config("standard_v1").effective_config["iterations"] == 8_000

    internal = resolve_internal_config("rtx4060_8gb_development_v1")
    assert internal.effective_config == second.effective_config
    assert internal.effective_config_hash == second.effective_config_hash


@pytest.mark.parametrize("profile", ["smoke_v1", "high_quality", ""])
def test_public_profile_rejects_unapproved_profiles(profile):
    with pytest.raises(GaussianConfigError, match="unsupported public"):
        resolve_public_config(profile)


def test_internal_override_is_validated_hashed_and_recorded():
    resolved = resolve_internal_config(
        overrides={"densification": {"every_iterations": 200}},
    )
    record = resolved_config_record(resolved)

    assert resolved.effective_config["densification"]["every_iterations"] == 200
    assert record == {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "requested_profile": "standard_v1",
        "effective_config": resolved.effective_config,
        "effective_config_hash": resolved.effective_config_hash,
    }
    assert record["effective_config"] is not resolved.effective_config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": 1}, "unknown Gaussian config field: unknown"),
        ({"loss": {"unknown": 0.1}}, "unknown Gaussian config field: loss.unknown"),
        ({"iterations": True}, "must be an integer: iterations"),
        ({"iterations": "30000"}, "must be an integer: iterations"),
        ({"loss": {"l1_weight": 1}}, "must be a finite float: loss.l1_weight"),
        ({"resolution": {"longest_edge": 1281}}, "exceeds 1280"),
        ({"gaussian_budget": {"max_count": 1_000_001}}, "exceeds 1000000"),
        (
            {"learning_rate": {"position": {"final": 0.001}}},
            "learning_rate.position.final cannot exceed initial",
        ),
        (
            {"sh_schedule": {"increase_every_iterations": 3_000}},
            "SH schedule cannot reach max_degree within iterations",
        ),
        (
            {"densification": {"end_iteration": 500}},
            "densification start must precede end",
        ),
        (
            {"opacity_reset": {"every_iterations": 8_001}},
            "exceeds 8000",
        ),
        (
            {"evaluation": {"validation_every_iterations": 0}},
            "below 1",
        ),
    ],
)
def test_internal_override_rejects_invalid_config(overrides, message):
    with pytest.raises(GaussianConfigError, match=message):
        resolve_internal_config(overrides=overrides)


def test_validation_rejects_missing_and_nonfinite_values():
    config = resolve_public_config("standard_v1").effective_config
    del config["evaluation"]
    with pytest.raises(GaussianConfigError, match="missing evaluation"):
        validate_effective_config(config)

    config = resolve_public_config("standard_v1").effective_config
    config["pruning"]["max_screen_fraction"] = float("nan")
    with pytest.raises(GaussianConfigError, match="finite float"):
        validate_effective_config(config)


def test_resolved_record_rejects_tampered_hash():
    resolved = resolve_public_config("standard_v1")
    tampered = ResolvedGaussianConfig(
        requested_profile=resolved.requested_profile,
        effective_config=resolved.effective_config,
        effective_config_hash="0" * 64,
    )

    with pytest.raises(GaussianConfigError, match="hash mismatch"):
        resolved_config_record(tampered)


def test_single_field_ablation_reports_one_changed_leaf():
    baseline = resolve_public_config("standard_v1").effective_config
    candidate = resolve_internal_config(
        overrides={"densification": {"every_iterations": 200}}
    ).effective_config

    assert assert_single_field_ablation(baseline, candidate) == "densification.every_iterations"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "densification": {"every_iterations": 200},
            "evaluation": {"validation_every_iterations": 2_000},
        },
    ],
)
def test_ablation_rejects_zero_or_multiple_changes(overrides):
    baseline = resolve_public_config("standard_v1").effective_config
    candidate = resolve_internal_config(overrides=overrides).effective_config

    with pytest.raises(GaussianConfigError, match="exactly one field"):
        assert_single_field_ablation(baseline, candidate)
