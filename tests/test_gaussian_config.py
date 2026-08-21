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


def test_public_profile_resolves_official_baseline_deterministically():
    first = resolve_public_config("standard_v1")
    second = resolve_public_config("standard_v1")

    assert first.requested_profile == "standard_v1"
    assert first.effective_config["schema_version"] == CONFIG_SCHEMA_VERSION == 9
    assert first.effective_config["iterations"] == 30_000
    assert first.effective_config["resolution"] == {
        "policy": "explicit_only",
        "longest_edge": 1280,
    }
    assert first.effective_config["learning_rate"] == {
        "position": {"initial": 0.00016, "final": 0.0000016},
        "feature": 0.0025,
        "opacity": 0.025,
        "scaling": 0.005,
        "rotation": 0.001,
    }
    assert first.effective_config["loss"]["clamp_render"] is True
    assert first.effective_config["sh_schedule"]["increase_every_iterations"] == 1_000
    assert first.effective_config["densification"] == {
        "enabled": True,
        "start_iteration": 500,
        "end_iteration": 15_000,
        "every_iterations": 100,
        "gradient_threshold": 0.0002,
        "scale_threshold": 0.01,
    }
    assert first.effective_config["pruning"] == {
        "enabled": True,
        "opacity_threshold": 0.005,
        "max_world_scale": 0.1,
        "screen_radius_enabled": False,
        "max_screen_radius_pixels": 20.0,
    }
    assert first.effective_config["opacity_reset"] == {
        "enabled": True,
        "every_iterations": 3_000,
        # 2.0 keeps the pre-schema-8 reset floor (0.005 * 2.0 = 0.01).
        "floor_multiplier": 2.0,
        # Disabled by default: schema 9 adds the leaf without changing behavior.
        "recovery_prune": {
            "enabled": False,
            "window_iterations": 500,
            "opacity_threshold": 0.05,
        },
    }
    assert first.effective_config["evaluation"] == {
        "validation_iterations": [
            3_000,
            5_000,
            7_000,
            10_000,
            15_000,
            20_000,
            25_000,
            30_000,
        ]
    }
    assert "gaussian_budget" not in first.effective_config
    assert first.effective_config == second.effective_config
    assert first.effective_config is not second.effective_config
    assert first.effective_config_hash == second.effective_config_hash

    first.effective_config["iterations"] = 1
    assert resolve_public_config("standard_v1").effective_config["iterations"] == 30_000
    assert resolve_internal_config("rtx4060_8gb_development_v1").effective_config == second.effective_config


def test_public_profile_hashes_selected_training_resolution():
    resolved = resolve_public_config("standard_v1", longest_edge=3072)

    assert resolved.effective_config["resolution"]["longest_edge"] == 3072
    assert resolved.effective_config_hash != resolve_public_config("standard_v1").effective_config_hash


@pytest.mark.parametrize("profile", ["smoke_v1", "high_quality", ""])
def test_public_profile_rejects_unapproved_profiles(profile):
    with pytest.raises(GaussianConfigError, match="unsupported public"):
        resolve_public_config(profile)


def test_internal_override_is_validated_hashed_and_recorded():
    resolved = resolve_internal_config(overrides={"densification": {"every_iterations": 200}})
    record = resolved_config_record(resolved)

    assert resolved.effective_config["densification"]["every_iterations"] == 200
    assert record == {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "requested_profile": "standard_v1",
        "effective_config": resolved.effective_config,
        "effective_config_hash": resolved.effective_config_hash,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": 1}, "unknown Gaussian config field: unknown"),
        ({"loss": {"unknown": 0.1}}, "unknown Gaussian config field: loss.unknown"),
        ({"iterations": True}, "must be an integer: iterations"),
        ({"iterations": "30000"}, "must be an integer: iterations"),
        ({"loss": {"l1_weight": 1}}, "must be a finite float: loss.l1_weight"),
        ({"resolution": {"longest_edge": 3073}}, "exceeds 3072"),
        (
            {"learning_rate": {"position": {"final": 0.001}}},
            "learning_rate.position.final cannot exceed initial",
        ),
        ({"densification": {"end_iteration": 500}}, "densification start must precede end"),
        ({"opacity_reset": {"every_iterations": 30_001}}, "exceeds 30000"),
        ({"opacity_reset": {"floor_multiplier": 0.0}}, "greater than 0.0"),
        ({"opacity_reset": {"floor_multiplier": 10.5}}, "exceeds 10.0"),
        (
            {"pruning": {"opacity_threshold": 0.5}, "opacity_reset": {"floor_multiplier": 2.0}},
            "must stay below 1.0",
        ),
        ({"pruning": {"screen_radius_enabled": 0}}, "must be a boolean"),
        ({"opacity_reset": {"recovery_prune": {"window_iterations": 0}}}, "below 1"),
        (
            {"opacity_reset": {"recovery_prune": {"opacity_threshold": 0.01}}},
            "must exceed",
        ),
        (
            {"opacity_reset": {"recovery_prune": {"opacity_threshold": 1.0}}},
            "must stay below 1.0",
        ),
        ({"evaluation": {"validation_iterations": [0, 30_000]}}, "below 1"),
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
    config["pruning"]["max_world_scale"] = float("nan")
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
    [{}, {"densification": {"every_iterations": 200}, "evaluation": {"validation_iterations": [14_000, 30_000]}}],
)
def test_ablation_rejects_zero_or_multiple_changes(overrides):
    baseline = resolve_public_config("standard_v1").effective_config
    candidate = resolve_internal_config(overrides=overrides).effective_config
    with pytest.raises(GaussianConfigError, match="exactly one field"):
        assert_single_field_ablation(baseline, candidate)
