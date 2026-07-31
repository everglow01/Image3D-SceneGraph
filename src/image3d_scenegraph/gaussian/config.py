"""Versioned hyperparameter contract for project-owned 3DGS training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any


CONFIG_SCHEMA_VERSION = 3
PUBLIC_PROFILES = ("standard_v1",)
INTERNAL_PROFILES = ("standard_v1", "rtx4060_8gb_development_v1")

_STANDARD_V1: dict[str, Any] = {
    "schema_version": CONFIG_SCHEMA_VERSION,
    "seed": 20260729,
    "iterations": 3_000,
    "resolution": {
        "policy": "explicit_only",
        "longest_edge": 640,
    },
    "loss": {
        "name": "l1_ssim",
        "l1_weight": 0.8,
        "ssim_weight": 0.2,
    },
    "learning_rate": {
        "schedule": "exponential",
        "delay_multiplier": 0.01,
        "position": {"initial": 0.00016, "final": 0.0000016},
        "feature": {"initial": 0.0025, "final": 0.000025},
        "opacity": {"initial": 0.05, "final": 0.005},
        "scaling": {"initial": 0.005, "final": 0.00005},
        "rotation": {"initial": 0.001, "final": 0.00001},
    },
    "sh_schedule": {
        "initial_degree": 0,
        "max_degree": 3,
        "increase_every_iterations": 1000,
    },
    "densification": {
        "enabled": True,
        "start_iteration": 200,
        "end_iteration": 1_500,
        "every_iterations": 100,
        "gradient_threshold": 0.0002,
        "duplicate_scale_threshold": 0.01,
        "split_children": 2,
    },
    "pruning": {
        "enabled": True,
        "opacity_threshold": 0.005,
        "screen_size_enabled": False,
        "max_screen_fraction": 0.1,
    },
    "opacity_reset": {
        "enabled": True,
        "every_iterations": 500,
        "value": 0.01,
    },
    "gaussian_budget": {"max_count": 250_000},
    "evaluation": {"validation_every_iterations": 500},
}


class GaussianConfigError(ValueError):
    """Raised when a 3DGS configuration violates the versioned contract."""


@dataclass(frozen=True)
class ResolvedGaussianConfig:
    requested_profile: str
    effective_config: dict[str, Any]
    effective_config_hash: str


def canonical_config_json(config: dict[str, Any]) -> str:
    validate_effective_config(config)
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def effective_config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config).encode()).hexdigest()


def resolve_public_config(profile: str) -> ResolvedGaussianConfig:
    if profile not in PUBLIC_PROFILES:
        raise GaussianConfigError(f"unsupported public Gaussian quality profile: {profile}")
    return _resolve(profile, None)


def resolve_internal_config(
    profile: str = "standard_v1",
    overrides: dict[str, Any] | None = None,
) -> ResolvedGaussianConfig:
    if profile not in INTERNAL_PROFILES:
        raise GaussianConfigError(f"unsupported internal Gaussian base profile: {profile}")
    if overrides is not None and not isinstance(overrides, dict):
        raise GaussianConfigError("Gaussian config overrides must be an object")
    resolved = _resolve(profile, None)
    if overrides:
        effective = copy.deepcopy(resolved.effective_config)
        _apply_overrides(effective, overrides, "")
        validate_effective_config(effective)
        resolved = ResolvedGaussianConfig(profile, effective, effective_config_hash(effective))
    return resolved


def resolved_config_record(resolved: ResolvedGaussianConfig) -> dict[str, Any]:
    if not isinstance(resolved, ResolvedGaussianConfig):
        raise GaussianConfigError("expected a resolved Gaussian configuration")
    if resolved.requested_profile not in INTERNAL_PROFILES:
        raise GaussianConfigError(f"unsupported resolved Gaussian profile: {resolved.requested_profile}")
    validate_effective_config(resolved.effective_config)
    expected_hash = effective_config_hash(resolved.effective_config)
    if resolved.effective_config_hash != expected_hash:
        raise GaussianConfigError("resolved Gaussian config hash mismatch")
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "requested_profile": resolved.requested_profile,
        "effective_config": copy.deepcopy(resolved.effective_config),
        "effective_config_hash": expected_hash,
    }


def validate_effective_config(config: dict[str, Any]) -> None:
    root = _mapping(
        config,
        "config",
        {
            "schema_version",
            "seed",
            "iterations",
            "resolution",
            "loss",
            "learning_rate",
            "sh_schedule",
            "densification",
            "pruning",
            "opacity_reset",
            "gaussian_budget",
            "evaluation",
        },
    )
    if _integer(root["schema_version"], "schema_version") != CONFIG_SCHEMA_VERSION:
        raise GaussianConfigError(f"unsupported Gaussian config schema version: {root['schema_version']}")
    _integer(root["seed"], "seed", minimum=0, maximum=2**63 - 1)
    iterations = _integer(root["iterations"], "iterations", minimum=1, maximum=1_000_000)

    resolution = _mapping(root["resolution"], "resolution", {"policy", "longest_edge"})
    _choice(resolution["policy"], "resolution.policy", {"explicit_only"})
    _integer(resolution["longest_edge"], "resolution.longest_edge", minimum=64, maximum=1280)

    loss = _mapping(root["loss"], "loss", {"name", "l1_weight", "ssim_weight"})
    _choice(loss["name"], "loss.name", {"l1_ssim"})
    l1_weight = _number(loss["l1_weight"], "loss.l1_weight", minimum=0.0, maximum=1.0)
    ssim_weight = _number(loss["ssim_weight"], "loss.ssim_weight", minimum=0.0, maximum=1.0)
    if not math.isclose(l1_weight + ssim_weight, 1.0, abs_tol=1e-12):
        raise GaussianConfigError("loss weights must sum to 1.0")

    learning_rate = _mapping(
        root["learning_rate"],
        "learning_rate",
        {"schedule", "delay_multiplier", "position", "feature", "opacity", "scaling", "rotation"},
    )
    _choice(learning_rate["schedule"], "learning_rate.schedule", {"exponential"})
    _number(
        learning_rate["delay_multiplier"],
        "learning_rate.delay_multiplier",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
    )
    for group_name in ("position", "feature", "opacity", "scaling", "rotation"):
        group = _mapping(
            learning_rate[group_name],
            f"learning_rate.{group_name}",
            {"initial", "final"},
        )
        initial = _number(
            group["initial"],
            f"learning_rate.{group_name}.initial",
            minimum=0.0,
            minimum_exclusive=True,
        )
        final = _number(
            group["final"],
            f"learning_rate.{group_name}.final",
            minimum=0.0,
            minimum_exclusive=True,
        )
        if final > initial:
            raise GaussianConfigError(f"learning_rate.{group_name}.final cannot exceed initial")

    sh_schedule = _mapping(
        root["sh_schedule"],
        "sh_schedule",
        {"initial_degree", "max_degree", "increase_every_iterations"},
    )
    initial_degree = _integer(sh_schedule["initial_degree"], "sh_schedule.initial_degree", minimum=0, maximum=3)
    max_degree = _integer(sh_schedule["max_degree"], "sh_schedule.max_degree", minimum=0, maximum=3)
    sh_interval = _integer(
        sh_schedule["increase_every_iterations"],
        "sh_schedule.increase_every_iterations",
        minimum=1,
        maximum=iterations,
    )
    if initial_degree > max_degree:
        raise GaussianConfigError("sh_schedule.initial_degree cannot exceed max_degree")
    if (max_degree - initial_degree) * sh_interval > iterations:
        raise GaussianConfigError("SH schedule cannot reach max_degree within iterations")

    densification = _mapping(
        root["densification"],
        "densification",
        {
            "enabled",
            "start_iteration",
            "end_iteration",
            "every_iterations",
            "gradient_threshold",
            "duplicate_scale_threshold",
            "split_children",
        },
    )
    _boolean(densification["enabled"], "densification.enabled")
    densify_start = _integer(
        densification["start_iteration"], "densification.start_iteration", minimum=1, maximum=iterations
    )
    densify_end = _integer(
        densification["end_iteration"], "densification.end_iteration", minimum=1, maximum=iterations
    )
    densify_interval = _integer(
        densification["every_iterations"], "densification.every_iterations", minimum=1, maximum=iterations
    )
    _number(
        densification["gradient_threshold"],
        "densification.gradient_threshold",
        minimum=0.0,
        minimum_exclusive=True,
    )
    _number(
        densification["duplicate_scale_threshold"],
        "densification.duplicate_scale_threshold",
        minimum=0.0,
        minimum_exclusive=True,
    )
    _integer(
        densification["split_children"],
        "densification.split_children",
        minimum=2,
        maximum=4,
    )
    if densify_start >= densify_end:
        raise GaussianConfigError("densification start must precede end")
    if densify_interval > densify_end - densify_start:
        raise GaussianConfigError("densification cadence exceeds its active range")

    pruning = _mapping(
        root["pruning"],
        "pruning",
        {"enabled", "opacity_threshold", "screen_size_enabled", "max_screen_fraction"},
    )
    _boolean(pruning["enabled"], "pruning.enabled")
    _number(pruning["opacity_threshold"], "pruning.opacity_threshold", minimum=0.0, maximum=1.0)
    _boolean(pruning["screen_size_enabled"], "pruning.screen_size_enabled")
    _number(
        pruning["max_screen_fraction"],
        "pruning.max_screen_fraction",
        minimum=0.0,
        maximum=1.0,
        minimum_exclusive=True,
    )

    opacity_reset = _mapping(
        root["opacity_reset"], "opacity_reset", {"enabled", "every_iterations", "value"}
    )
    _boolean(opacity_reset["enabled"], "opacity_reset.enabled")
    _integer(
        opacity_reset["every_iterations"],
        "opacity_reset.every_iterations",
        minimum=1,
        maximum=iterations,
    )
    _number(opacity_reset["value"], "opacity_reset.value", minimum=0.0, maximum=1.0)

    gaussian_budget = _mapping(root["gaussian_budget"], "gaussian_budget", {"max_count"})
    _integer(gaussian_budget["max_count"], "gaussian_budget.max_count", minimum=1, maximum=1_000_000)

    evaluation = _mapping(
        root["evaluation"], "evaluation", {"validation_every_iterations"}
    )
    _integer(
        evaluation["validation_every_iterations"],
        "evaluation.validation_every_iterations",
        minimum=1,
        maximum=iterations,
    )


def assert_single_field_ablation(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    validate_effective_config(baseline)
    validate_effective_config(candidate)
    baseline_leaves = dict(_leaf_items(baseline))
    candidate_leaves = dict(_leaf_items(candidate))
    if baseline_leaves.keys() != candidate_leaves.keys():
        raise GaussianConfigError("ablation configurations must have identical fields")
    changed = [path for path in baseline_leaves if baseline_leaves[path] != candidate_leaves[path]]
    if len(changed) != 1:
        raise GaussianConfigError(f"ablation must change exactly one field; found {len(changed)}")
    return changed[0]


def _resolve(profile: str, overrides: dict[str, Any] | None) -> ResolvedGaussianConfig:
    effective = copy.deepcopy(_STANDARD_V1)
    if overrides:
        _apply_overrides(effective, overrides, "")
    validate_effective_config(effective)
    return ResolvedGaussianConfig(
        requested_profile=profile,
        effective_config=effective,
        effective_config_hash=effective_config_hash(effective),
    )


def _apply_overrides(target: dict[str, Any], overrides: dict[str, Any], prefix: str) -> None:
    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise GaussianConfigError(f"unknown Gaussian config field: {path}")
        existing = target[key]
        if isinstance(existing, dict):
            if not isinstance(value, dict):
                raise GaussianConfigError(f"Gaussian config field must be an object: {path}")
            _apply_overrides(existing, value, path)
        elif isinstance(value, dict):
            raise GaussianConfigError(f"Gaussian config field cannot be an object: {path}")
        else:
            target[key] = copy.deepcopy(value)


def _mapping(value: Any, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GaussianConfigError(f"Gaussian config field must be an object: {path}")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise GaussianConfigError(f"invalid Gaussian config fields at {path}: {'; '.join(details)}")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise GaussianConfigError(f"Gaussian config field must be an integer: {path}")
    if minimum is not None and value < minimum:
        raise GaussianConfigError(f"Gaussian config field is below {minimum}: {path}")
    if maximum is not None and value > maximum:
        raise GaussianConfigError(f"Gaussian config field exceeds {maximum}: {path}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise GaussianConfigError(f"Gaussian config field must be a finite float: {path}")
    if minimum is not None and (value <= minimum if minimum_exclusive else value < minimum):
        comparison = "greater than" if minimum_exclusive else "at least"
        raise GaussianConfigError(f"Gaussian config field must be {comparison} {minimum}: {path}")
    if maximum is not None and value > maximum:
        raise GaussianConfigError(f"Gaussian config field exceeds {maximum}: {path}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise GaussianConfigError(f"Gaussian config field must be a boolean: {path}")
    return value


def _choice(value: Any, path: str, choices: set[str]) -> str:
    if type(value) is not str or value not in choices:
        raise GaussianConfigError(f"unsupported Gaussian config value for {path}: {value}")
    return value


def _leaf_items(value: dict[str, Any], prefix: str = ""):
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else key
        child = value[key]
        if isinstance(child, dict):
            yield from _leaf_items(child, path)
        else:
            yield path, child
