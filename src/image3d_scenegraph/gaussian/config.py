"""Versioned hyperparameter contract for project-owned 3DGS training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any


CONFIG_SCHEMA_VERSION = 7
PUBLIC_PROFILES = ("standard_v1",)
INTERNAL_PROFILES = ("standard_v1", "rtx4060_8gb_development_v1")

_STANDARD_V1: dict[str, Any] = {
    "schema_version": CONFIG_SCHEMA_VERSION,
    "seed": 20260729,
    "iterations": 30_000,
    "resolution": {"policy": "explicit_only", "longest_edge": 1280},
    "loss": {
        "name": "l1_ssim",
        "l1_weight": 0.8,
        "ssim_weight": 0.2,
        "clamp_render": True,
    },
    "learning_rate": {
        "position": {"initial": 0.00016, "final": 0.0000016},
        "feature": 0.0025,
        "opacity": 0.025,
        "scaling": 0.005,
        "rotation": 0.001,
    },
    "sh_schedule": {
        "initial_degree": 0,
        "max_degree": 3,
        "increase_every_iterations": 1_000,
    },
    "densification": {
        "enabled": True,
        "start_iteration": 500,
        "end_iteration": 15_000,
        "every_iterations": 100,
        "gradient_threshold": 0.0002,
        "scale_threshold": 0.01,
    },
    "pruning": {
        "enabled": True,
        "opacity_threshold": 0.005,
        "max_world_scale": 0.1,
        "screen_radius_enabled": False,
        "max_screen_radius_pixels": 20.0,
    },
    "opacity_reset": {"enabled": True, "every_iterations": 3_000},
    "evaluation": {
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
    },
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
    return _resolve(profile, overrides)


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

    loss = _mapping(
        root["loss"], "loss", {"name", "l1_weight", "ssim_weight", "clamp_render"}
    )
    _choice(loss["name"], "loss.name", {"l1_ssim"})
    _boolean(loss["clamp_render"], "loss.clamp_render")
    l1_weight = _number(loss["l1_weight"], "loss.l1_weight", minimum=0.0, maximum=1.0)
    ssim_weight = _number(loss["ssim_weight"], "loss.ssim_weight", minimum=0.0, maximum=1.0)
    if not math.isclose(l1_weight + ssim_weight, 1.0, abs_tol=1e-12):
        raise GaussianConfigError("loss weights must sum to 1.0")

    learning_rate = _mapping(
        root["learning_rate"],
        "learning_rate",
        {"position", "feature", "opacity", "scaling", "rotation"},
    )
    position = _mapping(
        learning_rate["position"],
        "learning_rate.position",
        {"initial", "final"},
    )
    initial = _positive(position["initial"], "learning_rate.position.initial")
    final = _positive(position["final"], "learning_rate.position.final")
    if final > initial:
        raise GaussianConfigError("learning_rate.position.final cannot exceed initial")
    for name in ("feature", "opacity", "scaling", "rotation"):
        _positive(learning_rate[name], f"learning_rate.{name}")

    sh_schedule = _mapping(
        root["sh_schedule"],
        "sh_schedule",
        {"initial_degree", "max_degree", "increase_every_iterations"},
    )
    initial_degree = _integer(sh_schedule["initial_degree"], "sh_schedule.initial_degree", minimum=0, maximum=3)
    max_degree = _integer(sh_schedule["max_degree"], "sh_schedule.max_degree", minimum=0, maximum=3)
    interval = _integer(
        sh_schedule["increase_every_iterations"],
        "sh_schedule.increase_every_iterations",
        minimum=1,
        maximum=iterations,
    )
    if initial_degree > max_degree:
        raise GaussianConfigError("sh_schedule.initial_degree cannot exceed max_degree")
    if (max_degree - initial_degree) * interval > iterations:
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
            "scale_threshold",
        },
    )
    _boolean(densification["enabled"], "densification.enabled")
    start = _integer(densification["start_iteration"], "densification.start_iteration", minimum=1, maximum=iterations)
    end = _integer(densification["end_iteration"], "densification.end_iteration", minimum=1, maximum=iterations)
    every = _integer(densification["every_iterations"], "densification.every_iterations", minimum=1, maximum=iterations)
    _positive(densification["gradient_threshold"], "densification.gradient_threshold")
    _positive(densification["scale_threshold"], "densification.scale_threshold")
    if start >= end:
        raise GaussianConfigError("densification start must precede end")
    if every > end - start:
        raise GaussianConfigError("densification cadence exceeds its active range")

    pruning = _mapping(
        root["pruning"],
        "pruning",
        {
            "enabled",
            "opacity_threshold",
            "max_world_scale",
            "screen_radius_enabled",
            "max_screen_radius_pixels",
        },
    )
    _boolean(pruning["enabled"], "pruning.enabled")
    _boolean(pruning["screen_radius_enabled"], "pruning.screen_radius_enabled")
    _number(pruning["opacity_threshold"], "pruning.opacity_threshold", minimum=0.0, maximum=1.0)
    _positive(pruning["max_world_scale"], "pruning.max_world_scale", maximum=1.0)
    _positive(
        pruning["max_screen_radius_pixels"],
        "pruning.max_screen_radius_pixels",
    )

    opacity_reset = _mapping(
        root["opacity_reset"],
        "opacity_reset",
        {"enabled", "every_iterations"},
    )
    _boolean(opacity_reset["enabled"], "opacity_reset.enabled")
    _integer(opacity_reset["every_iterations"], "opacity_reset.every_iterations", minimum=1, maximum=iterations)

    evaluation = _mapping(root["evaluation"], "evaluation", {"validation_iterations"})
    validation_iterations = evaluation["validation_iterations"]
    if not isinstance(validation_iterations, list) or not validation_iterations:
        raise GaussianConfigError("evaluation.validation_iterations must be a non-empty array")
    validated_iterations = [
        _integer(value, "evaluation.validation_iterations", minimum=1, maximum=iterations)
        for value in validation_iterations
    ]
    if validated_iterations != sorted(set(validated_iterations)):
        raise GaussianConfigError("evaluation.validation_iterations must be sorted and unique")
    if validated_iterations[-1] != iterations:
        raise GaussianConfigError("evaluation.validation_iterations must include the final iteration")


def assert_single_field_ablation(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
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
    return ResolvedGaussianConfig(profile, effective, effective_config_hash(effective))


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


def _positive(value: Any, path: str, maximum: float | None = None) -> float:
    return _number(value, path, minimum=0.0, maximum=maximum, minimum_exclusive=True)


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
