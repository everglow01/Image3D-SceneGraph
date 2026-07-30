"""Project-owned 3D Gaussian training contracts."""

from .config import (
    CONFIG_SCHEMA_VERSION,
    PUBLIC_PROFILES,
    GaussianConfigError,
    ResolvedGaussianConfig,
    assert_single_field_ablation,
    canonical_config_json,
    effective_config_hash,
    resolve_internal_config,
    resolve_public_config,
    resolved_config_record,
    validate_effective_config,
)
from .dataset import (
    DatasetContractError,
    build_colmap_contract,
    deterministic_spatial_split,
    validate_contract,
    write_contract,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "PUBLIC_PROFILES",
    "DatasetContractError",
    "GaussianConfigError",
    "ResolvedGaussianConfig",
    "assert_single_field_ablation",
    "build_colmap_contract",
    "canonical_config_json",
    "deterministic_spatial_split",
    "effective_config_hash",
    "resolve_internal_config",
    "resolve_public_config",
    "resolved_config_record",
    "validate_contract",
    "validate_effective_config",
    "write_contract",
]
