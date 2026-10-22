"""固定 B0 的冻结依赖桥；旧文件只作来源验证和既有工具复用。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[6]
MANIFEST_PATH = Path(__file__).with_name("gpu-source.json")
MANIFEST_SHA256 = "5f4cd0426ce359b6f6f57f41ab33dee59576b0a8aa64b42e3dafffcd37f1f3d2"


def verify(root=None):
    root = PROJECT if root is None else Path(root)
    if MANIFEST_PATH.is_symlink() or hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen bridge manifest differs")
    manifest = json.loads(MANIFEST_PATH.read_text())
    for name, expected in manifest["files"].items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts
                or any(p.is_symlink() for p in (path, *path.parents))
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError("unsafe frozen dependency path")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("frozen dependency differs: " + name)
    return manifest


# 先验证冻结字节，只加载不引入张量库的旧工具。
MANIFEST = verify()
LEGACY_FILES = tuple(MANIFEST["files"])
CORE = MANIFEST["core_sha256"]
CORE_FILES = tuple(MANIFEST["core_files"])
from scripts import mcmc_first_divergence as snapshots  # noqa: E402
from scripts.prepare_gsplat_camera_sizes import file_hashes  # noqa: E402, F401

digest_state = snapshots.digest_state
rng_state = snapshots.rng_state
unchanged_rng = snapshots.unchanged_rng


def read(path):
    return json.loads(path.read_text())


LIMITS = {"max_observed_global_gaussians": 3_000_000,
          "max_main_stage_wall_seconds": 21600,
          "validation_progress_timeout_seconds": 300,
          "max_per_rank_peak_reserved_bytes": [18 * 1024**3] * 2}


def unlimited_host_cgroup(unit: str) -> Path:
    expected = f"/system.slice/{unit}"
    if f"0::{expected}" not in Path("/proc/self/cgroup").read_text().splitlines():
        raise ValueError("unlimited host memory requires its own systemd cgroup")
    group = Path("/sys/fs/cgroup") / expected.lstrip("/")
    for directory in (group, group.parent, group.parent.parent):
        for name in ("memory.max", "memory.high", "memory.swap.max"):
            path = directory / name
            if directory == group.parent.parent and not path.exists():
                continue
            if path.read_text().strip() != "max":
                raise ValueError(f"host memory is still limited: {path}")
    return group


def core_hash():
    digest = hashlib.sha256()
    for name in CORE_FILES:
        digest.update(name.encode())
        digest.update((PROJECT / "src/image3d_scenegraph/gaussian" / name).read_bytes())
    return digest.hexdigest()


def _load_source():
    verify()
    from scripts import run_mcmc_fixed_b0 as frozen
    if frozen.PROJECT != PROJECT or frozen.native.lifecycle.CORE != CORE:
        raise ValueError("frozen project/core identity differs")
    return frozen


def verify_source():
    return _load_source().verify_source()


def validate_boundary_schema(record, stage):
    return _load_source().boundary.validate_boundary_schema(record, stage)


boundary = SimpleNamespace(SAVED=tuple(MANIFEST["saved_tensors"]),
                           ORDER=tuple(MANIFEST["boundary_order"]),
                           OUTPUTS=tuple(MANIFEST["boundary_outputs"]),
                           validate_boundary_schema=validate_boundary_schema)


def run_stage(*args, **kwargs):
    from image3d_scenegraph.gaussian.absgrad_resources import run_stage as execute
    return execute(*args, **kwargs)
