"""Self-contained replay bundle for native Gaussian training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .dataset import sha256_file, validate_contract
from .initialization import load_frozen_initialization


REPLAY_SCHEMA_VERSION = 1


class ReplayError(ValueError):
    """Raised when a Gaussian replay bundle is incomplete or modified."""


def build_replay_bundle(
    *,
    contract: dict[str, Any],
    dataset_path: Path,
    dataset_root: Path,
    initialization_path: Path,
    diagnostics_path: Path,
    replay_root: Path,
) -> dict[str, Any]:
    if (replay_root / "replay.json").is_file():
        return validate_replay_bundle(replay_root)
    if replay_root.exists():
        raise ReplayError("incomplete Gaussian replay directory already exists")

    replay_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".replay-", dir=replay_root.parent))
    try:
        _link_file(dataset_path, temporary / "dataset.json")
        camera_relative = _safe_relative(contract["source"]["camera_path"])
        _link_file(dataset_root / camera_relative, temporary / camera_relative)

        initialization_relative = _safe_relative(contract["initialization"]["asset"])
        _link_file(initialization_path, temporary / initialization_relative)
        diagnostics_relative = initialization_relative.with_suffix(".json")
        _link_file(diagnostics_path, temporary / diagnostics_relative)

        image_bytes = 0
        for image in contract["images"]:
            relative = _safe_relative(image["path"])
            source = dataset_root / relative
            if sha256_file(source) != image["sha256"]:
                raise ReplayError(f"source image hash mismatch: {relative.as_posix()}")
            _link_file(source, temporary / relative)
            image_bytes += source.stat().st_size

        fixed_bytes = sum(
            path.stat().st_size
            for path in (
                dataset_path,
                dataset_root / camera_relative,
                initialization_path,
                diagnostics_path,
            )
        )
        record = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "dataset_id": str(contract["dataset_id"]),
            "dataset_hash": str(contract["dataset_hash"]),
            "dataset_sha256": sha256_file(dataset_path),
            "camera_path": camera_relative.as_posix(),
            "camera_sha256": sha256_file(dataset_root / camera_relative),
            "initialization_path": initialization_relative.as_posix(),
            "initialization_sha256": str(contract["initialization"]["sha256"]),
            "initialization_diagnostics_path": diagnostics_relative.as_posix(),
            "initialization_diagnostics_sha256": sha256_file(diagnostics_path),
            "image_count": len(contract["images"]),
            "image_bytes": image_bytes,
            "total_bytes": fixed_bytes + image_bytes,
        }
        (temporary / "replay.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, replay_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_replay_bundle(replay_root)


def validate_replay_bundle(replay_root: Path) -> dict[str, Any]:
    try:
        record = json.loads((replay_root / "replay.json").read_text(encoding="utf-8"))
        contract = json.loads((replay_root / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read Gaussian replay bundle: {exc}") from exc
    if record.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayError("unsupported Gaussian replay schema")
    if sha256_file(replay_root / "dataset.json") != record.get("dataset_sha256"):
        raise ReplayError("Gaussian replay dataset hash mismatch")
    try:
        validate_contract(contract, replay_root)
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    if contract["dataset_hash"] != record.get("dataset_hash"):
        raise ReplayError("Gaussian replay contract hash mismatch")
    if len(contract["images"]) != record.get("image_count"):
        raise ReplayError("Gaussian replay image count mismatch")

    camera_relative = _recorded_relative(record, "camera_path")
    if sha256_file(replay_root / camera_relative) != record.get("camera_sha256"):
        raise ReplayError("Gaussian replay camera hash mismatch")
    initialization_relative = _recorded_relative(record, "initialization_path")
    diagnostics_relative = _recorded_relative(record, "initialization_diagnostics_path")
    if contract["initialization"]["asset"] != initialization_relative.as_posix():
        raise ReplayError("Gaussian replay initialization path mismatch")
    if contract["initialization"]["sha256"] != record.get("initialization_sha256"):
        raise ReplayError("Gaussian replay initialization contract mismatch")
    if sha256_file(replay_root / diagnostics_relative) != record.get(
        "initialization_diagnostics_sha256"
    ):
        raise ReplayError("Gaussian replay initialization diagnostics hash mismatch")
    load_frozen_initialization(
        replay_root / initialization_relative,
        replay_root / diagnostics_relative,
        expected_sha256=str(record["initialization_sha256"]),
    )
    image_bytes = sum(
        (replay_root / _safe_relative(image["path"])).stat().st_size
        for image in contract["images"]
    )
    if image_bytes != record.get("image_bytes"):
        raise ReplayError("Gaussian replay image byte count mismatch")
    fixed_bytes = sum(
        (replay_root / relative).stat().st_size
        for relative in (
            Path("dataset.json"),
            camera_relative,
            initialization_relative,
            diagnostics_relative,
        )
    )
    if fixed_bytes + image_bytes != record.get("total_bytes"):
        raise ReplayError("Gaussian replay total byte count mismatch")
    return record


def _link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ReplayError(f"missing replay source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_relative(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not str(value):
        raise ReplayError("Gaussian replay path must be project-relative")
    return path


def _recorded_relative(record: dict[str, Any], key: str) -> Path:
    if not isinstance(record.get(key), str):
        raise ReplayError(f"Gaussian replay record is missing {key}")
    return _safe_relative(record[key])
