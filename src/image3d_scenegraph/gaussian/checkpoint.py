"""Atomic attempt and checkpoint contract for project-owned 3DGS training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset import sha256_file


ATTEMPT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
ATTEMPT_KINDS = {"fresh", "retry", "resume"}
CHECKPOINT_PURPOSES = {"periodic", "best_validation", "final"}
_COMPONENT_FILES = {
    "model": "model.bin",
    "optimizer": "optimizer.bin",
    "scheduler": "scheduler.bin",
    "densification": "densification.bin",
    "rng": "rng.bin",
    "metric_history": "metrics.json",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_DIRECTORY = re.compile(r"iteration_([0-9]{9})\Z")


class CheckpointContractError(ValueError):
    """Raised when attempt or checkpoint state violates the R2.5 contract."""


@dataclass(frozen=True)
class CheckpointProvenance:
    dataset_hash: str
    effective_config_hash: str
    code_hash: str
    environment_hash: str


@dataclass(frozen=True)
class CheckpointReference:
    attempt_id: str
    iteration: int
    path: str
    checkpoint_hash: str


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    kind: str
    provenance: CheckpointProvenance
    parent_attempt_id: str | None
    resume_checkpoint: CheckpointReference | None


@dataclass(frozen=True)
class CheckpointState:
    model: bytes
    optimizer: bytes
    scheduler: bytes
    densification: bytes
    rng: bytes
    metric_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CheckpointRecord:
    attempt_id: str
    iteration: int
    purpose: str
    validation_score: float | None
    provenance: CheckpointProvenance
    checkpoint_hash: str


@dataclass(frozen=True)
class LoadedCheckpoint:
    record: CheckpointRecord
    state: CheckpointState


def attempt_dir(job_dir: Path, attempt_id: str) -> Path:
    _validate_identifier(attempt_id, "attempt_id")
    return job_dir / "attempts" / attempt_id


def checkpoint_dir(job_dir: Path, attempt_id: str, iteration: int) -> Path:
    _validate_iteration(iteration)
    return attempt_dir(job_dir, attempt_id) / "checkpoints" / f"iteration_{iteration:09d}"


def create_attempt(
    job_dir: Path,
    *,
    attempt_id: str,
    kind: str,
    provenance: CheckpointProvenance,
    parent_attempt_id: str | None = None,
    resume_iteration: int | None = None,
) -> AttemptRecord:
    """Create one immutable fresh, retry, or resume attempt descriptor."""
    _validate_identifier(attempt_id, "attempt_id")
    _validate_provenance(provenance)
    if kind not in ATTEMPT_KINDS:
        raise CheckpointContractError(f"unsupported attempt kind: {kind}")

    attempts_root = job_dir / "attempts"
    destination = attempt_dir(job_dir, attempt_id)
    if destination.exists():
        raise CheckpointContractError(f"attempt already exists: {attempt_id}")

    parent: AttemptRecord | None = None
    reference: CheckpointReference | None = None
    if kind == "fresh":
        if parent_attempt_id is not None or resume_iteration is not None:
            raise CheckpointContractError("fresh attempt cannot have parent or resume checkpoint")
        if attempts_root.is_dir() and any(
            entry.is_dir() and not entry.name.startswith(".") for entry in attempts_root.iterdir()
        ):
            raise CheckpointContractError("fresh attempt must be the first attempt")
    else:
        if parent_attempt_id is None:
            raise CheckpointContractError(f"{kind} attempt requires a parent attempt")
        if parent_attempt_id == attempt_id:
            raise CheckpointContractError("attempt cannot be its own parent")
        parent = load_attempt(job_dir, parent_attempt_id)
        if kind == "retry":
            if resume_iteration is not None:
                raise CheckpointContractError("retry attempt cannot load a checkpoint")
            _match_provenance(parent.provenance, provenance, ("dataset_hash", "effective_config_hash"))
        else:
            if resume_iteration is None:
                raise CheckpointContractError("resume attempt requires a checkpoint iteration")
            loaded = load_checkpoint(
                job_dir,
                parent_attempt_id,
                resume_iteration,
                expected_provenance=provenance,
            )
            reference = CheckpointReference(
                attempt_id=parent_attempt_id,
                iteration=resume_iteration,
                path=(
                    Path("attempts")
                    / parent_attempt_id
                    / "checkpoints"
                    / f"iteration_{resume_iteration:09d}"
                ).as_posix(),
                checkpoint_hash=loaded.record.checkpoint_hash,
            )

    record = AttemptRecord(
        attempt_id=attempt_id,
        kind=kind,
        provenance=provenance,
        parent_attempt_id=parent.attempt_id if parent is not None else None,
        resume_checkpoint=reference,
    )
    payload = _attempt_payload(record)
    payload["attempt_hash"] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    attempts_root.mkdir(parents=True, exist_ok=True)
    temporary = attempts_root / f".{attempt_id}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        _write_bytes_sync(temporary / "attempt.json", _json_bytes(payload, indent=2))
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        _fsync_directory(attempts_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def load_attempt(job_dir: Path, attempt_id: str) -> AttemptRecord:
    path = attempt_dir(job_dir, attempt_id)
    if path.is_symlink() or not path.is_dir():
        raise CheckpointContractError(f"attempt does not exist: {attempt_id}")
    payload = _read_json_object(path / "attempt.json", "attempt descriptor")
    _exact_fields(
        payload,
        {
            "schema_version",
            "attempt_id",
            "kind",
            "provenance",
            "parent_attempt_id",
            "resume_checkpoint",
            "attempt_hash",
        },
        "attempt descriptor",
    )
    recorded_hash = payload["attempt_hash"]
    _validate_hash(recorded_hash, "attempt_hash")
    hash_payload = {key: value for key, value in payload.items() if key != "attempt_hash"}
    if hashlib.sha256(_json_bytes(hash_payload)).hexdigest() != recorded_hash:
        raise CheckpointContractError("attempt descriptor hash mismatch")
    if payload["schema_version"] != ATTEMPT_SCHEMA_VERSION:
        raise CheckpointContractError(
            f"unsupported attempt schema version: {payload['schema_version']}"
        )
    if payload["attempt_id"] != attempt_id:
        raise CheckpointContractError("attempt descriptor ID mismatch")
    kind = payload["kind"]
    if kind not in ATTEMPT_KINDS:
        raise CheckpointContractError(f"unsupported attempt kind: {kind}")
    provenance = _parse_provenance(payload["provenance"])
    parent = payload["parent_attempt_id"]
    if parent is not None:
        _validate_identifier(parent, "parent_attempt_id")
    reference = _parse_reference(payload["resume_checkpoint"])
    if kind == "fresh" and (parent is not None or reference is not None):
        raise CheckpointContractError("invalid fresh attempt lineage")
    if kind == "retry" and (parent is None or reference is not None):
        raise CheckpointContractError("invalid retry attempt lineage")
    if kind == "resume" and (parent is None or reference is None or reference.attempt_id != parent):
        raise CheckpointContractError("invalid resume attempt lineage")
    return AttemptRecord(attempt_id, kind, provenance, parent, reference)


def write_checkpoint(
    job_dir: Path,
    *,
    attempt_id: str,
    iteration: int,
    purpose: str,
    provenance: CheckpointProvenance,
    state: CheckpointState,
    validation_score: float | None = None,
) -> CheckpointRecord:
    """Publish a complete checkpoint with one same-filesystem directory rename."""
    _validate_iteration(iteration)
    _validate_provenance(provenance)
    _validate_checkpoint_selection(purpose, validation_score)
    attempt = load_attempt(job_dir, attempt_id)
    _match_provenance(attempt.provenance, provenance, tuple(_provenance_payload(provenance)))
    components = _state_components(state)

    destination = checkpoint_dir(job_dir, attempt_id, iteration)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CheckpointContractError(
            f"checkpoint already exists: {attempt_id} iteration {iteration}"
        )
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        file_records: dict[str, dict[str, Any]] = {}
        for name, content in components.items():
            filename = _COMPONENT_FILES[name]
            _write_bytes_sync(temporary / filename, content)
            file_records[name] = {
                "path": filename,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "iteration": iteration,
            "purpose": purpose,
            "validation_score": validation_score,
            "provenance": _provenance_payload(provenance),
            "files": file_records,
        }
        checkpoint_hash = hashlib.sha256(_json_bytes(metadata)).hexdigest()
        metadata["checkpoint_hash"] = checkpoint_hash
        _write_bytes_sync(temporary / "checkpoint.json", _json_bytes(metadata, indent=2))
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return CheckpointRecord(
        attempt_id,
        iteration,
        purpose,
        validation_score,
        provenance,
        checkpoint_hash,
    )


def load_checkpoint(
    job_dir: Path,
    attempt_id: str,
    iteration: int,
    *,
    expected_provenance: CheckpointProvenance | None = None,
) -> LoadedCheckpoint:
    """Load only a complete checkpoint whose metadata and component hashes agree."""
    path = checkpoint_dir(job_dir, attempt_id, iteration)
    if path.is_symlink() or not path.is_dir():
        raise CheckpointContractError(
            f"checkpoint does not exist: {attempt_id} iteration {iteration}"
        )
    metadata = _read_json_object(path / "checkpoint.json", "checkpoint metadata")
    _exact_fields(
        metadata,
        {
            "schema_version",
            "attempt_id",
            "iteration",
            "purpose",
            "validation_score",
            "provenance",
            "files",
            "checkpoint_hash",
        },
        "checkpoint metadata",
    )
    if metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointContractError(
            f"unsupported checkpoint schema version: {metadata['schema_version']}"
        )
    if metadata["attempt_id"] != attempt_id or metadata["iteration"] != iteration:
        raise CheckpointContractError("checkpoint identity mismatch")
    _validate_checkpoint_selection(metadata["purpose"], metadata["validation_score"])
    provenance = _parse_provenance(metadata["provenance"])
    attempt = load_attempt(job_dir, attempt_id)
    _match_provenance(attempt.provenance, provenance, tuple(_provenance_payload(provenance)))
    if expected_provenance is not None:
        _validate_provenance(expected_provenance)
        _match_provenance(provenance, expected_provenance, tuple(_provenance_payload(provenance)))

    files = metadata["files"]
    if not isinstance(files, dict):
        raise CheckpointContractError("checkpoint files must be an object")
    _exact_fields(files, set(_COMPONENT_FILES), "checkpoint files")
    contents: dict[str, bytes] = {}
    for name, expected_path in _COMPONENT_FILES.items():
        entry = files[name]
        if not isinstance(entry, dict):
            raise CheckpointContractError(f"checkpoint file record must be an object: {name}")
        _exact_fields(entry, {"path", "bytes", "sha256"}, f"checkpoint file {name}")
        if entry["path"] != expected_path:
            raise CheckpointContractError(f"checkpoint file path mismatch: {name}")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise CheckpointContractError(f"invalid checkpoint file size: {name}")
        _validate_hash(entry["sha256"], f"checkpoint file hash: {name}")
        component_path = path / expected_path
        if component_path.is_symlink() or not component_path.is_file():
            raise CheckpointContractError(f"checkpoint file missing: {name}")
        if component_path.stat().st_size != entry["bytes"]:
            raise CheckpointContractError(f"checkpoint file size mismatch: {name}")
        if sha256_file(component_path) != entry["sha256"]:
            raise CheckpointContractError(f"checkpoint file hash mismatch: {name}")
        contents[name] = component_path.read_bytes()

    recorded_hash = metadata["checkpoint_hash"]
    _validate_hash(recorded_hash, "checkpoint_hash")
    hash_payload = {key: value for key, value in metadata.items() if key != "checkpoint_hash"}
    if hashlib.sha256(_json_bytes(hash_payload)).hexdigest() != recorded_hash:
        raise CheckpointContractError("checkpoint metadata hash mismatch")
    try:
        metric_history = json.loads(contents["metric_history"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointContractError("metric history is not valid JSON") from exc
    _validate_metric_history(metric_history)

    state = CheckpointState(
        model=contents["model"],
        optimizer=contents["optimizer"],
        scheduler=contents["scheduler"],
        densification=contents["densification"],
        rng=contents["rng"],
        metric_history=tuple(metric_history),
    )
    record = CheckpointRecord(
        attempt_id,
        iteration,
        metadata["purpose"],
        metadata["validation_score"],
        provenance,
        recorded_hash,
    )
    return LoadedCheckpoint(record, state)


def prune_attempt_checkpoints(
    job_dir: Path,
    attempt_id: str,
    *,
    keep_iterations: Iterable[int],
) -> None:
    """Remove committed checkpoints outside the current attempt's retained set."""
    attempt = load_attempt(job_dir, attempt_id)
    retained = set(keep_iterations)
    for iteration in retained:
        _validate_iteration(iteration)
    root = attempt_dir(job_dir, attempt.attempt_id) / "checkpoints"
    if not root.is_dir():
        return
    for entry in root.iterdir():
        match = _CHECKPOINT_DIRECTORY.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            continue
        if int(match.group(1)) not in retained:
            shutil.rmtree(entry)
    _fsync_directory(root)


def _state_components(state: CheckpointState) -> dict[str, bytes]:
    components = {
        "model": state.model,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "densification": state.densification,
        "rng": state.rng,
    }
    for name, value in components.items():
        if type(value) is not bytes:
            raise CheckpointContractError(f"checkpoint component must be bytes: {name}")
    history = list(state.metric_history)
    _validate_metric_history(history)
    components["metric_history"] = _json_bytes(history)
    return components


def _validate_metric_history(history: Any) -> None:
    if not isinstance(history, list) or any(not isinstance(entry, dict) for entry in history):
        raise CheckpointContractError("metric history must be a list of objects")
    try:
        json.dumps(history, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError("metric history must contain finite JSON values") from exc


def _validate_checkpoint_selection(purpose: Any, validation_score: Any) -> None:
    if purpose not in CHECKPOINT_PURPOSES:
        raise CheckpointContractError(f"unsupported checkpoint purpose: {purpose}")
    if purpose == "best_validation":
        if type(validation_score) is not float or not math.isfinite(validation_score):
            raise CheckpointContractError("best-validation checkpoint requires a finite score")
    elif validation_score is not None:
        raise CheckpointContractError("validation score is only valid for best-validation checkpoints")


def _attempt_payload(record: AttemptRecord) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": record.attempt_id,
        "kind": record.kind,
        "provenance": _provenance_payload(record.provenance),
        "parent_attempt_id": record.parent_attempt_id,
        "resume_checkpoint": (
            {
                "attempt_id": record.resume_checkpoint.attempt_id,
                "iteration": record.resume_checkpoint.iteration,
                "path": record.resume_checkpoint.path,
                "checkpoint_hash": record.resume_checkpoint.checkpoint_hash,
            }
            if record.resume_checkpoint is not None
            else None
        ),
    }


def _parse_reference(value: Any) -> CheckpointReference | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CheckpointContractError("resume checkpoint must be an object")
    _exact_fields(value, {"attempt_id", "iteration", "path", "checkpoint_hash"}, "resume checkpoint")
    _validate_identifier(value["attempt_id"], "resume checkpoint attempt_id")
    _validate_iteration(value["iteration"])
    expected_path = (
        Path("attempts")
        / value["attempt_id"]
        / "checkpoints"
        / f"iteration_{value['iteration']:09d}"
    ).as_posix()
    if value["path"] != expected_path:
        raise CheckpointContractError("resume checkpoint path mismatch")
    _validate_hash(value["checkpoint_hash"], "resume checkpoint hash")
    return CheckpointReference(
        value["attempt_id"], value["iteration"], value["path"], value["checkpoint_hash"]
    )


def _provenance_payload(provenance: CheckpointProvenance) -> dict[str, str]:
    return {
        "dataset_hash": provenance.dataset_hash,
        "effective_config_hash": provenance.effective_config_hash,
        "code_hash": provenance.code_hash,
        "environment_hash": provenance.environment_hash,
    }


def _parse_provenance(value: Any) -> CheckpointProvenance:
    if not isinstance(value, dict):
        raise CheckpointContractError("checkpoint provenance must be an object")
    _exact_fields(
        value,
        {"dataset_hash", "effective_config_hash", "code_hash", "environment_hash"},
        "checkpoint provenance",
    )
    provenance = CheckpointProvenance(**value)
    _validate_provenance(provenance)
    return provenance


def _validate_provenance(provenance: CheckpointProvenance) -> None:
    if not isinstance(provenance, CheckpointProvenance):
        raise CheckpointContractError("provenance must be CheckpointProvenance")
    for name, value in _provenance_payload(provenance).items():
        _validate_hash(value, name)


def _match_provenance(
    actual: CheckpointProvenance,
    expected: CheckpointProvenance,
    fields: tuple[str, ...],
) -> None:
    actual_values = _provenance_payload(actual)
    expected_values = _provenance_payload(expected)
    for field in fields:
        if actual_values[field] != expected_values[field]:
            raise CheckpointContractError(f"checkpoint {field} mismatch")


def _validate_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CheckpointContractError(f"invalid {name}")


def _validate_iteration(value: Any) -> None:
    if type(value) is not int or not 0 <= value <= 999_999_999:
        raise CheckpointContractError("iteration must be an integer from 0 to 999999999")


def _validate_hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise CheckpointContractError(f"invalid {name}")


def _exact_fields(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise CheckpointContractError(f"{name} fields must be exactly {sorted(expected)}")


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError("value is not finite JSON") from exc
    return (text + ("\n" if indent is not None else "")).encode("utf-8")


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointContractError(f"{name} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointContractError(f"cannot read {name}") from exc
    if not isinstance(value, dict):
        raise CheckpointContractError(f"{name} must be an object")
    return value


def _write_bytes_sync(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
