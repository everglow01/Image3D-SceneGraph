"""Import INRIA Gaussian PLY files into the project model snapshot."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import sha256_file
from .export import PLY_FIELDS, read_gaussian_ply
from .initialization import transform_points


class GaussianImportError(RuntimeError):
    """Raised when an external Gaussian PLY cannot be normalized safely."""


def import_inria_ply(
    source: Path,
    destination: Path,
    *,
    normalized_from_source: np.ndarray | None = None,
    trainer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = read_gaussian_ply(source)
    matrix = np.column_stack([fields[name] for name in PLY_FIELDS]).astype(np.float32)
    means = matrix[:, :3]
    log_scales = matrix[:, 55:58]
    quats = matrix[:, 58:62]
    if normalized_from_source is not None:
        transform = np.asarray(normalized_from_source, dtype=np.float64)
        _validate_similarity(transform)
        means = transform_points(means, transform)
        scale = float(np.linalg.norm(transform[:3, 0]))
        log_scales = log_scales + np.log(scale)
        rotation = transform[:3, :3] / scale
        quats = _multiply_quaternions(_rotation_quaternion(rotation), quats)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    sh0 = matrix[:, 6:9, None].transpose(0, 2, 1)
    rest = matrix[:, 9:54].reshape(len(matrix), 3, 15).transpose(0, 2, 1)
    state = {
        "means": means,
        "log_scales": log_scales,
        "quats": quats,
        "opacity_logits": matrix[:, 54],
        "sh_coeffs": np.concatenate((sh0, rest), axis=1),
    }
    if not all(np.isfinite(value).all() for value in state.values()):
        raise GaussianImportError("external Gaussian PLY contains non-finite tensors")

    import torch

    payload = {
        "max_sh_degree": 3,
        "state_dict": {
            name: torch.from_numpy(value.astype(np.float32, copy=False))
            for name, value in state.items()
        },
        "external_trainer": trainer,
        "source_sha256": sha256_file(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    destination.write_bytes(buffer.getvalue())
    record = {
        "source": str(source),
        "source_sha256": payload["source_sha256"],
        "model": str(destination),
        "model_sha256": sha256_file(destination),
        "gaussian_count": len(matrix),
        "trainer": trainer,
    }
    destination.with_suffix(".import.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _validate_similarity(transform: np.ndarray) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise GaussianImportError("external Gaussian transform must be finite 4 x 4")
    linear = transform[:3, :3]
    scale = float(np.linalg.norm(linear[:, 0]))
    if scale <= 0 or not np.allclose(linear.T @ linear, np.eye(3) * scale * scale, atol=1e-7):
        raise GaussianImportError("external Gaussian transform must be a similarity")
    if np.linalg.det(linear) <= 0:
        raise GaussianImportError("external Gaussian transform cannot reflect coordinates")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12):
        raise GaussianImportError("external Gaussian transform has invalid homogeneous row")


def _rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        root = np.sqrt(trace + 1.0) * 2
        quat = [0.25 * root, (rotation[2, 1] - rotation[1, 2]) / root,
                (rotation[0, 2] - rotation[2, 0]) / root,
                (rotation[1, 0] - rotation[0, 1]) / root]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            root = np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quat = [(rotation[2, 1] - rotation[1, 2]) / root, 0.25 * root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root]
        elif index == 1:
            root = np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quat = [(rotation[0, 2] - rotation[2, 0]) / root,
                    (rotation[0, 1] + rotation[1, 0]) / root, 0.25 * root,
                    (rotation[1, 2] + rotation[2, 1]) / root]
        else:
            root = np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quat = [(rotation[1, 0] - rotation[0, 1]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                    (rotation[1, 2] + rotation[2, 1]) / root, 0.25 * root]
    value = np.asarray(quat, dtype=np.float32)
    return value / np.linalg.norm(value)


def _multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right.T
    return np.column_stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    ).astype(np.float32)
