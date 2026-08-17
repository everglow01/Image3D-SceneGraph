"""Versioned dataset and camera contract for project-owned 3DGS training."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MIN_REGISTERED_IMAGES = 12
SPLIT_SEED = 20260729


class DatasetContractError(ValueError):
    """Raised when a 3DGS dataset contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qvec_to_rotmat(qvec: Any) -> np.ndarray:
    quaternion = np.asarray(qvec, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise DatasetContractError("COLMAP qvec must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise DatasetContractError("COLMAP qvec cannot be zero")
    qw, qx, qy, qz = quaternion / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def colmap_intrinsics(camera: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    model = str(camera.get("model", ""))
    params = np.asarray(camera.get("params"), dtype=np.float64)
    expected = {
        "SIMPLE_PINHOLE": 3,
        "PINHOLE": 4,
        "SIMPLE_RADIAL": 4,
        "RADIAL": 5,
        "OPENCV": 8,
    }
    if model not in expected or params.shape != (expected.get(model, -1),) or not np.isfinite(params).all():
        raise DatasetContractError(f"unsupported or invalid COLMAP camera model: {model}")
    if model.startswith("SIMPLE_") or model == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = params[3:]
    else:
        fx, fy, cx, cy = params[:4]
        distortion = params[4:]
    if min(fx, fy) <= 0:
        raise DatasetContractError("camera focal lengths must be positive")
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return intrinsic, {
        "state": "none" if len(distortion) == 0 or np.allclose(distortion, 0.0) else "modeled",
        "model": model,
        "params": distortion.tolist(),
    }


def spatial_order(image_ids: list[str], centers: np.ndarray, seed: int = SPLIT_SEED) -> list[int]:
    if centers.shape != (len(image_ids), 3) or not np.isfinite(centers).all():
        raise DatasetContractError("camera centers must be a finite N x 3 array")
    if not image_ids:
        return []
    tie_break = [hashlib.sha256(f"{seed}:{image_id}".encode()).digest() for image_id in image_ids]
    centroid = centers.mean(axis=0)
    first = max(range(len(image_ids)), key=lambda index: (float(np.linalg.norm(centers[index] - centroid)), tie_break[index]))
    selected = [first]
    remaining = set(range(len(image_ids))) - {first}
    min_distances = np.linalg.norm(centers - centers[first], axis=1)
    while remaining:
        next_index = max(remaining, key=lambda index: (float(min_distances[index]), tie_break[index]))
        selected.append(next_index)
        remaining.remove(next_index)
        min_distances = np.minimum(min_distances, np.linalg.norm(centers - centers[next_index], axis=1))
    return selected


def deterministic_temporal_group_split(
    images: list[dict[str, Any]],
    timestamps: dict[str, float],
    seed: int = SPLIT_SEED,
    group_seconds: float = 2.0,
) -> dict[str, list[str]]:
    if len(images) < MIN_REGISTERED_IMAGES:
        raise DatasetContractError(f"at least {MIN_REGISTERED_IMAGES} registered images are required")
    groups: dict[int, list[dict[str, Any]]] = {}
    for image in images:
        name = Path(str(image["path"])).name
        try:
            timestamp = float(timestamps[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetContractError(f"missing video timestamp for registered image: {name}") from exc
        if not np.isfinite(timestamp) or timestamp < 0:
            raise DatasetContractError(f"invalid video timestamp for registered image: {name}")
        groups.setdefault(int(timestamp // group_seconds), []).append(image)
    if len(groups) < 5:
        raise DatasetContractError("video split requires at least five temporal groups")
    group_ids = sorted(groups)
    centers = np.stack(
        [
            np.mean(
                [np.asarray(image["world_from_camera"], dtype=np.float64)[:3, 3] for image in groups[group]],
                axis=0,
            )
            for group in group_ids
        ]
    )
    order = spatial_order([str(group) for group in group_ids], centers, seed)
    heldout_count = min(max(2, int(round(len(groups) * 0.1))), (len(groups) - 1) // 2)
    heldout = order[: 2 * heldout_count]
    validation_groups = {group_ids[index] for index in heldout[1::2]}
    test_groups = {group_ids[index] for index in heldout[::2]}
    split = {"train": [], "validation": [], "test": []}
    for group, entries in groups.items():
        destination = (
            "validation"
            if group in validation_groups
            else "test"
            if group in test_groups
            else "train"
        )
        split[destination].extend(str(image["image_id"]) for image in entries)
    return {name: sorted(ids) for name, ids in split.items()}


def deterministic_spatial_split(images: list[dict[str, Any]], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    if len(images) < MIN_REGISTERED_IMAGES:
        raise DatasetContractError(f"at least {MIN_REGISTERED_IMAGES} registered images are required")
    image_ids = [str(image["image_id"]) for image in images]
    centers = np.stack([np.asarray(image["world_from_camera"], dtype=np.float64)[:3, 3] for image in images])
    heldout_count = max(2, int(round(len(images) * 0.1)))
    if len(images) - 2 * heldout_count < 1:
        raise DatasetContractError("not enough training images after held-out split")
    order = spatial_order(image_ids, centers, seed)
    heldout = order[: 2 * heldout_count]
    test_indices = heldout[::2]
    validation_indices = heldout[1::2]
    heldout_set = set(heldout)
    return {
        "train": sorted(image_ids[index] for index in range(len(images)) if index not in heldout_set),
        "validation": sorted(image_ids[index] for index in validation_indices),
        "test": sorted(image_ids[index] for index in test_indices),
    }


def camera_normalization(images: list[dict[str, Any]]) -> dict[str, Any]:
    centers = np.stack([np.asarray(image["world_from_camera"], dtype=np.float64)[:3, 3] for image in images])
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max())
    if not np.isfinite(radius) or radius <= 1e-12:
        raise DatasetContractError("camera centers do not define a usable scene extent")
    normalized_from_world = np.eye(4, dtype=np.float64)
    normalized_from_world[:3, :3] *= 1.0 / radius
    normalized_from_world[:3, 3] = -center / radius
    return {
        "method": "camera_center_max_radius_v1",
        "center_world": center.tolist(),
        "radius_world": radius,
        "normalized_from_world": normalized_from_world.tolist(),
        "world_from_normalized": np.linalg.inv(normalized_from_world).tolist(),
    }


def build_colmap_contract(
    *,
    dataset_id: str,
    dataset_root: Path,
    image_root: str,
    cameras_path: str,
    max_views: int | None = None,
    seed: int = SPLIT_SEED,
    temporal_timestamps: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    camera_file = (root / cameras_path).resolve()
    image_dir = (root / image_root).resolve()
    for path in (camera_file, image_dir):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DatasetContractError(f"dataset path escapes root: {path}") from exc
    try:
        source = json.loads(camera_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read camera source: {exc}") from exc
    if source.get("coordinate_system") != "colmap_world":
        raise DatasetContractError("camera source must use colmap_world")
    cameras = {int(camera["camera_id"]): camera for camera in source.get("cameras", [])}
    image_entries: list[dict[str, Any]] = []
    for source_image in source.get("images", []):
        camera = cameras.get(int(source_image["camera_id"]))
        if camera is None:
            raise DatasetContractError(f"missing camera {source_image['camera_id']}")
        image_path = Path(image_root) / str(source_image["name"])
        absolute_image = (root / image_path).resolve()
        if not absolute_image.is_file():
            raise DatasetContractError(f"missing registered image: {image_path.as_posix()}")
        intrinsic, distortion = colmap_intrinsics(camera)
        rotation = qvec_to_rotmat(source_image["qvec"])
        camera_from_world = np.eye(4, dtype=np.float64)
        camera_from_world[:3, :3] = rotation
        camera_from_world[:3, 3] = np.asarray(source_image["tvec"], dtype=np.float64)
        image_entries.append(
            {
                "image_id": str(source_image["image_id"]),
                "path": image_path.as_posix(),
                "width": int(camera["width"]),
                "height": int(camera["height"]),
                "sha256": sha256_file(absolute_image),
                "intrinsic": intrinsic.tolist(),
                "distortion": distortion,
                "camera_from_world": camera_from_world.tolist(),
                "world_from_camera": np.linalg.inv(camera_from_world).tolist(),
            }
        )
    if max_views is not None and len(image_entries) > max_views:
        ids = [str(image["image_id"]) for image in image_entries]
        centers = np.stack([np.asarray(image["world_from_camera"])[:3, 3] for image in image_entries])
        selected = set(spatial_order(ids, centers, seed)[:max_views])
        image_entries = [image for index, image in enumerate(image_entries) if index in selected]
    image_entries.sort(key=lambda image: image["image_id"])
    contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "coordinate_system": {
            "camera_convention": "opencv",
            "camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "world_frame": "raw",
            "world_units": "arbitrary",
            "raw_from_world": np.eye(4).tolist(),
            "world_from_raw": np.eye(4).tolist(),
        },
        "normalization": camera_normalization(image_entries),
        "source": {
            "camera_format": "colmap_cameras_json_v1",
            "camera_path": cameras_path,
            "camera_sha256": sha256_file(camera_file),
            "image_root": image_root,
        },
        "images": image_entries,
        "splits": (
            deterministic_temporal_group_split(image_entries, temporal_timestamps, seed)
            if temporal_timestamps is not None
            else deterministic_spatial_split(image_entries, seed)
        ),
        "initialization": {"coordinate_frame": "world", "asset": None, "sha256": None},
    }
    contract["dataset_hash"] = contract_hash(contract)
    validate_contract(contract, root)
    return contract


def contract_hash(contract: dict[str, Any]) -> str:
    payload = {key: value for key, value in contract.items() if key != "dataset_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_contract(contract: dict[str, Any], dataset_root: Path | None = None) -> None:
    required = {
        "schema_version",
        "dataset_id",
        "dataset_hash",
        "coordinate_system",
        "normalization",
        "source",
        "images",
        "splits",
        "initialization",
    }
    if set(contract) != required:
        raise DatasetContractError(f"contract fields must be exactly {sorted(required)}")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise DatasetContractError(f"unsupported dataset schema version: {contract['schema_version']}")
    images = contract["images"]
    if not isinstance(images, list) or len(images) < MIN_REGISTERED_IMAGES:
        raise DatasetContractError(f"at least {MIN_REGISTERED_IMAGES} registered images are required")
    image_ids = [str(image.get("image_id")) for image in images]
    image_paths = [str(image.get("path")) for image in images]
    if len(set(image_ids)) != len(images) or len(set(image_paths)) != len(images):
        raise DatasetContractError("image IDs and paths must be unique")
    for image in images:
        intrinsic = np.asarray(image.get("intrinsic"), dtype=np.float64)
        camera_from_world = np.asarray(image.get("camera_from_world"), dtype=np.float64)
        world_from_camera = np.asarray(image.get("world_from_camera"), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
            raise DatasetContractError(f"invalid intrinsics for image {image.get('image_id')}")
        if camera_from_world.shape != (4, 4) or world_from_camera.shape != (4, 4):
            raise DatasetContractError(f"invalid pose shape for image {image.get('image_id')}")
        if not np.allclose(camera_from_world @ world_from_camera, np.eye(4), atol=1e-7):
            raise DatasetContractError(f"camera round-trip failed for image {image.get('image_id')}")
        if dataset_root is not None:
            path = (dataset_root / str(image["path"])).resolve()
            try:
                path.relative_to(dataset_root.resolve())
            except ValueError as exc:
                raise DatasetContractError(f"image path escapes dataset root: {image['path']}") from exc
            if not path.is_file() or sha256_file(path) != image.get("sha256"):
                raise DatasetContractError(f"image hash mismatch: {image['path']}")
    split = contract["splits"]
    if set(split) != {"train", "validation", "test"}:
        raise DatasetContractError("splits must contain train, validation, and test")
    split_ids = [str(image_id) for name in ("train", "validation", "test") for image_id in split[name]]
    if len(split["validation"]) < 2 or len(split["test"]) < 2:
        raise DatasetContractError("validation and test require at least two views each")
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(image_ids):
        raise DatasetContractError("splits must cover every image exactly once")
    coordinate_system = contract["coordinate_system"]
    raw_from_world = np.asarray(coordinate_system.get("raw_from_world"), dtype=np.float64)
    world_from_raw = np.asarray(coordinate_system.get("world_from_raw"), dtype=np.float64)
    if raw_from_world.shape != (4, 4) or world_from_raw.shape != (4, 4):
        raise DatasetContractError("Raw/world transforms must be 4 x 4")
    if not np.allclose(raw_from_world @ world_from_raw, np.eye(4), atol=1e-7):
        raise DatasetContractError("Raw/world round-trip failed")
    if coordinate_system.get("world_units") != "arbitrary":
        raise DatasetContractError("Stage 2A world units must remain arbitrary")
    normalization = contract["normalization"]
    normalized_from_world = np.asarray(normalization.get("normalized_from_world"), dtype=np.float64)
    world_from_normalized = np.asarray(normalization.get("world_from_normalized"), dtype=np.float64)
    if normalized_from_world.shape != (4, 4) or world_from_normalized.shape != (4, 4):
        raise DatasetContractError("normalization transforms must be 4 x 4")
    if not np.allclose(normalized_from_world @ world_from_normalized, np.eye(4), atol=1e-7):
        raise DatasetContractError("normalization round-trip failed")
    initialization = contract["initialization"]
    if not isinstance(initialization, dict) or set(initialization) != {
        "coordinate_frame",
        "asset",
        "sha256",
    }:
        raise DatasetContractError("initialization must contain coordinate_frame, asset, and sha256")
    if initialization["coordinate_frame"] not in {"world", "normalized"}:
        raise DatasetContractError("unsupported initialization coordinate frame")
    asset = initialization["asset"]
    asset_hash = initialization["sha256"]
    if (asset is None) != (asset_hash is None):
        raise DatasetContractError("initialization asset and hash must both be set or both be null")
    if asset is not None:
        path = Path(str(asset))
        if path.is_absolute() or ".." in path.parts or not str(asset):
            raise DatasetContractError("initialization asset must be a fixed project-relative path")
        if type(asset_hash) is not str or len(asset_hash) != 64 or any(
            character not in "0123456789abcdef" for character in asset_hash
        ):
            raise DatasetContractError("initialization asset hash must be lowercase SHA-256")
    if contract["dataset_hash"] != contract_hash(contract):
        raise DatasetContractError("dataset contract hash mismatch")


def with_initialization(
    contract: dict[str, Any],
    *,
    asset: str,
    asset_sha256: str,
    coordinate_frame: str = "normalized",
) -> dict[str, Any]:
    """Return a new contract that identifies one verified initialization asset."""
    validate_contract(contract)
    path = Path(asset)
    if path.is_absolute() or ".." in path.parts or not asset:
        raise DatasetContractError("initialization asset must be a fixed project-relative path")
    if coordinate_frame not in {"world", "normalized"}:
        raise DatasetContractError("initialization coordinate frame must be world or normalized")
    if len(asset_sha256) != 64 or any(character not in "0123456789abcdef" for character in asset_sha256):
        raise DatasetContractError("initialization asset hash must be lowercase SHA-256")
    updated = copy.deepcopy(contract)
    updated["initialization"] = {
        "coordinate_frame": coordinate_frame,
        "asset": path.as_posix(),
        "sha256": asset_sha256,
    }
    updated["dataset_hash"] = contract_hash(updated)
    validate_contract(updated)
    return updated


def write_contract(path: Path, contract: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
