"""Shared native datasets for project, Graphdeco, and Nerfstudio trainers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import sha256_file, validate_contract
from .initialization import InitializationResult, transform_points


class TrainerDatasetError(RuntimeError):
    """Raised when trainer-native data cannot preserve the frozen contract."""


def prepare_external_dataset(
    *,
    trainer: str,
    contract: dict[str, Any],
    dataset_root: Path,
    initialization: InitializationResult,
    output_dir: Path,
) -> dict[str, Any]:
    validate_contract(contract, dataset_root)
    if trainer not in {"graphdeco", "nerfstudio"}:
        raise TrainerDatasetError(f"unsupported external trainer dataset: {trainer}")
    if output_dir.exists():
        raise TrainerDatasetError(f"trainer dataset already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    if trainer == "graphdeco":
        _prepare_graphdeco(contract, dataset_root, initialization, output_dir)
    else:
        _prepare_nerfstudio(contract, dataset_root, initialization, output_dir)
    record = _integrity_record(trainer, contract, dataset_root, initialization, output_dir)
    (output_dir / "integrity.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _prepare_graphdeco(
    contract: dict[str, Any],
    dataset_root: Path,
    initialization: InitializationResult,
    output_dir: Path,
) -> None:
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir()
    sparse_dir.mkdir(parents=True)
    selected_ids = set(contract["splits"]["train"]) | set(contract["splits"]["validation"])
    selected = [image for image in contract["images"] if image["image_id"] in selected_ids]
    for image in selected:
        source = (dataset_root / image["path"]).resolve()
        os.link(source, images_dir / Path(image["path"]).name)
    _write_colmap_text(sparse_dir, selected, initialization, contract)
    validation_names = sorted(
        Path(image["path"]).name
        for image in contract["images"]
        if image["image_id"] in contract["splits"]["validation"]
    )
    (sparse_dir / "test.txt").write_text("\n".join(validation_names) + "\n", encoding="utf-8")


def _prepare_nerfstudio(
    contract: dict[str, Any],
    dataset_root: Path,
    initialization: InitializationResult,
    output_dir: Path,
) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir()
    normalized_from_world = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    frames = []
    filenames: dict[str, str] = {}
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    for image in contract["images"]:
        name = Path(image["path"]).name
        source = (dataset_root / image["path"]).resolve()
        os.link(source, images_dir / name)
        filenames[image["image_id"]] = f"images/{name}"
        world_from_camera = np.asarray(image["world_from_camera"], dtype=np.float64)
        normalized_from_opengl_camera = (
            normalized_from_world @ world_from_camera @ opencv_to_opengl
        )
        intrinsic = np.asarray(image["intrinsic"], dtype=np.float64)
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": normalized_from_opengl_camera.tolist(),
                "fl_x": float(intrinsic[0, 0]),
                "fl_y": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "w": int(image["width"]),
                "h": int(image["height"]),
            }
        )
    payload = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": frames,
        "train_filenames": [filenames[value] for value in contract["splits"]["train"]],
        "val_filenames": [filenames[value] for value in contract["splits"]["validation"]],
        "test_filenames": [filenames[value] for value in contract["splits"]["test"]],
        "ply_file_path": "sparse_pc.ply",
    }
    (output_dir / "transforms.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_rgb_ply(output_dir / "sparse_pc.ply", initialization.points, initialization.colors)


def _write_colmap_text(
    sparse_dir: Path,
    images: list[dict[str, Any]],
    initialization: InitializationResult,
    contract: dict[str, Any],
) -> None:
    camera_rows = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_rows = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, image in enumerate(images, start=1):
        intrinsic = np.asarray(image["intrinsic"], dtype=np.float64)
        camera_rows.append(
            f"{index} PINHOLE {image['width']} {image['height']} "
            f"{intrinsic[0, 0]:.17g} {intrinsic[1, 1]:.17g} "
            f"{intrinsic[0, 2]:.17g} {intrinsic[1, 2]:.17g}"
        )
        camera_from_world = np.asarray(image["camera_from_world"], dtype=np.float64)
        quat = _rotmat_to_qvec(camera_from_world[:3, :3])
        translation = camera_from_world[:3, 3]
        image_rows.extend(
            (
                f"{index} {' '.join(f'{value:.17g}' for value in quat)} "
                f"{' '.join(f'{value:.17g}' for value in translation)} {index} "
                f"{Path(image['path']).name}",
                "",
            )
        )
    world_from_normalized = np.asarray(
        contract["normalization"]["world_from_normalized"], dtype=np.float64
    )
    world_points = transform_points(initialization.points, world_from_normalized)
    point_rows = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    for index, (point, color) in enumerate(
        zip(world_points, initialization.colors, strict=True), start=1
    ):
        point_rows.append(
            f"{index} {' '.join(f'{float(value):.9g}' for value in point)} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 0"
        )
    (sparse_dir / "cameras.txt").write_text("\n".join(camera_rows) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_rows) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("\n".join(point_rows) + "\n", encoding="utf-8")


def _write_rgb_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    rows = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    rows["x"], rows["y"], rows["z"] = points.T
    rows["red"], rows["green"], rows["blue"] = colors.T
    path.write_bytes(header + rows.tobytes())


def _integrity_record(
    trainer: str,
    contract: dict[str, Any],
    dataset_root: Path,
    initialization: InitializationResult,
    output_dir: Path,
) -> dict[str, Any]:
    image_hashes = {
        image["image_id"]: sha256_file(dataset_root / image["path"])
        for image in contract["images"]
    }
    if any(
        image_hashes[image["image_id"]] != image["sha256"]
        for image in contract["images"]
    ):
        raise TrainerDatasetError("trainer dataset source image hash mismatch")
    return {
        "trainer": trainer,
        "dataset_hash": contract["dataset_hash"],
        "image_hashes": image_hashes,
        "splits": contract["splits"],
        "initialization_count": len(initialization.points),
        "initialization_points_sha256": _array_hash(initialization.points),
        "initialization_colors_sha256": _array_hash(initialization.colors),
        "native_root": str(output_dir),
    }


def _array_hash(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).T
    symmetric = np.array(
        [
            [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0, 0, 0],
            [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0, 0],
            [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0],
            [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix.trace()],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    return -qvec if qvec[0] < 0 else qvec
