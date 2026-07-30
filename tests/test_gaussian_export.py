from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.dataset import contract_hash, sha256_file
from image3d_scenegraph.gaussian.export import (
    GaussianExportError,
    PLY_FIELDS,
    _camera_path,
    _model_rows,
    _write_binary_ply,
    read_gaussian_ply,
    write_deterministic_zip,
)
from image3d_scenegraph.gaussian.model import GaussianModel


def model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.full((2,), 0.1),
    )


def contract() -> dict:
    images = []
    for index in range(12):
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = [np.cos(index), np.sin(index), 0.0]
        camera_from_world = np.linalg.inv(world_from_camera)
        images.append(
            {
                "image_id": str(index),
                "path": f"images/{index}.png",
                "width": 64,
                "height": 64,
                "sha256": "a" * 64,
                "intrinsic": [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
                "distortion": {"state": "none", "model": "PINHOLE", "params": []},
                "camera_from_world": camera_from_world.tolist(),
                "world_from_camera": world_from_camera.tolist(),
            }
        )
    value = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "coordinate_system": {
            "camera_convention": "opencv",
            "camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "world_frame": "raw",
            "world_units": "arbitrary",
            "raw_from_world": np.eye(4).tolist(),
            "world_from_raw": np.eye(4).tolist(),
        },
        "normalization": {
            "method": "fixture",
            "center_world": [0.0, 0.0, 0.0],
            "radius_world": 1.0,
            "normalized_from_world": np.eye(4).tolist(),
            "world_from_normalized": np.eye(4).tolist(),
        },
        "source": {"camera_format": "fixture", "camera_path": "cameras.json", "camera_sha256": "b" * 64, "image_root": "images"},
        "images": images,
        "splits": {
            "train": [str(index) for index in range(8)],
            "validation": ["8", "9"],
            "test": ["10", "11"],
        },
        "initialization": {"coordinate_frame": "world", "asset": None, "sha256": None},
    }
    value["dataset_hash"] = contract_hash(value)
    return value


def test_canonical_ply_round_trips_all_owned_attributes(tmp_path):
    gaussian = model()
    rows = _model_rows(gaussian)
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"

    _write_binary_ply(first, rows)
    _write_binary_ply(second, rows)
    decoded = read_gaussian_ply(first)

    assert first.read_bytes() == second.read_bytes()
    assert tuple(decoded) == PLY_FIELDS
    assert np.array_equal(decoded["x"], gaussian.means[:, 0].detach().numpy())
    assert np.array_equal(decoded["f_dc_0"], gaussian.sh_coeffs[:, 0, 0].detach().numpy())
    assert np.array_equal(decoded["opacity"], gaussian.opacity_logits.detach().numpy())
    assert np.array_equal(decoded["scale_2"], gaussian.log_scales[:, 2].detach().numpy())
    assert np.array_equal(decoded["rot_0"], np.ones(2, dtype=np.float32))


def test_deterministic_bundle_has_safe_sorted_entries(tmp_path):
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    write_deterministic_zip(first, {"z/b.txt": source_b, "a.txt": source_a})
    write_deterministic_zip(second, {"a.txt": source_a, "z/b.txt": source_b})

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt", "z/b.txt"]
    with pytest.raises(GaussianExportError, match="unsafe bundle"):
        write_deterministic_zip(tmp_path / "bad.zip", {"../escape": source_a})


def test_camera_path_stays_in_normalized_trusted_bound():
    value = contract()
    path = _camera_path(value)
    assert len(path["keyframes"]) == 2

    value["images"][8]["world_from_camera"][0][3] = 3.0
    with pytest.raises(GaussianExportError, match="trusted"):
        _camera_path(value)
