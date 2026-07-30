from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.gaussian.dataset import (
    DatasetContractError,
    build_colmap_contract,
    contract_hash,
    deterministic_spatial_split,
    validate_contract,
)


def make_images(count: int = 20) -> list[dict]:
    images = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = [np.cos(angle), np.sin(angle), index / count]
        images.append({"image_id": f"view-{index:02d}", "world_from_camera": world_from_camera.tolist()})
    return images


def test_spatial_split_is_deterministic_disjoint_and_not_upload_order():
    images = make_images()

    split = deterministic_spatial_split(images)

    assert split == deterministic_spatial_split(list(reversed(images)))
    assert len(split["train"]) == 16
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert split["test"] != ["view-18", "view-19"]


def test_spatial_split_rejects_too_few_registered_images():
    with pytest.raises(DatasetContractError, match="at least 12"):
        deterministic_spatial_split(make_images(11))


def write_colmap_fixture(root: Path, count: int = 12) -> None:
    (root / "images").mkdir(parents=True)
    images = []
    for index in range(count):
        name = f"frame_{index:03d}.jpg"
        (root / "images" / name).write_bytes(f"image-{index}".encode())
        angle = 2 * np.pi * index / count
        images.append(
            {
                "image_id": index + 1,
                "qvec": [1.0, 0.0, 0.0, 0.0],
                "tvec": [-float(np.cos(angle)), -float(np.sin(angle)), -0.1 * index],
                "camera_id": 1,
                "name": name,
            }
        )
    payload = {
        "coordinate_system": "colmap_world",
        "cameras": [
            {
                "camera_id": 1,
                "model": "SIMPLE_RADIAL",
                "width": 640,
                "height": 480,
                "params": [500.0, 320.0, 240.0, 0.01],
            }
        ],
        "images": images,
    }
    (root / "cameras.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_colmap_contract_round_trips_and_hashes_sources(tmp_path):
    write_colmap_fixture(tmp_path)

    contract = build_colmap_contract(
        dataset_id="fixture",
        dataset_root=tmp_path,
        image_root="images",
        cameras_path="cameras.json",
    )

    assert contract["schema_version"] == 1
    assert contract["coordinate_system"]["camera_convention"] == "opencv"
    assert contract["coordinate_system"]["world_units"] == "arbitrary"
    assert np.allclose(
        np.asarray(contract["coordinate_system"]["raw_from_world"])
        @ np.asarray(contract["coordinate_system"]["world_from_raw"]),
        np.eye(4),
    )
    assert len(contract["images"]) == 12
    assert len(contract["splits"]["validation"]) == 2
    assert len(contract["splits"]["test"]) == 2
    assert contract["dataset_hash"] == contract_hash(contract)
    validate_contract(contract, tmp_path)


def test_contract_rejects_pose_split_and_source_tampering(tmp_path):
    write_colmap_fixture(tmp_path)
    contract = build_colmap_contract(
        dataset_id="fixture",
        dataset_root=tmp_path,
        image_root="images",
        cameras_path="cameras.json",
    )

    bad_pose = copy.deepcopy(contract)
    bad_pose["images"][0]["world_from_camera"][0][3] += 1.0
    bad_pose["dataset_hash"] = contract_hash(bad_pose)
    with pytest.raises(DatasetContractError, match="camera round-trip"):
        validate_contract(bad_pose)

    bad_split = copy.deepcopy(contract)
    bad_split["splits"]["test"].append(bad_split["splits"]["train"][0])
    bad_split["dataset_hash"] = contract_hash(bad_split)
    with pytest.raises(DatasetContractError, match="exactly once"):
        validate_contract(bad_split)

    (tmp_path / contract["images"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(DatasetContractError, match="image hash mismatch"):
        validate_contract(contract, tmp_path)
