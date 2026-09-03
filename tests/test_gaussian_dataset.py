from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.gaussian.dataset import (
    DatasetContractError,
    build_colmap_contract,
    camera_from_normalized_transform,
    contract_hash,
    deterministic_spatial_split,
    deterministic_temporal_group_split,
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


def test_video_split_keeps_two_second_groups_disjoint():
    images = make_images(20)
    timestamps = {}
    for index, image in enumerate(images):
        name = f"frame_{index:03d}.jpg"
        image["path"] = f"images/{name}"
        timestamps[name] = index * 1.1

    split = deterministic_temporal_group_split(images, timestamps)

    owners = {image_id: name for name, ids in split.items() for image_id in ids}
    assert set(owners) == {image["image_id"] for image in images}
    groups = {}
    for image in images:
        name = Path(image["path"]).name
        groups.setdefault(int(timestamps[name] // 2), set()).add(owners[image["image_id"]])
    assert all(len(destinations) == 1 for destinations in groups.values())
    assert len(split["validation"]) >= 2
    assert len(split["test"]) >= 2


def make_video_images(times: list[float]) -> tuple[list[dict], dict[str, float]]:
    images = []
    timestamps = {}
    for index, time in enumerate(times):
        name = f"frame_{index:03d}.jpg"
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = [float(index), 0.0, 0.0]
        images.append(
            {
                "image_id": f"view-{index:02d}",
                "path": f"images/{name}",
                "world_from_camera": world_from_camera.tolist(),
            }
        )
        timestamps[name] = time
    return images, timestamps


def test_video_split_keeps_gap_adjacent_groups_in_train():
    # 5s registration hole between frame 9 (t=9) and frame 10 (t=14).
    times = [float(index) for index in range(10)] + [
        float(index) + 4.0 for index in range(10, 20)
    ]
    images, timestamps = make_video_images(times)

    split = deterministic_temporal_group_split(images, timestamps)

    owners = {image_id: part for part, ids in split.items() for image_id in ids}
    assert set(owners) == {image["image_id"] for image in images}
    protected = {int(timestamps[f"frame_{index:03d}.jpg"] // 2.0) for index in (9, 10)}
    for image in images:
        name = Path(image["path"]).name
        if int(timestamps[name] // 2.0) in protected:
            assert owners[image["image_id"]] == "train"
    assert len(split["validation"]) >= 2
    assert len(split["test"]) >= 2


def test_video_split_shrinks_holdout_when_many_groups_protected():
    # 11 two-to-six-frame blocks separated by holes: only 6 of 26 groups
    # stay eligible, so the held-out budget shrinks from 3 groups to 2.
    times: list[float] = []
    group = 0
    for size in [6] + [2] * 10:
        for _ in range(size):
            times.append(group * 2.0 + 0.5)
            group += 1
        group += 2
    images, timestamps = make_video_images(times)

    split = deterministic_temporal_group_split(images, timestamps)

    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    assert len(split["train"]) == 22
    ordered = sorted(timestamps.values())
    protected = {
        int(time // 2.0)
        for left, right in zip(ordered, ordered[1:])
        if right - left > 2.0
        for time in (left, right)
    }
    owners = {image_id: part for part, ids in split.items() for image_id in ids}
    for image in images:
        name = Path(image["path"]).name
        if int(timestamps[name] // 2.0) in protected:
            assert owners[image["image_id"]] == "train"


def test_video_split_rejects_when_gap_avoidance_starves_holdout():
    # Six two-frame blocks separated by holes leave only two eligible groups,
    # which cannot fund two validation and two test groups.
    times: list[float] = []
    group = 0
    for _ in range(6):
        for _ in range(2):
            times.append(group * 2.0 + 0.5)
            group += 1
        group += 2
    images, timestamps = make_video_images(times)

    with pytest.raises(DatasetContractError, match="too few temporal groups"):
        deterministic_temporal_group_split(images, timestamps)


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


def test_camera_transform_uses_normalized_camera_units():
    radius = 1_297_235_666.7843318
    center = np.array([4.0e8, -8.0e8, 2.0e8])
    normalized_center = np.array([0.001, -0.002, 0.0005])
    camera_center = center + radius * normalized_center
    normalized_from_world = np.eye(4)
    normalized_from_world[:3, :3] /= radius
    normalized_from_world[:3, 3] = -center / radius
    camera_from_world = np.eye(4)
    camera_from_world[:3, 3] = -camera_center

    camera = camera_from_normalized_transform(
        camera_from_world,
        {"normalized_from_world": normalized_from_world.tolist()},
    )

    assert np.allclose(camera[:3, :3], np.eye(3))
    assert np.allclose(np.linalg.inv(camera)[:3, 3], normalized_center)
    assert np.allclose(np.linalg.norm(camera[:3, :3], axis=0), 1.0)
    legacy = camera_from_world @ np.linalg.inv(normalized_from_world)
    assert np.allclose(np.linalg.norm(legacy[:3, :3], axis=0), radius)


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
