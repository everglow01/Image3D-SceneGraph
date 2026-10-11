from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.gaussian.dataset import contract_hash
from scripts.build_gaussian_navigation import (
    load_sparse_initialization,
    walkable_collision_proxy,
    select_depth_view_ids,
    load_train_cameras,
    mask_to_polygons,
    point_in_polygon,
    protect_train_passages,
    remove_diagonal_pinches,
    train_trajectory_pairs,
    validate_inputs,
    validate_train_images,
)


def test_depth_selection_is_bounded_deterministic_train_only_and_pose_diverse():
    entries = []
    for index in range(800):
        pose = np.eye(4)
        pose[:3, 3] = [index % 20, index // 20, 0]
        if index % 2:
            pose[:3, :3] = np.diag([-1, 1, -1])
        entries.append({"image_id": str(index), "world_from_camera": pose.tolist()})
    train = [str(index) for index in range(600)]
    contract = {"images": entries}
    selected = select_depth_view_ids(contract, train)
    assert len(selected) == len(set(selected)) == 256
    assert set(selected) <= set(train)
    assert selected == select_depth_view_ids({"images": entries[::-1]}, train[::-1])
    assert {int(image_id) % 2 for image_id in selected} == {0, 1}
    assert min(int(image_id) // 20 for image_id in selected) == 0
    assert max(int(image_id) // 20 for image_id in selected) == 29
    assert set(select_depth_view_ids(contract, train[:12])) == set(train[:12])


def _proxy_fixture():
    mask = np.ones((12, 12), dtype=bool)
    mask[3:6, 3:6] = False
    mask[8:, 8:] = False
    basis = (np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]))
    vertices, triangles = walkable_collision_proxy(mask, np.zeros(2), 1.0, basis, height=1.0, max_triangles=50_000)
    return mask, basis, vertices, triangles


def test_walkable_proxy_preserves_holes_concavity_and_inward_guards():
    mask, basis, vertices, triangles = _proxy_fixture()
    points = vertices[triangles]
    floor = np.all(points[:, :, 2] == 0, axis=1)
    assert floor.sum() == 2 * int(mask.sum())
    centers = points.mean(axis=1)
    for center in centers[floor]:
        assert mask[int(center[1]), int(center[0])]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    assert np.all(normals[floor, 2] > 0)
    for center, normal in zip(centers[~floor], normals[~floor]):
        inside = center + normal / np.linalg.norm(normal) * 0.1
        assert mask[int(np.floor(inside[1])), int(np.floor(inside[0]))]
    edges = {}
    for triangle in triangles:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    assert all(len(uses) <= 2 for uses in edges.values())
    assert all(uses[0] == uses[1][::-1] for uses in edges.values() if len(uses) == 2)
    with pytest.raises(Exception, match="triangle budget"):
        walkable_collision_proxy(mask, np.zeros(2), 1.0, basis, height=1.0, max_triangles=10)
    with pytest.raises(Exception, match="empty"):
        walkable_collision_proxy(np.zeros((4, 4), bool), np.zeros(2), 1.0, basis, height=1.0, max_triangles=50_000)


def test_walkable_proxy_open3d_topology_and_glb_roundtrip(tmp_path):
    import open3d as o3d
    from scripts.build_gaussian_navigation import write_glb

    _, _, vertices, triangles = _proxy_fixture()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    path = tmp_path / "collision.glb"
    write_glb(path, mesh)
    decoded = o3d.io.read_triangle_mesh(str(path))
    assert len(decoded.triangles) == len(triangles)
    assert not decoded.is_self_intersecting()
    assert decoded.is_vertex_manifold()
    assert decoded.is_edge_manifold(allow_boundary_edges=True)
    assert decoded.is_orientable()


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _contract() -> dict:
    images = []
    for index in range(12):
        images.append(
            {
                "image_id": str(index),
                "path": f"images/{index}.png",
                "width": 64,
                "height": 64,
                "sha256": _sha(str(index)),
                "intrinsic": [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
                "distortion": {"state": "none", "model": "PINHOLE", "params": []},
                "camera_from_world": np.eye(4).tolist(),
                "world_from_camera": np.eye(4).tolist(),
            }
        )
    contract = {
        "schema_version": 1,
        "dataset_id": "navigation-fixture",
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
        "source": {"camera_format": "fixture", "camera_path": "cameras.json", "camera_sha256": _sha("cameras"), "image_root": "images"},
        "images": images,
        "splits": {"train": [str(index) for index in range(8)], "validation": ["8", "9"], "test": ["10", "11"]},
        "initialization": {"coordinate_frame": "world", "asset": None, "sha256": None},
    }
    contract["dataset_hash"] = contract_hash(contract)
    return contract


def test_navigation_cameras_use_rigid_normalized_units():
    contract = _contract()
    normalized_from_world = np.eye(4)
    normalized_from_world[:3, :3] /= 1_000.0
    contract["normalization"].update(
        radius_world=1_000.0,
        normalized_from_world=normalized_from_world.tolist(),
        world_from_normalized=np.linalg.inv(normalized_from_world).tolist(),
    )

    cameras = load_train_cameras(
        contract, contract["splits"]["train"], longest_edge=64
    )

    assert all(
        np.allclose(camera.camera_from_normalized[:3, :3], np.eye(3))
        for camera in cameras
    )


def test_navigation_inputs_enforce_train_only_and_provenance(tmp_path: Path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    contract = _contract()
    config_payload = {"schema_version": 1}
    config_hash = __import__("hashlib").sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config = {"effective_config": config_payload, "effective_config_hash": config_hash}
    export = {
        "coordinate_frame": "normalized",
        "world_units": "arbitrary",
        "dataset_hash": contract["dataset_hash"],
        "effective_config_hash": config_hash,
        "model_sha256": __import__("hashlib").sha256(b"model").hexdigest(),
    }

    assert validate_inputs(contract, config, export, model) == contract["splits"]["train"]

    contract["splits"]["train"].append("8")
    with pytest.raises(Exception, match="splits must cover every image exactly once"):
        validate_inputs(contract, config, export, model)


def test_sparse_floor_asset_is_hash_checked(tmp_path: Path):
    contract = _contract()
    initialization = tmp_path / "initialization"
    initialization.mkdir()
    asset = initialization / "sparse.npz"
    np.savez(asset, points=np.zeros((100, 3), dtype=np.float32))
    contract["initialization"] = {
        "coordinate_frame": "normalized",
        "asset": "initialization/sparse.npz",
        "sha256": __import__("hashlib").sha256(asset.read_bytes()).hexdigest(),
    }
    contract["dataset_hash"] = contract_hash(contract)
    contract_path = tmp_path / "dataset.json"

    points, digest = load_sparse_initialization(contract, contract_path)
    assert points.shape == (100, 3)
    assert digest == contract["initialization"]["sha256"]

    asset.write_bytes(b"tampered")
    with pytest.raises(Exception, match="hash mismatch"):
        load_sparse_initialization(contract, contract_path)


def test_train_image_validation_prefers_hash_match_over_existing_stale_path(tmp_path: Path):
    contract = _contract()
    train_ids = contract["splits"]["train"]
    preparation = tmp_path / "gaussian" / "preparation" / "train-001"
    contract_path = preparation / "dataset.json"
    for entry in contract["images"]:
        if entry["image_id"] not in train_ids:
            continue
        stale = tmp_path / entry["path"]
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"stale")
        frozen = preparation / "graphdeco-dataset" / "images" / Path(entry["path"]).name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_text(entry["image_id"], encoding="utf-8")

    layout = validate_train_images(contract, tmp_path, train_ids, contract_path)

    assert layout == "graphdeco_frozen_train_images"



def test_concave_boundary_does_not_bridge_unsupported_gap():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:3] = True
    mask[5:7, 3:7] = True

    polygons = mask_to_polygons(mask, np.zeros(2), 1.0)

    assert len(polygons) == 1
    assert point_in_polygon(np.array([1.5, 1.5]), np.asarray(polygons[0]))
    assert point_in_polygon(np.array([5.5, 5.5]), np.asarray(polygons[0]))
    assert not point_in_polygon(np.array([5.0, 2.0]), np.asarray(polygons[0]))


def test_trajectory_pairs_follow_filename_sequence_not_contract_order():
    contract = _contract()
    for index, entry in enumerate(contract["images"]):
        entry["path"] = f"images/other-{100 + index}.png"
    contract["images"][0]["path"] = "images/frame-0001.png"
    contract["images"][1]["path"] = "images/frame-0003.png"
    contract["images"][2]["path"] = "images/frame-0002.png"
    cameras = [type("Camera", (), {"image_id": str(index)})() for index in range(8)]

    pairs = train_trajectory_pairs(contract, contract["splits"]["train"], cameras)

    assert pairs[:2] == [(0, 2), (2, 1)]


def test_passage_protection_rejects_wall_and_level_change():
    floor = np.ones((20, 20), dtype=bool)
    support = floor.copy()
    obstacle = np.zeros_like(floor)
    obstacle[:, 10] = True
    pixels = np.asarray([[2, 5], [7, 5], [12, 5], [17, 5], [2, 15], [7, 15]])
    heights = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.3])

    protected, stats = protect_train_passages(
        floor,
        support,
        obstacle,
        pixels,
        heights,
        [(0, 1), (1, 2), (2, 3), (4, 5)],
        cell=0.1,
        height=1.0,
    )

    assert protected[5, 2:9].any()
    assert protected[5, 13:18].any()
    assert not protected[:, 9:12].any()
    assert stats["rejected_level_change_pairs"] == 1


def test_diagonal_pinch_removal_prevents_non_manifold_floor_vertices():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True

    cleaned, removed = remove_diagonal_pinches(mask)

    assert removed == 1
    assert cleaned.sum() == 1
