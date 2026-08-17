from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from image3d_scenegraph.geometry.vggt_ba import (
    WindowEdge,
    bridge_windows,
    filter_train_supported_points,
    optimize_window_graph,
    sequential_windows,
    supported_image_ids,
    write_initial_colmap_model,
)


def test_sequential_and_bridge_windows_are_bounded_and_deterministic():
    base = sequential_windows(20, window_size=8, overlap=4)
    assert [window.image_indices for window in base] == [
        tuple(range(0, 8)),
        tuple(range(4, 12)),
        tuple(range(8, 16)),
        tuple(range(12, 20)),
    ]
    descriptors = np.eye(20, dtype=np.float64)
    descriptors[0] = descriptors[16]
    first = bridge_windows(
        descriptors,
        base,
        window_size=8,
        minimum_index_gap=12,
        maximum_bridges=2,
    )
    second = bridge_windows(
        descriptors,
        base,
        window_size=8,
        minimum_index_gap=12,
        maximum_bridges=2,
    )
    assert first == second
    assert all(len(window.image_indices) <= 8 for window in first)


def test_window_pose_graph_recovers_connected_similarity_chain():
    first = np.eye(4)
    first[:3, :3] *= 2.0
    first[:3, 3] = [1, 0, 0]
    second = np.eye(4)
    second[:3, :3] *= 0.75
    second[:3, 3] = [0, 2, 0]
    edges = [
        WindowEdge("a", "b", first, (1, 2, 3), 0.01, 0.5),
        WindowEdge("b", "c", second, (4, 5, 6), 0.01, 0.5),
    ]

    transforms, diagnostics = optimize_window_graph(["a", "b", "c"], edges)

    assert set(transforms) == {"a", "b", "c"}
    assert diagnostics["connected"] is True
    assert diagnostics["final_cost"] <= diagnostics["initial_cost"] + 1e-12
    assert np.allclose(np.linalg.inv(transforms["b"]) @ transforms["a"], first)
    assert np.allclose(np.linalg.inv(transforms["c"]) @ transforms["b"], second)


def test_initial_model_and_supported_camera_parser(tmp_path):
    cameras = {}
    sizes = {}
    names = []
    for index in range(3):
        cameras[index] = {
            "extrinsic": np.column_stack((np.eye(3), np.array([-index, 0, 0]))),
            "intrinsic": np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]]),
        }
        sizes[index] = (100, 80)
        names.append(f"{index}.jpg")
    record = write_initial_colmap_model(tmp_path / "model", names, cameras, sizes)
    assert record["camera_model"] == "OPENCV"
    assert (tmp_path / "model" / "images.txt").is_file()

    observations = " ".join(f"{index} {index} {index + 1}" for index in range(32))
    (tmp_path / "images.txt").write_text(
        "# header\n1 1 0 0 0 0 0 0 1 a.jpg\n"
        + observations
        + "\n2 1 0 0 0 0 0 0 1 b.jpg\n\n",
        encoding="utf-8",
    )
    assert supported_image_ids(tmp_path / "images.txt", minimum_observations=32) == {1}


def test_train_supported_points_are_recolored_only_from_train(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image_root / "train-a.png")
    Image.new("RGB", (4, 4), (50, 60, 70)).save(image_root / "train-b.png")
    Image.new("RGB", (4, 4), (200, 210, 220)).save(image_root / "heldout.png")
    images_path = tmp_path / "images.txt"
    images_path.write_text(
        "# header\n"
        "1 1 0 0 0 0 0 0 1 train-a.png\n1 1 9\n"
        "2 1 0 0 0 0 0 0 1 train-b.png\n1 1 9\n"
        "3 1 0 0 0 0 0 0 1 heldout.png\n1 1 9\n",
        encoding="utf-8",
    )
    points_path = tmp_path / "points3D.txt"
    points_path.write_text(
        "# header\n9 0 0 0 255 255 255 1 1 0 2 0 3 0\n",
        encoding="utf-8",
    )

    diagnostics = filter_train_supported_points(
        points_path,
        images_path,
        image_root,
        {1, 2},
        tmp_path / "filtered.txt",
    )

    row = (tmp_path / "filtered.txt").read_text().splitlines()[1].split()
    assert row[4:7] == ["30", "40", "50"]
    assert diagnostics["counts"]["mixed_track_points"] == 1
