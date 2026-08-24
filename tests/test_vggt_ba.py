from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
sys.path.insert(0, scripts_dir)
try:
    run_vggt_ba_sparse = importlib.import_module("scripts.run_vggt_ba_sparse")
finally:
    sys.path.remove(scripts_dir)

from image3d_scenegraph.geometry.vggt_ba import (
    MIN_SUPPORTED_OBSERVATIONS,
    VGGT_BA_FALLBACK_REASONS,
    WindowEdge,
    classify_frame_support,
    bridge_windows,
    count_frame_inliers,
    filter_train_supported_points,
    optimize_window_graph,
    read_colmap_database_image_ids,
    recovery_windows,
    select_reliable_component,
    sequential_windows,
    supported_image_ids,
    write_initial_colmap_model,
)


def test_local_ba_inlier_gate_uses_bounded_per_frame_counts():
    mask = np.array(
        [
            [True, False, True, True],
            [False, False, True, False],
        ]
    )

    assert MIN_SUPPORTED_OBSERVATIONS == 32
    assert count_frame_inliers(mask) == [3, 1]
    assert classify_frame_support([32, 0, 31, 100]) == ([0, 3], [1, 2])


def test_recovery_windows_bridge_reliable_sides_once():
    base = sequential_windows(24, window_size=8, overlap=4)
    reliable = {
        "base-0000": set(range(0, 8)),
        "base-0001": set(range(4, 12)),
        "base-0002": set(range(8, 16)),
        "base-0003": {12, 13, 14, 15, 19},
        "base-0004": set(range(19, 24)),
    }

    recovery = recovery_windows(base, reliable)

    assert len(recovery) == 1
    assert recovery[0].kind == "recovery"
    assert len(recovery[0].image_indices) <= 8
    assert len(set(recovery[0].image_indices) & reliable["base-0003"]) >= 3
    assert len(set(recovery[0].image_indices) & reliable["base-0004"]) >= 3
    assert recovery == recovery_windows(base, reliable)


def test_forced_recovery_uses_distinct_evidence_from_both_sides():
    base = sequential_windows(20, window_size=8, overlap=4)
    reliable = {
        "base-0000": set(range(0, 8)),
        "base-0001": set(range(4, 12)),
        "base-0002": set(range(8, 16)),
        "base-0003": set(range(12, 20)),
    }

    recovery = recovery_windows(
        base,
        reliable,
        forced_pairs={("base-0002", "base-0003")},
    )

    assert len(recovery) == 1
    members = set(recovery[0].image_indices)
    assert len(members & (reliable["base-0002"] - reliable["base-0003"])) >= 3
    assert len(members & (reliable["base-0003"] - reliable["base-0002"])) >= 3
    assert len(members) <= 8


def test_reliable_component_requires_count_rate_and_temporal_coverage():
    windows = {
        "a": {index: {} for index in range(0, 6)},
        "b": {index: {} for index in range(4, 12)},
        "c": {index: {} for index in range(12, 20)},
    }
    identity = np.eye(4)
    edges = [WindowEdge("a", "b", identity, (4, 5, 6), 0.0, 0.0)]

    selected, diagnostics = select_reliable_component(windows, edges, 20)

    assert selected == []
    assert diagnostics["components"][0]["reliable_camera_count"] == 12
    assert diagnostics["components"][0]["temporal_coverage"] == 0.6

    edges.append(WindowEdge("b", "c", identity, (8, 9, 10), 0.0, 0.0))
    selected, diagnostics = select_reliable_component(windows, edges, 20)
    assert selected == ["a", "b", "c"]
    assert diagnostics["selected"]["reliable_camera_rate"] == 1.0


def test_colmap_fallback_reason_allowlist_is_narrow():
    assert VGGT_BA_FALLBACK_REASONS == {
        "vggt_graph_unusable_after_recovery",
        "vggt_seed_geometry_insufficient",
        "vggt_registration_gate_failed",
    }
    assert "cuda_oom" not in VGGT_BA_FALLBACK_REASONS
    assert "unexpected_error" not in VGGT_BA_FALLBACK_REASONS


def test_colmap_database_image_ids_are_used_for_partial_models(tmp_path):
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO images VALUES (?, ?)",
            [(17, "a.jpg"), (3, "b.jpg")],
        )

    assert read_colmap_database_image_ids(database_path) == {"a.jpg": 17, "b.jpg": 3}


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

    partial = write_initial_colmap_model(
        tmp_path / "partial-model",
        names,
        {0: cameras[0], 2: cameras[2]},
        sizes,
        image_ids_by_name={"0.jpg": 7, "1.jpg": 8, "2.jpg": 11},
    )
    partial_images = (tmp_path / "partial-model" / "images.txt").read_text()
    assert partial["camera_count"] == 2
    assert "7 1 0 0 0" in partial_images
    assert "11 1 0 0 0" in partial_images
    assert "8 1 0 0 0" not in partial_images

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


def test_vggt_ba_seeded_model_uses_shared_incremental_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = {
        "profile": "video_keyframes_standard_v2",
        "selected": [
            {
                "path": f"frames/selected/frame_{index:03d}.jpg",
                "time_seconds": float(index),
                "pts": index,
            }
            for index in range(12)
        ],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    initial_model = tmp_path / "seeded"
    recovered_model = tmp_path / "recovered"
    call = {}

    def fake_recovery(**kwargs):
        call.update(kwargs)
        return recovered_model, {"status": "recovered", "rounds": [{}]}, ["recovery-log"]

    monkeypatch.setattr(
        run_vggt_ba_sparse, "recover_video_registration", fake_recovery
    )
    logs = []
    model, diagnostics, fallback_registration = (
        run_vggt_ba_sparse.apply_video_registration_recovery(
            colmap="colmap",
            database_path=tmp_path / "database.db",
            image_dir=tmp_path / "images",
            final_model=initial_model,
            video_selection=selection,
            video_source=source,
            selection_path=selection_path,
            diagnostics_dir=tmp_path / "diagnostics",
            work_dir=tmp_path / "work",
            num_threads=8,
            progress_file=tmp_path / "progress.json",
            fallback_applied=False,
            command_logs=logs,
        )
    )

    assert model == recovered_model
    assert diagnostics["status"] == "recovered"
    assert fallback_registration is None
    assert call["initial_model"] == initial_model
    assert call["database_path"] == tmp_path / "database.db"
    assert logs == ["recovery-log"]


def test_vggt_ba_colmap_fallback_rechecks_recovered_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = {
        "profile": "video_keyframes_standard_v2",
        "selected": [
            {
                "path": f"frames/selected/frame_{index:03d}.jpg",
                "time_seconds": float(index),
                "pts": index,
            }
            for index in range(12)
        ],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    recovered_model = tmp_path / "fallback-recovered"
    inspected = {}

    monkeypatch.setattr(
        run_vggt_ba_sparse,
        "recover_video_registration",
        lambda **_kwargs: (
            recovered_model,
            {"status": "recovered", "rounds": [{}]},
            [],
        ),
    )
    monkeypatch.setattr(
        run_vggt_ba_sparse,
        "read_colmap_database_image_ids",
        lambda _path: {f"frame_{index:03d}.jpg": index + 100 for index in range(12)},
    )

    def fake_inspect(_colmap, model, _text, indices, count, _logs):
        inspected.update(model=model, indices=indices, count=count)
        return {"usable": True, "supported_camera_count": 12}

    monkeypatch.setattr(
        run_vggt_ba_sparse, "inspect_model_registration", fake_inspect
    )
    model, diagnostics, fallback_registration = (
        run_vggt_ba_sparse.apply_video_registration_recovery(
            colmap="colmap",
            database_path=tmp_path / "database.db",
            image_dir=tmp_path / "images",
            final_model=tmp_path / "fallback-initial",
            video_selection=selection,
            video_source=source,
            selection_path=selection_path,
            diagnostics_dir=tmp_path / "diagnostics",
            work_dir=tmp_path / "work",
            num_threads=8,
            progress_file=None,
            fallback_applied=True,
            command_logs=[],
        )
    )

    assert model == recovered_model
    assert diagnostics["status"] == "recovered"
    assert fallback_registration["usable"] is True
    assert inspected["model"] == recovered_model
    assert inspected["count"] == 12
    assert inspected["indices"] == {index + 100: index for index in range(12)}
