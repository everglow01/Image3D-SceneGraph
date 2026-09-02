from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.gaussian.dataset import with_initialization
from image3d_scenegraph.gaussian.initialization import (
    dense_initialization,
    graphdeco_nearest_neighbor_scales,
    sparse_initialization,
)
from test_gaussian_dataset import write_colmap_fixture
from image3d_scenegraph.gaussian.dataset import build_colmap_contract


def write_sparse(path: Path) -> None:
    path.write_text(
        "# point data\n"
        "1 0 0 2 255 0 0 0.2 1 1 2 1 3 1\n"
        "2 1 0 2 0 255 0 8.0 1 2 2 2 3 2\n"
        "3 0 1 2 0 0 255 0.3 1 3\n",
        encoding="utf-8",
    )


def test_graphdeco_scale_uses_three_nearest_neighbor_rms_without_clipping():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [100, 0, 0]],
        dtype=np.float32,
    )

    scales = graphdeco_nearest_neighbor_scales(points)

    assert scales[0] == pytest.approx(np.sqrt((1**2 + 2**2 + 3**2) / 3))
    assert scales[-1] > 50


def test_graphdeco_scale_excludes_self_for_duplicate_points():
    points = np.array(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32
    )

    scales = graphdeco_nearest_neighbor_scales(points)

    assert scales[0] == pytest.approx(np.sqrt((0**2 + 1**2 + 2**2) / 3))
    assert np.isfinite(scales).all()


def test_sparse_initialization_filters_and_hashes_deterministically(tmp_path):
    source = tmp_path / "points3D.txt"
    write_sparse(source)

    first = sparse_initialization(source, np.eye(4), max_points=2)
    second = sparse_initialization(source, np.eye(4), max_points=2)

    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["counts"] == {
        "input": 3,
        "rejected_non_finite": 0,
        "rejected_track_support": 1,
        "rejected_reprojection_error": 1,
        "accepted_before_budget": 1,
        "accepted": 1,
        "rejected_budget": 0,
    }
    assert first.points.tolist() == [[0.0, 0.0, 2.0]]
    assert np.all(first.scales > 0)


def test_sparse_initialization_default_rejects_two_view_points(tmp_path):
    source = tmp_path / "points3D.txt"
    source.write_text(
        "# point data\n"
        "1 0 0 2 255 0 0 0.2 1 1 2 1\n"
        "2 1 0 2 0 255 0 0.3 1 2 2 2 3 2\n",
        encoding="utf-8",
    )

    result = sparse_initialization(source, np.eye(4), max_points=10)

    assert result.points.tolist() == [[1.0, 0.0, 2.0]]
    assert result.diagnostics["counts"]["rejected_track_support"] == 1
    assert result.diagnostics["settings"]["min_track_length"] == 3


def test_dense_initialization_applies_support_voxel_and_budget(tmp_path, monkeypatch):
    source = tmp_path / "points.ply"
    source.write_bytes(b"source")
    points = np.array(
        [[0.0, 0.0, 2.0], [0.001, 0.0, 2.0], [0.2, 0.0, 2.0], [3.0, 3.0, 3.0]],
        dtype=np.float32,
    )
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.uint8)
    monkeypatch.setattr(
        "image3d_scenegraph.gaussian.initialization.read_rgb_ply",
        lambda _path: (points, colors),
    )
    diagnostics = tmp_path / "support.npz"
    np.savez(
        diagnostics,
        support_counts=np.array([2, 1, 3, 0]),
        confidence=np.array([1.0, 0.5, 2.0, 1.0]),
    )

    result = dense_initialization(
        source,
        np.eye(4),
        max_points=1,
        voxel_size=0.01,
        diagnostics_path=diagnostics,
        outlier_quantile=0.0,
    )

    assert len(result.points) == 1
    assert result.diagnostics["counts"]["rejected_support"] == 1
    assert result.diagnostics["counts"]["rejected_voxel"] == 1
    assert result.diagnostics["counts"]["rejected_budget"] == 1


def test_initialization_creates_new_hashed_dataset_contract(tmp_path):
    write_colmap_fixture(tmp_path)
    contract = build_colmap_contract(
        dataset_id="fixture", dataset_root=tmp_path, image_root="images", cameras_path="cameras.json"
    )

    updated = with_initialization(
        contract,
        asset="initialization/sparse.npz",
        asset_sha256="a" * 64,
    )

    assert contract["initialization"]["asset"] is None
    assert updated["initialization"]["asset"] == "initialization/sparse.npz"
    assert updated["dataset_hash"] != contract["dataset_hash"]
