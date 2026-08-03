from __future__ import annotations

import struct

from scripts.run_colmap_sparse import find_largest_sparse_model, read_sparse_model_counts


def _write_binary_count(path, count: int) -> None:
    path.write_bytes(struct.pack("<Q", count))


def test_sparse_model_selection_prefers_registered_images_then_points(tmp_path):
    first = tmp_path / "0"
    second = tmp_path / "1"
    third = tmp_path / "2"
    for path in (first, second, third):
        path.mkdir()

    _write_binary_count(first / "images.bin", 12)
    _write_binary_count(first / "points3D.bin", 5_000)
    _write_binary_count(second / "images.bin", 15)
    _write_binary_count(second / "points3D.bin", 1_000)
    _write_binary_count(third / "images.bin", 15)
    _write_binary_count(third / "points3D.bin", 2_000)

    assert read_sparse_model_counts(first) == (12, 5_000)
    assert find_largest_sparse_model(tmp_path) == (third, 15, 2_000)


def test_sparse_model_counts_support_text_models(tmp_path):
    model = tmp_path / "0"
    model.mkdir()
    (model / "images.txt").write_text(
        "# images\n1 pose.jpg\n1 2 3\n2 pose.jpg\n4 5 6\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "# points\n1 0 0 0 1 2 3 0.1 1 0\n2 0 0 0 1 2 3 0.1 1 0\n",
        encoding="utf-8",
    )

    assert read_sparse_model_counts(model) == (2, 2)
