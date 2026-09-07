from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from scripts.analyze_sfm_small_matches import read_pair, sampson_pixels


def test_pair_indices_follow_names_not_database_id_order():
    with sqlite3.connect(":memory:") as c:
        c.executescript("""
            CREATE TABLE images(image_id INTEGER, name TEXT);
            INSERT INTO images VALUES (7, 'a'), (2, 'b'), (9, 'c');
            CREATE TABLE matches(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
        """)
        pairs = np.array([[11, 22]], dtype="<u4")
        c.execute("INSERT INTO matches VALUES (?,1,2,?)", (2 * 2147483647 + 7, pairs.tobytes()))
        assert read_pair(c, "a", "b", "matches").tolist() == [[22, 11]]
        assert read_pair(c, "b", "a", "matches").tolist() == [[11, 22]]
        assert read_pair(c, "a", "c", "matches") is None
        c.execute("INSERT INTO matches VALUES (?,0,2,NULL)", (7 * 2147483647 + 9,))
        assert read_pair(c, "a", "c", "matches").shape == (0, 2)


def test_sampson_residual_distinguishes_correct_and_wrong_correspondences():
    x0 = np.array([[0.1, 0.2], [0.2, 0.3]])
    x1 = np.array([[0.2, 0.2], [0.3, 0.4]])
    errors = sampson_pixels(x0, x1, np.eye(3), np.array([1, 0, 0]), 1000)
    assert errors[0] == pytest.approx(0)
    assert errors[1] == pytest.approx(100 / np.sqrt(2))
    with pytest.raises(ValueError, match="degenerate"):
        sampson_pixels(x0, x1, np.eye(3), np.zeros(3), 1000)


def test_reference_comparison_handles_reordered_and_empty_features(monkeypatch):
    import runpy
    from pathlib import Path

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "run_sfm_reference_audit.py"))
    xy = np.array([[1.0, 2.0], [8.0, 9.0]])
    descriptors = np.eye(2)
    record = audit["compare_features"](xy, descriptors, xy[::-1], descriptors[::-1])
    assert record["fraction_within_half_pixel"] == 1.0
    assert record["descriptor_cosine_median_on_close_points"] == 1.0
    assert audit["compare_features"](xy[:0], descriptors[:0], xy, descriptors)["status"] == "empty"
    assert audit["compare_pairs"]({(1, 2)}, {(1, 2), (2, 3)})["jaccard"] == 0.5
