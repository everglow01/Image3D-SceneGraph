from pathlib import Path
import json
import runpy
import sqlite3

import numpy as np
import pytest


@pytest.fixture
def analyzer(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "analyze_aliked_dense_gaps.py"))


def test_direct_support_preserves_pair_orientation_and_deduplicates(analyzer):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE two_view_geometries(pair_id, rows, cols, data)")
        first = np.column_stack((np.arange(15), np.arange(15))).astype("<u4")
        second = first.copy()
        second[[0, 2], 1] = second[[2, 0], 1]
        for left, right, matches in ((10, 20, first), (20, 30, second), (10, 40, first[:14])):
            db.execute("INSERT INTO two_view_geometries VALUES (?,?,2,?)", (
                left * analyzer["PAIR_BASE"] + right, len(matches), matches.tobytes(),
            ))
        observations = {10: {0: 100, 2: 200}, 30: {0: 200, 2: 100}}
        counts = {10: 20, 20: 20, 30: 20, 40: 20}
        support = analyzer["direct_support"](db, observations, counts)
        assert support[20] == {
            "primary_verified_neighbors": 2,
            "primary_verified_correspondences": 30,
            "candidate_2d_count": 2,
            "candidate_3d_count": 2,
            "candidate_2d3d_pair_count": 2,
            "ambiguous_2d_count": 0,
            "one_to_one_support_upper_bound": 2,
        }
        assert support[40]["candidate_2d_count"] == 0
        observations[30][0] = 300
        support = analyzer["direct_support"](db, observations, counts)
        assert support[20]["ambiguous_2d_count"] == 1
        assert support[20]["candidate_3d_count"] == 3
        counts[20] = 1
        with pytest.raises(ValueError, match="match index"):
            analyzer["direct_support"](db, observations, counts)


def test_frozen_audit_reports_connected_graph_with_missing_registration(tmp_path, analyzer):
    root = tmp_path / "dense-bruteforce"
    model = root / "positive" / "model_0_txt"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("1 PINHOLE 100 100 50 50 50 50\n")
    (model / "images.txt").write_text(
        "10 1 0 0 0 0 0 0 1 a.jpg\n1 2 100\n"
        "30 1 0 0 0 -1 0 0 1 c.jpg\n1 2 100\n"
    )
    (model / "points3D.txt").write_text("100 0 0 5 255 255 255 0.2 10 0 30 0\n")
    entry = {"path": "dense-bruteforce/positive/model_0_txt"}
    (root / "results.json").write_text(json.dumps({
        "profile": "aliked_positive_dense_bruteforce_v1",
        "cell": {"models": [entry], "primary": entry},
    }))
    (root / "selection.json").write_text(json.dumps({"selected": [
        {"path": name, "time_seconds": time} for name, time in
        (("a.jpg", 1), ("b.jpg", 5), ("c.jpg", 10), ("d.jpg", 0))
    ]}))
    (root / "positive" / "mapping.log").write_text("frozen log\n")
    (root / "positive" / "mapping.request.json").write_text("{}\n")
    with sqlite3.connect(root / "positive" / "database.db") as db:
        db.executescript(
            "CREATE TABLE images(image_id, name); CREATE TABLE keypoints(image_id, rows);"
            "CREATE TABLE matches(pair_id, rows);"
            "CREATE TABLE two_view_geometries(pair_id, rows, cols, data, config);"
        )
        db.executemany("INSERT INTO images VALUES (?,?)", [(10, "a.jpg"), (20, "b.jpg"), (30, "c.jpg"), (40, "d.jpg")])
        db.executemany("INSERT INTO keypoints VALUES (?,20)", [(i,) for i in (10, 20, 30, 40)])
        matches = np.column_stack((np.arange(15), np.arange(15))).astype("<u4")
        for left, right in ((10, 20), (20, 30), (10, 40)):
            pair = left * analyzer["PAIR_BASE"] + right
            db.execute("INSERT INTO matches VALUES (?,15)", (pair,))
            db.execute("INSERT INTO two_view_geometries VALUES (?,15,2,?,2)", (pair, matches.tobytes()))
    record = analyzer["audit"](root)
    assert record["source_unchanged"]
    assert record["mapper_eligible_graph"]["connected_component_count"] == 1
    assert record["mapper_eligible_graph"]["registered_node_count"] == 2
    prefix, gap = record["missing_intervals"]
    assert prefix["kind"] == "prefix" and prefix["unregistered_count"] == 1
    assert gap["duration_seconds"] == 9 and gap["primary_tracks_spanning_interval"] == 1
    assert gap["direct_support"]["candidate_2d_count"]["max"] == 1
    assert record["models"][0]["track_unique_image_count"]["p50"] == 2
    assert record["models"][0]["first_last_ray_angle_degrees"]["p50"] == pytest.approx(11.30993247)
    assert not record["reconstruction_started"]
    (model / "images.txt").write_text(
        "10 1 0 0 0 0 0 0 1 a.jpg\n1 2 100 1.1 2.1 100\n"
        "30 1 0 0 0 -1 0 0 1 c.jpg\n1 2 100\n"
    )
    (model / "points3D.txt").write_text("100 0 0 5 255 255 255 0.2 10 0 10 1 30 0\n")
    repeated = analyzer["audit"](root)["models"][0]
    assert repeated["tracks_with_repeated_images"] == 1
    assert repeated["track_observation_count"]["p50"] == 3
    assert repeated["track_unique_image_count"]["p50"] == 2
    (model / "points3D.txt").write_text("100 0 0 5 255 255 255 0.2 10 0 10 0 30 0\n")
    with pytest.raises(ValueError, match="duplicate sparse observation"):
        analyzer["audit"](root)
    Path(str(root / "positive" / "database.db") + "-wal").write_bytes(b"pending")
    with pytest.raises(ValueError, match="pending WAL"):
        analyzer["audit"](root)
