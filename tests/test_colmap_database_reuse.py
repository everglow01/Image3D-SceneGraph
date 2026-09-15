from __future__ import annotations

import json
import sqlite3

import pytest

from image3d_scenegraph.file_integrity import sha256_file
from scripts.run_colmap_sparse import _database_table_digests, reuse_feature_database


def fixture(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("INSERT INTO images VALUES (1,'frame.jpg')")
        connection.execute("CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, data BLOB)")
        connection.execute("INSERT INTO descriptors VALUES (1,?)", (bytes(range(256)),))
        connection.execute("CREATE TABLE frame_data(frame_id INTEGER, sensor_id INTEGER, data BLOB, PRIMARY KEY(frame_id,sensor_id)) WITHOUT ROWID")
        connection.execute("INSERT INTO frame_data VALUES (2,1,?)", (b"frame-metadata",))
    frontend = {"profile": "sfm_frontend_contract_v1", "feature": {"profile": "sift_v1"}, "v2_mapper_seed_count": 1000}
    path = tmp_path / "frontend.json"
    path.write_text(json.dumps(frontend))
    requested = {**frontend, "v2_mapper_seed_count": 2500}
    return source, path, requested


@pytest.mark.parametrize("wal_mode", [False, True])
def test_reuse_copies_all_tables_without_modifying_source(tmp_path, wal_mode):
    source, frontend, requested = fixture(tmp_path)
    if wal_mode:
        connection = sqlite3.connect(source)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
    original = sha256_file(source)
    destination = tmp_path / "copy.db"
    record = reuse_feature_database(
        source, destination, source_frontend=frontend, requested_frontend=requested,
        expected_sha256=original, image_names={"frame.jpg"},
    )
    assert sha256_file(source) == original
    assert destination.stat().st_ino != source.stat().st_ino
    assert record["table_digests"] == _database_table_digests(source) == _database_table_digests(destination)
    assert set(record["table_digests"]) == {"images", "descriptors", "frame_data"}
    assert record["feature_extraction"] == record["feature_matching"] == "reused"
    with pytest.raises(ValueError, match="new destination"):
        reuse_feature_database(source, destination, source_frontend=frontend, requested_frontend=requested,
                              expected_sha256=original, image_names={"frame.jpg"})


@pytest.mark.parametrize("failure", ["hash", "frontend", "images", "wal"])
def test_reuse_rejects_unfrozen_or_mismatched_sources(tmp_path, failure):
    source, frontend, requested = fixture(tmp_path)
    expected = sha256_file(source)
    names = {"frame.jpg"}
    if failure == "hash":
        expected = "0" * 64
    elif failure == "frontend":
        requested["feature"] = {"profile": "aliked_n16rot_v1"}
    elif failure == "images":
        names = {"other.jpg"}
    elif failure == "wal":
        source.with_name(source.name + "-wal").write_bytes(b"active")
    with pytest.raises(ValueError):
        reuse_feature_database(source, tmp_path / "copy.db", source_frontend=frontend,
                              requested_frontend=requested, expected_sha256=expected, image_names=names)
    assert not (tmp_path / "copy.db").exists()
