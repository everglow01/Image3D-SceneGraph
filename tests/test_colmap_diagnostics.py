from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.geometry.colmap_diagnostics import (
    MAX_IMAGE_ID,
    ColmapDiagnosticsError,
    export_colmap_diagnostics,
)


def test_export_colmap_diagnostics_writes_final_sharded_frontend_data(tmp_path: Path) -> None:
    job = _fixture_job(tmp_path)
    output = job / "diagnostics" / "sfm"

    manifest_path, metrics = export_colmap_diagnostics(
        job_dir=job,
        database_path=job / "colmap" / "database.db",
        source_image_root=job / "frames" / "selected",
        dataset_contract_path=job / "dataset.json",
        output_dir=output,
        matcher="sequential",
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "sfm_frontend_diagnostics_v1"
    assert manifest["counts"] == {
        "images": 3,
        "registered_images": 2,
        "keypoints": 5,
        "pairs": 2,
        "candidate_matches": 2,
        "inliers": 1,
        "outliers": 1,
    }
    assert metrics["sfm_diagnostics_status"] == "available"
    assert metrics["sfm_diagnostics_keypoint_count"] == 5

    images = {image["colmap_image_id"]: image for image in manifest["images"]}
    assert images[1]["registered"] is True
    assert images[1]["split"] == "train"
    assert images[1]["source_time_seconds"] == pytest.approx(0.1)
    assert images[1]["center_normalized"] == pytest.approx([0.0, 0.0, 0.0])
    assert images[1]["forward_normalized"] == pytest.approx([0.0, 0.0, 1.0])
    assert images[1]["up_normalized"] == pytest.approx([0.0, -1.0, 0.0])
    assert images[2]["center_normalized"] == pytest.approx([0.5, 0.0, 0.0])
    assert images[3]["registered"] is False
    assert images[3]["split"] is None
    assert "center_normalized" not in images[3]

    run = manifest["runs"][0]
    assert run["matcher"]["name"] == "sequential"
    assert run["detector"]["keypoint_fields"] == ["x", "y"]
    feature_index = _read_gzip_json(job / run["feature_index_path"])
    feature_entry = next(item for item in feature_index["images"] if item["image_id"] == 1)
    feature_shard = _read_gzip_json(job / feature_entry["detail_shard"])
    assert feature_shard["images"]["1"]["points"] == [[10.12, 20.46], [30.0, 40.0]]

    pair_index = _read_gzip_json(job / run["pair_index_path"])
    direct = next(item for item in pair_index["pairs"] if item["pair_key"] == "1-2")
    assert direct["candidate_match_count"] == 2
    assert direct["inlier_count"] == 1
    pair_shard = _read_gzip_json(job / direct["detail_shard"])
    assert pair_shard["pairs"]["1-2"] == {
        "inliers": [[0, 0]],
        "outliers": [[1, 1]],
    }
    tested_empty = next(item for item in pair_index["pairs"] if item["pair_key"] == "2-3")
    assert tested_empty["candidate_match_count"] == 0
    assert tested_empty["inlier_count"] == 0

    paths = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert not any("descriptor" in path or path.endswith("database.db") for path in paths)

    first_hashes = _tree_hashes(output)
    shutil.rmtree(output)
    export_colmap_diagnostics(
        job_dir=job,
        database_path=job / "colmap" / "database.db",
        source_image_root=job / "frames" / "selected",
        dataset_contract_path=job / "dataset.json",
        output_dir=output,
        matcher="sequential",
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )
    assert _tree_hashes(output) == first_hashes


def test_export_colmap_diagnostics_rejects_verified_match_not_in_candidates(
    tmp_path: Path,
) -> None:
    job = _fixture_job(tmp_path)
    database = sqlite3.connect(job / "colmap" / "database.db")
    pair_id = 1 * MAX_IMAGE_ID + 2
    invalid = np.asarray([[0, 1]], dtype="<u4")
    database.execute(
        "UPDATE two_view_geometries SET rows=?, data=? WHERE pair_id=?",
        (1, invalid.tobytes(), pair_id),
    )
    database.commit()
    database.close()
    output = job / "diagnostics" / "sfm"

    with pytest.raises(ColmapDiagnosticsError, match="not a subset"):
        export_colmap_diagnostics(
            job_dir=job,
            database_path=job / "colmap" / "database.db",
            source_image_root=job / "frames" / "selected",
            dataset_contract_path=job / "dataset.json",
            output_dir=output,
            matcher="sequential",
            colmap_build="COLMAP 4.0.0",
            video_selection_path=job / "frames" / "selection.json",
        )

    assert not output.exists()
    assert not (job / "diagnostics" / ".sfm.tmp").exists()


def _fixture_job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    image_root = job / "frames" / "selected"
    image_root.mkdir(parents=True)
    names = ["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"]
    selected = []
    for index, name in enumerate(names, start=1):
        payload = f"image-{index}".encode()
        (image_root / name).write_bytes(payload)
        selected.append(
            {
                "path": f"frames/selected/{name}",
                "time_seconds": index / 10,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "selection_reason": "base",
            }
        )
    (job / "frames" / "selection.json").write_text(
        json.dumps({"selected": selected}), encoding="utf-8"
    )
    (job / "diagnostics").mkdir()
    (job / "colmap").mkdir()
    _write_database(job / "colmap" / "database.db")
    identity = np.eye(4).tolist()
    second = np.eye(4)
    second[0, 3] = 1.0
    contract = {
        "dataset_hash": "a" * 64,
        "coordinate_system": {"camera_convention": "opencv"},
        "normalization": {
            "normalized_from_world": [
                [0.5, 0, 0, 0],
                [0, 0.5, 0, 0],
                [0, 0, 0.5, 0],
                [0, 0, 0, 1],
            ]
        },
        "images": [
            {
                "image_id": "1",
                "path": "undistorted/frame-1.jpg",
                "width": 100,
                "height": 80,
                "world_from_camera": identity,
                "intrinsic": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
            },
            {
                "image_id": "2",
                "path": "undistorted/frame-2.jpg",
                "width": 100,
                "height": 80,
                "world_from_camera": second.tolist(),
                "intrinsic": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
            },
        ],
        "splits": {"train": ["1"], "validation": ["2"], "test": []},
    }
    (job / "dataset.json").write_text(json.dumps(contract), encoding="utf-8")
    return job


def _write_database(path: Path) -> None:
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE cameras (
          camera_id INTEGER PRIMARY KEY, model INTEGER NOT NULL, width INTEGER NOT NULL,
          height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE images (
          image_id INTEGER PRIMARY KEY, name TEXT NOT NULL, camera_id INTEGER NOT NULL
        );
        CREATE TABLE keypoints (
          image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE matches (
          pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE two_view_geometries (
          pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
          config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
        );
        """
    )
    database.execute("INSERT INTO cameras VALUES (1, 4, 100, 80, NULL, 0)")
    for image_id in (1, 2, 3):
        database.execute(
            "INSERT INTO images VALUES (?, ?, 1)", (image_id, f"frame-{image_id}.jpg")
        )
    keypoints = {
        1: [[10.123, 20.456, 1, 0, 0, 1], [30, 40, 1, 0, 0, 1]],
        2: [[11, 21, 1, 0, 0, 1], [31, 41, 1, 0, 0, 1]],
        3: [[12, 22, 1, 0, 0, 1]],
    }
    for image_id, values in keypoints.items():
        array = np.asarray(values, dtype="<f4")
        database.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, array.shape[0], array.shape[1], array.tobytes()),
        )
    pair_12 = 1 * MAX_IMAGE_ID + 2
    candidates = np.asarray([[0, 0], [1, 1]], dtype="<u4")
    inliers = np.asarray([[0, 0]], dtype="<u4")
    database.execute(
        "INSERT INTO matches VALUES (?, ?, 2, ?)",
        (pair_12, len(candidates), candidates.tobytes()),
    )
    database.execute(
        "INSERT INTO two_view_geometries VALUES (?, ?, 2, ?, 3, NULL, NULL, NULL, NULL, NULL)",
        (pair_12, len(inliers), inliers.tobytes()),
    )
    pair_23 = 2 * MAX_IMAGE_ID + 3
    database.execute("INSERT INTO matches VALUES (?, 0, 2, NULL)", (pair_23,))
    database.execute(
        "INSERT INTO two_view_geometries VALUES (?, 0, 2, NULL, 0, NULL, NULL, NULL, NULL, NULL)",
        (pair_23,),
    )
    database.commit()
    database.close()


def _read_gzip_json(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
