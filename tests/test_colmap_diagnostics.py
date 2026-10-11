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
    _run_id,
    export_colmap_diagnostics,
)


SIFT_FEATURE = {
    "profile": "sift_v1",
    "extractor": "SIFT",
    "descriptor": "SIFT",
    "local_matcher_profile": "bruteforce",
    "local_matcher": "SIFT_BRUTEFORCE",
    "max_features": 8_192,
    "extractor_model_sha256": None,
    "matcher_model_sha256": None,
}
DEFAULT_GEOMETRIC_VERIFICATION = {
    "profile": "default_v1",
    "guided_matching": False,
    "skip_geometric_verification": False,
    "raw_parameter_policy": "colmap_build_defaults",
}
GUIDED_GEOMETRIC_VERIFICATION = {
    **DEFAULT_GEOMETRIC_VERIFICATION,
    "profile": "guided_v1",
    "guided_matching": True,
}
SHARED_OPENCV_CALIBRATION = {
    "profile": "shared_opencv_v1",
    "camera_model": "OPENCV",
    "sharing_policy": "single_camera",
    "grouping_key_policy": "all_images",
    "initial_focal_policy": "colmap_exif_or_default",
}


def test_run_id_tracks_camera_profile_not_diagnostic_outcomes() -> None:
    camera = {
        **SHARED_OPENCV_CALIBRATION,
        "planned_camera_count": 1,
        "initial_camera_count": 1,
        "final_camera_count": 1,
        "prior_focal_camera_count": 0,
        "warning_count": 0,
    }

    def run_id(value):
        return _run_id(
            "image-set",
            SIFT_FEATURE,
            "exhaustive",
            None,
            DEFAULT_GEOMETRIC_VERIFICATION,
            value,
            "incremental",
            "COLMAP 4.0.0",
        )

    assert run_id({**camera, "warning_count": 2}) == run_id(camera)
    assert run_id(
        {
            **camera,
            "profile": "shared_simple_radial_v1",
            "camera_model": "SIMPLE_RADIAL",
        }
    ) != run_id(camera)


def test_export_colmap_diagnostics_writes_final_sharded_frontend_data(tmp_path: Path) -> None:
    job = _fixture_job(tmp_path)
    output = job / "diagnostics" / "sfm"

    manifest_path, metrics = export_colmap_diagnostics(
        job_dir=job,
        database_path=job / "colmap" / "database.db",
        source_image_root=job / "frames" / "selected",
        dataset_contract_path=job / "dataset.json",
        output_dir=output,
        feature=SIFT_FEATURE,
        pairing="sequential_loop",
        pairing_vocab_tree_sha256="b" * 64,
        geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
        camera_calibration=SHARED_OPENCV_CALIBRATION,
        camera_calibration_diagnostics_path=(
            job / "diagnostics" / "sfm_camera_calibration.json"
        ),
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "sfm_frontend_diagnostics_v4"
    assert manifest["counts"] == {
        "images": 3,
        "registered_images": 2,
        "keypoints": 5,
        "pairs": 2,
        "candidate_matches": 2,
        "candidate_inliers": 1,
        "guided_inliers": 0,
        "inliers": 1,
        "outliers": 1,
    }
    assert metrics["sfm_diagnostics_status"] == "available"
    assert metrics["sfm_diagnostics_keypoint_count"] == 5
    assert metrics["sfm_feature_profile"] == "sift_v1"
    assert metrics["sfm_local_matcher"] == "SIFT_BRUTEFORCE"
    assert metrics["sfm_pairing"] == "sequential_loop"
    assert metrics["sfm_pairing_vocab_tree_sha256"] == "b" * 64

    images = {image["colmap_image_id"]: image for image in manifest["images"]}
    assert images[1]["registered"] is True
    assert images[1]["camera_id"] == 1
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
    assert run["feature"]["profile"] == "sift_v1"
    assert run["feature"]["keypoint_fields"] == ["x", "y"]
    assert run["local_matcher"]["profile"] == "bruteforce"
    assert run["local_matcher"]["name"] == "SIFT_BRUTEFORCE"
    assert run["pairing"]["name"] == "sequential_loop"
    assert run["pairing"]["vocab_tree_sha256"] == "b" * 64
    assert run["geometric_verification"]["profile"] == "default_v1"
    assert run["geometric_verification"]["guided_matching"] is False
    assert run["camera_calibration"]["profile"] == "shared_opencv_v1"
    assert run["camera_calibration"]["camera_model"] == "OPENCV"
    assert run["camera_calibration"]["initial_camera_count"] == 1
    assert run["mapper"]["name"] == "incremental"
    assert run["view_graph"]["verified_edge_count"] == 1
    assert run["view_graph"]["connected_component_count"] == 2
    assert run["view_graph"]["largest_component_node_count"] == 2
    assert run["view_graph"]["isolated_node_count"] == 1
    assert run["view_graph"]["degree_distribution"]["p50"] == 1.0
    assert run["view_graph"]["video"]["verified_edge_time_span_seconds"][
        "p50"
    ] == pytest.approx(0.1)
    feature_index = _read_gzip_json(job / run["feature_index_path"])
    feature_entry = next(item for item in feature_index["images"] if item["image_id"] == 1)
    feature_shard = _read_gzip_json(job / feature_entry["detail_shard"])
    assert feature_shard["images"]["1"]["points"] == [[10.12, 20.46], [30.0, 40.0]]

    pair_index = _read_gzip_json(job / run["pair_index_path"])
    direct = next(item for item in pair_index["pairs"] if item["pair_key"] == "1-2")
    assert direct["candidate_match_count"] == 2
    assert direct["candidate_inlier_count"] == 1
    assert direct["guided_inlier_count"] == 0
    assert direct["inlier_count"] == 1
    assert direct["outlier_count"] == 1
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
        feature=SIFT_FEATURE,
        pairing="sequential_loop",
        pairing_vocab_tree_sha256="b" * 64,
        geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
        camera_calibration=SHARED_OPENCV_CALIBRATION,
        camera_calibration_diagnostics_path=(
            job / "diagnostics" / "sfm_camera_calibration.json"
        ),
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )
    assert _tree_hashes(output) == first_hashes

    shutil.rmtree(output)
    learned_manifest, learned_metrics = export_colmap_diagnostics(
        job_dir=job,
        database_path=job / "colmap" / "database.db",
        source_image_root=job / "frames" / "selected",
        dataset_contract_path=job / "dataset.json",
        output_dir=output,
        feature={
            "profile": "aliked_n16rot_v1",
            "extractor": "ALIKED_N16ROT",
            "descriptor": "ALIKED",
            "local_matcher_profile": "bruteforce",
            "local_matcher": "ALIKED_BRUTEFORCE",
            "max_features": 8_192,
            "extractor_model_sha256": "a" * 64,
            "matcher_model_sha256": "b" * 64,
        },
        pairing="sequential_loop",
        pairing_vocab_tree_sha256="b" * 64,
        geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
        camera_calibration=SHARED_OPENCV_CALIBRATION,
        camera_calibration_diagnostics_path=(
            job / "diagnostics" / "sfm_camera_calibration.json"
        ),
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )
    learned = json.loads(learned_manifest.read_text(encoding="utf-8"))
    assert learned["default_run_id"] != run["run_id"]
    assert learned["runs"][0]["feature"]["profile"] == "aliked_n16rot_v1"
    assert learned_metrics["sfm_feature_profile"] == "aliked_n16rot_v1"

    shutil.rmtree(output)
    lightglue_manifest, lightglue_metrics = export_colmap_diagnostics(
        job_dir=job,
        database_path=job / "colmap" / "database.db",
        source_image_root=job / "frames" / "selected",
        dataset_contract_path=job / "dataset.json",
        output_dir=output,
        feature={
            **SIFT_FEATURE,
            "local_matcher_profile": "lightglue",
            "local_matcher": "SIFT_LIGHTGLUE",
            "matcher_model_sha256": "c" * 64,
        },
        pairing="sequential_loop",
        pairing_vocab_tree_sha256="b" * 64,
        geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
        camera_calibration=SHARED_OPENCV_CALIBRATION,
        camera_calibration_diagnostics_path=(
            job / "diagnostics" / "sfm_camera_calibration.json"
        ),
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )
    lightglue = json.loads(lightglue_manifest.read_text(encoding="utf-8"))
    assert lightglue["default_run_id"] not in {
        run["run_id"],
        learned["default_run_id"],
    }
    assert lightglue["runs"][0]["local_matcher"]["profile"] == "lightglue"
    assert lightglue["runs"][0]["local_matcher"]["name"] == "SIFT_LIGHTGLUE"
    assert lightglue_metrics["sfm_local_matcher_profile"] == "lightglue"


def test_export_colmap_diagnostics_rejects_camera_mapping_drift(
    tmp_path: Path,
) -> None:
    job = _fixture_job(tmp_path)
    calibration = {
        "profile": "auto_grouped_simple_radial_v1",
        "camera_model": "SIMPLE_RADIAL",
        "sharing_policy": "focal_aware_groups",
        "grouping_key_policy": "exif_device_lens_focal_size_orientation_v1",
        "initial_focal_policy": "colmap_exif_or_default",
    }
    camera_path = job / "diagnostics" / "sfm_camera_calibration.json"
    camera_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "sfm_camera_calibration_diagnostics_v1",
                "calibration": calibration,
                "grouping": {"planned_camera_count": 2},
                "initial": {
                    "camera_count": 2,
                    "prior_focal_camera_count": 0,
                },
                "final": {"camera_count": 1},
                "plausibility": {
                    "warning_count": 0,
                    "warnings_are_job_gates": False,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ColmapDiagnosticsError, match="image mapping"):
        export_colmap_diagnostics(
            job_dir=job,
            database_path=job / "colmap" / "database.db",
            source_image_root=job / "frames" / "selected",
            dataset_contract_path=job / "dataset.json",
            output_dir=job / "diagnostics" / "sfm",
            feature=SIFT_FEATURE,
            pairing="sequential",
            geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
            camera_calibration=calibration,
            camera_calibration_diagnostics_path=camera_path,
            colmap_build="COLMAP 4.0.0",
            video_selection_path=job / "frames" / "selection.json",
        )


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
            feature=SIFT_FEATURE,
            pairing="sequential",
            geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
            camera_calibration=SHARED_OPENCV_CALIBRATION,
            camera_calibration_diagnostics_path=(
                job / "diagnostics" / "sfm_camera_calibration.json"
            ),
            colmap_build="COLMAP 4.0.0",
            video_selection_path=job / "frames" / "selection.json",
        )

    assert not output.exists()
    assert not (job / "diagnostics" / ".sfm.tmp").exists()


def test_export_colmap_diagnostics_records_guided_matches_outside_candidates(
    tmp_path: Path,
) -> None:
    job = _fixture_job(tmp_path)
    pair_id = 1 * MAX_IMAGE_ID + 2
    verified = np.asarray([[0, 0], [0, 1]], dtype="<u4")
    with sqlite3.connect(job / "colmap" / "database.db") as database:
        database.execute(
            "UPDATE two_view_geometries SET rows=?, data=? WHERE pair_id=?",
            (len(verified), verified.tobytes(), pair_id),
        )
    output = job / "diagnostics" / "sfm"

    manifest_path, metrics = export_colmap_diagnostics(
        job_dir=job,
        database_path=job / "colmap" / "database.db",
        source_image_root=job / "frames" / "selected",
        dataset_contract_path=job / "dataset.json",
        output_dir=output,
        feature=SIFT_FEATURE,
        pairing="sequential_loop",
        geometric_verification=GUIDED_GEOMETRIC_VERIFICATION,
        camera_calibration=SHARED_OPENCV_CALIBRATION,
        camera_calibration_diagnostics_path=(
            job / "diagnostics" / "sfm_camera_calibration.json"
        ),
        pairing_vocab_tree_sha256="b" * 64,
        colmap_build="COLMAP 4.0.0",
        video_selection_path=job / "frames" / "selection.json",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest["runs"][0]
    pair_index = _read_gzip_json(job / run["pair_index_path"])
    direct = next(item for item in pair_index["pairs"] if item["pair_key"] == "1-2")
    pair_shard = _read_gzip_json(job / direct["detail_shard"])

    assert direct["candidate_match_count"] == 2
    assert direct["candidate_inlier_count"] == 1
    assert direct["guided_inlier_count"] == 1
    assert direct["inlier_count"] == 2
    assert direct["outlier_count"] == 1
    assert pair_shard["pairs"]["1-2"] == {
        "inliers": [[0, 0], [0, 1]],
        "outliers": [[1, 1]],
    }
    assert run["geometric_verification"]["profile"] == "guided_v1"
    assert run["view_graph"]["match_totals"]["guided_inliers"] == 1
    assert metrics["sfm_view_graph_guided_inlier_count"] == 1


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
    (job / "diagnostics" / "sfm_camera_calibration.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "sfm_camera_calibration_diagnostics_v1",
                "calibration": SHARED_OPENCV_CALIBRATION,
                "grouping": {"planned_camera_count": 1},
                "initial": {
                    "camera_count": 1,
                    "prior_focal_camera_count": 0,
                },
                "final": {"camera_count": 1},
                "plausibility": {
                    "warning_count": 0,
                    "warnings_are_job_gates": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (job / "colmap").mkdir()
    _write_database(job / "colmap" / "database.db")
    identity = np.eye(4).tolist()
    second = np.eye(4)
    second[0, 3] = 1.0
    contract = {
        "dataset_hash": "a" * 64,
        "source": {"image_root": "undistorted"},
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


@pytest.mark.parametrize("wrong_folder", [False, True])
def test_diagnostics_matches_full_relative_image_identity(tmp_path, wrong_folder):
    job = _fixture_job(tmp_path)
    contract_path = job / "dataset.json"
    contract = json.loads(contract_path.read_text())
    image_root = job / "frames/selected"
    with sqlite3.connect((job / "colmap/database.db").as_uri() + "?mode=rw", uri=True) as db:
        for image_id in (1, 2):
            name = f"camera-{image_id}/same.jpg"
            (image_root / name).parent.mkdir()
            (image_root / f"frame-{image_id}.jpg").rename(image_root / name)
            db.execute("UPDATE images SET name=? WHERE image_id=?", (name, image_id))
            contract["images"][image_id - 1]["path"] = f"undistorted/{name}"
    if wrong_folder:
        contract["images"][0]["path"] = "undistorted/camera-2/same.jpg"
    contract_path.write_text(json.dumps(contract))
    kwargs = dict(
        job_dir=job, database_path=job / "colmap/database.db", source_image_root=image_root,
        dataset_contract_path=contract_path, output_dir=job / "diagnostics/sfm",
        feature=SIFT_FEATURE, pairing="exhaustive",
        geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
        camera_calibration=SHARED_OPENCV_CALIBRATION,
        camera_calibration_diagnostics_path=job / "diagnostics/sfm_camera_calibration.json",
        colmap_build="COLMAP 4.0.0",
    )
    if wrong_folder:
        with pytest.raises(ColmapDiagnosticsError, match="does not match"):
            export_colmap_diagnostics(**kwargs)
    else:
        path, metrics = export_colmap_diagnostics(**kwargs)
        assert metrics["sfm_diagnostics_status"] == "available"
        images = json.loads(path.read_text())["images"]
        assert [entry["name"] for entry in images[:2]] == ["camera-1/same.jpg", "camera-2/same.jpg"]


def test_completed_job_repair_is_locked_atomic_and_preserves_model_and_geometry(tmp_path):
    from image3d_scenegraph.gpu_lease import FileLease, LeaseBusy
    from scripts.repair_job_sfm_diagnostics import repair_diagnostics

    job = _fixture_job(tmp_path)
    (job / "input").mkdir()
    (job / "frames/selected").rename(job / "input/images")
    (job / "diagnostics/colmap_timing.json").write_text(json.dumps({
        "colmap_build": "COLMAP 4.0.0", "feature": SIFT_FEATURE,
        "geometric_verification": DEFAULT_GEOMETRIC_VERIFICATION,
        "camera_calibration": SHARED_OPENCV_CALIBRATION,
    }))
    (job / "model.pt").write_bytes(b"unchanged model; must never be loaded")
    (job / "diagnostics/geometry_bundle.json").write_bytes(b"frozen geometry bundle")
    manifest = {
        "job_id": "job", "status": "done", "mode": "multi_image",
        "geometry_backend": "project_3dgs", "output_type": "gaussian_splat",
        "gaussian_geometry_effective_source": "colmap", "navigation_status": "unavailable",
        "sfm_feature_effective_profile": "sift_v1", "sfm_local_matcher_effective": "bruteforce",
        "sfm_pairing_effective": "exhaustive", "sfm_geometric_verification_effective": "default_v1",
        "sfm_camera_calibration_effective": "shared_opencv_v1", "sfm_mapper_effective": "incremental",
        "inputs": [], "assets": {"gaussian_dataset": "dataset.json", "gaussian_model": "model.pt"},
        "metrics": {"sfm_diagnostics_status": "unavailable", "sfm_diagnostics_reason": "old failure", "psnr": 27.5},
    }
    path = job / "manifest.json"
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    hashes = _tree_hashes(job)
    with FileLease(tmp_path / ".worker.lock"), pytest.raises(LeaseBusy):
        repair_diagnostics(tmp_path, "job")
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="directory name"):
        repair_diagnostics(tmp_path, "../job")
    dataset_before = (job / "dataset.json").read_bytes()
    bad_dataset = json.loads(dataset_before)
    bad_dataset["images"][0]["path"] = "undistorted/wrong/frame-1.jpg"
    (job / "dataset.json").write_text(json.dumps(bad_dataset))
    with pytest.raises(ValueError, match="does not match"):
        repair_diagnostics(tmp_path, "job")
    assert path.read_bytes() == before
    (job / "dataset.json").write_bytes(dataset_before)
    repaired = repair_diagnostics(tmp_path, "job")
    current = json.loads(path.read_text())
    assert current["status"] == "done" and current["navigation_status"] == "unavailable"
    assert current["assets"]["sfm_diagnostics"] == repaired["sfm_diagnostics"]
    assert current["metrics"]["sfm_diagnostics_status"] == "available"
    assert "sfm_diagnostics_reason" not in current["metrics"]
    assert current["metrics"]["psnr"] == 27.5
    assert (job / repaired["sfm_diagnostics"]).parent.joinpath("manifest-before-repair.json").read_bytes() == before
    after_hashes = _tree_hashes(job)
    assert all(after_hashes[name] == digest for name, digest in hashes.items() if name != "manifest.json")
    with pytest.raises(ValueError, match="already published"):
        repair_diagnostics(tmp_path, "job")


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


def test_export_colmap_diagnostics_preserves_folder_grouped_profile(tmp_path):
    job = _fixture_job(tmp_path)
    calibration = {
        **SHARED_OPENCV_CALIBRATION,
        "profile": "folder_grouped_opencv_v1",
        "sharing_policy": "folder_groups",
        "grouping_key_policy": "relative_parent_directory_v1",
    }
    camera_path = job / "diagnostics/sfm_camera_calibration.json"
    camera = json.loads(camera_path.read_text())
    camera["calibration"] = calibration
    camera_path.write_text(json.dumps(camera))
    manifest_path, metrics = export_colmap_diagnostics(
        job_dir=job, database_path=job / "colmap/database.db",
        source_image_root=job / "frames/selected", dataset_contract_path=job / "dataset.json",
        output_dir=job / "diagnostics/sfm", feature=SIFT_FEATURE, pairing="exhaustive",
        geometric_verification=DEFAULT_GEOMETRIC_VERIFICATION,
        camera_calibration=calibration, camera_calibration_diagnostics_path=camera_path,
        colmap_build="COLMAP 4.0.0",
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["runs"][0]["camera_calibration"]["profile"] == "folder_grouped_opencv_v1"
    assert metrics["sfm_camera_calibration_profile"] == "folder_grouped_opencv_v1"
