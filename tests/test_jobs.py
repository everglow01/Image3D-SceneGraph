from __future__ import annotations

import zipfile

import pytest

from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput


def test_create_image_job_and_read_outputs(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_mock_job(
        "image",
        [
            UploadedInput(
                filename="room.jpg",
                content=b"fake-image",
                content_type="image/jpeg",
            )
        ],
    )

    job_id = manifest["job_id"]
    assert manifest["status"] == "done"
    assert manifest["mode"] == "image"
    assert manifest["geometry_backend"] == "mock"
    assert manifest["output_type"] == "point_cloud"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"

    loaded_manifest = store.get_manifest(job_id)
    assert loaded_manifest["metrics"]["num_points"] == 5

    scene = store.get_scene(job_id)
    assert scene["objects"][0]["label"] == "scene_proxy"

    asset_path = store.get_asset_path(job_id, "geometry/points.ply")
    assert asset_path.read_text(encoding="utf-8").startswith("ply\n")

    bundle_path = store.build_zip(job_id)
    assert bundle_path == tmp_path / "jobs" / f"{job_id}.zip"
    assert bundle_path.exists()
    with zipfile.ZipFile(bundle_path) as archive:
        assert "manifest.json" in archive.namelist()


def test_create_panorama_job(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_mock_job(
        "panorama",
        [
            UploadedInput(
                filename="office_360.jpg",
                content=b"fake-panorama",
                content_type="image/jpeg",
            )
        ],
    )

    assert manifest["mode"] == "panorama"
    assert manifest["input_type"] == "equirectangular_panorama"
    assert manifest["inputs"][0]["path"] == "input/images/office_360.jpg"


def test_preserve_multi_image_folder_paths(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_mock_job(
        "multi_image",
        [
            UploadedInput(filename="scan/front/frame_001.jpg", content=b"front"),
            UploadedInput(filename="scan/back/frame_001.jpg", content=b"back"),
        ],
    )

    input_paths = [item["path"] for item in manifest["inputs"]]
    assert input_paths == [
        "input/images/scan/front/frame_001.jpg",
        "input/images/scan/back/frame_001.jpg",
    ]


def test_reject_multiple_panorama_inputs(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="panorama mode requires exactly one file"):
        store.create_mock_job(
            "panorama",
            [
                UploadedInput(filename="front.jpg", content=b"fake-image"),
                UploadedInput(filename="back.jpg", content=b"fake-image"),
            ],
        )


def test_reject_invalid_mode(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="unsupported mode"):
        store.create_mock_job(
            "rgbd",
            [UploadedInput(filename="room.jpg", content=b"fake-image")],
        )


def test_reject_unimplemented_reconstruction_option(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="not implemented"):
        store.create_job(
            "image",
            [UploadedInput(filename="room.jpg", content=b"fake-image")],
            geometry_backend="vggt",
            output_type="point_cloud",
        )


def test_reject_invalid_output_type(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")

    with pytest.raises(JobError, match="unsupported output_type"):
        store.create_job(
            "image",
            [UploadedInput(filename="room.jpg", content=b"fake-image")],
            geometry_backend="mock",
            output_type="rgbd",
        )


def test_asset_path_cannot_escape_job_dir(tmp_path):
    store = JobStore(output_root=tmp_path / "jobs")
    manifest = store.create_mock_job(
        "image",
        [UploadedInput(filename="room.jpg", content=b"fake-image")],
    )

    with pytest.raises(JobError, match="escapes job directory"):
        store.get_asset_path(manifest["job_id"], "../manifest.json")
