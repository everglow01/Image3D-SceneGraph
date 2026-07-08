from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

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
    assert manifest["assets"]["point_cloud_aligned"] == "geometry/points_aligned.ply"
    assert manifest["assets"]["alignment_diagnostics"] == "diagnostics/alignment.json"
    assert manifest["metrics"]["alignment_status"] == "aligned"

    loaded_manifest = store.get_manifest(job_id)
    assert loaded_manifest["metrics"]["num_points"] == 5

    scene = store.get_scene(job_id)
    assert scene["objects"][0]["label"] == "scene_proxy"

    asset_path = store.get_asset_path(job_id, "geometry/points.ply")
    assert asset_path.read_text(encoding="utf-8").startswith("ply\n")
    aligned_asset_path = store.get_asset_path(job_id, "geometry/points_aligned.ply")
    assert aligned_asset_path.read_bytes().startswith(b"ply\n")

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
            geometry_backend="dust3r",
            output_type="point_cloud",
        )


def test_create_vggt_point_cloud_job_uses_adapter_contract(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
        (job_dir / "geometry" / "cameras.json").write_text("{}\n", encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=vggt",
                    "num_images=1",
                    "num_groups=1",
                    "batch_size=8",
                    "overlap_size=4",
                    "num_points=123",
                    "inference_seconds=0.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "image",
        [UploadedInput(filename="room.jpg", content=b"fake-image", content_type="image/jpeg")],
        geometry_backend="vggt",
        output_type="point_cloud",
        options={"vggt_max_images": 225, "vggt_batch_size": 8, "vggt_overlap_size": 4},
    )

    assert manifest["stage"] == "vggt_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["cameras"] == "geometry/cameras.json"
    assert manifest["metrics"]["num_points"] == 123
    assert manifest["metrics"]["num_groups"] == 1
    assert manifest["metrics"]["batch_size"] == 8
    assert manifest["metrics"]["overlap_size"] == 4
    assert manifest["metrics"]["inference_seconds"] == 0.5
    assert captured_command[captured_command.index("--max-images") + 1] == "225"
    assert captured_command[captured_command.index("--batch-size") + 1] == "8"
    assert captured_command[captured_command.index("--overlap-size") + 1] == "4"


def test_create_colmap_point_cloud_job_uses_adapter_contract(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 9",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (job_dir / "geometry" / "cameras.json").write_text('{"images": []}\n', encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=colmap",
                    "num_images=2",
                    "registered_images=2",
                    "num_points=9",
                    "matcher=sequential",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "multi_image",
        [
            UploadedInput(filename="frame_001.jpg", content=b"fake-image", content_type="image/jpeg"),
            UploadedInput(filename="frame_002.jpg", content=b"fake-image", content_type="image/jpeg"),
        ],
        geometry_backend="colmap",
        output_type="point_cloud",
    )

    assert manifest["stage"] == "colmap_sparse_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["cameras"] == "geometry/cameras.json"
    assert manifest["metrics"]["registered_images"] == 2
    assert manifest["metrics"]["num_points"] == 9
    assert captured_command[captured_command.index("--matcher") + 1] == "sequential"


def test_create_colmap_vggt_point_cloud_job_uses_adapter_contract(tmp_path, monkeypatch):
    captured_command = []

    def fake_run(command, **kwargs):
        captured_command[:] = command
        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
        (job_dir / "geometry" / "cameras.json").write_text('{"images": []}\n', encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=colmap_vggt",
                    "num_images=2",
                    "registered_images=2",
                    "scaled_images=2",
                    "num_points=99",
                    "scale_median=0.25",
                    "vggt_batch_size=4",
                    "conf_percentile=45.0",
                    "max_points=1500000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "multi_image",
        [
            UploadedInput(filename="frame_001.jpg", content=b"fake-image", content_type="image/jpeg"),
            UploadedInput(filename="frame_002.jpg", content=b"fake-image", content_type="image/jpeg"),
        ],
        geometry_backend="colmap_vggt",
        output_type="point_cloud",
        options={
            "vggt_batch_size": 4,
            "colmap_vggt_max_points": 1_500_000,
            "colmap_vggt_conf_percentile": 45.0,
        },
    )

    assert manifest["stage"] == "colmap_vggt_dense_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["cameras"] == "geometry/cameras.json"
    assert manifest["metrics"]["registered_images"] == 2
    assert manifest["metrics"]["scaled_images"] == 2
    assert manifest["metrics"]["num_points"] == 99
    assert manifest["metrics"]["scale_median"] == 0.25
    assert manifest["metrics"]["vggt_batch_size"] == 4
    assert manifest["metrics"]["max_points"] == 1500000
    assert manifest["metrics"]["conf_percentile"] == 45.0
    assert captured_command[captured_command.index("--matcher") + 1] == "exhaustive"
    assert captured_command[captured_command.index("--vggt-batch-size") + 1] == "4"
    assert captured_command[captured_command.index("--max-points") + 1] == "1500000"
    assert captured_command[captured_command.index("--conf-percentile") + 1] == "45.0"


def test_create_colmap_vggt_mesh_job_runs_mesh_postprocess(tmp_path, monkeypatch):
    captured_commands = []

    def fake_run(command, **kwargs):
        captured_commands.append(command)
        if str(command[1]).endswith("mesh_from_pointcloud.py"):
            output_path = Path(command[3])
            diagnostics_path = Path(command[command.index("--diagnostics-output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-glb")
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "vertices": 42,
                        "triangles": 80,
                        "processed_points": 120,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="mesh ok\n", stderr="")

        output_dir = tmp_path
        for index, value in enumerate(command):
            if value == "--output-dir":
                output_dir = command[index + 1]
                break
        job_dir = Path(output_dir)
        (job_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry" / "points.ply").write_text("ply\n", encoding="utf-8")
        (job_dir / "geometry" / "cameras.json").write_text('{"images": []}\n', encoding="utf-8")
        (job_dir / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "backend=colmap_vggt",
                    "num_images=2",
                    "num_points=99",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="points ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = JobStore(output_root=tmp_path / "jobs")

    manifest = store.create_job(
        "multi_image",
        [
            UploadedInput(filename="frame_001.jpg", content=b"fake-image", content_type="image/jpeg"),
            UploadedInput(filename="frame_002.jpg", content=b"fake-image", content_type="image/jpeg"),
        ],
        geometry_backend="colmap_vggt",
        output_type="mesh",
    )

    assert manifest["output_type"] == "mesh"
    assert manifest["stage"] == "mesh_reconstruction"
    assert manifest["assets"]["point_cloud"] == "geometry/points.ply"
    assert manifest["assets"]["mesh"] == "geometry/mesh.glb"
    assert manifest["assets"]["mesh_diagnostics"] == "diagnostics/mesh.json"
    assert manifest["metrics"]["mesh_status"] == "built"
    assert manifest["metrics"]["mesh_vertices"] == 42
    assert manifest["metrics"]["mesh_triangles"] == 80
    assert any(str(command[1]).endswith("run_colmap_vggt_dense.py") for command in captured_commands)
    assert any(str(command[1]).endswith("mesh_from_pointcloud.py") for command in captured_commands)


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
