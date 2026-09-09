from __future__ import annotations

import sqlite3
import zipfile

import pytest

from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput


def test_publish_preserves_raw_sfm_without_public_download(tmp_path):
    store = JobStore(tmp_path / "jobs")
    queued = store.enqueue_job("image", [UploadedInput("input.jpg", b"image")])
    job_dir = store.job_dir(queued["job_id"])
    workspace = store._create_attempt_workspace(job_dir, "attempt-001")
    database = workspace / "colmap/database.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE matches(pair_id INTEGER, data BLOB)")
        connection.execute("INSERT INTO matches VALUES(1, ?)", (b"matches",))
    originals = {"colmap/database.db": database.read_bytes()}
    for name in (
        "colmap/sparse/0/points3D.bin",
        "colmap/sparse_raw_txt/points3D.txt",
        "colmap/pose_recovery/global/database.db",
        "colmap/undistorted/sparse_txt/images.txt",
    ):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        originals[name] = name.encode()
        path.write_bytes(originals[name])
    (workspace / "logs/run.log").write_text("done\n")
    (workspace / "geometry/points.ply").write_bytes(b"public points")
    inode = database.stat().st_ino

    store._publish_workspace(job_dir, workspace, "attempt-001")

    assert not workspace.exists()
    assert (job_dir / "colmap/database.db").stat().st_ino == inode
    for name, content in originals.items():
        assert (job_dir / name).read_bytes() == content
        with pytest.raises(JobError, match="internal evidence"):
            store.get_asset_path(queued["job_id"], name)
    assert store.get_asset_path(queued["job_id"], "geometry/points.ply").read_bytes() == b"public points"
    with zipfile.ZipFile(store.build_zip(queued["job_id"])) as archive:
        assert "geometry/points.ply" in archive.namelist()
        assert not any(name.startswith("colmap/") for name in archive.namelist())


def test_interrupted_publication_quarantines_raw_sfm_before_retry(tmp_path):
    store = JobStore(tmp_path / "jobs")
    queued = store.enqueue_job("image", [UploadedInput("input.jpg", b"image")])
    job_dir = store.job_dir(queued["job_id"])
    workspace = store._create_attempt_workspace(job_dir, "attempt-001")
    (workspace / "colmap").mkdir()
    (workspace / "colmap/database.db").write_bytes(b"original")

    with pytest.raises(JobError, match="logs/run.log"):
        store._publish_workspace(job_dir, workspace, "attempt-001")
    store._quarantine_unpublished_outputs(job_dir, "attempt-001")

    retained = job_dir / "lifecycle/attempts/attempt-001/partial_published/colmap/database.db"
    assert retained.read_bytes() == b"original"
    assert not (job_dir / "colmap").exists()
    retry = store._create_attempt_workspace(job_dir, "attempt-002")
    (retry / "colmap").mkdir()
    (retry / "colmap/database.db").write_bytes(b"retry")
    (retry / "logs/run.log").write_text("done\n")
    store._publish_workspace(job_dir, retry, "attempt-002")
    assert (job_dir / "colmap/database.db").read_bytes() == b"retry"
    assert retained.read_bytes() == b"original"
