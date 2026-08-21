from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.main import create_app


class FakeWorker:
    def __init__(self) -> None:
        self.notifications = 0

    def notify(self) -> None:
        self.notifications += 1


class FakeJobStore:
    def __init__(self) -> None:
        self.options: dict[str, int | float | str] | None = None
        self.navigation_job_id: str | None = None

    def enqueue_job(self, mode, files, *, geometry_backend, output_type, options):
        self.options = options
        return {
            "job_id": "test-job",
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "mode": mode,
            "geometry_backend": geometry_backend,
            "output_type": output_type,
            "metrics": {},
        }

    def request_navigation_assets(self, job_id):
        self.navigation_job_id = job_id
        return {
            "job_id": job_id,
            "status": "done",
            "navigation_status": "queued",
        }


def test_list_jobs_returns_store_summaries(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    store.list_jobs = lambda: [
        {
            "job_id": "job-2",
            "status": "done",
            "geometry_backend": "colmap",
            "output_type": "mesh",
            "updated_at": "2026-08-13T00:00:00Z",
        }
    ]
    app.state.job_store = store

    response = TestClient(app).get("/api/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["job_id"] == "job-2"


def test_public_job_schema_exposes_only_bounded_gaussian_controls(tmp_path):
    schema = create_app(tmp_path / "jobs").openapi()
    body_schema = schema["paths"]["/api/jobs"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    if "$ref" in body_schema:
        body_schema = schema["components"]["schemas"][body_schema["$ref"].rsplit("/", 1)[-1]]

    properties = body_schema["properties"]
    assert "quality_profile" not in properties
    assert "gaussian_trainer" in properties
    assert properties["gaussian_trainer"]["enum"] == ["project", "graphdeco"]
    assert properties["gaussian_trainer"]["default"] == "graphdeco"
    assert properties["gaussian_geometry_source"]["enum"] == ["colmap", "vggt_ba"]
    assert properties["gaussian_geometry_source"]["default"] == "colmap"
    assert properties["gaussian_postprocess"]["enum"] == [
        "none",
        "vggt_visibility_v1",
    ]
    assert properties["gaussian_postprocess"]["default"] == "none"
    assert properties["gaussian_sor_filter"].get("default") is None
    assert properties["gaussian_recovery_prune"].get("default") is None
    gaussian_names = {name for name in properties if "gaussian" in name}
    assert gaussian_names == {
        "gaussian_trainer",
        "gaussian_geometry_source",
        "gaussian_postprocess",
        "gaussian_sor_filter",
        "gaussian_recovery_prune",
        "gaussian_longest_edge",
    }


def test_create_job_forwards_gaussian_trainer(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    app.state.job_store = store
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "project_3dgs",
            "output_type": "gaussian_splat",
            "gaussian_trainer": "graphdeco",
            "gaussian_geometry_source": "vggt_ba",
            "gaussian_postprocess": "vggt_visibility_v1",
            "gaussian_longest_edge": "3072",
        },
        files=[
            ("files", (f"{index}.jpg", b"image", "image/jpeg"))
            for index in range(12)
        ],
    )

    assert response.status_code == 202
    assert store.options == {
        "gaussian_trainer": "graphdeco",
        "gaussian_geometry_source": "vggt_ba",
        "gaussian_postprocess": "vggt_visibility_v1",
        "gaussian_longest_edge": 3072,
    }


def test_create_job_forwards_gaussian_sor_filter(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    app.state.job_store = store
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "project_3dgs",
            "output_type": "gaussian_splat",
            "gaussian_sor_filter": "off",
        },
        files=[("files", ("room.jpg", b"image", "image/jpeg"))],
    )

    assert response.status_code == 202
    assert store.options["gaussian_sor_filter"] == "off"


def test_create_job_forwards_gaussian_recovery_prune(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    app.state.job_store = store
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "project_3dgs",
            "output_type": "gaussian_splat",
            "gaussian_recovery_prune": "on",
        },
        files=[("files", ("room.jpg", b"image", "image/jpeg"))],
    )

    assert response.status_code == 202
    assert store.options["gaussian_recovery_prune"] == "on"


def test_create_video_job_forwards_colmap_matcher(tmp_path):
    root = tmp_path / "jobs"
    app = create_app(root, start_worker=False)
    response = TestClient(app).post(
        "/api/jobs",
        data={
            "mode": "video",
            "geometry_backend": "project_3dgs",
            "output_type": "gaussian_splat",
            "colmap_matcher": "sequential",
        },
        files=[("files", ("room.mp4", b"video", "video/mp4"))],
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    request = json.loads((root / job_id / "request.json").read_text())
    assert request["options"]["colmap_matcher"] == "sequential"


def test_create_job_omits_colmap_matcher_for_non_video_jobs(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    app.state.job_store = store
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "project_3dgs",
            "output_type": "gaussian_splat",
            "colmap_matcher": "sequential",
        },
        files=[("files", ("room.jpg", b"image", "image/jpeg"))],
    )

    assert response.status_code == 202
    assert "colmap_matcher" not in store.options


def test_create_video_job_streams_to_persisted_input(tmp_path):
    root = tmp_path / "jobs"
    app = create_app(root, start_worker=False)
    response = TestClient(app).post(
        "/api/jobs",
        data={
            "mode": "video",
            "geometry_backend": "project_3dgs",
            "output_type": "gaussian_splat",
            "video_rotation": "counterclockwise_90",
        },
        files=[("files", ("portrait.mp4", b"video-content", "video/mp4"))],
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    request = json.loads((root / job_id / "request.json").read_text())
    assert (root / job_id / "input" / "portrait.mp4").read_bytes() == b"video-content"
    assert request["options"]["video_keyframe_profile"] == "standard_v1"
    assert request["options"]["video_rotation"] == "counterclockwise_90"
    assert not list((root / ".uploads").glob("*.upload"))


def test_create_job_rejects_invalid_gaussian_trainer(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    client = TestClient(app)
    response = client.post(
        "/api/jobs",
        data={"gaussian_trainer": "unknown"},
        files=[("files", ("room.jpg", b"image", "image/jpeg"))],
    )
    assert response.status_code == 422


def test_create_job_rejects_invalid_gaussian_experimental_options(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    client = TestClient(app)
    files = [("files", ("room.jpg", b"image", "image/jpeg"))]

    geometry_response = client.post(
        "/api/jobs",
        data={"gaussian_geometry_source": "unknown"},
        files=files,
    )
    postprocess_response = client.post(
        "/api/jobs",
        data={"gaussian_postprocess": "unknown"},
        files=files,
    )
    matcher_response = client.post(
        "/api/jobs",
        data={"colmap_matcher": "unknown"},
        files=files,
    )
    sor_response = client.post(
        "/api/jobs",
        data={"gaussian_sor_filter": "unknown"},
        files=files,
    )
    recovery_prune_response = client.post(
        "/api/jobs",
        data={"gaussian_recovery_prune": "unknown"},
        files=files,
    )

    assert geometry_response.status_code == 422
    assert postprocess_response.status_code == 422
    assert matcher_response.status_code == 422
    assert sor_response.status_code == 422
    assert recovery_prune_response.status_code == 422


def test_create_job_rejects_invalid_gaussian_resolution(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    response = TestClient(app).post(
        "/api/jobs",
        data={"gaussian_longest_edge": "3073"},
        files=[("files", ("room.jpg", b"image", "image/jpeg"))],
    )
    assert response.status_code == 422


def test_create_job_omits_unspecified_colmap_vggt_options(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    app.state.job_store = store
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "colmap_vggt",
            "output_type": "point_cloud",
        },
        files=[
            ("files", ("first.jpg", b"first", "image/jpeg")),
            ("files", ("second.jpg", b"second", "image/jpeg")),
        ],
    )

    assert response.status_code == 202
    assert store.options == {}


def test_create_job_forwards_independent_colmap_vggt_policies(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    app.state.job_store = store
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "colmap_vggt",
            "output_type": "point_cloud",
            "colmap_vggt_grouping": "covisibility",
            "colmap_vggt_overlap_size": "1",
            "colmap_vggt_confidence_threshold_scope": "per_frame",
            "colmap_vggt_consistency_support_policy": "adaptive_two",
            "colmap_vggt_point_budget_policy": "spatial_balanced",
        },
        files=[
            ("files", ("first.jpg", b"first", "image/jpeg")),
            ("files", ("second.jpg", b"second", "image/jpeg")),
        ],
    )

    assert response.status_code == 202
    assert store.options == {
        "colmap_vggt_grouping": "covisibility",
        "colmap_vggt_overlap_size": 1,
        "colmap_vggt_confidence_threshold_scope": "per_frame",
        "colmap_vggt_consistency_support_policy": "adaptive_two",
        "colmap_vggt_point_budget_policy": "spatial_balanced",
    }


def test_create_job_rejects_invalid_colmap_vggt_grouping(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    app.state.job_store = FakeJobStore()
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "colmap_vggt",
            "output_type": "point_cloud",
            "colmap_vggt_grouping": "unknown",
        },
        files=[
            ("files", ("first.jpg", b"first", "image/jpeg")),
            ("files", ("second.jpg", b"second", "image/jpeg")),
        ],
    )

    assert response.status_code == 422


def test_navigation_assets_route_queues_worker(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    store = FakeJobStore()
    worker = FakeWorker()
    app.state.job_store = store
    app.state.job_worker = worker
    client = TestClient(app)

    response = client.post("/api/jobs/retained-job/navigation-assets")

    assert response.status_code == 202
    assert response.json()["navigation_status"] == "queued"
    assert store.navigation_job_id == "retained-job"
    assert worker.notifications == 1



def test_async_job_cancel_retry_and_status_routes(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    client = TestClient(app)

    created = client.post(
        "/api/jobs",
        files=[("files", ("room.jpg", b"image", "image/jpeg"))],
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    status_response = client.get(f"/api/jobs/{job_id}")
    assert status_response.json()["status"] == "queued"
    assert status_response.json()["active_attempt_id"] == "attempt-001"

    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    retried = client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 202
    assert retried.json()["active_attempt_id"] == "attempt-002"


def test_create_job_rejects_invalid_colmap_vggt_policy(tmp_path):
    app = create_app(tmp_path / "jobs", start_worker=False)
    app.state.job_store = FakeJobStore()
    app.state.job_worker = FakeWorker()
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={
            "mode": "multi_image",
            "geometry_backend": "colmap_vggt",
            "output_type": "point_cloud",
            "colmap_vggt_point_budget_policy": "unknown",
        },
        files=[
            ("files", ("first.jpg", b"first", "image/jpeg")),
            ("files", ("second.jpg", b"second", "image/jpeg")),
        ],
    )

    assert response.status_code == 422
