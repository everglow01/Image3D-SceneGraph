from __future__ import annotations

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


def test_public_job_schema_has_no_raw_gaussian_hyperparameters(tmp_path):
    schema = create_app(tmp_path / "jobs").openapi()
    body_schema = schema["paths"]["/api/jobs"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    if "$ref" in body_schema:
        body_schema = schema["components"]["schemas"][body_schema["$ref"].rsplit("/", 1)[-1]]

    properties = body_schema["properties"]
    assert "quality_profile" not in properties
    assert not any("gaussian" in name for name in properties)


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
