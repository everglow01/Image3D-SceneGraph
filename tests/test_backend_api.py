from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


class FakeJobStore:
    def __init__(self) -> None:
        self.options: dict[str, int | float | str] | None = None

    def create_job(self, mode, files, *, geometry_backend, output_type, options):
        self.options = options
        return {
            "job_id": "test-job",
            "status": "done",
            "stage": "colmap_vggt_dense_reconstruction",
            "progress": 1.0,
            "mode": mode,
            "geometry_backend": geometry_backend,
            "output_type": output_type,
            "metrics": {},
        }


def test_create_job_forwards_independent_colmap_vggt_policies(tmp_path):
    app = create_app(tmp_path / "jobs")
    store = FakeJobStore()
    app.state.job_store = store
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

    assert response.status_code == 200
    assert store.options == {
        "colmap_vggt_grouping": "covisibility",
        "colmap_vggt_overlap_size": 1,
        "colmap_vggt_confidence_threshold_scope": "per_frame",
        "colmap_vggt_consistency_support_policy": "adaptive_two",
        "colmap_vggt_point_budget_policy": "spatial_balanced",
    }


def test_create_job_rejects_invalid_colmap_vggt_grouping(tmp_path):
    app = create_app(tmp_path / "jobs")
    app.state.job_store = FakeJobStore()
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


def test_create_job_rejects_invalid_colmap_vggt_policy(tmp_path):
    app = create_app(tmp_path / "jobs")
    app.state.job_store = FakeJobStore()
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
