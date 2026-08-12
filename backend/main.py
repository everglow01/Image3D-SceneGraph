from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from image3d_scenegraph.geometry.backends import get_backend_status_payload
from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput
from image3d_scenegraph.worker import LocalJobWorker


class MeshVariantRequest(BaseModel):
    method: Literal["poisson", "ball_pivoting", "alpha_shape"] = "poisson"
    voxel_size: float = Field(default=0.05, ge=0.005, le=2.0)
    normal_radius: float = Field(default=0.2, ge=0.01, le=5.0)
    statistical_std_ratio: float = Field(default=2.0, ge=0.1, le=10.0)
    poisson_depth: int = Field(default=8, ge=5, le=12)
    density_trim_quantile: float = Field(default=0.1, ge=0.0, lt=1.0)
    component_min_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    edge_trim_factor: float = Field(default=2.5, ge=0.5, le=10.0)
    max_triangles: int = Field(default=120_000, ge=1_000, le=1_000_000)
    alpha: float = Field(default=0.12, ge=0.0, le=10.0)


def create_app(output_root: Path | str | None = None, *, start_worker: bool = True) -> FastAPI:
    store = JobStore(output_root)
    worker = LocalJobWorker(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                worker.stop()

    app = FastAPI(title="Image3D-SceneGraph API", lifespan=lifespan)
    app.state.job_store = store
    app.state.job_worker = worker

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/backends")
    def get_backends() -> dict:
        return get_backend_status_payload()

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        files: Annotated[list[UploadFile], File()],
        mode: Annotated[str, Form()] = "image",
        geometry_backend: Annotated[str, Form()] = "mock",
        output_type: Annotated[str, Form()] = "point_cloud",
        gaussian_trainer: Annotated[
            Literal["project", "graphdeco"], Form()
        ] = "graphdeco",
        vggt_max_images: Annotated[int | None, Form()] = None,
        vggt_batch_size: Annotated[int | None, Form()] = None,
        vggt_overlap_size: Annotated[int | None, Form()] = None,
        colmap_vggt_grouping: Annotated[
            Literal["sequential", "covisibility"] | None, Form()
        ] = None,
        colmap_vggt_overlap_size: Annotated[int | None, Form(gt=0)] = None,
        colmap_vggt_max_points: Annotated[int | None, Form()] = None,
        colmap_vggt_conf_percentile: Annotated[float | None, Form()] = None,
        colmap_vggt_confidence_threshold_scope: Annotated[
            Literal["global", "per_frame"] | None, Form()
        ] = None,
        colmap_vggt_consistency_support_policy: Annotated[
            Literal["any_support", "adaptive_two"] | None, Form()
        ] = None,
        colmap_vggt_point_budget_policy: Annotated[
            Literal["random", "spatial_balanced"] | None, Form()
        ] = None,
    ) -> dict:
        uploaded: list[UploadedInput] = []
        for file in files:
            uploaded.append(
                UploadedInput(
                    filename=file.filename or "",
                    content=await file.read(),
                    content_type=file.content_type,
                )
            )

        options = {
            key: value
            for key, value in {
                "gaussian_trainer": (
                    gaussian_trainer if geometry_backend == "project_3dgs" else None
                ),
                "vggt_max_images": vggt_max_images,
                "vggt_batch_size": vggt_batch_size,
                "vggt_overlap_size": vggt_overlap_size,
                "colmap_vggt_grouping": (
                    colmap_vggt_grouping if geometry_backend == "colmap_vggt" else None
                ),
                "colmap_vggt_overlap_size": colmap_vggt_overlap_size,
                "colmap_vggt_max_points": colmap_vggt_max_points,
                "colmap_vggt_conf_percentile": colmap_vggt_conf_percentile,
                "colmap_vggt_confidence_threshold_scope": (
                    colmap_vggt_confidence_threshold_scope
                    if geometry_backend == "colmap_vggt"
                    else None
                ),
                "colmap_vggt_consistency_support_policy": (
                    colmap_vggt_consistency_support_policy
                    if geometry_backend == "colmap_vggt"
                    else None
                ),
                "colmap_vggt_point_budget_policy": (
                    colmap_vggt_point_budget_policy
                    if geometry_backend == "colmap_vggt"
                    else None
                ),
            }.items()
            if value is not None
        }

        try:
            manifest = app.state.job_store.enqueue_job(
                mode,
                uploaded,
                geometry_backend=geometry_backend,
                output_type=output_type,
                options=options,
            )
            app.state.job_worker.notify()
            return manifest
        except JobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            manifest = app.state.job_store.get_manifest(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return {
            "job_id": manifest["job_id"],
            "status": manifest["status"],
            "stage": manifest["stage"],
            "progress": manifest["progress"],
            "mode": manifest["mode"],
            "geometry_backend": manifest["geometry_backend"],
            "output_type": manifest["output_type"],
            "active_attempt_id": manifest.get("active_attempt_id"),
            "created_at": manifest.get("created_at"),
            "updated_at": manifest.get("updated_at", manifest.get("created_at")),
            "started_at": manifest.get("started_at"),
            "completed_at": manifest.get("completed_at"),
            "error": manifest.get("error"),
            "navigation_status": manifest.get("navigation_status"),
            "navigation_reason": manifest.get("navigation_reason"),
            "metrics": manifest["metrics"],
        }

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        try:
            return app.state.job_store.cancel_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
    def retry_job(job_id: str) -> dict:
        try:
            manifest = app.state.job_store.retry_job(job_id)
            app.state.job_worker.notify()
            return manifest
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/manifest")
    def get_manifest(job_id: str) -> dict:
        try:
            return app.state.job_store.get_manifest(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/api/jobs/{job_id}/scene")
    def get_scene(job_id: str) -> dict:
        try:
            return app.state.job_store.get_scene(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="scene not found") from exc

    @app.post("/api/jobs/{job_id}/mesh-variants")
    def build_mesh_variant(job_id: str, request: MeshVariantRequest) -> dict:
        try:
            return app.state.job_store.build_mesh_variant(job_id, request.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except JobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/jobs/{job_id}/navigation-assets",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def build_navigation_assets(job_id: str) -> dict:
        try:
            manifest = app.state.job_store.request_navigation_assets(job_id)
            app.state.job_worker.notify()
            return manifest
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/assets/{asset_path:path}")
    def get_asset(job_id: str, asset_path: str) -> FileResponse:
        try:
            path = app.state.job_store.get_asset_path(job_id, asset_path)
        except JobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="asset not found") from exc
        return FileResponse(path)

    @app.get("/api/jobs/{job_id}/download")
    def download_job(job_id: str) -> FileResponse:
        try:
            path = app.state.job_store.build_zip(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return FileResponse(path, filename=path.name, media_type="application/zip")

    return app


app = create_app()
