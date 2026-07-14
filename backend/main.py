from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from image3d_scenegraph.geometry.backends import get_backend_status_payload
from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput


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


def create_app(output_root: Path | str | None = None) -> FastAPI:
    app = FastAPI(title="Image3D-SceneGraph API")
    app.state.job_store = JobStore(output_root)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/backends")
    def get_backends() -> dict:
        return get_backend_status_payload()

    @app.post("/api/jobs")
    async def create_job(
        files: Annotated[list[UploadFile], File()],
        mode: Annotated[str, Form()] = "image",
        geometry_backend: Annotated[str, Form()] = "mock",
        output_type: Annotated[str, Form()] = "point_cloud",
        vggt_max_images: Annotated[int | None, Form()] = None,
        vggt_batch_size: Annotated[int | None, Form()] = None,
        vggt_overlap_size: Annotated[int | None, Form()] = None,
        colmap_vggt_overlap_size: Annotated[int | None, Form()] = None,
        colmap_vggt_max_points: Annotated[int | None, Form()] = None,
        colmap_vggt_conf_percentile: Annotated[float | None, Form()] = None,
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
                "vggt_max_images": vggt_max_images,
                "vggt_batch_size": vggt_batch_size,
                "vggt_overlap_size": vggt_overlap_size,
                "colmap_vggt_overlap_size": colmap_vggt_overlap_size,
                "colmap_vggt_max_points": colmap_vggt_max_points,
                "colmap_vggt_conf_percentile": colmap_vggt_conf_percentile,
            }.items()
            if value is not None
        }

        try:
            return app.state.job_store.create_job(
                mode,
                uploaded,
                geometry_backend=geometry_backend,
                output_type=output_type,
                options=options,
            )
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
            "metrics": manifest["metrics"],
        }

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
