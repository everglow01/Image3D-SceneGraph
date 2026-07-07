from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from image3d_scenegraph.geometry.backends import get_backend_status_payload
from image3d_scenegraph.jobs import JobError, JobStore, UploadedInput


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

        try:
            return app.state.job_store.create_job(
                mode,
                uploaded,
                geometry_backend=geometry_backend,
                output_type=output_type,
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
