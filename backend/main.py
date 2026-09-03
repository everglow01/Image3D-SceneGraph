from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from image3d_scenegraph.geometry.backends import get_backend_status_payload
from image3d_scenegraph.jobs import (
    COLMAP_FEATURE_BACKENDS,
    DEFAULT_VIDEO_PROFILE,
    MAX_VIDEO_BYTES,
    VIDEO_SUFFIXES,
    JobError,
    JobStore,
    UploadedInput,
)
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


async def _stage_video_upload(store: JobStore, upload: UploadFile) -> UploadedInput:
    staging_dir = store.output_root / ".uploads"
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / f"{uuid.uuid4().hex}.upload"
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("xb") as handle:
            while chunk := await upload.read(8 * 1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_VIDEO_BYTES:
                    raise JobError("video exceeds the 2 GiB limit")
                digest.update(chunk)
                handle.write(chunk)
        return UploadedInput(
            filename=upload.filename or "",
            content_type=upload.content_type,
            staged_path=path,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise


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

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, list[dict[str, object]]]:
        return {"jobs": app.state.job_store.list_jobs()}

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        files: Annotated[list[UploadFile], File()],
        mode: Annotated[str, Form()] = "image",
        geometry_backend: Annotated[str, Form()] = "mock",
        output_type: Annotated[str, Form()] = "point_cloud",
        gaussian_trainer: Annotated[
            Literal["project", "graphdeco", "mcmc"], Form()
        ] = "graphdeco",
        gaussian_geometry_source: Annotated[
            Literal["colmap", "vggt_ba"], Form()
        ] = "colmap",
        gaussian_postprocess: Annotated[
            Literal["none", "vggt_visibility_v1"], Form()
        ] = "none",
        gaussian_sor_filter: Annotated[
            Literal["on", "off"] | None, Form()
        ] = None,
        gaussian_recovery_prune: Annotated[
            Literal["on", "off"] | None, Form()
        ] = None,
        gaussian_longest_edge: Annotated[int | None, Form(ge=1280, le=3072)] = None,
        colmap_matcher: Annotated[
            Literal["exhaustive", "sequential"] | None, Form()
        ] = None,
        sfm_feature_profile: Annotated[
            Literal["sift_v1", "aliked_n16rot_v1"], Form()
        ] = "sift_v1",
        sfm_local_matcher: Annotated[
            Literal["bruteforce", "lightglue"], Form()
        ] = "bruteforce",
        sfm_pairing: Annotated[
            Literal["exhaustive", "sequential_loop", "vocab_tree"] | None,
            Form(),
        ] = None,
        sfm_geometric_verification: Annotated[
            Literal["default_v1", "guided_v1"], Form()
        ] = "default_v1",
        sfm_camera_calibration: Annotated[
            Literal[
                "shared_opencv_v1",
                "shared_simple_radial_v1",
                "auto_grouped_simple_radial_v1",
            ]
            | None,
            Form(),
        ] = None,
        video_keyframe_profile: Annotated[
            Literal["standard_v1", "standard_v2"], Form()
        ] = DEFAULT_VIDEO_PROFILE,
        video_rotation: Annotated[
            Literal["auto", "clockwise_90", "counterclockwise_90", "180"], Form()
        ] = "auto",
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
        if mode == "video" and (
            geometry_backend != "project_3dgs" or output_type != "gaussian_splat"
        ):
            raise HTTPException(
                status_code=400,
                detail="video mode currently requires project_3dgs + gaussian_splat",
            )
        if mode == "video" and len(files) != 1:
            raise HTTPException(status_code=400, detail="video mode requires exactly one file")
        if (
            mode == "video"
            and Path(files[0].filename or "").suffix.lower() not in VIDEO_SUFFIXES
        ):
            raise HTTPException(
                status_code=400, detail="video must use MP4, MOV, M4V, or WebM"
            )
        uploaded: list[UploadedInput] = []
        for file in files:
            if mode == "video":
                try:
                    uploaded.append(await _stage_video_upload(app.state.job_store, file))
                except JobError as exc:
                    for item in uploaded:
                        if item.staged_path is not None:
                            item.staged_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            else:
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
                "gaussian_geometry_source": (
                    gaussian_geometry_source
                    if geometry_backend == "project_3dgs"
                    else None
                ),
                "gaussian_postprocess": (
                    gaussian_postprocess if geometry_backend == "project_3dgs" else None
                ),
                "gaussian_sor_filter": (
                    gaussian_sor_filter if geometry_backend == "project_3dgs" else None
                ),
                "gaussian_recovery_prune": (
                    gaussian_recovery_prune
                    if geometry_backend == "project_3dgs"
                    else None
                ),
                "gaussian_longest_edge": (
                    gaussian_longest_edge if geometry_backend == "project_3dgs" else None
                ),
                "sfm_feature_profile": (
                    sfm_feature_profile
                    if geometry_backend in COLMAP_FEATURE_BACKENDS
                    else None
                ),
                "sfm_local_matcher": (
                    sfm_local_matcher
                    if geometry_backend in COLMAP_FEATURE_BACKENDS
                    else None
                ),
                "sfm_pairing": (
                    sfm_pairing
                    if geometry_backend in COLMAP_FEATURE_BACKENDS
                    else None
                ),
                "sfm_geometric_verification": (
                    sfm_geometric_verification
                    if geometry_backend in COLMAP_FEATURE_BACKENDS
                    else None
                ),
                "sfm_camera_calibration": (
                    sfm_camera_calibration
                    if geometry_backend in COLMAP_FEATURE_BACKENDS
                    else None
                ),
                "colmap_matcher": (
                    colmap_matcher
                    if mode == "video" and geometry_backend == "project_3dgs"
                    else None
                ),
                "video_keyframe_profile": (
                    video_keyframe_profile if mode == "video" else None
                ),
                "video_rotation": video_rotation if mode == "video" else None,
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
        finally:
            for item in uploaded:
                if item.staged_path is not None:
                    item.staged_path.unlink(missing_ok=True)

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
        if path.name.endswith(".json.gz"):
            return FileResponse(
                path,
                media_type="application/json",
                headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
            )
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
