from __future__ import annotations

import asyncio
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from image3d_scenegraph.gaussian import cloud_media
from image3d_scenegraph.gaussian.cloud_render import CloudRenderError
from image3d_scenegraph.gaussian.editing import EditConflict, GaussianEditError
from image3d_scenegraph.gaussian.editor_session import offload
from image3d_scenegraph.gpu_lease import LeaseBusy


class EditorRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request: Request):
            try:
                if request.method not in {"GET", "HEAD"}:
                    origin = request.headers.get("origin", "")
                    allowed = {
                        x.strip().rstrip("/")
                        for x in os.environ.get(
                            "IMAGE3D_EDITOR_ORIGINS",
                            "http://localhost:8081,http://127.0.0.1:8081",
                        ).split(",")
                        if x.strip()
                    }
                    if (
                        origin not in allowed
                        or request.headers.get("x-image3d-editor") != "1"
                    ):
                        raise HTTPException(
                            403,
                            "页面来源未列入 IMAGE3D_EDITOR_ORIGINS，或缺少编辑请求头",
                        )
                    body = bytearray()
                    async for chunk in request.stream():
                        body.extend(chunk)
                        if len(body) > 96 * 1024:
                            raise HTTPException(413, "编辑请求超过 96 KiB 上限")
                    request._body = bytes(body)
                response = await original(request)
            except PermissionError as exc:
                response = JSONResponse({"detail": str(exc)}, status_code=403)
            except (EditConflict, LeaseBusy, FileExistsError) as exc:
                response = JSONResponse({"detail": str(exc)}, status_code=409)
            except FileNotFoundError:
                response = JSONResponse(
                    {"detail": "编辑文档、版本或资产不存在"}, status_code=404
                )
            except (GaussianEditError, CloudRenderError, ValueError) as exc:
                response = JSONResponse({"detail": str(exc)}, status_code=422)
            response.headers["Cache-Control"] = "no-store"
            return response

        return handle


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Source(Input):
    job_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    variant_id: str | None = Field(default=None, max_length=128)
    asset_role: Literal["scene_splat", "scene_splat_vggt_filtered"] = "scene_splat"


class SessionInput(Input):
    edit_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    version: str | None = Field(default=None, pattern=r"^v[0-9]{8}$")


class Freeze(Input):
    sequence: int = Field(ge=0, le=2**53 - 1)
    camera: dict


class FrameInput(Input):
    ticket: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)


class Selection(FrameInput):
    mode: Literal["visible", "depth", "through"] = "through"
    layer_tolerance: float = Field(default=0.02, ge=0, le=1)
    shape: Literal["polygon", "box", "clear"]
    polygon: list[list[float]] = Field(default_factory=list, max_length=128)
    depth_range: list[float] = Field(
        default_factory=lambda: [0.01, 100.0], min_length=2, max_length=2
    )
    minimum: list[float] = Field(default_factory=list, max_length=3)
    maximum: list[float] = Field(default_factory=list, max_length=3)
    coverage: bool = False
    combine: Literal["replace", "add", "subtract"] = "replace"


class Protection(FrameInput):
    kind: Literal["add", "remove", "clear"]
    selection_token: str | None = Field(default=None, max_length=128)


class DepthPick(FrameInput):
    pixel: list[float] = Field(min_length=2, max_length=2)


class Preview(FrameInput):
    selection_token: str = Field(min_length=1, max_length=128)
    mode: Literal["highlight", "isolated", "after_delete"]


class Operation(FrameInput):
    operation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    kind: Literal["delete", "undo", "redo"]
    selection_token: str | None = Field(default=None, max_length=128)
    confirm_large: bool = False


class Revision(Input):
    expected_revision: int = Field(ge=0)


class Offer(Input):
    type: Literal["offer"]
    sdp: str = Field(min_length=1, max_length=65536)


def editor_router(service, *, capability_provider=cloud_media.capabilities):
    router = APIRouter(prefix="/api", route_class=EditorRoute)
    disk_lock = asyncio.Lock()

    async def disk(function, *args, **kwargs):
        if disk_lock.locked():
            raise EditConflict("文档读取/创建尚未结束，请稍后重试")
        async with disk_lock:
            return await offload(function, *args, **kwargs)

    def session(request, session_id):
        return service.authenticate(session_id, request.headers.get("x-editor-token"))

    @router.get("/gaussian-editor/capabilities")
    async def capability():
        return capability_provider()

    @router.post("/gaussian-edits", status_code=201)
    async def create_edit(value: Source):
        document = await disk(service.edits.create, **value.model_dump())
        return await disk(service.edits.describe, document["edit_id"])

    @router.get("/gaussian-edits")
    async def list_edits(job_id: str):
        return {"edits": await disk(service.edits.list_documents, job_id)}

    @router.get("/gaussian-edits/{edit_id}")
    async def get_edit(edit_id: str):
        return await disk(service.edits.describe, edit_id)

    @router.post("/gaussian-render-sessions", status_code=202)
    async def create_session(value: SessionInput):
        capability = capability_provider()
        if not capability["cloud_available"]:
            raise HTTPException(503, capability["reason"])
        return await service.create(value.edit_id, value.version)

    @router.get("/gaussian-render-sessions/{session_id}")
    async def status(request: Request, session_id: str):
        return service.status(session(request, session_id))

    @router.delete("/gaussian-render-sessions/{session_id}")
    async def close(request: Request, session_id: str):
        await service.close(session(request, session_id))
        return {"state": "closed"}

    @router.post("/gaussian-render-sessions/{session_id}/prepare-frame")
    async def prepare(request: Request, session_id: str, value: Freeze):
        return await service.prepare(
            session(request, session_id), value.sequence, value.camera
        )

    @router.post("/gaussian-render-sessions/{session_id}/freeze-prepared")
    async def freeze_prepared(request: Request, session_id: str, value: FrameInput):
        return await service.freeze_prepared(
            session(request, session_id), value.ticket, value.expected_revision
        )

    @router.post("/gaussian-render-sessions/{session_id}/protection")
    async def protection(request: Request, session_id: str, value: Protection):
        return await service.protection(
            session(request, session_id), value.model_dump()
        )

    @router.post("/gaussian-render-sessions/{session_id}/depth-pick")
    async def depth_pick(request: Request, session_id: str, value: DepthPick):
        return await service.depth_pick(
            session(request, session_id), value.model_dump()
        )

    @router.post("/gaussian-render-sessions/{session_id}/freeze-frame")
    async def freeze(request: Request, session_id: str, value: Freeze):
        return await service.freeze(
            session(request, session_id), value.sequence, value.camera
        )

    @router.post("/gaussian-render-sessions/{session_id}/selection")
    async def selection(request: Request, session_id: str, value: Selection):
        return await service.selection(session(request, session_id), value.model_dump())

    @router.post("/gaussian-render-sessions/{session_id}/preview")
    async def preview(request: Request, session_id: str, value: Preview):
        return await service.preview(session(request, session_id), value.model_dump())

    @router.post("/gaussian-render-sessions/{session_id}/operations")
    async def operation(request: Request, session_id: str, value: Operation):
        return await service.operation(session(request, session_id), value.model_dump())

    @router.post("/gaussian-render-sessions/{session_id}/invalidate-frame")
    async def invalidate(request: Request, session_id: str):
        s = session(request, session_id)
        async with service.action(s):
            s["tickets"].invalidate()
            s["selection"] = None
        return {"invalidated": True}

    @router.post("/gaussian-render-sessions/{session_id}/resume")
    async def resume(request: Request, session_id: str):
        await service.resume(session(request, session_id))
        return service.status(session(request, session_id))

    @router.get("/gaussian-render-sessions/{session_id}/ice")
    async def ice(request: Request, session_id: str):
        session(request, session_id)
        return cloud_media.browser_ice(session_id)

    @router.post("/gaussian-render-sessions/{session_id}/offer")
    async def offer(request: Request, session_id: str, value: Offer):
        s = session(request, session_id)
        async with service.action(s):
            if s["media"] is not None:
                raise EditConflict("媒体已连接；恢复观看会先关闭旧连接")
            if not capability_provider()["cloud_available"]:
                raise HTTPException(503, "媒体依赖或 TURN 配置不可用")
            media = cloud_media.CloudMedia(service, s)
            s["media"] = media
            try:
                return await asyncio.wait_for(media.answer(value.sdp), timeout=20)
            except BaseException:
                await media.close()
                s["media"] = None
                raise HTTPException(
                    502, "WebRTC 协商失败，请核验同机 TURN / ICE 网络"
                ) from None

    @router.post("/gaussian-render-sessions/{session_id}/versions")
    async def save(request: Request, session_id: str, value: Revision):
        return await service.save(session(request, session_id), value.expected_revision)

    @router.post(
        "/gaussian-render-sessions/{session_id}/exports/{version}", status_code=202
    )
    async def export(request: Request, session_id: str, version: str):
        return await service.start_export(session(request, session_id), version)

    @router.get("/gaussian-edits/{edit_id}/versions/{version}/export")
    async def export_status(edit_id: str, version: str):
        state = service.export_state
        if state and state["edit_id"] == edit_id and state["version"] == version:
            return state.copy()
        try:
            await disk(service.edits.export_asset, edit_id, version, "export.json")
            return {"status": "done"}
        except FileNotFoundError:
            return {"status": "not_exported"}

    @router.get("/gaussian-edits/{edit_id}/versions/{version}/assets/{name}")
    async def download(edit_id: str, version: str, name: str):
        path = await disk(service.edits.export_asset, edit_id, version, name)
        return FileResponse(path, filename=f"{edit_id}-{version}-{name}")

    return router
