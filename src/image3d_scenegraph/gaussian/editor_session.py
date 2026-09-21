"""Single-user editing lifecycle; CUDA stays in CloudRenderProcess."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

from image3d_scenegraph.gpu_lease import gpu_lease_path
from .cloud_render import (
    CloudCamera,
    CloudRenderError,
    CloudRenderProcess,
    FrozenFrameTickets,
)
from .editing import (
    EditConflict,
    GaussianEditError,
    GaussianEditStore,
    load_edit_source_rows,
    select_box,
    select_polygon,
)


async def offload(function, *args, **kwargs):
    # Cancellation must not release a session fence while its thread still uses the renderer.
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def png_data(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(
        "ascii"
    )


class EditorSessions:
    def __init__(self, jobs, *, renderer_factory=CloudRenderProcess):
        self.edits = GaussianEditStore(jobs)
        self.renderer_factory = renderer_factory
        self.active = None
        self.last_closed = None
        self.export_task = None
        self.export_state = None
        self._reaper = None

    def start(self):
        self._reaper = asyncio.create_task(self._expire())

    async def shutdown(self):
        if self._reaper:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
        if self.active:
            await self.close(self.active)
        if self.export_task:
            await asyncio.shield(self.export_task)

    async def _expire(self):
        while True:
            await asyncio.sleep(5)
            session = self.active
            if session and (
                time.monotonic() - session["heartbeat"] > 30
                or time.monotonic() - session["interaction"] > 600
            ):
                await self.close(session)

    async def create(self, edit_id, version=None):
        if self.active is not None:
            raise EditConflict("已有云端会话（含加载/关闭中）；请先关闭或等待过期")
        session = {
            "id": uuid.uuid4().hex,
            "token": secrets.token_urlsafe(32),
            "edit_id": edit_id,
            "version": version,
            "state": "loading",
            "error": None,
            "heartbeat": time.monotonic(),
            "interaction": time.monotonic(),
            "lock": asyncio.Lock(),
            "tickets": FrozenFrameTickets(),
            "camera": None,
            "seq": -1,
            "selection": None,
            "media": None,
            "renderer": None,
            "render_mask": None,
        }
        self.active = session
        session["loader"] = asyncio.create_task(self._load(session))
        return self.status(session) | {"token": session["token"]}

    async def _load(self, s):
        try:
            async with s["lock"]:
                document = await offload(self.edits.get, s["edit_id"])
                source = document["source"]
                rows = await offload(load_edit_source_rows, self.edits.jobs, source)
                s["means"] = rows[:, :3].copy()
                s["radii"] = 3 * np.exp(np.clip(rows[:, 55:58].max(axis=1), -30, 30))
                del rows
                s["source"] = source
                s["revision"] = document["revision"]
                if s["version"]:
                    saved = await offload(
                        self.edits.version, s["edit_id"], s["version"]
                    )
                    s["revision"] = saved["revision"]
                    s["visible"] = await offload(
                        self.edits.version_visible, s["edit_id"], s["version"]
                    )
                else:
                    s["visible"] = await offload(self.edits.visible, s["edit_id"])
                scratch = self.edits.root / ".render-scratch"
                scratch.mkdir(exist_ok=True)
                s["renderer"] = self.renderer_factory(
                    source=self.edits.jobs.get_asset_path(
                        source["job_id"], source["ply_asset"]
                    ),
                    source_sha256=source["ply_sha256"],
                    count=source["gaussian_count"],
                    lease_path=gpu_lease_path(self.edits.jobs.output_root),
                    scratch_root=scratch,
                )
                await offload(s["renderer"].start)
                if s["state"] == "loading":
                    s["state"] = "viewing"
        except Exception as exc:
            s["error"] = str(exc)
            s["state"] = "error"
            if s["renderer"]:
                await offload(s["renderer"].close)

    def authenticate(self, session_id, token):
        s = (
            self.active
            if self.active and self.active["id"] == session_id
            else self.last_closed
        )
        if (
            s is None
            or s["id"] != session_id
            or not isinstance(token, str)
            or not token.isascii()
            or not secrets.compare_digest(s["token"], token)
        ):
            raise PermissionError("会话凭证无效或会话已关闭")
        s["heartbeat"] = time.monotonic()
        return s

    def status(self, s):
        return {
            "session_id": s["id"],
            "edit_id": s["edit_id"],
            "version": s["version"],
            "state": s["state"],
            "error": s["error"],
            "revision": s.get("revision"),
            "visible_count": int(s["visible"].sum()) if "visible" in s else None,
        }

    @asynccontextmanager
    async def action(self, s, *, writable=False):
        if s is not self.active or s["state"] not in {"viewing", "editing_frozen"}:
            raise EditConflict("会话尚未就绪或正在关闭")
        if writable and s["version"]:
            raise EditConflict("保存版本只读；请打开当前编辑文档")
        if s.get("action_pending"):
            raise EditConflict("上一项操作尚未完成，请稍后重试")
        s["action_pending"] = True
        try:
            async with s["lock"]:
                if s["state"] not in {"viewing", "editing_frozen"}:
                    raise EditConflict("会话正在关闭或已失败")
                s["interaction"] = time.monotonic()
                yield
        finally:
            s["action_pending"] = False

    async def close(self, s):
        if s["state"] == "closed":
            return
        if s.get("closer"):
            return await asyncio.shield(s["closer"])
        s["state"] = "closing"
        s["closer"] = asyncio.create_task(self._close(s))
        await asyncio.shield(s["closer"])

    async def _close(self, s):
        await asyncio.shield(s["loader"])
        s["state"] = "closing"
        if s["media"]:
            await s["media"].close()
            s["media"] = None
        async with s["lock"]:
            if s["renderer"]:
                await offload(s["renderer"].close)
            s["tickets"].invalidate()
            s["selection"] = None
            s.pop("means", None)
            s.pop("radii", None)
            s["state"] = "closed"
            self.last_closed = {
                key: s.get(key)
                for key in (
                    "id",
                    "token",
                    "edit_id",
                    "version",
                    "state",
                    "error",
                    "revision",
                )
            }
            if self.active is s:
                self.active = None

    def camera_input(self, s, sequence, value):
        if (
            s["state"] != "viewing"
            or type(sequence) is not int
            or not 0 <= sequence <= 2**53 - 1
        ):
            return False
        if sequence <= s["seq"]:
            return False
        camera = CloudCamera.from_json(value)
        s.update(camera=camera, seq=sequence, interaction=time.monotonic())
        return True

    async def render(self, s, camera, visible):
        mask = s["render_mask"]
        changed = mask is None or not np.array_equal(mask, visible)
        try:
            frame = await offload(
                s["renderer"].render, camera, visible=visible if changed else None
            )
        except Exception:
            s["state"] = "error"
            s["error"] = "渲染进程失败，请关闭会话后重新连接；已确认的编辑仍然保留"
            raise
        s["render_mask"] = visible.copy() if changed else mask
        return frame["rgb"]

    async def video_frame(self, s):
        while s is self.active and s["state"] in {"viewing", "editing_frozen"}:
            if (
                s["state"] == "viewing"
                and s["camera"]
                and not s["lock"].locked()
                and not s.get("action_pending")
                and not s.get("paused")
            ):
                async with s["lock"]:
                    camera, sequence = s["camera"], s["seq"]
                    value = camera.to_json()
                    limit = 1280 if time.monotonic() - s["interaction"] < 0.3 else 1920
                    ratio = min(1, limit / max(camera.width, camera.height))
                    w, h = (
                        max(16, int(camera.width * ratio)),
                        max(16, int(camera.height * ratio)),
                    )
                    k = camera.intrinsic.copy()
                    k[0] *= w / camera.width
                    k[1] *= h / camera.height
                    value.update(width=w, height=h, intrinsic=k.tolist())
                    rgb = await self.render(
                        s, CloudCamera.from_json(value), s["visible"]
                    )
                    return rgb, sequence, s["revision"]
            await asyncio.sleep(0.03)
        raise CloudRenderError("会话已结束")

    async def freeze(self, s, sequence, camera_value):
        async with self.action(s):
            camera = CloudCamera.from_json(camera_value)
            if type(sequence) is not int or sequence < s["seq"] or sequence < 0:
                raise EditConflict("固定帧相机序号已过期")
            s.update(
                state="editing_frozen", camera=camera, seq=sequence, selection=None
            )
            return await self._frozen(s)

    async def _frozen(self, s):
        rgb = await self.render(s, s["camera"], s["visible"])
        ticket = s["tickets"].issue(
            source_sha256=s["source"]["ply_sha256"],
            revision=s["revision"],
            camera_seq=s["seq"],
            camera=s["camera"],
        )
        s["frame_ticket"] = ticket
        return {
            "ticket": ticket,
            "revision": s["revision"],
            "camera_seq": s["seq"],
            "width": s["camera"].width,
            "height": s["camera"].height,
            "image": await offload(png_data, rgb),
        }

    def validate_frame(self, s, ticket, revision):
        if s["state"] != "editing_frozen" or revision != s["revision"]:
            raise EditConflict("固定帧或编辑版本已变化，请重新固定画面")
        s["tickets"].validate(
            ticket,
            source_sha256=s["source"]["ply_sha256"],
            revision=revision,
            camera_seq=s["seq"],
            camera=s["camera"],
        )

    async def selection(self, s, request):
        async with self.action(s):
            self.validate_frame(s, request["ticket"], request["expected_revision"])
            kind = request["shape"]
            if kind == "clear":
                selected = np.zeros(len(s["visible"]), dtype=bool)
            elif kind == "box":
                selected = await offload(
                    select_box, s["means"], request["minimum"], request["maximum"]
                )
            else:
                selected = await offload(
                    select_polygon,
                    s["means"],
                    s["camera"],
                    request["polygon"],
                    request["depth_range"],
                    radii=s["radii"] if request["coverage"] else None,
                )
            old = s["selection"]
            if kind != "clear" and request["combine"] != "replace":
                previous = old["mask"] if old else np.zeros_like(selected)
                selected = (
                    (previous | selected)
                    if request["combine"] == "add"
                    else (previous & ~selected)
                )
            selected &= s["visible"]
            token = secrets.token_urlsafe(32)
            s["selection"] = {
                "token": token,
                "mask": selected,
                "ticket": request["ticket"],
                "previewed": False,
            }
            return {
                "selection_token": token,
                "selected_count": int(selected.sum()),
                "visible_count": int(s["visible"].sum()),
                "revision": s["revision"],
            }

    def selected(self, s, token, ticket, revision):
        self.validate_frame(s, ticket, revision)
        selection = s["selection"]
        if (
            not selection
            or selection["token"] != token
            or selection["ticket"] != ticket
        ):
            raise EditConflict("选择已失效，请重新预览")
        return selection["mask"]

    async def preview(self, s, request):
        async with self.action(s):
            selected = self.selected(
                s,
                request["selection_token"],
                request["ticket"],
                request["expected_revision"],
            )
            kind = request["mode"]
            mask = s["visible"] & ~selected if kind == "after_delete" else selected
            if not mask.any():
                raise GaussianEditError("预览集合为空，请调整选择")
            rgb = await self.render(s, s["camera"], mask)
            if kind == "highlight":
                original = await self.render(s, s["camera"], s["visible"])
                strength = np.clip(rgb.max(axis=2).astype(np.float32) / 255, 0, 0.65)[
                    ..., None
                ]
                rgb = (
                    original * (1 - strength) + np.array([255, 80, 160]) * strength
                ).astype(np.uint8)
            image = await offload(png_data, rgb)
            s["selection"]["previewed"] = True
            return {"image": image, "revision": s["revision"]}

    async def refresh_committed(self, s):
        document = await offload(self.edits.get, s["edit_id"])
        if document["revision"] != s["revision"]:
            s["visible"] = await offload(self.edits.visible, s["edit_id"])
            s["revision"] = document["revision"]
            s["selection"] = None
            s["tickets"].invalidate()

    async def operation(self, s, request):
        async with self.action(s, writable=True):
            key = request["operation_id"]
            digest = hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest()
            old = await offload(
                self.edits.acknowledged_request, s["edit_id"], key, digest
            )
            if old is not None:
                await self.refresh_committed(s)
                return old
            self.validate_frame(s, request["ticket"], request["expected_revision"])
            mask = None
            if request["kind"] == "delete":
                mask = self.selected(
                    s,
                    request["selection_token"],
                    request["ticket"],
                    request["expected_revision"],
                )
                if not s["selection"]["previewed"]:
                    raise EditConflict("必须先预览当前选择，再确认删除")
            try:
                ack = await offload(
                    self.edits.apply,
                    s["edit_id"],
                    operation_id=key,
                    expected_revision=request["expected_revision"],
                    kind=request["kind"],
                    selected=mask,
                    confirm_large=request["confirm_large"],
                    api_request_sha256=digest,
                )
            finally:
                await self.refresh_committed(s)
            return ack

    async def resume(self, s):
        async with self.action(s):
            if s["media"]:
                await s["media"].close()
                s["media"] = None
            s["selection"] = None
            s["tickets"].invalidate()
            s["state"] = "viewing"

    async def save(self, s, revision):
        async with self.action(s, writable=True):
            return await offload(
                self.edits.save_version, s["edit_id"], expected_revision=revision
            )

    async def start_export(self, s, version):
        async with self.action(s):
            await offload(self.edits.version, s["edit_id"], version)
            if self.export_task and not self.export_task.done():
                if (
                    self.export_state["edit_id"] == s["edit_id"]
                    and self.export_state["version"] == version
                ):
                    return self.export_state.copy()
                raise EditConflict("另一个导出正在执行")
            state = {
                "edit_id": s["edit_id"],
                "version": version,
                "status": "running",
                "error": None,
            }
            self.export_state = state
            self.export_task = asyncio.create_task(self._export(state))
            return state.copy()

    async def _export(self, state):
        try:
            try:
                await offload(
                    self.edits.export_asset,
                    state["edit_id"],
                    state["version"],
                    "export.json",
                )
            except FileNotFoundError:
                await offload(
                    self.edits.export_version, state["edit_id"], state["version"]
                )
            state["status"] = "done"
        except Exception as exc:
            state.update(status="error", error=str(exc))
