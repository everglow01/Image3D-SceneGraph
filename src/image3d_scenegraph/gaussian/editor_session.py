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
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from PIL import Image

from image3d_scenegraph.gpu_lease import FileLease, gpu_lease_path
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


if TYPE_CHECKING:
    from .cloud_media import CloudMedia


class SessionIdentity(TypedDict):
    id: str
    token: str
    edit_id: str
    version: str | None
    state: Literal["loading", "viewing", "editing_frozen", "error", "closing", "closed"]
    error: str | None


class SelectionState(TypedDict):
    token: str
    mask: np.ndarray
    ticket: str
    previewed: bool


class RenderSession(SessionIdentity, total=False):
    # Loading sessions and retained closed identities do not contain model state.
    revision: int | None
    heartbeat: float
    interaction: float
    lock: asyncio.Lock
    loader: asyncio.Task[None]
    closer: asyncio.Task[None]
    action_pending: bool
    tickets: FrozenFrameTickets
    camera: CloudCamera | None
    seq: int
    selection: SelectionState | None
    media: CloudMedia | None
    renderer: CloudRenderProcess | None
    source: dict[str, Any]
    means: np.ndarray
    radii: np.ndarray
    visible: np.ndarray
    protected: np.ndarray | None
    render_mask: np.ndarray | None
    render_cache: tuple[str, np.ndarray] | None
    prepared: dict[str, Any] | None
    render_ms: float
    motion_limit: int
    quality_samples: int
    paused: bool
    document_lease: FileLease


class LocalAuthorization(TypedDict):
    edit_id: str
    token: str
    identity: dict[str, Any]
    expires: float
    lock: asyncio.Lock
    lease: FileLease
    ready: bool
    closing: bool


LOCAL_AUTH_SECONDS = 600
LOCAL_AUTH_LIMIT = 64


async def offload(function, *args, **kwargs):
    # Cancellation must not release a session fence while its thread still uses the renderer.
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
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
        self.active: RenderSession | None = None
        self.last_closed: RenderSession | None = None
        self.export_task = None
        self.export_state = None
        self._reaper = None
        self.local: dict[str, LocalAuthorization] = {}

    def start(self):
        self._reaper = asyncio.create_task(self._expire())

    async def shutdown(self):
        if self._reaper:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
        if self.active:
            await self.close(self.active)
        for authorization in list(self.local.values()):
            await self.close_local(authorization)
        if self.export_task:
            await asyncio.shield(self.export_task)

    async def _expire(self):
        while True:
            await asyncio.sleep(5)
            self.expire_local()
            session = self.active
            if session and (
                time.monotonic() - session["heartbeat"] > 30
                or time.monotonic() - session["interaction"] > 600
            ):
                await self.close(session)

    def _document_lease(self, edit_id: str) -> FileLease:
        self.expire_local()
        return FileLease(self.edits._directory(edit_id) / ".writer.lock").acquire()

    def _release_local(self, authorization: LocalAuthorization):
        if self.local.get(authorization["edit_id"]) is authorization:
            del self.local[authorization["edit_id"]]
        authorization["lease"].close()

    def expire_local(self):
        for authorization in list(self.local.values()):
            if (time.monotonic() >= authorization["expires"]
                    and not authorization["lock"].locked()):
                self._release_local(authorization)

    async def open_local(self, edit_id: str, identity: dict):
        self.expire_local()
        if len(self.local) >= LOCAL_AUTH_LIMIT:
            raise EditConflict("本地文档授权已达上限，请关闭闲置文档")
        lease = self._document_lease(edit_id)
        authorization: LocalAuthorization = {
            "edit_id": edit_id, "token": secrets.token_urlsafe(32),
            "identity": identity.copy(), "expires": time.monotonic() + LOCAL_AUTH_SECONDS,
            "lock": asyncio.Lock(), "lease": lease, "ready": False, "closing": False,
        }
        self.local[edit_id] = authorization
        try:
            async with authorization["lock"]:
                state, _ = await offload(self.edits.local_snapshot, edit_id, identity)
                authorization["ready"] = True
                return {
                    "edit_id": edit_id, "token": authorization["token"],
                    "identity": identity, **state,
                    "expires_in_seconds": max(0, int(authorization["expires"] - time.monotonic())),
                }
        except BaseException:
            self._release_local(authorization)
            raise

    def authenticate_local(self, edit_id: str, token: str | None) -> LocalAuthorization:
        authorization = self.local.get(edit_id)
        if (
            authorization is None or not authorization["ready"]
            or authorization["closing"] or time.monotonic() >= authorization["expires"]
            or not isinstance(token, str) or not token.isascii()
            or not secrets.compare_digest(authorization["token"], token)
        ):
            raise PermissionError("本地文档凭证无效、过期或已关闭；请重新授权并核对revision")
        return authorization

    @asynccontextmanager
    async def local_action(self, authorization: LocalAuthorization):
        self.authenticate_local(authorization["edit_id"], authorization["token"])
        if authorization["lock"].locked():
            raise EditConflict("本地文档上一项操作尚未完成")
        try:
            async with authorization["lock"]:
                yield
        finally:
            self.expire_local()

    async def close_local(self, authorization: LocalAuthorization):
        authorization["closing"] = True
        async def close():
            async with authorization["lock"]:
                self._release_local(authorization)
        await asyncio.shield(asyncio.create_task(close()))

    async def read_local(self, authorization: LocalAuthorization, version: str | None = None):
        async with self.local_action(authorization):
            return await offload(
                self.edits.local_snapshot, authorization["edit_id"], authorization["identity"],
                version=version,
            )

    async def submit_local(self, authorization: LocalAuthorization, content: bytes, request: dict):
        async with self.local_action(authorization):
            return await offload(
                self.edits.submit_snapshot, authorization["edit_id"],
                identity=authorization["identity"], content=content, **request,
            )

    async def save_local(self, authorization: LocalAuthorization, revision: int):
        async with self.local_action(authorization):
            await offload(self.edits.check_source, authorization["edit_id"], authorization["identity"])
            return await offload(
                self.edits.save_version, authorization["edit_id"], expected_revision=revision
            )

    async def export_local(self, authorization: LocalAuthorization, version: str):
        async with self.local_action(authorization):
            await offload(self.edits.check_source, authorization["edit_id"], authorization["identity"])
            return await self._start_export(authorization["edit_id"], version)

    async def create(self, edit_id, version=None):
        if self.active is not None:
            raise EditConflict("已有云端会话（含加载/关闭中）；请先关闭或等待过期")
        lease = self._document_lease(edit_id)
        session: RenderSession = {
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
            "render_cache": None,
            "prepared": None,
            "protected": None,
            "render_ms": 0.0,
            "motion_limit": 1280,
            "quality_samples": 0,
            "document_lease": lease,
        }
        self.active = session
        session["loader"] = asyncio.create_task(self._load(session))
        return self.status(session) | {"token": session["token"]}

    async def _load(self, s: RenderSession):
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
                s["protected"] = np.zeros(source["gaussian_count"], dtype=bool)
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

    def authenticate(self, session_id, token) -> RenderSession:
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

    def status(self, s: RenderSession):
        return {
            "session_id": s["id"],
            "edit_id": s["edit_id"],
            "version": s["version"],
            "state": s["state"],
            "error": s["error"],
            "revision": s.get("revision"),
            "visible_count": int(s["visible"].sum()) if "visible" in s else None,
            "protected_count": int(s["protected"].sum())
            if s.get("protected") is not None
            else 0,
            "render_ms": round(s.get("render_ms", 0), 2),
        }

    @asynccontextmanager
    async def action(self, s: RenderSession, *, writable=False):
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

    async def close(self, s: RenderSession):
        if s["state"] == "closed":
            return
        if s.get("closer"):
            return await asyncio.shield(s["closer"])
        s["state"] = "closing"
        s["closer"] = asyncio.create_task(self._close(s))
        await asyncio.shield(s["closer"])

    async def _close(self, s: RenderSession):
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
                "id": s["id"], "token": s["token"], "edit_id": s["edit_id"],
                "version": s["version"], "state": s["state"], "error": s["error"],
                "revision": s.get("revision"),
            }
            s["document_lease"].close()
            if self.active is s:
                self.active = None

    def camera_input(self, s: RenderSession, sequence, value):
        if (
            s["state"] != "viewing"
            or type(sequence) is not int
            or not 0 <= sequence <= 2**53 - 1
        ):
            return False
        if sequence <= s["seq"]:
            return False
        camera = CloudCamera.from_json(value)
        s["prepared"] = None
        s["tickets"].invalidate()
        s.update(camera=camera, seq=sequence, interaction=time.monotonic())
        return True

    async def render(self, s: RenderSession, camera, visible):
        mask = s["render_mask"]
        changed = mask is None or not np.array_equal(mask, visible)
        cached = s.get("render_cache")
        if not changed and cached and cached[0] == camera.digest:
            return cached[1]
        started = time.monotonic()
        try:
            frame = await offload(
                s["renderer"].render, camera, visible=visible if changed else None
            )
        except Exception:
            s["state"] = "error"
            s["error"] = "渲染进程失败，请关闭会话后重新连接；已确认的编辑仍然保留"
            raise
        s["render_ms"] = (time.monotonic() - started) * 1000
        s["render_mask"] = visible.copy() if changed else mask
        s["render_cache"] = (camera.digest, frame["rgb"])
        return frame["rgb"]

    async def video_frame(self, s: RenderSession):
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
                    moving = time.monotonic() - s["interaction"] < 0.4
                    limit = s.get("motion_limit", 1280) if moving else 1920
                    ratio = min(1, limit / max(camera.width, camera.height))
                    w, h = (
                        max(16, int(camera.width * ratio)),
                        max(16, int(camera.height * ratio)),
                    )
                    k = camera.intrinsic.copy()
                    k[0] *= w / camera.width
                    k[1] *= h / camera.height
                    value.update(width=w, height=h, intrinsic=k.tolist())
                    before = s.get("render_cache")
                    rgb = await self.render(
                        s, CloudCamera.from_json(value), s["visible"]
                    )
                    if moving and s.get("render_cache") is not before:
                        s["quality_samples"] = s.get("quality_samples", 0) + 1
                        bands = (640, 960, 1280)
                        index = bands.index(s.get("motion_limit", 1280))
                        ms = s.get("render_ms", 0)
                        if ms > 40 and s["quality_samples"] >= 4 and index:
                            s["motion_limit"] = bands[index - 1]
                            s["quality_samples"] = 0
                        elif ms < 18 and s["quality_samples"] >= 30 and index < 2:
                            s["motion_limit"] = bands[index + 1]
                            s["quality_samples"] = 0
                    return rgb, sequence, s["revision"]
            await asyncio.sleep(0.03)
        raise CloudRenderError("会话已结束")

    async def prepare(self, s: RenderSession, sequence, camera_value):
        async with self.action(s):
            if s["state"] != "viewing":
                raise EditConflict("当前不在导航模式")
            camera = CloudCamera.from_json(camera_value)
            if sequence < s["seq"] or (
                sequence == s["seq"]
                and s["camera"]
                and camera.digest != s["camera"].digest
            ):
                raise EditConflict("相机已变化，等待最新高清画面")
            s.update(camera=camera, seq=sequence, prepared=None, selection=None)
            result = await self._frozen(s)
            s["prepared"] = {
                key: result[key] for key in ("ticket", "revision", "camera_seq")
            }
            return result

    async def freeze_prepared(self, s: RenderSession, ticket, revision):
        async with self.action(s):
            prepared = s.get("prepared")
            if s["state"] != "viewing" or not prepared or prepared["ticket"] != ticket:
                raise EditConflict("高清画面已变化，请等待最新画面")
            if revision != s["revision"] or prepared["camera_seq"] != s["seq"]:
                raise EditConflict("高清画面版本或相机已变化")
            s["tickets"].validate(
                ticket,
                source_sha256=s["source"]["ply_sha256"],
                revision=revision,
                camera_seq=s["seq"],
                camera=s["camera"],
            )
            s.update(state="editing_frozen", selection=None, prepared=None)
            return {"ticket": ticket, "revision": revision, "camera_seq": s["seq"]}

    async def freeze(self, s: RenderSession, sequence, camera_value):
        async with self.action(s):
            camera = CloudCamera.from_json(camera_value)
            if type(sequence) is not int or sequence < s["seq"] or sequence < 0:
                raise EditConflict("固定帧相机序号已过期")
            s.update(
                state="editing_frozen", camera=camera, seq=sequence, selection=None
            )
            return await self._frozen(s)

    async def _frozen(self, s: RenderSession):
        camera, sequence, revision = s["camera"], s["seq"], s["revision"]
        rgb = await self.render(s, camera, s["visible"])
        image = await offload(png_data, rgb)
        if (
            s["seq"] != sequence
            or s["revision"] != revision
            or s["camera"].digest != camera.digest
        ):
            raise EditConflict("相机已变化，丢弃过期高清画面")
        ticket = s["tickets"].issue(
            source_sha256=s["source"]["ply_sha256"],
            revision=revision,
            camera_seq=sequence,
            camera=camera,
        )
        return {
            "ticket": ticket,
            "revision": revision,
            "camera_seq": sequence,
            "camera_digest": camera.digest,
            "width": camera.width,
            "height": camera.height,
            "image": image,
            "render_ms": round(s.get("render_ms", 0), 2),
        }

    async def invalidate_frame(self, s: RenderSession):
        async with self.action(s):
            s["tickets"].invalidate()
            s["selection"] = None

    def validate_frame(self, s: RenderSession, ticket, revision):
        if s["state"] != "editing_frozen" or revision != s["revision"]:
            raise EditConflict("固定帧或编辑版本已变化，请重新固定画面")
        s["tickets"].validate(
            ticket,
            source_sha256=s["source"]["ply_sha256"],
            revision=revision,
            camera_seq=s["seq"],
            camera=s["camera"],
        )

    async def visible_pick(self, s: RenderSession, polygon, depth_range, tolerance=0.02):
        try:
            return await offload(
                s["renderer"].pick,
                s["camera"],
                visible=s["visible"],
                polygon=polygon,
                depth_range=depth_range,
                tolerance=tolerance,
            )
        finally:
            # A pick may change the renderer's active mask after an isolated preview.
            s["render_mask"] = None
            s["render_cache"] = None

    async def selection(self, s: RenderSession, request):
        async with self.action(s):
            self.validate_frame(s, request["ticket"], request["expected_revision"])
            kind = request["shape"]
            if kind == "clear":
                selected = np.zeros(len(s["visible"]), dtype=bool)
            elif kind == "box":
                if request.get("mode", "through") != "through":
                    raise GaussianEditError("三维盒是穿透体积选择，请显式使用穿透模式")
                selected = await offload(
                    select_box, s["means"], request["minimum"], request["maximum"]
                )
            elif request.get("mode", "through") == "visible":
                if request["coverage"]:
                    raise GaussianEditError("可见表层不支持扩大覆盖候选")
                if any(len(point) != 2 for point in request["polygon"]):
                    raise GaussianEditError("可见选区顶点必须为二维坐标")
                polygon = np.asarray(request["polygon"], dtype=np.float64)
                depth = np.asarray(request["depth_range"], dtype=np.float64)
                if (
                    polygon.ndim != 2
                    or polygon.shape[1:] != (2,)
                    or not 3 <= len(polygon) <= 128
                    or not np.isfinite(polygon).all()
                    or (polygon < 0).any()
                    or (polygon > [s["camera"].width, s["camera"].height]).any()
                    or abs(
                        np.sum(
                            polygon[:, 0] * np.roll(polygon[:, 1], 1)
                            - polygon[:, 1] * np.roll(polygon[:, 0], 1)
                        )
                    )
                    < 2
                    or depth.shape != (2,)
                    or not np.isfinite(depth).all()
                    or not 0.01 <= depth[0] < depth[1] <= 1e6
                ):
                    raise GaussianEditError("可见选区或深度无效；未启动GPU选择")
                picked = await self.visible_pick(
                    s,
                    request["polygon"],
                    request["depth_range"],
                    request.get("layer_tolerance", 0.02),
                )
                ids = np.asarray(picked["ids"])
                if (
                    ids.ndim != 1
                    or not np.issubdtype(ids.dtype, np.integer)
                    or (ids < 0).any()
                    or (ids >= len(s["visible"])).any()
                ):
                    raise GaussianEditError("可见选择返回无效源索引；未应用选择")
                selected = np.zeros(len(s["visible"]), dtype=bool)
                selected[ids] = True
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
                "deletable_count": int((selected & ~s["protected"]).sum()),
                "visible_count": int(s["visible"].sum()),
                "revision": s["revision"],
            }

    async def protection(self, s: RenderSession, request):
        async with self.action(s):
            self.validate_frame(s, request["ticket"], request["expected_revision"])
            if request["kind"] == "clear":
                s["protected"][:] = False
            else:
                selected = self.selected(
                    s,
                    request["selection_token"],
                    request["ticket"],
                    request["expected_revision"],
                )
                if request["kind"] == "add":
                    s["protected"] |= selected
                else:
                    s["protected"] &= ~selected
            s["selection"] = None
            return {"protected_count": int(s["protected"].sum())}

    async def depth_pick(self, s: RenderSession, request):
        async with self.action(s):
            self.validate_frame(s, request["ticket"], request["expected_revision"])
            x, y = request["pixel"]
            if (
                not 1 <= x < s["camera"].width - 1
                or not 1 <= y < s["camera"].height - 1
            ):
                raise GaussianEditError("请点击固定图片内部")
            polygon = [[x - 1, y - 1], [x + 1, y - 1], [x + 1, y + 1], [x - 1, y + 1]]
            result = await self.visible_pick(s, polygon, [0.01, 1e6])
            if result["depth"] is None:
                raise GaussianEditError("此处表层深度不确定，请换个位置或手动设置深度")
            return {"depth": result["depth"]}

    def selected(self, s: RenderSession, token, ticket, revision):
        self.validate_frame(s, ticket, revision)
        selection = s["selection"]
        if (
            not selection
            or selection["token"] != token
            or selection["ticket"] != ticket
        ):
            raise EditConflict("选择已失效，请重新预览")
        return selection["mask"]

    async def preview(self, s: RenderSession, request):
        async with self.action(s):
            selected = (
                self.selected(
                    s,
                    request["selection_token"],
                    request["ticket"],
                    request["expected_revision"],
                )
                & ~s["protected"]
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
            s["selection"]["previewed"] = kind in {"isolated", "after_delete"}
            return {"image": image, "revision": s["revision"]}

    async def refresh_committed(self, s: RenderSession):
        document = await offload(self.edits.get, s["edit_id"])
        if document["revision"] != s["revision"]:
            s["visible"] = await offload(self.edits.visible, s["edit_id"])
            s["revision"] = document["revision"]
            s["selection"] = None
            s["tickets"].invalidate()

    async def operation(self, s: RenderSession, request):
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
                mask = (
                    self.selected(
                        s,
                        request["selection_token"],
                        request["ticket"],
                        request["expected_revision"],
                    )
                    & ~s["protected"]
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

    async def resume(self, s: RenderSession):
        async with self.action(s):
            if s["media"]:
                await s["media"].close()
                s["media"] = None
            s["selection"] = None
            s["tickets"].invalidate()
            s["state"] = "viewing"

    async def save(self, s: RenderSession, revision):
        async with self.action(s, writable=True):
            return await offload(
                self.edits.save_version, s["edit_id"], expected_revision=revision
            )

    async def start_export(self, s: RenderSession, version):
        async with self.action(s):
            return await self._start_export(s["edit_id"], version)

    async def _start_export(self, edit_id: str, version: str):
        await offload(self.edits.version, edit_id, version)
        if self.export_task and not self.export_task.done():
            if (
                self.export_state["edit_id"] == edit_id
                and self.export_state["version"] == version
            ):
                return self.export_state.copy()
            raise EditConflict("另一个导出正在执行")
        state = {
            "edit_id": edit_id,
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
