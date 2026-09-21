import asyncio
import json
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.gaussian_editor import editor_router
from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.editor_session import EditorSessions
from image3d_scenegraph.gaussian.editing import EditConflict
from test_gaussian_editing import setup_source, camera


@pytest.fixture(autouse=True)
def editor_origin(monkeypatch):
    monkeypatch.setenv("IMAGE3D_EDITOR_ORIGINS", "http://testserver")


class FakeRenderer:
    closed = False

    def __init__(self, **kwargs):
        self.count = kwargs["count"]

    def start(self):
        pass

    def render(self, camera, *, visible=None):
        if visible is not None:
            self.count = int(visible.sum())
        return {
            "rgb": np.full((camera.height, camera.width, 3), self.count, dtype=np.uint8)
        }

    def close(self):
        self.closed = True


def make_client(tmp_path):
    store, original, _ = setup_source(tmp_path)
    service = EditorSessions(store.jobs, renderer_factory=FakeRenderer)
    app = FastAPI()
    app.include_router(
        editor_router(service, capability_provider=lambda: {"cloud_available": True})
    )
    client = TestClient(app)
    client.headers.update({"origin": "http://testserver", "x-image3d-editor": "1"})
    return client, service, original


def opened(client):
    document = client.post("/api/gaussian-edits", json={"job_id": "source"})
    assert document.status_code == 201, document.text
    edit_id = document.json()["edit_id"]
    response = client.post("/api/gaussian-render-sessions", json={"edit_id": edit_id})
    assert response.status_code == 202, response.text
    data = response.json()
    client.headers["x-editor-token"] = data["token"]
    base = "/api/gaussian-render-sessions/" + data["session_id"]
    for _ in range(100):
        response = client.get(base)
        if response.json()["state"] != "loading":
            break
        time.sleep(0.01)
    assert response.json()["state"] == "viewing", response.text
    return edit_id, base


def freeze(client, base, sequence=1):
    response = client.post(
        base + "/freeze-frame",
        json={"sequence": sequence, "camera": camera().to_json()},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_editor_api_complete_cycle_and_guards(tmp_path):
    client, service, original = make_client(tmp_path)
    before = sha256_file(original)
    with client:
        edit_id, base = opened(client)
        assert (
            client.post(
                "/api/gaussian-render-sessions", json={"edit_id": edit_id}
            ).status_code
            == 409
        )
        assert (
            client.get(base, headers={"x-editor-token": "foreign"}).status_code == 403
        )
        frame = freeze(client, base)
        binding = {"ticket": frame["ticket"], "expected_revision": 0}
        selection = {
            **binding,
            "shape": "box",
            "minimum": [-1, -1, 1],
            "maximum": [-0.2, 1, 3],
        }
        response = client.post(base + "/selection", json=selection)
        assert response.status_code == 200, response.text
        selected = response.json()
        assert selected["selected_count"] == 2
        assert (
            client.post(
                base + "/selection", json={**selection, "mask": [True]}
            ).status_code
            == 422
        )
        assert (
            client.post(
                base + "/selection", json={**selection, "ticket": "old"}
            ).status_code
            == 422
        )
        for mode in ("highlight", "isolated", "after_delete"):
            response = client.post(
                base + "/preview",
                json={
                    **binding,
                    "selection_token": selected["selection_token"],
                    "mode": mode,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["image"].startswith("data:image/png;base64,")
        operation = {
            **binding,
            "selection_token": selected["selection_token"],
            "operation_id": "delete-1",
            "kind": "delete",
        }
        ack = client.post(base + "/operations", json=operation)
        assert ack.status_code == 200, ack.text
        assert ack.json()["visible_count"] == 6
        assert client.post(base + "/operations", json=operation).json() == ack.json()
        assert (
            client.post(
                base + "/operations", json={**operation, "kind": "undo"}
            ).status_code
            == 409
        )
        assert client.post(base + "/selection", json=selection).status_code == 409
        for revision, kind, count in ((1, "undo", 8), (2, "redo", 6)):
            frame = freeze(client, base)
            response = client.post(
                base + "/operations",
                json={
                    "ticket": frame["ticket"],
                    "expected_revision": revision,
                    "kind": kind,
                    "operation_id": kind,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["visible_count"] == count
        response = client.post(base + "/versions", json={"expected_revision": 3})
        assert response.status_code == 200, response.text
        version = response.json()["version"]
        assert client.post(base + "/exports/" + version).status_code == 202
        state_url = f"/api/gaussian-edits/{edit_id}/versions/{version}/export"
        for _ in range(100):
            state = client.get(state_url).json()
            if state["status"] != "running":
                break
            time.sleep(0.01)
        assert state["status"] == "done", state
        download = client.get(
            f"/api/gaussian-edits/{edit_id}/versions/{version}/assets/bundle.zip"
        )
        assert download.status_code == 200
        assert download.content[:2] == b"PK"
        assert client.delete(base).status_code == 200
        assert service.active is None
        reopened = client.get(f"/api/gaussian-edits/{edit_id}").json()
        assert reopened["visible_count"] == 6
        assert reopened["quality_role"] == "manual_edit_not_evaluated"
        assert len(client.get("/api/gaussian-edits?job_id=source").json()["edits"]) == 1
        assert sha256_file(original) == before
        assert reopened["versions"][0]["exported"] is True
        session_info = client.post(
            "/api/gaussian-render-sessions", json={"edit_id": edit_id}
        ).json()
        client.headers["x-editor-token"] = session_info["token"]
        reopened_base = "/api/gaussian-render-sessions/" + session_info["session_id"]
        for _ in range(100):
            if client.get(reopened_base).json()["state"] == "viewing":
                break
            time.sleep(0.01)
        assert (
            client.post(reopened_base + "/operations", json=operation).json()
            == ack.json()
        )
        assert client.get(reopened_base).json()["revision"] == 3
        assert client.delete(reopened_base).status_code == 200


def test_origin_limits_and_optional_dependencies(tmp_path, monkeypatch):
    from backend.main import create_app
    from image3d_scenegraph.gaussian.cloud_media import browser_ice, capabilities

    monkeypatch.delenv("IMAGE3D_CLOUD_ENABLED", raising=False)
    with TestClient(create_app(tmp_path / "jobs", start_worker=False)) as client:
        assert client.get("/api/health").status_code == 200
        assert not client.get("/api/gaussian-editor/capabilities").json()[
            "cloud_available"
        ]
        assert (
            client.post("/api/gaussian-edits", json={"job_id": "source"}).status_code
            == 403
        )
        headers = {"origin": "http://testserver", "x-image3d-editor": "1"}
        assert (
            client.post(
                "/api/gaussian-edits",
                json={"job_id": "source"},
                headers={
                    **headers,
                    "origin": "http://foreign.example",
                    "host": "foreign.example",
                },
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/gaussian-edits", content=b"x" * 100000, headers=headers
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/api/gaussian-edits", json={"job_id": "../escape"}, headers=headers
            ).status_code
            == 422
        )
    monkeypatch.setenv("IMAGE3D_TURN_HOST", "turn.internal")
    monkeypatch.setenv("IMAGE3D_TURN_SECRET", "private-secret-" * 4)
    ice = browser_ice("session")
    assert ice["iceTransportPolicy"] == "relay"
    assert "private-secret" not in json.dumps(ice)
    assert all(":8082?" in url for url in ice["iceServers"][0]["urls"])
    assert capabilities()["editing_core"]


def test_close_fences_loading_and_camera_latest_only(tmp_path):
    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    gate = threading.Event()

    class SlowRenderer(FakeRenderer):
        def start(self):
            assert gate.wait(5)

    async def run():
        service = EditorSessions(store.jobs, renderer_factory=SlowRenderer)
        await service.create(edit_id)
        s = service.active
        closer = asyncio.create_task(service.close(s))
        await asyncio.sleep(0.01)
        with pytest.raises(EditConflict):
            await service.create(edit_id)
        gate.set()
        await closer
        assert s["renderer"].closed and service.active is None
        await service.create(edit_id)
        s = service.active
        await s["loader"]
        assert service.camera_input(s, 3, camera().to_json())
        assert not service.camera_input(s, 2, camera().to_json())
        await service.freeze(s, 3, camera().to_json())
        assert not service.camera_input(s, 4, camera().to_json())
        await service.close(s)

    asyncio.run(run())


def test_selection_requires_preview_and_saved_version_is_read_only(tmp_path):
    client, service, _ = make_client(tmp_path)
    with client:
        edit_id, base = opened(client)
        frame = freeze(client, base)
        binding = {"ticket": frame["ticket"], "expected_revision": 0}
        shape = {
            **binding,
            "shape": "box",
            "minimum": [-1, -1, 1],
            "maximum": [-0.2, 1, 3],
        }
        empty = client.post(
            base + "/selection", json={**shape, "combine": "subtract"}
        ).json()
        assert empty["selected_count"] == 0
        selection = client.post(base + "/selection", json=shape).json()
        operation = {
            **binding,
            "kind": "delete",
            "selection_token": selection["selection_token"],
            "operation_id": "unpreviewed",
        }
        assert client.post(base + "/operations", json=operation).status_code == 409
        assert client.post(base + "/invalidate-frame").status_code == 200
        assert client.post(base + "/selection", json=shape).status_code == 422
        saved = client.post(base + "/versions", json={"expected_revision": 0}).json()
        assert client.delete(base).status_code == 200
        assert client.delete(base).status_code == 200
        response = client.post(
            "/api/gaussian-render-sessions",
            json={"edit_id": edit_id, "version": saved["version"]},
        ).json()
        client.headers["x-editor-token"] = response["token"]
        base = "/api/gaussian-render-sessions/" + response["session_id"]
        for _ in range(100):
            if client.get(base).json()["state"] == "viewing":
                break
            time.sleep(0.01)
        assert (
            client.post(base + "/versions", json={"expected_revision": 0}).status_code
            == 409
        )
        assert client.delete(base).status_code == 200
        assert service.active is None


def test_optional_media_signaling_is_authenticated_and_recreated(tmp_path, monkeypatch):
    from backend import gaussian_editor

    events = []

    class Media:
        def __init__(self, service, session):
            events.append("create")

        async def answer(self, sdp):
            assert sdp == "fake-offer"
            return {"type": "answer", "sdp": "fake-answer"}

        async def close(self):
            events.append("close")

    monkeypatch.setattr(gaussian_editor.cloud_media, "CloudMedia", Media)
    client, _, _ = make_client(tmp_path)
    with client:
        _, base = opened(client)
        offer = {"type": "offer", "sdp": "fake-offer"}
        assert (
            client.post(
                base + "/offer", json=offer, headers={"x-editor-token": "wrong"}
            ).status_code
            == 403
        )
        assert client.post(base + "/offer", json=offer).json()["sdp"] == "fake-answer"
        assert client.post(base + "/offer", json=offer).status_code == 409
        assert client.post(base + "/resume").status_code == 200
        assert client.post(base + "/offer", json=offer).status_code == 200
        client.delete(base)
        assert events == ["create", "close", "create", "close"]


def test_media_track_and_camera_channel_without_ice_or_cuda(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from image3d_scenegraph.gaussian.cloud_media import CloudMedia

    class Emitter:
        def __init__(self, *args, **kwargs):
            self.handlers = {}
            self.readyState = "open"
            self.bufferedAmount = 0
            self.label = "camera"
            self.connectionState = "connected"

        def on(self, name):
            def install(callback):
                self.handlers[name] = callback
                return callback

            return install

        def addTrack(self, track):
            self.track = track

        def send(self, value):
            self.message = json.loads(value)

        async def close(self):
            self.connectionState = "closed"

    class Track:
        async def next_timestamp(self):
            return 0, 1 / 90000

        def stop(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "aiortc",
        SimpleNamespace(
            RTCPeerConnection=Emitter,
            RTCConfiguration=lambda **kwargs: kwargs,
            VideoStreamTrack=Track,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(
            VideoFrame=SimpleNamespace(
                from_ndarray=lambda rgb, format: SimpleNamespace(rgb=rgb)
            )
        ),
    )
    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]

    async def run():
        service = EditorSessions(store.jobs, renderer_factory=FakeRenderer)
        await service.create(edit_id)
        s = service.active
        await s["loader"]
        media = CloudMedia(service, s)
        channel = Emitter()
        media.pc.handlers["datachannel"](channel)
        channel.handlers["message"](
            json.dumps({"sequence": 7, "camera": camera().to_json()})
        )
        channel.handlers["message"](
            json.dumps({"sequence": 9, "camera": camera().to_json()})
        )
        channel.handlers["message"](
            json.dumps({"sequence": 8, "camera": camera().to_json()})
        )
        await asyncio.sleep(0.02)
        assert s["seq"] == 9
        frame = await media.track.recv()
        assert frame.rgb.shape == (64, 64, 3)
        assert channel.message == {"camera_seq": 9, "revision": 0}
        media.last_input = 0
        channel.handlers["message"](json.dumps({"paused": True}))
        assert s["paused"]
        media.last_input = 0
        channel.handlers["message"]("[]")
        assert s["seq"] == 9
        await media.close()
        await service.close(s)

    asyncio.run(run())


def test_loader_failure_and_idle_expiry_release_only_session(tmp_path):
    store, original, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    before = sha256_file(original)

    class FailedRenderer(FakeRenderer):
        def start(self):
            raise RuntimeError("simulated GPU busy")

    async def run():
        service = EditorSessions(store.jobs, renderer_factory=FailedRenderer)
        await service.create(edit_id)
        failed = service.active
        await failed["loader"]
        assert failed["state"] == "error" and failed["renderer"].closed
        await service.close(failed)
        service.renderer_factory = FakeRenderer
        await service.create(edit_id)
        active = service.active
        await active["loader"]
        active["interaction"] = time.monotonic() - 601
        active["heartbeat"] = time.monotonic()
        service.start()
        for _ in range(70):
            if service.active is None:
                break
            await asyncio.sleep(0.1)
        assert service.active is None and active["renderer"].closed
        assert service.edits.get(edit_id)["revision"] == 0
        await service.shutdown()

    asyncio.run(run())
    assert sha256_file(original) == before
