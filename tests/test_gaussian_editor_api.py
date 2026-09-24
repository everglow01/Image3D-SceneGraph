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


def opened(client, edit_id=None):
    if edit_id is None:
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


def local_open(client, edit_id=None):
    if edit_id is None:
        response = client.post("/api/gaussian-edits", json={"job_id": "source"})
        assert response.status_code == 201, response.text
        document = response.json()
    else:
        document = client.get(f"/api/gaussian-edits/{edit_id}").json()
    base = f"/api/gaussian-edits/{document['edit_id']}"
    identity = {key: document["source"][key] for key in (
        "ply_sha256", "metadata_sha256", "gaussian_count"
    )}
    response = client.post(base + "/local-authorization", json=identity)
    assert response.status_code == 201, response.text
    client.headers["x-editor-token"] = response.json()["token"]
    return base, identity, response.json()


def test_local_historical_mask_is_source_bound_and_does_not_change_current(tmp_path):
    client, service, original = make_client(tmp_path)
    before = sha256_file(original)
    with client:
        base, identity, authorization = local_open(client)
        headers = {"content-type": "application/octet-stream"}
        params = identity | {"expected_revision": 0, "operation_id": "hide"}
        assert client.put(base + "/local-mask", params=params, content=b"\xfe", headers=headers).status_code == 200
        version = client.post(base + "/local-versions", json={"expected_revision": 1}).json()["version"]
        assert client.put(base + "/local-mask", params=identity | {
            "expected_revision": 1, "operation_id": "restore",
        }, content=b"\xff", headers=headers).status_code == 200
        historical = client.get(base + "/local-mask", params={"version": version})
        assert historical.status_code == 200
        assert historical.content == b"\xfe" and historical.headers["x-edit-revision"] == "1"
        assert historical.headers["x-source-sha256"] == identity["ply_sha256"]
        assert historical.headers["cache-control"] == "no-store"
        current = client.get(base + "/local-mask")
        assert current.content == b"\xff" and current.headers["x-edit-revision"] == "2"
        assert client.get(base + "/local-mask", params={"version": "../edit.json"}).status_code == 422
        assert client.get(base + "/local-mask", params={"version": "v99999999"}).status_code == 404
        assert client.get(base + "/local-mask", params={"version": version}, headers={"x-editor-token": "wrong"}).status_code == 403
        assert service.edits.get(authorization["edit_id"])["revision"] == 2
        assert service.active is None and sha256_file(original) == before
        assert client.delete(base + "/local-authorization").status_code == 200


def test_local_cpu_protocol_save_restore_export_without_cloud(tmp_path, monkeypatch):
    store, original, rows = setup_source(tmp_path)
    before = sha256_file(original)
    def no_renderer(**kwargs):
        raise AssertionError("local protocol must not create a renderer")
    service = EditorSessions(store.jobs, renderer_factory=no_renderer)
    app = FastAPI()
    app.include_router(editor_router(service, capability_provider=lambda: {
        "cloud_available": False, "reason": "CPU only",
    }))
    # Local export uses the same disk gate without allocating its production reserve in this fixture.
    import image3d_scenegraph.gaussian.editing as editing
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(editing.shutil, "disk_usage", lambda _: usage(10**10, 0, 10**10))
    with TestClient(app) as client:
        client.headers.update({"origin": "http://testserver", "x-image3d-editor": "1"})
        base, identity, authorization = local_open(client)
        assert authorization["revision"] == 0 and 0 < authorization["expires_in_seconds"] <= 600
        assert service.active is None
        mask = client.get(base + "/local-mask")
        assert mask.content == b"\xff" and mask.headers["x-edit-revision"] == "0"
        assert mask.headers["cache-control"] == "no-store"
        params = identity | {"expected_revision": 0, "operation_id": "snapshot"}
        headers = {"content-type": "application/octet-stream"}
        response = client.put(base + "/local-mask", params=params, content=b"\xfe", headers=headers)
        assert response.status_code == 200, response.text
        ack = response.json()
        assert ack["revision"] == 1 and ack["visible_count"] == 7
        assert client.delete(base + "/local-authorization").status_code == 200
        _, _, reopened = local_open(client, authorization["edit_id"])
        assert reopened["revision"] == 1
        assert client.put(base + "/local-mask", params=params, content=b"\xfe", headers=headers).json() == ack
        assert client.put(base + "/local-mask", params=params | {"operation_id": "stale"},
                          content=b"\xfd", headers=headers).status_code == 409
        version = client.post(base + "/local-versions", json={"expected_revision": 1})
        assert version.status_code == 200, version.text
        assert client.post(base + "/local-versions", json={"expected_revision": 1}).json() == version.json()
        name = version.json()["version"]
        assert client.post(base + "/local-exports/" + name).status_code == 202
        for _ in range(200):
            status = client.get(base + f"/versions/{name}/export").json()
            if status["status"] != "running":
                break
            time.sleep(0.01)
        assert status["status"] == "done", status
        ply = client.get(base + f"/versions/{name}/assets/scene.ply").content
        assert ply.split(b"end_header\n", 1)[1] == rows[1:].astype("<f4").tobytes()
        assert client.get(base + f"/versions/{name}/assets/bundle.zip").content[:2] == b"PK"
        restore = client.put(base + "/local-mask", params=identity | {
            "expected_revision": 1, "operation_id": "restore",
        }, content=b"\xff", headers=headers)
        assert restore.json()["revision"] == 2 and restore.json()["visible_count"] == 8
        assert client.get(base + "/local-mask").headers["x-edit-revision"] == "2"
        large = identity | {"expected_revision": 2, "operation_id": "large", "confirm_large": "true"}
        confirmed = client.put(base + "/local-mask", params=large, content=b"\x01", headers=headers)
        assert confirmed.status_code == 200 and confirmed.json()["visible_count"] == 1
        assert client.put(base + "/local-mask", params=large | {"confirm_large": "false"},
                          content=b"\x01", headers=headers).status_code == 409
        assert client.delete(base + "/local-authorization").status_code == 200
        assert not service.local and service.active is None
        assert sha256_file(original) == before


def test_local_protocol_identity_tokens_body_limits_and_cloud_guards(tmp_path):
    client, service, _ = make_client(tmp_path)
    headers = {"content-type": "application/octet-stream"}
    with client:
        base, identity, authorization = local_open(client)
        params = identity | {"expected_revision": 0, "operation_id": "one"}
        assert client.get(base + "/local-mask", headers={"x-editor-token": "wrong"}).status_code == 403
        foreign = client.post("/api/gaussian-edits", json={"job_id": "source"}).json()["edit_id"]
        assert client.get(f"/api/gaussian-edits/{foreign}/local-mask").status_code == 403
        assert client.post(base + "/local-authorization", json=identity).status_code == 409
        assert client.post("/api/gaussian-render-sessions", json={"edit_id": authorization["edit_id"]}).status_code == 409
        assert client.put(base + "/local-mask", params=params, content=b"\xfe").status_code == 415
        assert client.put(base + "/local-mask", params=params, content=b"\xfe",
                          headers=headers | {"content-encoding": "gzip"}).status_code == 415
        for content in (b"", b"\x00", b"\xff\x00"):
            assert client.put(base + "/local-mask", params=params, content=content, headers=headers).status_code == 422
        assert client.put(base + "/local-mask", params=params, content=b"\x01", headers=headers).status_code == 422
        assert client.put(base + "/local-mask", params=params | {"metadata_sha256": "0" * 64},
                          content=b"\xfe", headers=headers).status_code == 409
        assert client.put(base + "/local-mask", params=params | {"path": "/untrusted"},
                          content=b"\xfe", headers=headers).status_code == 422
        assert client.put(base + "/local-mask", params=params, content=b"\xfe",
                          headers=headers | {"origin": "http://foreign.example"}).status_code == 403
        assert client.put(base + "/local-mask", params=params, content=iter([b"x" * (300 * 1024)] * 2),
                          headers=headers).status_code == 413
        # Only the binary route exceeds the unchanged 96 KiB JSON ceiling.
        assert client.put(base + "/local-mask", params=params, content=b"x" * (100 * 1024),
                          headers=headers).status_code == 422
        assert client.post(base + "/local-authorization", content=b"x" * (100 * 1024),
                           headers={"content-type": "application/json"}).status_code == 413
        service.local[authorization["edit_id"]]["expires"] = time.monotonic() - 1
        assert client.get(base + "/local-mask").status_code == 403
        base, _, renewed = local_open(client, authorization["edit_id"])
        assert renewed["token"] != authorization["token"] and renewed["revision"] == 0
        assert client.get(base + "/local-mask", headers={"x-editor-token": authorization["token"]}).status_code == 403
        assert client.delete(base + "/local-authorization").status_code == 200
        _, cloud = opened(client, renewed["edit_id"])
        assert client.post(base + "/local-authorization", json=identity).status_code == 409
        frame = freeze(client, cloud)
        assert client.post(cloud + "/operations", json={
            "ticket": frame["ticket"], "expected_revision": 0,
            "operation_id": "mask-in-old-api", "kind": "delete", "mask": [True],
        }).status_code == 422
        assert client.delete(cloud).status_code == 200


def test_local_document_fence_survives_expiry_and_repeated_cancellation(tmp_path, monkeypatch):
    from image3d_scenegraph.gpu_lease import LeaseBusy

    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    identity = store.source_identity(edit_id)
    service = EditorSessions(store.jobs, renderer_factory=FakeRenderer)
    other = EditorSessions(store.jobs, renderer_factory=FakeRenderer)
    started, release = threading.Event(), threading.Event()
    submit = service.edits.submit_snapshot
    def blocked(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("fixture release was not signalled")
        return submit(*args, **kwargs)
    monkeypatch.setattr(service.edits, "submit_snapshot", blocked)

    async def scenario():
        response = await service.open_local(edit_id, identity)
        authorization = service.authenticate_local(edit_id, response["token"])
        task = asyncio.create_task(service.submit_local(authorization, b"\xfe", {
            "expected_revision": 0, "operation_id": "cancelled-ack",
        }))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            with pytest.raises(EditConflict):
                await service.read_local(authorization)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
            authorization["expires"] = time.monotonic() - 1
            service.expire_local()
            assert edit_id in service.local and not task.done()
            with pytest.raises(LeaseBusy):
                await other.open_local(edit_id, identity)
            with pytest.raises(LeaseBusy):
                await other.create(edit_id)
            closing = asyncio.create_task(service.close_local(authorization))
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        await closing
        assert edit_id not in service.local
        reopened = await other.open_local(edit_id, identity)
        assert reopened["revision"] == 1
        await service.shutdown()
        await other.shutdown()

    asyncio.run(scenario())


def test_local_authorization_failed_open_and_cloud_loading_hold_same_fence(tmp_path, monkeypatch):
    from image3d_scenegraph.gpu_lease import LeaseBusy

    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    identity = store.source_identity(edit_id)
    service = EditorSessions(store.jobs, renderer_factory=FakeRenderer)
    other = EditorSessions(store.jobs, renderer_factory=FakeRenderer)

    async def scenario():
        with pytest.raises(EditConflict):
            await service.open_local(edit_id, identity | {"ply_sha256": "0" * 64})
        assert not service.local
        loader = service._load
        release = asyncio.Event()
        async def waiting(session):
            await release.wait()
            await loader(session)
        monkeypatch.setattr(service, "_load", waiting)
        await service.create(edit_id)
        with pytest.raises(LeaseBusy):
            await other.open_local(edit_id, identity)
        closing = asyncio.create_task(service.close(service.active))
        await asyncio.sleep(0)
        with pytest.raises(LeaseBusy):
            await other.open_local(edit_id, identity)
        release.set()
        await closing
        info = await other.open_local(edit_id, identity)
        assert info["revision"] == 0
        await other.shutdown()
        await service.shutdown()

    asyncio.run(scenario())


def test_local_and_cloud_share_export_budget_and_allow_distinct_documents(tmp_path, monkeypatch):
    store, _, _ = setup_source(tmp_path)
    first = store.create("source")["edit_id"]
    second = store.create("source")["edit_id"]
    for edit_id in (first, second):
        store.save_version(edit_id, expected_revision=0)
    service = EditorSessions(store.jobs, renderer_factory=FakeRenderer)

    async def scenario():
        release = asyncio.Event()
        async def held_export(state):
            await release.wait()
            state["status"] = "done"
        monkeypatch.setattr(service, "_export", held_export)
        info = await service.open_local(first, store.source_identity(first))
        authorization = service.authenticate_local(first, info["token"])
        await service.create(second)
        cloud = service.active
        await cloud["loader"]
        assert cloud["state"] == "viewing"
        state = await service.export_local(authorization, "v00000000")
        assert state == await service.export_local(authorization, "v00000000")
        with pytest.raises(EditConflict):
            await service.start_export(cloud, "v00000000")
        release.set()
        await service.export_task
        release.clear()
        await service.start_export(cloud, "v00000000")
        with pytest.raises(EditConflict):
            await service.export_local(authorization, "v00000000")
        release.set()
        await service.shutdown()
        assert not service.local and service.active is None

    asyncio.run(scenario())


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


def test_prepared_frame_and_protection_are_session_bound(tmp_path):
    class PickerRenderer(FakeRenderer):
        def pick(self, camera, *, visible, polygon, depth_range, tolerance=0.02):
            assert visible.dtype == np.bool_
            return {
                "ids": np.array([0, 1]),
                "depth": 1.0,
                "confident_pixels": 4,
                "uncertain_pixels": 0,
                "selection_ms": 1.0,
                "projection_ms": 1.0,
            }

    client, service, original = make_client(tmp_path)
    service.renderer_factory = PickerRenderer
    before = sha256_file(original)
    with client:
        _, base = opened(client)
        prepared = client.post(
            base + "/prepare-frame", json={"sequence": 1, "camera": camera().to_json()}
        )
        assert prepared.status_code == 200, prepared.text
        frame = prepared.json()
        binding = {"ticket": frame["ticket"], "expected_revision": 0}
        assert frame["camera_digest"]
        assert client.post(base + "/freeze-prepared", json=binding).status_code == 200
        assert client.post(base + "/freeze-prepared", json=binding).status_code == 409
        polygon = [
            [2, 2],
            [camera().width - 2, 2],
            [camera().width - 2, camera().height - 2],
            [2, camera().height - 2],
        ]
        selected = client.post(
            base + "/selection",
            json={**binding, "shape": "polygon", "mode": "visible", "polygon": polygon},
        ).json()
        assert selected["selected_count"] == 2
        assert (
            client.post(
                base + "/preview",
                json={
                    **binding,
                    "selection_token": selected["selection_token"],
                    "mode": "highlight",
                },
            ).status_code
            == 200
        )
        operation = {
            **binding,
            "kind": "delete",
            "selection_token": selected["selection_token"],
            "operation_id": "highlight-not-enough",
        }
        assert client.post(base + "/operations", json=operation).status_code == 409
        protected = client.post(
            base + "/protection",
            json={
                **binding,
                "kind": "add",
                "selection_token": selected["selection_token"],
            },
        )
        assert protected.json()["protected_count"] == 2
        assert client.post(base + "/operations", json=operation).status_code == 409
        selected = client.post(
            base + "/selection",
            json={**binding, "shape": "polygon", "mode": "visible", "polygon": polygon},
        ).json()
        assert selected["selected_count"] == 2 and selected["deletable_count"] == 0
        assert (
            client.post(
                base + "/preview",
                json={
                    **binding,
                    "selection_token": selected["selection_token"],
                    "mode": "isolated",
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                base + "/protection",
                json={
                    **binding,
                    "kind": "remove",
                    "selection_token": selected["selection_token"],
                },
            ).json()["protected_count"]
            == 0
        )
        assert client.get(base).json()["revision"] == 0
        service.active["renderer"].pick = lambda *args, **kwargs: {
            "ids": np.array([-1])
        }
        assert (
            client.post(
                base + "/selection",
                json={
                    **binding,
                    "shape": "polygon",
                    "mode": "visible",
                    "polygon": polygon,
                },
            ).status_code
            == 422
        )
        assert client.get(base).json()["revision"] == 0
        assert client.delete(base).status_code == 200
        assert sha256_file(original) == before
        assert service.active is None


def test_prepared_frame_expires_when_camera_advances(tmp_path):
    client, service, _ = make_client(tmp_path)
    with client:
        _, base = opened(client)
        response = client.post(
            base + "/prepare-frame", json={"sequence": 1, "camera": camera().to_json()}
        )
        assert response.status_code == 200
        ticket = response.json()["ticket"]
        assert service.camera_input(service.active, 2, camera().to_json())
        assert (
            client.post(
                base + "/freeze-prepared",
                json={"ticket": ticket, "expected_revision": 0},
            ).status_code
            == 409
        )
        assert client.delete(base).status_code == 200


def test_protection_survives_resume_but_not_new_session(tmp_path):
    client, _, original = make_client(tmp_path)
    before = sha256_file(original)
    with client:
        edit_id, base = opened(client)
        frame = freeze(client, base)
        binding = {"ticket": frame["ticket"], "expected_revision": 0}
        box = {
            **binding,
            "shape": "box",
            "minimum": [-1, -1, 1],
            "maximum": [-0.2, 1, 3],
        }
        selection = client.post(base + "/selection", json=box).json()
        response = client.post(
            base + "/protection",
            json={
                **binding,
                "kind": "add",
                "selection_token": selection["selection_token"],
            },
        )
        assert response.json()["protected_count"] == 2
        assert client.post(base + "/resume").status_code == 200
        assert client.get(base).json()["protected_count"] == 2
        frame = freeze(client, base, sequence=2)
        box.update(ticket=frame["ticket"])
        selection = client.post(base + "/selection", json=box).json()
        assert selection["selected_count"] == 2 and selection["deletable_count"] == 0
        client.delete(base)
        response = client.post(
            "/api/gaussian-render-sessions", json={"edit_id": edit_id}
        )
        client.headers["x-editor-token"] = response.json()["token"]
        new_base = "/api/gaussian-render-sessions/" + response.json()["session_id"]
        assert client.get(new_base).json()["protected_count"] == 0
        client.delete(new_base)
    assert sha256_file(original) == before


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
