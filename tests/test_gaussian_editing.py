from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.cloud_render import (
    CloudCamera,
    CloudRenderError,
    CloudRenderProcess,
    FrozenFrameTickets,
)
from image3d_scenegraph.gaussian.editing import (
    EditConflict,
    GaussianEditError,
    GaussianEditStore,
    resolve_edit_source,
    select_box,
    select_polygon,
)
from image3d_scenegraph.jobs import JobStore


def camera():
    return CloudCamera(
        np.eye(4), np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]]), 64, 64
    )


def setup_source(tmp_path, count=8):
    from image3d_scenegraph.gaussian.export import write_binary_ply

    jobs = JobStore(tmp_path / "jobs")
    directory = jobs.output_root / "source"
    directory.mkdir(parents=True)
    rows = np.arange(count * 62, dtype=np.float32).reshape(count, 62) / 1000
    rows[:, :3] = np.column_stack(
        (np.linspace(-0.3, 0.3, count), np.zeros(count), np.full(count, 2))
    )
    rows[:, 55:58] = -3
    rows[:, 58:62] = [1, 0, 0, 0]
    original = tmp_path / "original.ply"
    write_binary_ply(original, rows)
    os.link(original, directory / "scene.ply")
    metadata = {
        "coordinate_frame": "normalized",
        "world_units": "arbitrary",
        "sh_degree": 3,
        "world_from_normalized": np.eye(4).tolist(),
        "gaussian_count": len(rows),
        "browser_sha256": sha256_file(original),
    }
    (directory / "export.json").write_text(json.dumps(metadata))
    manifest = {
        "job_id": "source",
        "status": "done",
        "result_kind": "gaussian_comparison",
        "output_type": "gaussian_splat",
        "assets": {
            "scene_splat": "scene.ply",
            "gaussian_export_metadata": "export.json",
        },
        "gaussian_variants": [
            {
                "id": "project-train-only",
                "scene_splat": "scene.ply",
                "export_metadata": "export.json",
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return GaussianEditStore(jobs), original, rows


def test_local_snapshot_restore_replay_and_branch(tmp_path):
    store, original, _ = setup_source(tmp_path)
    before = sha256_file(original)
    edit_id = store.create("source")["edit_id"]
    identity = store.source_identity(edit_id)
    request = dict(identity=identity, content=b"\xfe", expected_revision=0,
                   operation_id="local-1")
    first = store.submit_snapshot(edit_id, **request)
    assert first["revision"] == 1 and first["visible_count"] == 7
    version = store.save_version(edit_id, expected_revision=1)
    restored = store.submit_snapshot(edit_id, identity=identity, content=b"\xff",
                                     expected_revision=1, operation_id="restore")
    assert restored["revision"] == 2 and restored["visible_count"] == 8
    assert store.submit_snapshot(edit_id, **request) == first
    with pytest.raises(EditConflict):
        store.submit_snapshot(edit_id, **(request | {"content": b"\xfc"}))
    with pytest.raises(EditConflict):
        store.submit_snapshot(edit_id, **(request | {"operation_id": "stale"}))
    store.apply(edit_id, expected_revision=2, operation_id="undo", kind="undo")
    store.submit_snapshot(edit_id, identity=identity, content=b"\xfd",
                          expected_revision=3, operation_id="branch")
    assert not store.describe(edit_id)["can_redo"]
    assert store.version(edit_id, version["version"]) == version
    state, content = store.local_snapshot(edit_id, identity)
    assert state == {"revision": 4, "visible_count": 7} and content == b"\xfd"
    assert sha256_file(original) == before


def test_local_snapshot_validation_and_atomic_retry(tmp_path, monkeypatch):
    import image3d_scenegraph.gaussian.editing as editing

    store, _, _ = setup_source(tmp_path, count=9)
    edit_id = store.create("source")["edit_id"]
    identity = store.source_identity(edit_id)
    request = dict(identity=identity, expected_revision=0, operation_id="snapshot")
    for content in (b"", b"\xff", b"\xff\x03", b"\x00\x00", b"x" * (512 * 1024 + 1)):
        with pytest.raises(GaussianEditError):
            store.submit_snapshot(edit_id, **request, content=content)
    with pytest.raises(EditConflict):
        store.submit_snapshot(edit_id, **(request | {
            "identity": identity | {"metadata_sha256": "0" * 64}
        }), content=b"\xfe\x01")
    with pytest.raises(GaussianEditError):
        store.submit_snapshot(edit_id, **request, content=b"\x01\x00")
    replace = editing._replace_json
    def fail(*args):
        raise OSError("simulated pre-commit failure")
    monkeypatch.setattr(editing, "_replace_json", fail)
    with pytest.raises(OSError):
        store.submit_snapshot(edit_id, **request, content=b"\x01\x00", confirm_large=True)
    assert store.get(edit_id)["revision"] == 0
    monkeypatch.setattr(editing, "_replace_json", replace)
    result = store.submit_snapshot(edit_id, **request, content=b"\x01\x00", confirm_large=True)
    assert result["revision"] == 1 and result["visible_count"] == 1
    metadata = store.jobs.output_root / "source" / "export.json"
    metadata.write_text(metadata.read_text() + "\n")
    with pytest.raises(EditConflict):
        store.submit_snapshot(edit_id, **request, content=b"\x01\x00", confirm_large=True)


def test_local_snapshot_history_and_operation_budgets(tmp_path):
    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    identity = store.source_identity(edit_id)
    for revision in range(102):
        store.submit_snapshot(
            edit_id, identity=identity, content=b"\xfe" if revision % 2 else b"\xff",
            expected_revision=revision, operation_id=f"local-{revision}",
        )
    state = store.get(edit_id)
    assert state["revision"] == 102 and len(state["history"]) == 101
    assert state["cursor"] == 100
    while len(state["requests"]) < 1000:
        state["requests"][f"fixture-{len(state['requests'])}"] = {}
    (store.root / edit_id / "edit.json").write_text(json.dumps(state))
    with pytest.raises(GaussianEditError, match="budget"):
        store.submit_snapshot(edit_id, identity=identity, content=b"\xff",
                              expected_revision=102, operation_id="over-budget")
    ack = store.submit_snapshot(edit_id, identity=identity, content=b"\xfe",
                                expected_revision=101, operation_id="local-101")
    assert ack["revision"] == 102
    assert store.save_version(edit_id, expected_revision=102)["revision"] == 102


def test_local_selection_shared_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/gaussian_local_selection.json").read_text()
    )
    view = CloudCamera(
        np.array(fixture["camera_from_normalized"]),
        np.array(fixture["intrinsic"]), fixture["width"], fixture["height"],
    )
    for case in fixture["cases"]:
        if case["mode"] == "box":
            selected = select_box(fixture["points"], case["min"], case["max"])
        else:
            selected = select_polygon(
                fixture["points"], view, case["polygon"], case["depth"]
            )
        assert np.flatnonzero(selected).tolist() == case["selected"], case["name"]


def test_selection_center_depth_and_conservative_extent():
    points = np.array([[0, 0, 2], [0, 0, 5], [0, 0, -2], [1, 0, 2]])
    polygon = [[28, 28], [36, 28], [36, 36], [28, 36]]
    assert select_polygon(points, camera(), polygon, [1, 3]).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert select_box(points, [-0.1, -0.1, 1], [0.1, 0.1, 3]).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert select_polygon(
        points, camera(), polygon, [1, 3], radii=[0.1, 0.1, 0.1, 1.0]
    ).tolist() == [True, False, False, True]
    assert select_polygon(
        [[0, 0, 0.005]], camera(), polygon, [0.01, 3], radii=[0.1]
    ).tolist() == [True]


def test_concave_polygon_and_frozen_camera_copy():
    polygon = [[10, 10], [50, 10], [50, 20], [20, 20], [20, 50], [10, 50]]
    xy = (np.array([[15, 15], [40, 40], [10, 30]]) - 32) / 50 * 2
    assert select_polygon(
        np.column_stack((xy, np.full(3, 2))), camera(), polygon, [1, 3]
    ).tolist() == [True, False, True]
    value = camera().to_json()
    parsed = CloudCamera.from_json(value)
    value["camera_from_normalized"][0][3] = 7
    assert parsed.camera_from_normalized[0, 3] == 0
    with pytest.raises(ValueError):
        parsed.intrinsic[0, 0] = 0


@pytest.mark.parametrize(
    "change", ["reflection", "scale", "nan", "dimensions", "budget", "skew"]
)
def test_bad_cameras_rejected(change):
    value = camera().to_json()
    if change == "reflection":
        value["camera_from_normalized"][0][0] = -1
    elif change == "scale":
        value["camera_from_normalized"][0][0] = 2
    elif change == "nan":
        value["camera_from_normalized"][0][3] = float("nan")
    elif change == "dimensions":
        value["width"] = True
    elif change == "budget":
        value.update(width=1920, height=1920)
    else:
        value["intrinsic"][0][1] = 2
    with pytest.raises(CloudRenderError):
        CloudCamera.from_json(value)


@pytest.mark.parametrize(
    "polygon,depth",
    [
        ([[0, 0]] * 3, [1, 2]),
        ([[0, 0], [65, 0], [0, 40]], [1, 2]),
        ([[0, 0], [40, 0], [0, 40]], [0, 2]),
    ],
)
def test_bad_selections_rejected(polygon, depth):
    with pytest.raises(GaussianEditError):
        select_polygon([[0, 0, 1]], camera(), polygon, depth)


def test_frame_ticket_rejects_old_camera_revision_source_and_expiry():
    clock = [0.0]
    tickets = FrozenFrameTickets(clock=lambda: clock[0])
    args = dict(source_sha256="a" * 64, revision=0, camera_seq=1, camera=camera())
    token = tickets.issue(**args)
    tickets.validate(token, **args)
    for changed in ({"revision": 1}, {"camera_seq": 2}, {"source_sha256": "b" * 64}):
        with pytest.raises(CloudRenderError, match="stale"):
            tickets.validate(token, **(args | changed))
    resized = camera().to_json() | {"width": 65}
    with pytest.raises(CloudRenderError):
        tickets.validate(token, **(args | {"camera": CloudCamera.from_json(resized)}))
    tickets.issue(**args)
    with pytest.raises(CloudRenderError):
        tickets.validate(token, **args)
    token = tickets.issue(**args)
    clock[0] = 301
    with pytest.raises(CloudRenderError):
        tickets.validate(token, **args)


def test_edit_undo_redo_branch_and_idempotency(tmp_path):
    store, original, _ = setup_source(tmp_path)
    source_hash = sha256_file(original)
    edit = store.create("source", variant_id="project-train-only")
    edit_id = edit["edit_id"]
    selected = np.arange(8) == 0
    args = dict(
        operation_id="first", expected_revision=0, kind="delete", selected=selected
    )
    first = store.apply(edit_id, **args)
    assert first["visible_count"] == 7
    assert store.apply(edit_id, **args) == first
    with pytest.raises(EditConflict):
        store.apply(edit_id, **(args | {"selected": np.arange(8) == 1}))
    with pytest.raises(EditConflict):
        store.apply(edit_id, **(args | {"operation_id": "stale"}))
    store.apply(edit_id, operation_id="undo", expected_revision=1, kind="undo")
    assert store.visible(edit_id).all()
    store.apply(edit_id, operation_id="redo", expected_revision=2, kind="redo")
    assert not store.visible(edit_id)[0]
    saved = store.save_version(edit_id, expected_revision=3)
    assert store.save_version(edit_id, expected_revision=3) == saved
    store.apply(edit_id, operation_id="undo-again", expected_revision=3, kind="undo")
    store.apply(
        edit_id,
        operation_id="branch",
        expected_revision=4,
        kind="delete",
        selected=np.arange(8) == 1,
    )
    with pytest.raises(GaussianEditError, match="nothing to redo"):
        store.apply(edit_id, operation_id="no-redo", expected_revision=5, kind="redo")
    reopened = GaussianEditStore(store.jobs, store.root)
    assert reopened.visible(edit_id).tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert reopened.version(edit_id, saved["version"]) == saved
    assert sha256_file(original) == source_hash


def test_export_preserves_rows_hardlink_and_metric_identity(tmp_path):
    from image3d_scenegraph.gaussian.export import PLY_FIELDS, read_gaussian_ply

    store, original, rows = setup_source(tmp_path)
    source_hash = sha256_file(original)
    edit = store.create("source")
    edit_id = edit["edit_id"]
    store.apply(
        edit_id,
        operation_id="trim",
        expected_revision=0,
        kind="delete",
        selected=np.arange(8) < 2,
    )
    version = store.save_version(edit_id, expected_revision=1)["version"]
    record = store.export_version(edit_id, version)
    assert record["quality_role"] == "manual_edit_not_evaluated"
    assert record["navigation_status"] == "not_inherited"
    assert not {"psnr", "ssim", "evaluation_sha256"} & record.keys()
    fields = read_gaussian_ply(
        store.root / edit_id / "versions" / version / "export" / "scene.ply"
    )
    assert np.array_equal(
        np.column_stack([fields[name] for name in PLY_FIELDS]), rows[2:]
    )
    assert sha256_file(original) == source_hash
    assert (
        original.stat().st_ino
        == (store.jobs.output_root / "source/scene.ply").stat().st_ino
    )
    with pytest.raises(FileExistsError):
        store.export_version(edit_id, version)


def test_empty_large_and_bad_masks_rejected(tmp_path):
    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    for mask in (
        np.zeros(8, dtype=bool),
        np.ones(8, dtype=bool),
        np.ones(7, dtype=bool),
        np.arange(8) < 6,
    ):
        with pytest.raises(GaussianEditError):
            store.apply(
                edit_id,
                operation_id="bad",
                expected_revision=0,
                kind="delete",
                selected=mask,
            )
    assert store.get(edit_id)["revision"] == 0
    result = store.apply(
        edit_id,
        operation_id="large",
        expected_revision=0,
        kind="delete",
        selected=np.arange(8) < 6,
        confirm_large=True,
    )
    assert result["visible_count"] == 2


def test_source_drift_export_and_source_path_rejected(tmp_path):
    store, original, _ = setup_source(tmp_path)
    with pytest.raises(GaussianEditError):
        resolve_edit_source(store.jobs, "source", variant_id="missing")
    edit_id = store.create("source")["edit_id"]
    version = store.save_version(edit_id, expected_revision=0)["version"]
    with original.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises((EditConflict, CloudRenderError)):
        store.export_version(edit_id, version)
    assert not (store.root / edit_id / "versions" / version / "export").exists()
    manifest_path = store.jobs.output_root / "source/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"]["scene_splat"] = "../../original.ply"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(GaussianEditError, match="relative"):
        resolve_edit_source(store.jobs, "source")
    with pytest.raises(GaussianEditError):
        store.get("../outside")


def test_mask_corruption_cannot_be_used(tmp_path):
    store, _, _ = setup_source(tmp_path)
    edit = store.create("source")
    path = store.root / edit["edit_id"] / edit["history"][0]["path"]
    path.write_bytes(b"\x00")
    with pytest.raises(GaussianEditError, match="hash"):
        store.visible(edit["edit_id"])


def test_last_100_edits_remain_undoable(tmp_path):
    store, _, _ = setup_source(tmp_path, count=128)
    edit_id = store.create("source")["edit_id"]
    for revision in range(101):
        store.apply(
            edit_id,
            operation_id=f"delete-{revision}",
            expected_revision=revision,
            kind="delete",
            selected=np.arange(128) == revision,
        )
    assert len(store.get(edit_id)["history"]) == 101
    for revision in range(101, 201):
        store.apply(
            edit_id,
            operation_id=f"undo-{revision}",
            expected_revision=revision,
            kind="undo",
        )
    assert store.visible(edit_id).sum() == 127
    with pytest.raises(GaussianEditError, match="nothing to undo"):
        store.apply(edit_id, operation_id="too-far", expected_revision=201, kind="undo")


def test_failed_commit_leaves_previous_revision_and_can_retry(tmp_path, monkeypatch):
    import image3d_scenegraph.gaussian.editing as module

    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    original = module._replace_json
    args = dict(
        operation_id="retry",
        expected_revision=0,
        kind="delete",
        selected=np.arange(8) == 0,
    )

    def fail(*_args):
        raise OSError("simulated commit failure")

    monkeypatch.setattr(module, "_replace_json", fail)
    with pytest.raises(OSError):
        store.apply(edit_id, **args)
    assert store.get(edit_id)["revision"] == 0
    assert store.visible(edit_id).all()
    monkeypatch.setattr(module, "_replace_json", original)
    assert store.apply(edit_id, **args)["revision"] == 1


def test_saved_mask_tamper_and_operation_directory_symlink_rejected(tmp_path):
    store, _, _ = setup_source(tmp_path)
    edit_id = store.create("source")["edit_id"]
    version = store.save_version(edit_id, expected_revision=0)["version"]
    base = store.root / edit_id
    (base / "versions" / version / "visible-mask.bin").write_bytes(b"\x01")
    with pytest.raises(GaussianEditError, match="mask"):
        store.export_version(edit_id, version)
    (base / "operations").rename(base / "retained-operations")
    (base / "operations").symlink_to(
        base / "retained-operations", target_is_directory=True
    )
    with pytest.raises(GaussianEditError, match="escape"):
        store.visible(edit_id)


def test_render_process_spawn_lifecycle_without_cuda(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import image3d_scenegraph.gaussian.cloud_render as module

    _, original, _ = setup_source(tmp_path)
    parent, child, process, context = Mock(), Mock(), Mock(), Mock()
    parent.poll.return_value = True
    parent.recv.side_effect = [
        {"status": "ready"},
        {
            "status": "frame",
            "camera_hash": camera().digest,
            "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "gaussian_count": 8,
        },
    ]
    process.pid = 123
    process.is_alive.return_value = False
    context.Pipe.return_value = (parent, child)
    context.Process.return_value = process
    factory = Mock(return_value=context)
    monkeypatch.setattr(module.multiprocessing, "get_context", factory)
    renderer = CloudRenderProcess(
        source=original,
        source_sha256=sha256_file(original),
        count=8,
        lease_path=tmp_path / "gpu.lock",
        scratch_root=tmp_path,
    )
    renderer.start()
    factory.assert_called_once_with("spawn")
    assert context.Process.call_args.kwargs["daemon"] is True
    with pytest.raises(CloudRenderError, match="empty"):
        renderer.render(camera(), visible=np.zeros(8, dtype=bool))
    result = renderer.render(camera())
    assert result["rgb"].shape == (64, 64, 3)
    renderer.close()
    renderer.close()
    process.start.assert_called_once()
    process.close.assert_called_once()
    child.close.assert_called_once()
    process.terminate.assert_not_called()


def test_render_process_timeout_closes_only_owned_child(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import image3d_scenegraph.gaussian.cloud_render as module

    _, original, _ = setup_source(tmp_path)
    parent, child, process, context = Mock(), Mock(), Mock(), Mock()
    parent.poll.return_value = False
    process.pid = 123
    process.is_alive.return_value = True
    context.Pipe.return_value = (parent, child)
    context.Process.return_value = process
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _: context)
    renderer = CloudRenderProcess(
        source=original,
        source_sha256=sha256_file(original),
        count=8,
        lease_path=tmp_path / "gpu.lock",
        scratch_root=tmp_path,
    )
    with pytest.raises(CloudRenderError, match="timed out"):
        renderer.start(timeout=0.01)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    process.close.assert_called_once()
    assert renderer._process is None


def test_render_preflight_rejects_missing_source_before_process_start(tmp_path):
    renderer = CloudRenderProcess(
        source=tmp_path / "missing.ply",
        source_sha256="a" * 64,
        count=3,
        lease_path=tmp_path / "gpu.lock",
        scratch_root=tmp_path,
    )
    with pytest.raises(CloudRenderError):
        renderer.start()
    renderer.close()
    assert renderer._process is None
