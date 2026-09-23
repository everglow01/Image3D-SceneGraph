from __future__ import annotations

import fcntl
import json
import subprocess
from pathlib import Path

import pytest

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian import browser_assets as browser
from image3d_scenegraph.jobs import JobStore


def job(tmp_path, comparison=False):
    root = tmp_path / "jobs" / "sample"
    root.mkdir(parents=True)
    variants = []
    for name in ("original", "filtered"):
        folder = root / name
        folder.mkdir()
        # Publication tests mock the converter; real SH3/KSPLAT decoding is covered by Node tests.
        (folder / "scene.ply").write_bytes(b"ply\nelement vertex 2\nend_header\n" + name.encode())
        (folder / "export.json").write_text(json.dumps({
            "sh_degree": 3, "gaussian_count": 2,
            "browser_sha256": sha256_file(folder / "scene.ply"),
        }))
        variants.append({"id": name, "scene_splat": f"{name}/scene.ply", "export_metadata": f"{name}/export.json"})
    manifest = {"job_id": "sample", "status": "done", "assets": {
        "scene_splat": "original/scene.ply", "gaussian_export_metadata": "original/export.json",
        "scene_splat_vggt_filtered": "filtered/scene.ply", "gaussian_vggt_filtered_export_metadata": "filtered/export.json",
    }}
    if comparison:
        manifest.update(result_kind="gaussian_comparison", gaussian_variants=variants, default_gaussian_variant="original")
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def fake_converter(command, *, check, timeout):
    assert command[:2] == ["node", "--max-old-space-size=6144"]
    assert command[-1] == "2" and check and timeout == 600
    Path(command[-2]).write_bytes(b"mock validated KSPLAT")


@pytest.mark.parametrize("comparison", [False, True])
def test_publication_is_source_bound_and_get_manifest_never_rewrites(tmp_path, monkeypatch, comparison):
    root, manifest = job(tmp_path, comparison)
    originals = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(browser.subprocess, "run", fake_converter)
    destination = browser.publish_browser_asset(root, "filtered/scene.ply", Path("converter.mjs"))
    published = {p: sha256_file(p) for p in destination.iterdir()}
    with pytest.raises(FileExistsError):
        browser.publish_browser_asset(root, "filtered/scene.ply", Path("converter.mjs"))
    assert {p: sha256_file(p) for p in destination.iterdir()} == published
    store = JobStore(output_root=root.parent)
    response = store.get_manifest("sample")
    assert len(response["gaussian_browser_assets"]) == 1
    asset = response["gaussian_browser_assets"][0]
    assert asset["source"] == "filtered/scene.ply"
    assert asset["sh_degree"] == 2 and asset["compression_level"] == 2
    assert store.get_asset_path("sample", asset["path"]) == destination / "scene.ksplat"
    assert response["assets"]["scene_splat"] == manifest["assets"]["scene_splat"]
    assert {p: sha256_file(p) for p in originals} == originals
    assert "gaussian_browser_assets" not in manifest

    original_hash = browser.sha256_file
    def small_records_only(path):
        assert Path(path).suffix not in {".ply", ".ksplat"}, "GET must not hash large assets"
        return original_hash(path)
    monkeypatch.setattr(browser, "sha256_file", small_records_only)
    assert browser.with_browser_assets(root, manifest)["gaussian_browser_assets"] == response["gaussian_browser_assets"]


@pytest.mark.parametrize("damage", ["source", "metadata", "output", "record", "symlink", "directory_symlink"])
def test_invalid_or_changed_publication_is_not_advertised(tmp_path, monkeypatch, damage):
    root, manifest = job(tmp_path)
    monkeypatch.setattr(browser.subprocess, "run", fake_converter)
    destination = browser.publish_browser_asset(root, "original/scene.ply", Path("converter.mjs"))
    if damage == "source":
        with (root / "original/scene.ply").open("ab") as handle:
            handle.write(b"changed")
    elif damage == "metadata":
        (root / "original/export.json").write_text("{}")
    elif damage == "output":
        (destination / "scene.ksplat").write_bytes(b"corrupt")
    elif damage == "record":
        (destination / "record.json").write_text("{")
    elif damage == "symlink":
        (destination / "scene.ksplat").unlink()
        (destination / "scene.ksplat").symlink_to(root / "original/scene.ply")
    else:
        moved = destination.with_name("moved")
        destination.rename(moved)
        destination.symlink_to(moved, target_is_directory=True)
    response = browser.with_browser_assets(root, manifest)
    assert "gaussian_browser_assets" not in response
    assert response["assets"]["scene_splat"] == "original/scene.ply"


def test_failure_and_concurrent_publication_leave_no_complete_asset(tmp_path, monkeypatch):
    root, manifest = job(tmp_path)
    def failed(command, **kwargs):
        Path(command[-2]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(browser.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        browser.publish_browser_asset(root, "original/scene.ply", Path("converter.mjs"))
    assert "gaussian_browser_assets" not in browser.with_browser_assets(root, manifest)
    assert len(list((root / "lifecycle/browser").glob(".staging-*"))) == 1
    with (root / "lifecycle/browser/.publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            browser.publish_browser_asset(root, "original/scene.ply", Path("converter.mjs"))


def test_source_mutation_during_conversion_refuses_publication(tmp_path, monkeypatch):
    root, manifest = job(tmp_path)
    def mutate(command, **kwargs):
        fake_converter(command, **kwargs)
        Path(command[-3]).write_bytes(b"changed during conversion")
    monkeypatch.setattr(browser.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="source changed"):
        browser.publish_browser_asset(root, "original/scene.ply", Path("converter.mjs"))
    assert "gaussian_browser_assets" not in browser.with_browser_assets(root, manifest)


@pytest.mark.parametrize("damage", ["hash", "count", "sh_degree", "queued"])
def test_source_identity_errors_fail_before_conversion(tmp_path, monkeypatch, damage):
    root, manifest = job(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("converter must not start")
    monkeypatch.setattr(browser.subprocess, "run", forbidden)
    path = root / "original/export.json"
    metadata = json.loads(path.read_text())
    if damage == "hash":
        metadata["browser_sha256"] = "0" * 64
    elif damage == "count":
        metadata["gaussian_count"] = 3
    elif damage == "sh_degree":
        metadata["sh_degree"] = 2
    else:
        manifest["status"] = "queued"
        (root / "manifest.json").write_text(json.dumps(manifest))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        browser.publish_browser_asset(root, "original/scene.ply", Path("converter.mjs"))
    assert not (root / "lifecycle").exists()


def test_invalid_source_and_symlink_paths_are_rejected(tmp_path, monkeypatch):
    root, manifest = job(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("converter must not start")
    monkeypatch.setattr(browser.subprocess, "run", forbidden)
    for path in ("../scene.ply", "/scene.ply", "original//scene.ply", "original/./scene.ply", "a%2fb", "a\\b"):
        with pytest.raises(ValueError):
            browser.contained_path(root, path)
    with pytest.raises(ValueError, match="one completed Job variant"):
        browser.publish_browser_asset(root, "missing.ply", Path("converter.mjs"))
    (root / "original/scene.ply").unlink()
    (root / "original/scene.ply").symlink_to(root / "filtered/scene.ply")
    with pytest.raises(ValueError, match="symlink"):
        browser.publish_browser_asset(root, "original/scene.ply", Path("converter.mjs"))
    assert "gaussian_browser_assets" not in browser.with_browser_assets(root, manifest)
