from __future__ import annotations

from pathlib import Path

from image3d_scenegraph.geometry.colmap import resolve_colmap_executable


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def test_colmap_resolver_prefers_explicit_override(tmp_path, monkeypatch):
    explicit = _make_executable(tmp_path / "explicit" / "colmap")
    local = _make_executable(
        tmp_path / "external" / "colmap-cuda" / "install" / "bin" / "colmap"
    )
    monkeypatch.setenv("IMAGE3D_COLMAP_BIN", str(explicit))
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(local.parents[4]))

    assert resolve_colmap_executable(tmp_path) == explicit


def test_colmap_resolver_prefers_project_local_before_path(tmp_path, monkeypatch):
    local = _make_executable(
        tmp_path / "external" / "colmap-cuda" / "install" / "bin" / "colmap"
    )
    path_colmap = _make_executable(tmp_path / "path" / "colmap")
    monkeypatch.delenv("IMAGE3D_COLMAP_BIN", raising=False)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))
    monkeypatch.setenv("PATH", str(path_colmap.parent))

    assert resolve_colmap_executable(tmp_path) == local


def test_colmap_resolver_rejects_non_executable_override(tmp_path, monkeypatch):
    configured = tmp_path / "colmap"
    configured.write_text("binary", encoding="utf-8")
    monkeypatch.setenv("IMAGE3D_COLMAP_BIN", str(configured))

    assert resolve_colmap_executable(tmp_path) is None
