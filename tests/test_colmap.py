from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from image3d_scenegraph.geometry import colmap
from image3d_scenegraph.geometry.colmap import (
    ColmapFeatureAsset,
    ColmapFeatureError,
    resolve_colmap_executable,
    resolve_colmap_feature_profile,
)


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def test_colmap_resolver_prefers_explicit_override(tmp_path, monkeypatch):
    explicit = _make_executable(tmp_path / "explicit" / "colmap")
    local = _make_executable(
        tmp_path / "external" / "colmap-4-cuda" / "install" / "bin" / "colmap"
    )
    monkeypatch.setenv("IMAGE3D_COLMAP_BIN", str(explicit))
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(local.parents[4]))

    assert resolve_colmap_executable(tmp_path) == explicit


def test_colmap_resolver_prefers_project_local_before_path(tmp_path, monkeypatch):
    local = _make_executable(
        tmp_path / "external" / "colmap-4-cuda" / "install" / "bin" / "colmap"
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


def test_sift_feature_profile_requires_no_models(tmp_path):
    profile = resolve_colmap_feature_profile("sift_v1", tmp_path)

    assert profile.extractor == "SIFT"
    assert profile.local_matcher == "SIFT_BRUTEFORCE"
    assert profile.extractor_model_sha256 is None
    assert profile.extraction_options == (
        "--FeatureExtraction.type",
        "SIFT",
        "--SiftExtraction.max_num_features",
        "8192",
    )


def test_aliked_feature_profile_verifies_pinned_models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    extractor_bytes = b"extractor"
    matcher_bytes = b"matcher"
    assets = {
        "aliked_n16rot": ColmapFeatureAsset(
            "aliked.onnx",
            "https://example.test/aliked",
            len(extractor_bytes),
            hashlib.sha256(extractor_bytes).hexdigest(),
        ),
        "aliked_bruteforce": ColmapFeatureAsset(
            "matcher.onnx",
            "https://example.test/matcher",
            len(matcher_bytes),
            hashlib.sha256(matcher_bytes).hexdigest(),
        ),
    }
    (root / "aliked.onnx").write_bytes(extractor_bytes)
    (root / "matcher.onnx").write_bytes(matcher_bytes)
    monkeypatch.setattr(colmap, "COLMAP_FEATURE_ASSETS", assets)
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))

    profile = resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)

    assert profile.extractor == "ALIKED_N16ROT"
    assert profile.local_matcher == "ALIKED_BRUTEFORCE"
    assert str((root / "aliked.onnx").resolve()) in profile.extraction_options
    assert str((root / "matcher.onnx").resolve()) in profile.matching_options


def test_aliked_feature_profile_rejects_missing_or_tampered_model(
    tmp_path, monkeypatch
):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))

    with pytest.raises(ColmapFeatureError, match="model missing"):
        resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)

    asset = colmap.COLMAP_FEATURE_ASSETS["aliked_n16rot"]
    (root / asset.filename).write_bytes(b"tampered")
    with pytest.raises(ColmapFeatureError, match="size mismatch"):
        resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)


def test_feature_profile_rejects_unknown_id(tmp_path):
    with pytest.raises(ColmapFeatureError, match="unsupported"):
        resolve_colmap_feature_profile("unknown", tmp_path)
