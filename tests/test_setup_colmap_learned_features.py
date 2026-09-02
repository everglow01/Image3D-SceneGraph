from __future__ import annotations

import hashlib
import sys
import urllib.error
from pathlib import Path

import pytest

from image3d_scenegraph.geometry.colmap import (
    COLMAP_FEATURE_ASSETS,
    ColmapFeatureAsset,
)
from scripts import setup_colmap_learned_features as setup


def _asset(payload: bytes) -> ColmapFeatureAsset:
    return ColmapFeatureAsset(
        filename="model.onnx",
        url="https://example.test/model.onnx",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_lightglue_assets_are_pinned() -> None:
    assert COLMAP_FEATURE_ASSETS["sift_lightglue"].size_bytes == 45_806_253
    assert COLMAP_FEATURE_ASSETS["sift_lightglue"].sha256 == (
        "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e"
    )
    assert COLMAP_FEATURE_ASSETS["aliked_lightglue"].size_bytes == 45_804_950
    assert COLMAP_FEATURE_ASSETS["aliked_lightglue"].sha256 == (
        "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d"
    )


def test_setup_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    root = tmp_path / "models"
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))
    monkeypatch.setattr(sys, "argv", ["setup_colmap_learned_features.py"])

    setup.main()

    assert "dry_run=true" in capsys.readouterr().out
    assert not root.exists()


def test_setup_downloads_verifies_and_atomically_publishes(
    tmp_path, monkeypatch
):
    payload = b"model"
    asset = _asset(payload)
    root = tmp_path / "models"
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))
    monkeypatch.setattr(setup, "COLMAP_FEATURE_ASSETS", {"model": asset})
    attempts = 0

    def retrieve(_url, path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(path).write_bytes(b"partial")
            raise urllib.error.ContentTooShortError("incomplete", b"partial")
        Path(path).write_bytes(payload)

    monkeypatch.setattr(setup.urllib.request, "urlretrieve", retrieve)
    monkeypatch.setattr(
        sys, "argv", ["setup_colmap_learned_features.py", "--install"]
    )

    setup.main()

    assert attempts == 2
    assert (root / asset.filename).read_bytes() == payload
    assert not (root / f"{asset.filename}.part").exists()


def test_setup_rejects_tampered_existing_model(tmp_path, monkeypatch):
    payload = b"model"
    asset = _asset(payload)
    root = tmp_path / "models"
    root.mkdir()
    (root / asset.filename).write_bytes(b"wrong")
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))
    monkeypatch.setattr(setup, "COLMAP_FEATURE_ASSETS", {"model": asset})
    monkeypatch.setattr(
        sys, "argv", ["setup_colmap_learned_features.py", "--install"]
    )

    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        setup.main()
