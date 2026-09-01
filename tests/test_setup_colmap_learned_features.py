from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from image3d_scenegraph.geometry.colmap import ColmapFeatureAsset
from scripts import setup_colmap_learned_features as setup


def _asset(payload: bytes) -> ColmapFeatureAsset:
    return ColmapFeatureAsset(
        filename="model.onnx",
        url="https://example.test/model.onnx",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
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
    monkeypatch.setattr(
        setup.urllib.request,
        "urlretrieve",
        lambda _url, path: Path(path).write_bytes(payload),
    )
    monkeypatch.setattr(
        sys, "argv", ["setup_colmap_learned_features.py", "--install"]
    )

    setup.main()

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
