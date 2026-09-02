from __future__ import annotations

import hashlib
import sys
import urllib.error
from pathlib import Path

from image3d_scenegraph import file_integrity
from image3d_scenegraph.geometry.colmap import ColmapFeatureAsset
from scripts import setup_colmap_vocab_tree as setup


def _asset(name: str, payload: bytes) -> ColmapFeatureAsset:
    return ColmapFeatureAsset(
        filename=name,
        url=f"https://example.test/{name}",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_aliked_vocab_tree_asset_is_pinned() -> None:
    asset = setup.COLMAP_VOCAB_TREE_ASSETS["aliked_n16rot_v1"]

    assert asset.filename == ("vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin")
    assert asset.size_bytes == 18_764_565
    assert asset.sha256 == (
        "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e"
    )


def test_vocab_setup_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    root = tmp_path / "vocab"
    monkeypatch.setattr(setup, "colmap_vocab_tree_root", lambda: root)
    monkeypatch.setattr(sys, "argv", ["setup_colmap_vocab_tree.py"])

    setup.main()

    assert "dry_run=true" in capsys.readouterr().out
    assert not root.exists()


def test_vocab_setup_retries_verifies_and_publishes(tmp_path, monkeypatch):
    payload = b"tree"
    asset = _asset("tree.bin", payload)
    root = tmp_path / "vocab"
    attempts = 0

    def retrieve(_url, path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(path).write_bytes(b"partial")
            raise urllib.error.ContentTooShortError("incomplete", b"partial")
        Path(path).write_bytes(payload)

    monkeypatch.setattr(setup, "colmap_vocab_tree_root", lambda: root)
    monkeypatch.setattr(setup, "COLMAP_VOCAB_TREE_ASSETS", {"sift_v1": asset})
    monkeypatch.setattr(file_integrity.urllib.request, "urlretrieve", retrieve)
    monkeypatch.setattr(sys, "argv", ["setup_colmap_vocab_tree.py", "--install"])

    setup.main()

    assert attempts == 2
    assert (root / asset.filename).read_bytes() == payload
    assert not (root / f"{asset.filename}.part").exists()


def test_vocab_setup_replaces_tampered_existing_tree(tmp_path, monkeypatch):
    payload = b"tree"
    asset = _asset("tree.bin", payload)
    root = tmp_path / "vocab"
    root.mkdir()
    destination = root / asset.filename
    destination.write_bytes(b"bad!")

    def retrieve(_url, path):
        Path(path).write_bytes(payload)

    monkeypatch.setattr(setup, "colmap_vocab_tree_root", lambda: root)
    monkeypatch.setattr(setup, "COLMAP_VOCAB_TREE_ASSETS", {"sift_v1": asset})
    monkeypatch.setattr(file_integrity.urllib.request, "urlretrieve", retrieve)
    monkeypatch.setattr(sys, "argv", ["setup_colmap_vocab_tree.py", "--install"])

    setup.main()

    assert destination.read_bytes() == payload
