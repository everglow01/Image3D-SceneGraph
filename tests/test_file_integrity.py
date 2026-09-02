from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from image3d_scenegraph import file_integrity
from image3d_scenegraph.file_integrity import (
    FileIntegrityError,
    install_verified_file,
)


def metadata(payload: bytes) -> tuple[int, str]:
    return len(payload), hashlib.sha256(payload).hexdigest()


def test_install_verified_file_keeps_valid_existing_asset(tmp_path, monkeypatch):
    payload = b"valid"
    size, digest = metadata(payload)
    destination = tmp_path / "asset.bin"
    destination.write_bytes(payload)

    def unexpected_download(_url, _path):
        raise AssertionError("valid asset must not be downloaded again")

    monkeypatch.setattr(
        file_integrity.urllib.request,
        "urlretrieve",
        unexpected_download,
    )

    downloaded = install_verified_file(
        "https://example.test/asset.bin",
        destination,
        expected_size=size,
        expected_sha256=digest,
    )

    assert downloaded is False
    assert destination.read_bytes() == payload


def test_failed_repair_preserves_existing_asset_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    payload = b"valid"
    size, digest = metadata(payload)
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"old-corrupt")
    attempts = 0

    def retrieve(_url, path):
        nonlocal attempts
        attempts += 1
        Path(path).write_bytes(b"partial")

    monkeypatch.setattr(file_integrity.urllib.request, "urlretrieve", retrieve)

    with pytest.raises(FileIntegrityError, match="size mismatch"):
        install_verified_file(
            "https://example.test/asset.bin",
            destination,
            expected_size=size,
            expected_sha256=digest,
        )

    assert attempts == 3
    assert destination.read_bytes() == b"old-corrupt"
    assert not (tmp_path / "asset.bin.part").exists()
