from __future__ import annotations

import hashlib

import pytest

from scripts import setup_model
from scripts.setup_model import download_file, verify_sha256


def test_setup_checkpoint_hash_gate_rejects_existing_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"pinned-weight")
    expected = hashlib.sha256(b"pinned-weight").hexdigest()

    verify_sha256(checkpoint, expected)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_file(
            "https://example.invalid/checkpoint.pt",
            checkpoint,
            expected_sha256="0" * 64,
        )


def test_setup_lightglue_install_does_not_upgrade_torch(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(setup_model, "get_free_gb", lambda _path: 100.0)
    monkeypatch.setattr(setup_model, "ensure_repo", lambda *_args: None)
    monkeypatch.setattr(setup_model, "ensure_venv", lambda *_args: None)
    monkeypatch.setattr(setup_model, "download_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(setup_model, "verify_sha256", lambda *_args: None)
    monkeypatch.setattr(setup_model, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(setup_model, "run", commands.append)

    setup_model.setup_vggt(
        project_root=tmp_path,
        install=True,
        force=False,
        min_free_gb=20.0,
        model_id=setup_model.VGGT_MODEL_ID,
        torch_index_url=setup_model.PYTORCH_CUDA_INDEX,
    )

    lightglue_path = str(tmp_path / "external" / "lightglue")
    command = next(command for command in commands if lightglue_path in command)
    assert "--no-deps" in command
