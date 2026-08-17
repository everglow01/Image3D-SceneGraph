from __future__ import annotations

import hashlib

import pytest

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
