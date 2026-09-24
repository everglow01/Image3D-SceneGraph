from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import profile_gaussian_delivery as delivery


def ply_bytes(format_line="format binary_little_endian 1.0"):
    header = "\n".join([
        "ply", format_line, "element vertex 1",
        *(f"property float {field}" for field in delivery.PLY_FIELDS), "end_header", "",
    ]).encode("ascii")
    return header + bytes(4 * len(delivery.PLY_FIELDS))


def test_optimized_cli_still_rejects_invalid_ply_before_writing(tmp_path):
    source, output = tmp_path / "source.ply", tmp_path / "output.ply"
    source.write_bytes(ply_bytes("format ascii 1.0"))
    result = subprocess.run([
        sys.executable, "-O", str(Path(delivery.__file__).resolve()),
        "--source", str(source), "--output", str(output),
        "--operation", "ply", "--writer", "previous",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "PLY must be binary little-endian" in result.stderr
    assert not output.exists()


def test_profile_never_marks_mismatched_outputs_passed(tmp_path, monkeypatch):
    source = tmp_path / "source" / "scene.ply"
    source.parent.mkdir()
    source.write_bytes(ply_bytes())
    output = tmp_path / "profile"
    monkeypatch.setattr(sys, "argv", ["profile", "--source", str(source),
                                     "--output", str(output), "--lease", str(tmp_path / "lease")])
    monkeypatch.setattr(delivery, "FileLease", lambda _: nullcontext())
    monkeypatch.setattr(delivery, "require_idle_gpu", lambda: None)

    def fake_trial(command, **kwargs):
        target = Path(command[command.index("--output") + 1])
        writer = command[command.index("--writer") + 1]
        target.write_bytes(source.read_bytes() if writer == "previous" else b"changed")
        return SimpleNamespace(stdout=json.dumps({"writer": writer}))

    monkeypatch.setattr(delivery.subprocess, "run", fake_trial)
    with pytest.raises(ValueError, match="ply output bytes changed"):
        delivery.main()
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "failed"
    assert report["error"] == "ply output bytes changed"
    assert source.read_bytes() == ply_bytes()
