from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "setup_gaussian_trainer.py"
SPEC = importlib.util.spec_from_file_location("setup_gaussian_trainer", SCRIPT_PATH)
assert SPEC and SPEC.loader
SETUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SETUP
SPEC.loader.exec_module(SETUP)


def test_graphdeco_setup_uses_cuda_12_toolchain(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    repo = tmp_path / "external" / "gaussian-splatting"
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SETUP, "_cuda_status", lambda _root: {"available": True, "reason": None})
    monkeypatch.setattr(SETUP, "_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(SETUP.shutil, "disk_usage", lambda _root: type("Usage", (), {"free": 20 * 1024**3})())
    monkeypatch.setattr(SETUP, "_run", commands.append)
    monkeypatch.setattr(
        SETUP.subprocess,
        "run",
        lambda *_args, **_kwargs: SETUP.subprocess.CompletedProcess([], 0, "2.3.1+cu121 12.1 True\n", ""),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup_gaussian_trainer.py",
            "--trainer",
            "graphdeco",
            "--install",
            "--accept-research-license",
        ],
    )

    SETUP.main()

    torch_install = commands[0]
    assert "torch==2.3.1" in torch_install
    assert "torchvision==0.18.1" in torch_install
    assert "https://download.pytorch.org/whl/cu121" in torch_install
    extension_installs = commands[2:]
    assert len(extension_installs) == 3
    assert all("CUDA_HOME=/usr/local/cuda-12.2" in command for command in extension_installs)
