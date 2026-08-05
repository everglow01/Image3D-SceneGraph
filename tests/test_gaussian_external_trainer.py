from __future__ import annotations

from pathlib import Path

from image3d_scenegraph.gaussian.external_trainer import (
    _graphdeco_command,
    _nerfstudio_command,
    _parse_losses,
)


def test_graphdeco_command_uses_explicit_split_wrapper(tmp_path):
    command = _graphdeco_command(
        tmp_path, tmp_path / "dataset", tmp_path / "output", 40, 20260729
    )

    assert command[:2] == [
        str(tmp_path / "external/gaussian-splatting/.venv/bin/python"),
        str(tmp_path / "scripts/run_graphdeco_training.py"),
    ]
    assert "--graphdeco-root" in command
    assert command[command.index("--seed") + 1] == "20260729"
    assert command[command.index("--iterations") + 1] == "40"
    assert "--disable_viewer" in command
    assert "--eval" in command


def test_nerfstudio_command_freezes_pose_processing_and_camera_optimization(tmp_path):
    command = _nerfstudio_command(
        tmp_path, tmp_path / "dataset", tmp_path / "output", 40, 20260729
    )

    assert command[1] == "splatfacto"
    assert command[command.index("--machine.seed") + 1] == "20260729"
    assert command[
        command.index("--pipeline.datamanager.train-cameras-sampling-seed") + 1
    ] == "20260729"
    assert command[command.index("--max-num-iterations") + 1] == "40"
    assert command[command.index("--pipeline.model.camera-optimizer.mode") + 1] == "off"
    assert command[-8:] == [
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
        "--downscale-factor",
        "1",
    ]


def test_loss_parser_accepts_finite_native_loss_lines():
    assert _parse_losses("Loss 0.8\ntrain loss: 0.25\nLoss nan") == [0.8, 0.25]
