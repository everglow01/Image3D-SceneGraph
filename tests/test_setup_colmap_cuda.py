from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "setup_colmap_cuda.py"
SPEC = importlib.util.spec_from_file_location("setup_colmap_cuda", SCRIPT_PATH)
assert SPEC and SPEC.loader
SETUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SETUP
SPEC.loader.exec_module(SETUP)


def test_stable_profile_preserves_colmap_391_cuda_117():
    profile = SETUP.PROFILES["stable"]

    assert profile.tag == "3.9.1"
    assert profile.root == Path("external/colmap-cuda")
    assert profile.cuda_root == Path("/usr/local/cuda-11.7")
    assert profile.cuda_architecture == "86"
    assert profile.onnx is False


def test_learned_profile_isolated_colmap_400_cuda_122_onnx():
    profile = SETUP.PROFILES["learned"]

    assert profile.tag == "4.0.0"
    assert profile.root == Path("external/colmap-4-cuda")
    assert profile.cuda_root == Path("/usr/local/cuda-12.2")
    assert profile.cuda_architecture == "89"
    assert profile.onnx is True
