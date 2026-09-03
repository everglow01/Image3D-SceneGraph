from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "setup_colmap_cuda.py"
SPEC = importlib.util.spec_from_file_location("setup_colmap_cuda", SCRIPT_PATH)
assert SPEC and SPEC.loader
SETUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SETUP
SPEC.loader.exec_module(SETUP)


def test_learned_profile_is_colmap_400_cuda_122_onnx():
    assert list(SETUP.PROFILES) == ["learned"]
    profile = SETUP.PROFILES["learned"]

    assert profile.tag == "4.0.0"
    assert profile.root == Path("external/colmap-4-cuda")
    assert profile.cuda_root == Path("/usr/local/cuda-12.2")
    assert profile.cuda_architecture == "89"
    assert profile.onnx is True


def test_install_verification_requires_geometry_flags_on_matches_importer(
    tmp_path, monkeypatch
):
    feature_markers = "AlikedExtraction.max_num_features"
    matcher_markers = " ".join(
        [
            "AlikedMatching.bruteforce_model_path",
            "SiftMatching.lightglue_model_path",
            "AlikedMatching.lightglue_model_path",
            "FeatureMatching.guided_matching",
            "FeatureMatching.skip_geometric_verification",
        ]
    )
    outputs = {
        "feature_extractor": feature_markers,
        "exhaustive_matcher": matcher_markers,
        "sequential_matcher": matcher_markers
        + " SequentialMatching.vocab_tree_path",
        "vocab_tree_matcher": matcher_markers
        + " VocabTreeMatching.vocab_tree_path",
        "matches_importer": matcher_markers.replace(
            "FeatureMatching.skip_geometric_verification", ""
        ),
    }

    def capture(command, **_kwargs):
        if command[1:] == ["-h"]:
            return "COLMAP 4.0.0 with CUDA"
        return outputs[command[1]]

    monkeypatch.setattr(SETUP, "capture", capture)

    with pytest.raises(SystemExit, match="skip_geometric_verification"):
        SETUP.verify_install(tmp_path / "colmap", SETUP.PROFILES["learned"])
