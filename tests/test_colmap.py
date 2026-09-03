from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from image3d_scenegraph.geometry import colmap
from image3d_scenegraph.geometry.colmap import (
    ColmapFeatureAsset,
    ColmapFeatureError,
    ResolvedColmapFeatureProfile,
    colmap_camera_calibration_support_reason,
    colmap_geometric_verification_support_reason,
    colmap_local_matcher_support_reason,
    colmap_pairing_support_reason,
    resolve_colmap_executable,
    resolve_colmap_camera_calibration,
    resolve_colmap_feature_profile,
    resolve_colmap_geometric_verification,
    resolve_colmap_local_matcher,
    resolve_colmap_pairing,
)


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def test_colmap_resolver_prefers_explicit_override(tmp_path, monkeypatch):
    explicit = _make_executable(tmp_path / "explicit" / "colmap")
    local = _make_executable(
        tmp_path / "external" / "colmap-4-cuda" / "install" / "bin" / "colmap"
    )
    monkeypatch.setenv("IMAGE3D_COLMAP_BIN", str(explicit))
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(local.parents[4]))

    assert resolve_colmap_executable(tmp_path) == explicit


def test_colmap_resolver_prefers_project_local_before_path(tmp_path, monkeypatch):
    local = _make_executable(
        tmp_path / "external" / "colmap-4-cuda" / "install" / "bin" / "colmap"
    )
    path_colmap = _make_executable(tmp_path / "path" / "colmap")
    monkeypatch.delenv("IMAGE3D_COLMAP_BIN", raising=False)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))
    monkeypatch.setenv("PATH", str(path_colmap.parent))

    assert resolve_colmap_executable(tmp_path) == local


def test_colmap_resolver_rejects_non_executable_override(tmp_path, monkeypatch):
    configured = tmp_path / "colmap"
    configured.write_text("binary", encoding="utf-8")
    monkeypatch.setenv("IMAGE3D_COLMAP_BIN", str(configured))

    assert resolve_colmap_executable(tmp_path) is None


def test_local_matcher_capability_probe_is_descriptor_specific(
    tmp_path, monkeypatch
):
    executable = _make_executable(tmp_path / "colmap")
    monkeypatch.setattr(
        colmap,
        "_capture_help",
        lambda _executable, _command: "SiftMatching.lightglue_model_path",
    )

    assert (
        colmap_local_matcher_support_reason(
            executable, "sift_v1", "lightglue"
        )
        is None
    )
    monkeypatch.setattr(
        colmap,
        "_capture_help",
        lambda _executable, command: (
            "SiftMatching.lightglue_model_path"
            if command == "exhaustive_matcher"
            else ""
        ),
    )
    assert "SiftMatching.lightglue_model_path" in (
        colmap_local_matcher_support_reason(
            executable, "sift_v1", "lightglue"
        )
        or ""
    )
    monkeypatch.setattr(
        colmap,
        "_capture_help",
        lambda _executable, _command: "SiftMatching.lightglue_model_path",
    )
    assert "AlikedMatching.lightglue_model_path" in (
        colmap_local_matcher_support_reason(
            executable, "aliked_n16rot_v1", "lightglue"
        )
        or ""
    )
    assert (
        colmap_local_matcher_support_reason(
            executable, "sift_v1", "bruteforce"
        )
        is None
    )


def test_sift_feature_profile_requires_no_models(tmp_path):
    profile = resolve_colmap_feature_profile("sift_v1", tmp_path)
    matcher = resolve_colmap_local_matcher(profile, "bruteforce", tmp_path)

    assert profile.extractor == "SIFT"
    assert matcher.name == "SIFT_BRUTEFORCE"
    assert profile.extractor_model_sha256 is None
    assert matcher.model_sha256 is None
    assert profile.extraction_options == (
        "--FeatureExtraction.type",
        "SIFT",
        "--SiftExtraction.max_num_features",
        "8192",
    )


def test_aliked_feature_profile_verifies_pinned_models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    extractor_bytes = b"extractor"
    matcher_bytes = b"matcher"
    assets = {
        "aliked_n16rot": ColmapFeatureAsset(
            "aliked.onnx",
            "https://example.test/aliked",
            len(extractor_bytes),
            hashlib.sha256(extractor_bytes).hexdigest(),
        ),
        "aliked_bruteforce": ColmapFeatureAsset(
            "matcher.onnx",
            "https://example.test/matcher",
            len(matcher_bytes),
            hashlib.sha256(matcher_bytes).hexdigest(),
        ),
    }
    (root / "aliked.onnx").write_bytes(extractor_bytes)
    (root / "matcher.onnx").write_bytes(matcher_bytes)
    monkeypatch.setattr(colmap, "COLMAP_FEATURE_ASSETS", assets)
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))

    profile = resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)
    matcher = resolve_colmap_local_matcher(profile, "bruteforce", tmp_path)

    assert profile.extractor == "ALIKED_N16ROT"
    assert matcher.name == "ALIKED_BRUTEFORCE"
    assert str((root / "aliked.onnx").resolve()) in profile.extraction_options
    assert str((root / "matcher.onnx").resolve()) in matcher.matching_options


def test_aliked_feature_profile_rejects_missing_or_tampered_model(
    tmp_path, monkeypatch
):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))

    with pytest.raises(ColmapFeatureError, match="model missing"):
        resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)

    asset = colmap.COLMAP_FEATURE_ASSETS["aliked_n16rot"]
    (root / asset.filename).write_bytes(b"tampered")
    with pytest.raises(ColmapFeatureError, match="size mismatch"):
        resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)


def test_lightglue_matchers_use_descriptor_specific_models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    payloads = {
        "sift_lightglue": b"sift-lightglue",
        "aliked_n16rot": b"aliked-extractor",
        "aliked_lightglue": b"aliked-lightglue",
    }
    assets = {
        key: ColmapFeatureAsset(
            f"{key}.onnx",
            f"https://example.test/{key}",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        for key, payload in payloads.items()
    }
    for key, payload in payloads.items():
        (root / assets[key].filename).write_bytes(payload)
    monkeypatch.setattr(colmap, "COLMAP_FEATURE_ASSETS", assets)
    monkeypatch.setenv("IMAGE3D_COLMAP_FEATURE_ROOT", str(root))

    sift = resolve_colmap_feature_profile("sift_v1", tmp_path)
    sift_matcher = resolve_colmap_local_matcher(sift, "lightglue", tmp_path)
    aliked = resolve_colmap_feature_profile("aliked_n16rot_v1", tmp_path)
    aliked_matcher = resolve_colmap_local_matcher(aliked, "lightglue", tmp_path)

    assert sift_matcher.name == "SIFT_LIGHTGLUE"
    assert "--SiftMatching.lightglue_min_score" in sift_matcher.matching_options
    assert str((root / "sift_lightglue.onnx").resolve()) in sift_matcher.matching_options
    assert aliked_matcher.name == "ALIKED_LIGHTGLUE"
    assert "--AlikedMatching.lightglue_min_score" in aliked_matcher.matching_options
    assert str((root / "aliked_lightglue.onnx").resolve()) in aliked_matcher.matching_options


def test_local_matcher_rejects_unknown_id(tmp_path):
    feature = resolve_colmap_feature_profile("sift_v1", tmp_path)

    with pytest.raises(ColmapFeatureError, match="unsupported"):
        resolve_colmap_local_matcher(feature, "unknown", tmp_path)


def test_feature_profile_rejects_unknown_id(tmp_path):
    with pytest.raises(ColmapFeatureError, match="unsupported"):
        resolve_colmap_feature_profile("unknown", tmp_path)


def test_pairing_resolver_uses_descriptor_specific_vocab_trees(
    tmp_path, monkeypatch
):
    root = tmp_path / "external" / "colmap-vocab"
    root.mkdir(parents=True)
    assets = {
        "sift_v1": ColmapFeatureAsset(
            "sift.bin",
            "https://example.test/sift.bin",
            4,
            hashlib.sha256(b"sift").hexdigest(),
        ),
        "aliked_n16rot_v1": ColmapFeatureAsset(
            "aliked.bin",
            "https://example.test/aliked.bin",
            6,
            hashlib.sha256(b"aliked").hexdigest(),
        ),
    }
    (root / "sift.bin").write_bytes(b"sift")
    (root / "aliked.bin").write_bytes(b"aliked")
    monkeypatch.setattr(colmap, "COLMAP_VOCAB_TREE_ASSETS", assets)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))

    sift = resolve_colmap_feature_profile("sift_v1", tmp_path)
    aliked = ResolvedColmapFeatureProfile(
        profile_id="aliked_n16rot_v1",
        extractor="ALIKED_N16ROT",
        descriptor="ALIKED",
        max_features=8_192,
        extraction_options=(),
        extractor_model_sha256="a" * 64,
    )
    sequential = resolve_colmap_pairing(sift, "sequential_loop", tmp_path)
    retrieval = resolve_colmap_pairing(aliked, "vocab_tree", tmp_path)

    assert sequential.command == "sequential_matcher"
    assert str((root / "sift.bin").resolve()) in sequential.pairing_options
    assert sequential.vocab_tree_sha256 == assets["sift_v1"].sha256
    assert retrieval.command == "vocab_tree_matcher"
    assert str((root / "aliked.bin").resolve()) in retrieval.pairing_options
    assert retrieval.vocab_tree_sha256 == assets["aliked_n16rot_v1"].sha256


def test_pairing_resolver_rejects_missing_or_tampered_tree(
    tmp_path, monkeypatch
):
    root = tmp_path / "external" / "colmap-vocab"
    root.mkdir(parents=True)
    asset = ColmapFeatureAsset(
        "sift.bin",
        "https://example.test/sift.bin",
        4,
        hashlib.sha256(b"sift").hexdigest(),
    )
    monkeypatch.setattr(
        colmap,
        "COLMAP_VOCAB_TREE_ASSETS",
        {"sift_v1": asset, "aliked_n16rot_v1": asset},
    )
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))
    feature = resolve_colmap_feature_profile("sift_v1", tmp_path)

    with pytest.raises(ColmapFeatureError, match="tree missing"):
        resolve_colmap_pairing(feature, "vocab_tree", tmp_path)

    (root / "sift.bin").write_bytes(b"bad")
    with pytest.raises(ColmapFeatureError, match="size mismatch"):
        resolve_colmap_pairing(feature, "vocab_tree", tmp_path)


def test_pairing_capability_probe_is_profile_specific(tmp_path, monkeypatch):
    executable = _make_executable(tmp_path / "colmap")

    def help_output(_executable, command):
        return {
            "exhaustive_matcher": "FeatureMatching.type",
            "sequential_matcher": (
                "FeatureMatching.type SequentialMatching.vocab_tree_path "
                "SiftMatching.lightglue_model_path"
            ),
            "vocab_tree_matcher": (
                "FeatureMatching.type VocabTreeMatching.vocab_tree_path "
                "AlikedMatching.lightglue_model_path"
            ),
        }[command]

    monkeypatch.setattr(colmap, "_capture_help", help_output)

    assert (
        colmap_pairing_support_reason(
            executable, "sift_v1", "lightglue", "sequential_loop"
        )
        is None
    )
    assert "AlikedMatching.lightglue_model_path" in (
        colmap_pairing_support_reason(
            executable,
            "aliked_n16rot_v1",
            "lightglue",
            "sequential_loop",
        )
        or ""
    )
    assert (
        colmap_pairing_support_reason(
            executable,
            "aliked_n16rot_v1",
            "lightglue",
            "vocab_tree",
        )
        is None
    )


def test_camera_calibration_profiles_freeze_model_and_sharing() -> None:
    opencv = resolve_colmap_camera_calibration("shared_opencv_v1")
    radial = resolve_colmap_camera_calibration("shared_simple_radial_v1")
    grouped = resolve_colmap_camera_calibration("auto_grouped_simple_radial_v1")

    assert opencv.camera_model == "OPENCV"
    assert radial.camera_model == "SIMPLE_RADIAL"
    assert opencv.sharing_policy == radial.sharing_policy == "single_camera"
    assert opencv.image_reader_options[1] == "OPENCV"
    assert radial.image_reader_options[1] == "SIMPLE_RADIAL"
    assert grouped.provenance() == {
        "profile": "auto_grouped_simple_radial_v1",
        "camera_model": "SIMPLE_RADIAL",
        "sharing_policy": "focal_aware_groups",
        "grouping_key_policy": "exif_device_lens_focal_size_orientation_v1",
        "initial_focal_policy": "colmap_exif_or_default",
    }


def test_camera_calibration_capability_checks_grouped_batch_options(
    tmp_path, monkeypatch
) -> None:
    executable = _make_executable(tmp_path / "colmap")
    common = "ImageReader.camera_model ImageReader.single_camera"
    monkeypatch.setattr(colmap, "_capture_help", lambda _executable, _command: common)

    assert (
        colmap_camera_calibration_support_reason(
            executable, "shared_simple_radial_v1"
        )
        is None
    )
    reason = colmap_camera_calibration_support_reason(
        executable, "auto_grouped_simple_radial_v1"
    )
    assert reason is not None
    assert "image_list_path" in reason
    assert "ImageReader.single_camera_per_image" in reason


def test_geometric_verification_profiles_only_change_guided_matching() -> None:
    default = resolve_colmap_geometric_verification("default_v1")
    guided = resolve_colmap_geometric_verification("guided_v1")

    assert default.provenance() == {
        "profile": "default_v1",
        "guided_matching": False,
        "skip_geometric_verification": False,
        "raw_parameter_policy": "colmap_build_defaults",
    }
    assert default.matching_options == (
        "--FeatureMatching.guided_matching",
        "0",
        "--FeatureMatching.skip_geometric_verification",
        "0",
    )
    assert guided.matching_options == (
        "--FeatureMatching.guided_matching",
        "1",
        "--FeatureMatching.skip_geometric_verification",
        "0",
    )


def test_geometric_verification_capability_checks_pairing_and_recovery(
    tmp_path, monkeypatch
) -> None:
    executable = _make_executable(tmp_path / "colmap")
    markers = (
        "FeatureMatching.guided_matching "
        "FeatureMatching.skip_geometric_verification"
    )
    monkeypatch.setattr(colmap, "_capture_help", lambda _executable, _command: markers)

    assert (
        colmap_geometric_verification_support_reason(
            executable, "exhaustive", "guided_v1"
        )
        is None
    )

    monkeypatch.setattr(
        colmap,
        "_capture_help",
        lambda _executable, command: markers if command != "matches_importer" else "",
    )
    assert "matches_importer:FeatureMatching.guided_matching" in (
        colmap_geometric_verification_support_reason(
            executable, "exhaustive", "guided_v1"
        )
        or ""
    )
