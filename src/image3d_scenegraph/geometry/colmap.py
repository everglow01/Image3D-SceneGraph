from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from image3d_scenegraph.file_integrity import sha256_file


COLMAP_FEATURE_PROFILE_IDS = ("sift_v1", "aliked_n16rot_v1")
COLMAP_LOCAL_MATCHER_IDS = ("bruteforce", "lightglue")
COLMAP_PAIRING_IDS = ("exhaustive", "sequential_loop", "vocab_tree")
COLMAP_GEOMETRIC_VERIFICATION_IDS = ("default_v1", "guided_v1")
COLMAP_CAMERA_CALIBRATION_IDS = (
    "shared_opencv_v1",
    "shared_simple_radial_v1",
    "auto_grouped_simple_radial_v1",
)
COLMAP_LEGACY_MATCHER_IDS = ("exhaustive", "sequential")
COLMAP_LEGACY_MATCHER_TO_PAIRING = {
    "exhaustive": "exhaustive",
    "sequential": "sequential_loop",
}
_LOCAL_MATCHER_MARKERS = {
    ("sift_v1", "bruteforce"): None,
    ("sift_v1", "lightglue"): "SiftMatching.lightglue_model_path",
    ("aliked_n16rot_v1", "bruteforce"): "AlikedMatching.bruteforce_model_path",
    ("aliked_n16rot_v1", "lightglue"): "AlikedMatching.lightglue_model_path",
}
_LOCAL_MATCHER_REQUIREMENTS = {
    ("sift_v1", "bruteforce"): (),
    ("sift_v1", "lightglue"): (
        ("exhaustive_matcher", "SiftMatching.lightglue_model_path"),
        ("matches_importer", "SiftMatching.lightglue_model_path"),
    ),
    ("aliked_n16rot_v1", "bruteforce"): (
        ("exhaustive_matcher", "AlikedMatching.bruteforce_model_path"),
        ("matches_importer", "AlikedMatching.bruteforce_model_path"),
    ),
    ("aliked_n16rot_v1", "lightglue"): (
        ("exhaustive_matcher", "AlikedMatching.lightglue_model_path"),
        ("matches_importer", "AlikedMatching.lightglue_model_path"),
    ),
}
COLMAP_LEARNED_FEATURE_SETUP_COMMAND = (
    "uv run python scripts/setup_colmap_learned_features.py --install"
)
COLMAP_VOCAB_TREE_SETUP_COMMAND = (
    "uv run python scripts/setup_colmap_vocab_tree.py --install"
)


class ColmapFeatureError(ValueError):
    """Raised when a COLMAP feature profile cannot be resolved safely."""


@dataclass(frozen=True)
class ColmapFeatureAsset:
    filename: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ResolvedColmapFeatureProfile:
    profile_id: str
    extractor: str
    descriptor: str
    max_features: int
    extraction_options: tuple[str, ...]
    extractor_model_sha256: str | None = None

    def provenance(self) -> dict[str, str | int | None]:
        return {
            "profile": self.profile_id,
            "extractor": self.extractor,
            "descriptor": self.descriptor,
            "max_features": self.max_features,
            "extractor_model_sha256": self.extractor_model_sha256,
        }


@dataclass(frozen=True)
class ResolvedColmapLocalMatcher:
    profile_id: str
    name: str
    matching_options: tuple[str, ...]
    model_sha256: str | None = None


@dataclass(frozen=True)
class ResolvedColmapPairing:
    profile_id: str
    command: str
    pairing_options: tuple[str, ...]
    vocab_tree_path: Path | None = None
    vocab_tree_sha256: str | None = None

    def provenance(self) -> dict[str, str | None]:
        return {
            "profile": self.profile_id,
            "command": self.command,
            "vocab_tree": (
                self.vocab_tree_path.name
                if self.vocab_tree_path is not None
                else None
            ),
            "vocab_tree_sha256": self.vocab_tree_sha256,
        }


@dataclass(frozen=True)
class ResolvedColmapGeometricVerification:
    profile_id: str
    guided_matching: bool
    matching_options: tuple[str, ...]

    def provenance(self) -> dict[str, str | bool]:
        return {
            "profile": self.profile_id,
            "guided_matching": self.guided_matching,
            "skip_geometric_verification": False,
            "raw_parameter_policy": "colmap_build_defaults",
        }


@dataclass(frozen=True)
class ResolvedColmapCameraCalibration:
    profile_id: str
    camera_model: str
    sharing_policy: str
    grouping_key_policy: str
    image_reader_options: tuple[str, ...]

    def provenance(self) -> dict[str, str]:
        return {
            "profile": self.profile_id,
            "camera_model": self.camera_model,
            "sharing_policy": self.sharing_policy,
            "grouping_key_policy": self.grouping_key_policy,
            "initial_focal_policy": "colmap_exif_or_default",
        }


def colmap_frontend_provenance(
    feature: ResolvedColmapFeatureProfile,
    local_matcher: ResolvedColmapLocalMatcher,
) -> dict[str, str | int | None]:
    return {
        **feature.provenance(),
        "local_matcher_profile": local_matcher.profile_id,
        "local_matcher": local_matcher.name,
        "matcher_model_sha256": local_matcher.model_sha256,
    }


COLMAP_FEATURE_ASSETS = {
    "aliked_n16rot": ColmapFeatureAsset(
        filename="aliked-n16rot.onnx",
        url=(
            "https://github.com/colmap/colmap/releases/download/3.13.0/"
            "aliked-n16rot.onnx"
        ),
        size_bytes=2_997_054,
        sha256="39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    ),
    "aliked_bruteforce": ColmapFeatureAsset(
        filename="bruteforce-matcher.onnx",
        url=(
            "https://github.com/colmap/colmap/releases/download/3.13.0/"
            "bruteforce-matcher.onnx"
        ),
        size_bytes=5_014,
        sha256="3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
    ),
    "sift_lightglue": ColmapFeatureAsset(
        filename="sift-lightglue.onnx",
        url=(
            "https://github.com/colmap/colmap/releases/download/3.13.0/"
            "sift-lightglue.onnx"
        ),
        size_bytes=45_806_253,
        sha256="e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
    ),
    "aliked_lightglue": ColmapFeatureAsset(
        filename="aliked-lightglue.onnx",
        url=(
            "https://github.com/colmap/colmap/releases/download/3.13.0/"
            "aliked-lightglue.onnx"
        ),
        size_bytes=45_804_950,
        sha256="b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    ),
}

COLMAP_VOCAB_TREE_ASSETS = {
    "sift_v1": ColmapFeatureAsset(
        filename="vocab_tree_faiss_flickr100K_words256K.bin",
        url=(
            "https://github.com/colmap/colmap/releases/download/3.11.1/"
            "vocab_tree_faiss_flickr100K_words256K.bin"
        ),
        size_bytes=72_412_636,
        sha256="96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc",
    ),
    "aliked_n16rot_v1": ColmapFeatureAsset(
        filename="vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin",
        url=(
            "https://github.com/colmap/colmap/releases/download/3.13.0/"
            "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin"
        ),
        size_bytes=18_764_565,
        sha256="8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e",
    ),
}


def resolve_colmap_executable(
    project_root: Path | str | None = None,
) -> Path | None:
    """Resolve an explicit, project-local, or PATH COLMAP executable."""
    configured = os.environ.get("IMAGE3D_COLMAP_BIN")
    if configured:
        return _executable(Path(configured).expanduser())

    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    external_root = Path(
        os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external")
    ).expanduser()
    local = external_root / "colmap-4-cuda" / "install" / "bin" / "colmap"
    if resolved := _executable(local):
        return resolved

    found = shutil.which("colmap")
    return _executable(Path(found)) if found else None


def resolve_colmap_vocab_tree(
    project_root: Path | str | None = None,
    *,
    feature_profile_id: str = "sift_v1",
) -> Path | None:
    """Resolve a descriptor-compatible COLMAP vocabulary tree."""
    feature_profile_id = validate_colmap_feature_profile(feature_profile_id)
    configured_name = (
        "IMAGE3D_COLMAP_VOCAB_TREE"
        if feature_profile_id == "sift_v1"
        else "IMAGE3D_COLMAP_ALIKED_VOCAB_TREE"
    )
    configured = os.environ.get(configured_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None

    asset = COLMAP_VOCAB_TREE_ASSETS[feature_profile_id]
    path = (colmap_vocab_tree_root(project_root) / asset.filename).resolve()
    if not path.is_file():
        return None
    _verify_vocab_tree_asset(path, asset)
    return path


def colmap_vocab_tree_root(
    project_root: Path | str | None = None,
) -> Path:
    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    external_root = Path(
        os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external")
    ).expanduser()
    return (external_root / "colmap-vocab").resolve()


def _verify_vocab_tree_asset(path: Path, asset: ColmapFeatureAsset) -> None:
    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        raise ColmapFeatureError(
            f"COLMAP vocabulary tree size mismatch for {path}: "
            f"expected {asset.size_bytes}, got {actual_size}; run "
            f"`{COLMAP_VOCAB_TREE_SETUP_COMMAND}`"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise ColmapFeatureError(
            f"COLMAP vocabulary tree SHA-256 mismatch for {path}: "
            f"expected {asset.sha256}, got {actual_sha256}; run "
            f"`{COLMAP_VOCAB_TREE_SETUP_COMMAND}`"
        )


def validate_colmap_feature_profile(profile_id: str) -> str:
    if profile_id not in COLMAP_FEATURE_PROFILE_IDS:
        raise ColmapFeatureError(f"unsupported COLMAP feature profile: {profile_id}")
    return profile_id


def validate_colmap_local_matcher(profile_id: str) -> str:
    if profile_id not in COLMAP_LOCAL_MATCHER_IDS:
        raise ColmapFeatureError(f"unsupported COLMAP local matcher: {profile_id}")
    return profile_id


def validate_colmap_pairing(profile_id: str) -> str:
    if profile_id not in COLMAP_PAIRING_IDS:
        raise ColmapFeatureError(f"unsupported COLMAP pairing: {profile_id}")
    return profile_id


def validate_colmap_geometric_verification(profile_id: str) -> str:
    if profile_id not in COLMAP_GEOMETRIC_VERIFICATION_IDS:
        raise ColmapFeatureError(
            f"unsupported COLMAP geometric verification: {profile_id}"
        )
    return profile_id


def validate_colmap_camera_calibration(profile_id: str) -> str:
    if profile_id not in COLMAP_CAMERA_CALIBRATION_IDS:
        raise ColmapFeatureError(
            f"unsupported COLMAP camera calibration: {profile_id}"
        )
    return profile_id


def resolve_colmap_feature_profile(
    profile_id: str,
    project_root: Path | str | None = None,
) -> ResolvedColmapFeatureProfile:
    profile_id = validate_colmap_feature_profile(profile_id)
    if profile_id == "sift_v1":
        return ResolvedColmapFeatureProfile(
            profile_id=profile_id,
            extractor="SIFT",
            descriptor="SIFT",
            max_features=8_192,
            extraction_options=(
                "--FeatureExtraction.type",
                "SIFT",
                "--SiftExtraction.max_num_features",
                "8192",
            ),
        )

    extractor_asset = COLMAP_FEATURE_ASSETS["aliked_n16rot"]
    extractor_path = resolve_colmap_feature_asset(extractor_asset, project_root)
    return ResolvedColmapFeatureProfile(
        profile_id=profile_id,
        extractor="ALIKED_N16ROT",
        descriptor="ALIKED",
        max_features=8_192,
        extraction_options=(
            "--FeatureExtraction.type",
            "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features",
            "8192",
            "--AlikedExtraction.min_score",
            "0.2",
            "--AlikedExtraction.n16rot_model_path",
            str(extractor_path),
        ),
        extractor_model_sha256=extractor_asset.sha256,
    )


def resolve_colmap_local_matcher(
    feature: ResolvedColmapFeatureProfile,
    profile_id: str,
    project_root: Path | str | None = None,
) -> ResolvedColmapLocalMatcher:
    profile_id = validate_colmap_local_matcher(profile_id)
    if feature.descriptor == "SIFT":
        if profile_id == "bruteforce":
            return ResolvedColmapLocalMatcher(
                profile_id=profile_id,
                name="SIFT_BRUTEFORCE",
                matching_options=("--FeatureMatching.type", "SIFT_BRUTEFORCE"),
            )
        asset = COLMAP_FEATURE_ASSETS["sift_lightglue"]
        model_path = resolve_colmap_feature_asset(asset, project_root)
        return ResolvedColmapLocalMatcher(
            profile_id=profile_id,
            name="SIFT_LIGHTGLUE",
            matching_options=(
                "--FeatureMatching.type",
                "SIFT_LIGHTGLUE",
                "--SiftMatching.lightglue_min_score",
                "0.1",
                "--SiftMatching.lightglue_model_path",
                str(model_path),
            ),
            model_sha256=asset.sha256,
        )
    if feature.descriptor == "ALIKED":
        asset_key = (
            "aliked_bruteforce" if profile_id == "bruteforce" else "aliked_lightglue"
        )
        asset = COLMAP_FEATURE_ASSETS[asset_key]
        model_path = resolve_colmap_feature_asset(asset, project_root)
        matcher_name = (
            "ALIKED_BRUTEFORCE" if profile_id == "bruteforce" else "ALIKED_LIGHTGLUE"
        )
        option_prefix = (
            "bruteforce" if profile_id == "bruteforce" else "lightglue"
        )
        matching_options = ["--FeatureMatching.type", matcher_name]
        if profile_id == "lightglue":
            matching_options.extend(("--AlikedMatching.lightglue_min_score", "0.1"))
        matching_options.extend(
            (f"--AlikedMatching.{option_prefix}_model_path", str(model_path))
        )
        return ResolvedColmapLocalMatcher(
            profile_id=profile_id,
            name=matcher_name,
            matching_options=tuple(matching_options),
            model_sha256=asset.sha256,
        )
    raise ColmapFeatureError(
        f"unsupported COLMAP descriptor for local matching: {feature.descriptor}"
    )


def resolve_colmap_pairing(
    feature: ResolvedColmapFeatureProfile,
    profile_id: str,
    project_root: Path | str | None = None,
) -> ResolvedColmapPairing:
    profile_id = validate_colmap_pairing(profile_id)
    if profile_id == "exhaustive":
        return ResolvedColmapPairing(
            profile_id=profile_id,
            command="exhaustive_matcher",
            pairing_options=(),
        )

    tree_path = resolve_colmap_vocab_tree(
        project_root,
        feature_profile_id=feature.profile_id,
    )
    if tree_path is None:
        raise ColmapFeatureError(
            f"COLMAP vocabulary tree missing for {feature.profile_id}; run "
            f"`{COLMAP_VOCAB_TREE_SETUP_COMMAND}`"
        )
    tree_sha256 = sha256_file(tree_path)
    if profile_id == "sequential_loop":
        return ResolvedColmapPairing(
            profile_id=profile_id,
            command="sequential_matcher",
            pairing_options=(
                "--SequentialMatching.loop_detection",
                "1",
                "--SequentialMatching.vocab_tree_path",
                str(tree_path),
            ),
            vocab_tree_path=tree_path,
            vocab_tree_sha256=tree_sha256,
        )
    return ResolvedColmapPairing(
        profile_id=profile_id,
        command="vocab_tree_matcher",
        pairing_options=(
            "--VocabTreeMatching.vocab_tree_path",
            str(tree_path),
        ),
        vocab_tree_path=tree_path,
        vocab_tree_sha256=tree_sha256,
    )


def resolve_colmap_geometric_verification(
    profile_id: str,
) -> ResolvedColmapGeometricVerification:
    profile_id = validate_colmap_geometric_verification(profile_id)
    guided_matching = profile_id == "guided_v1"
    return ResolvedColmapGeometricVerification(
        profile_id=profile_id,
        guided_matching=guided_matching,
        matching_options=(
            "--FeatureMatching.guided_matching",
            "1" if guided_matching else "0",
            "--FeatureMatching.skip_geometric_verification",
            "0",
        ),
    )


def resolve_colmap_camera_calibration(
    profile_id: str,
) -> ResolvedColmapCameraCalibration:
    profile_id = validate_colmap_camera_calibration(profile_id)
    if profile_id == "shared_opencv_v1":
        return ResolvedColmapCameraCalibration(
            profile_id=profile_id,
            camera_model="OPENCV",
            sharing_policy="single_camera",
            grouping_key_policy="all_images",
            image_reader_options=(
                "--ImageReader.camera_model",
                "OPENCV",
                "--ImageReader.single_camera",
                "1",
            ),
        )
    if profile_id == "shared_simple_radial_v1":
        return ResolvedColmapCameraCalibration(
            profile_id=profile_id,
            camera_model="SIMPLE_RADIAL",
            sharing_policy="single_camera",
            grouping_key_policy="all_images",
            image_reader_options=(
                "--ImageReader.camera_model",
                "SIMPLE_RADIAL",
                "--ImageReader.single_camera",
                "1",
            ),
        )
    return ResolvedColmapCameraCalibration(
        profile_id=profile_id,
        camera_model="SIMPLE_RADIAL",
        sharing_policy="focal_aware_groups",
        grouping_key_policy="exif_device_lens_focal_size_orientation_v1",
        image_reader_options=(
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
        ),
    )


def resolve_colmap_feature_asset(
    asset: ColmapFeatureAsset,
    project_root: Path | str | None = None,
) -> Path:
    path = colmap_feature_asset_root(project_root) / asset.filename
    if not path.is_file():
        raise ColmapFeatureError(
            f"COLMAP learned feature model missing: {path}; run "
            f"`{COLMAP_LEARNED_FEATURE_SETUP_COMMAND}`"
        )
    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        raise ColmapFeatureError(
            f"COLMAP learned feature model size mismatch for {path}: "
            f"expected {asset.size_bytes}, got {actual_size}; run "
            f"`{COLMAP_LEARNED_FEATURE_SETUP_COMMAND}`"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise ColmapFeatureError(
            f"COLMAP learned feature model SHA-256 mismatch for {path}: "
            f"expected {asset.sha256}, got {actual_sha256}; run "
            f"`{COLMAP_LEARNED_FEATURE_SETUP_COMMAND}`"
        )
    return path.resolve()


def colmap_feature_asset_root(
    project_root: Path | str | None = None,
) -> Path:
    configured = os.environ.get("IMAGE3D_COLMAP_FEATURE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    external_root = Path(
        os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external")
    ).expanduser()
    return (external_root / "colmap-features").resolve()


def colmap_learned_feature_support_reason(executable: Path) -> str | None:
    return _colmap_support_reason(
        executable,
        (("feature_extractor", "AlikedExtraction.max_num_features"),),
    )


def colmap_local_matcher_support_reason(
    executable: Path,
    feature_profile_id: str,
    local_matcher_id: str,
) -> str | None:
    feature_profile_id = validate_colmap_feature_profile(feature_profile_id)
    local_matcher_id = validate_colmap_local_matcher(local_matcher_id)
    return colmap_local_matcher_support_reasons(executable)[
        (feature_profile_id, local_matcher_id)
    ]


def colmap_local_matcher_support_reasons(
    executable: Path,
) -> dict[tuple[str, str], str | None]:
    commands = {
        command
        for requirements in _LOCAL_MATCHER_REQUIREMENTS.values()
        for command, _marker in requirements
    }
    try:
        outputs = {
            command: _capture_help(executable, command)
            for command in sorted(commands)
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        reason = f"cannot inspect COLMAP learned feature support: {exc}"
        return {
            key: None if not requirements else reason
            for key, requirements in _LOCAL_MATCHER_REQUIREMENTS.items()
        }
    return {
        key: _support_reason_from_outputs(requirements, outputs)
        for key, requirements in _LOCAL_MATCHER_REQUIREMENTS.items()
    }


def colmap_pairing_support_reason(
    executable: Path,
    feature_profile_id: str,
    local_matcher_id: str,
    pairing_id: str,
) -> str | None:
    feature_profile_id = validate_colmap_feature_profile(feature_profile_id)
    local_matcher_id = validate_colmap_local_matcher(local_matcher_id)
    pairing_id = validate_colmap_pairing(pairing_id)
    return colmap_pairing_support_reasons(executable)[
        (feature_profile_id, local_matcher_id, pairing_id)
    ]


def colmap_pairing_support_reasons(
    executable: Path,
) -> dict[tuple[str, str, str], str | None]:
    commands = {
        "exhaustive": "exhaustive_matcher",
        "sequential_loop": "sequential_matcher",
        "vocab_tree": "vocab_tree_matcher",
    }
    pairing_markers = {
        "exhaustive": None,
        "sequential_loop": "SequentialMatching.vocab_tree_path",
        "vocab_tree": "VocabTreeMatching.vocab_tree_path",
    }
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    for command in commands.values():
        try:
            outputs[command] = _capture_help(executable, command)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors[command] = f"cannot inspect COLMAP pairing support: {exc}"

    result: dict[tuple[str, str, str], str | None] = {}
    for feature_id in COLMAP_FEATURE_PROFILE_IDS:
        for matcher_id in COLMAP_LOCAL_MATCHER_IDS:
            matcher_marker = _LOCAL_MATCHER_MARKERS[(feature_id, matcher_id)]
            for pairing_id in COLMAP_PAIRING_IDS:
                command = commands[pairing_id]
                if command in errors:
                    reason = errors[command]
                else:
                    required = ["FeatureMatching.type"]
                    if matcher_marker is not None:
                        required.append(matcher_marker)
                    pairing_marker = pairing_markers[pairing_id]
                    if pairing_marker is not None:
                        required.append(pairing_marker)
                    missing = [
                        marker
                        for marker in required
                        if marker not in outputs[command]
                    ]
                    reason = (
                        "COLMAP build is missing pairing options: "
                        + ", ".join(f"{command}:{marker}" for marker in missing)
                        if missing
                        else None
                    )
                result[(feature_id, matcher_id, pairing_id)] = reason
    return result


def colmap_geometric_verification_support_reason(
    executable: Path,
    pairing_id: str,
    profile_id: str,
) -> str | None:
    pairing_id = validate_colmap_pairing(pairing_id)
    profile_id = validate_colmap_geometric_verification(profile_id)
    return colmap_geometric_verification_support_reasons(executable)[
        (pairing_id, profile_id)
    ]


def colmap_geometric_verification_support_reasons(
    executable: Path,
) -> dict[tuple[str, str], str | None]:
    commands = {
        "exhaustive": "exhaustive_matcher",
        "sequential_loop": "sequential_matcher",
        "vocab_tree": "vocab_tree_matcher",
    }
    required_markers = (
        "FeatureMatching.guided_matching",
        "FeatureMatching.skip_geometric_verification",
    )
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    for command in (*commands.values(), "matches_importer"):
        try:
            outputs[command] = _capture_help(executable, command)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors[command] = (
                f"cannot inspect COLMAP geometric verification support: {exc}"
            )

    result: dict[tuple[str, str], str | None] = {}
    for pairing_id, pairing_command in commands.items():
        checked_commands = (pairing_command, "matches_importer")
        for profile_id in COLMAP_GEOMETRIC_VERIFICATION_IDS:
            command_error = next(
                (errors[command] for command in checked_commands if command in errors),
                None,
            )
            if command_error is not None:
                reason = command_error
            else:
                missing = [
                    f"{command}:{marker}"
                    for command in checked_commands
                    for marker in required_markers
                    if marker not in outputs[command]
                ]
                reason = (
                    "COLMAP build is missing geometric verification options: "
                    + ", ".join(missing)
                    if missing
                    else None
                )
            result[(pairing_id, profile_id)] = reason
    return result


def colmap_camera_calibration_support_reason(
    executable: Path,
    profile_id: str,
) -> str | None:
    profile_id = validate_colmap_camera_calibration(profile_id)
    return colmap_camera_calibration_support_reasons(executable)[profile_id]


def colmap_camera_calibration_support_reasons(
    executable: Path,
) -> dict[str, str | None]:
    try:
        output = _capture_help(executable, "feature_extractor")
    except (OSError, subprocess.CalledProcessError) as exc:
        reason = f"cannot inspect COLMAP camera calibration support: {exc}"
        return {profile_id: reason for profile_id in COLMAP_CAMERA_CALIBRATION_IDS}

    common = ("ImageReader.camera_model", "ImageReader.single_camera")
    requirements = {
        "shared_opencv_v1": common,
        "shared_simple_radial_v1": common,
        "auto_grouped_simple_radial_v1": (
            *common,
            "image_list_path",
            "ImageReader.single_camera_per_image",
        ),
    }
    result: dict[str, str | None] = {}
    for profile_id, markers in requirements.items():
        missing = [marker for marker in markers if marker not in output]
        result[profile_id] = (
            "COLMAP build is missing camera calibration options: "
            + ", ".join(f"feature_extractor:{marker}" for marker in missing)
            if missing
            else None
        )
    return result


def _colmap_support_reason(
    executable: Path,
    requirements: tuple[tuple[str, str], ...],
) -> str | None:
    outputs: dict[str, str] = {}
    try:
        for command, _marker in requirements:
            if command not in outputs:
                outputs[command] = _capture_help(executable, command)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"cannot inspect COLMAP learned feature support: {exc}"
    return _support_reason_from_outputs(requirements, outputs)


def _support_reason_from_outputs(
    requirements: tuple[tuple[str, str], ...],
    outputs: dict[str, str],
) -> str | None:
    missing = [
        f"{command}:{marker}"
        for command, marker in requirements
        if marker not in outputs[command]
    ]
    if missing:
        return "COLMAP build is missing learned feature options: " + ", ".join(
            missing
        )
    return None


@lru_cache(maxsize=32)
def _capture_help(executable: Path, command: str) -> str:
    completed = subprocess.run(
        [str(executable), command, "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def _executable(path: Path) -> Path | None:
    resolved = path.resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    return None
