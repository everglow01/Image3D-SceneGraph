from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


COLMAP_FEATURE_PROFILE_IDS = ("sift_v1", "aliked_n16rot_v1")
COLMAP_LEARNED_FEATURE_SETUP_COMMAND = (
    "uv run python scripts/setup_colmap_learned_features.py --install"
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
    local_matcher: str
    max_features: int
    extraction_options: tuple[str, ...]
    matching_options: tuple[str, ...]
    extractor_model_sha256: str | None = None
    matcher_model_sha256: str | None = None

    def provenance(self) -> dict[str, str | int | None]:
        return {
            "profile": self.profile_id,
            "extractor": self.extractor,
            "descriptor": self.descriptor,
            "local_matcher": self.local_matcher,
            "max_features": self.max_features,
            "extractor_model_sha256": self.extractor_model_sha256,
            "matcher_model_sha256": self.matcher_model_sha256,
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
}


def resolve_colmap_executable(project_root: Path | str | None = None) -> Path | None:
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


def resolve_colmap_vocab_tree(project_root: Path | str | None = None) -> Path | None:
    """Resolve the COLMAP vocab tree used for sequential matching loop detection."""
    configured = os.environ.get("IMAGE3D_COLMAP_VOCAB_TREE")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None

    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    external_root = Path(
        os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external")
    ).expanduser()
    local = (
        external_root
        / "colmap-vocab"
        / "vocab_tree_faiss_flickr100K_words256K.bin"
    ).resolve()
    return local if local.is_file() else None


def validate_colmap_feature_profile(profile_id: str) -> str:
    if profile_id not in COLMAP_FEATURE_PROFILE_IDS:
        raise ColmapFeatureError(f"unsupported COLMAP feature profile: {profile_id}")
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
            local_matcher="SIFT_BRUTEFORCE",
            max_features=8_192,
            extraction_options=(
                "--FeatureExtraction.type",
                "SIFT",
                "--SiftExtraction.max_num_features",
                "8192",
            ),
            matching_options=(
                "--FeatureMatching.type",
                "SIFT_BRUTEFORCE",
            ),
        )

    extractor_asset = COLMAP_FEATURE_ASSETS["aliked_n16rot"]
    matcher_asset = COLMAP_FEATURE_ASSETS["aliked_bruteforce"]
    extractor_path = resolve_colmap_feature_asset(extractor_asset, project_root)
    matcher_path = resolve_colmap_feature_asset(matcher_asset, project_root)
    return ResolvedColmapFeatureProfile(
        profile_id=profile_id,
        extractor="ALIKED_N16ROT",
        descriptor="ALIKED",
        local_matcher="ALIKED_BRUTEFORCE",
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
        matching_options=(
            "--FeatureMatching.type",
            "ALIKED_BRUTEFORCE",
            "--AlikedMatching.bruteforce_model_path",
            str(matcher_path),
        ),
        extractor_model_sha256=extractor_asset.sha256,
        matcher_model_sha256=matcher_asset.sha256,
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
    try:
        extraction_help = _capture_help(executable, "feature_extractor")
        matching_help = _capture_help(executable, "exhaustive_matcher")
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"cannot inspect COLMAP learned feature support: {exc}"
    markers = (
        (extraction_help, "AlikedExtraction.max_num_features"),
        (matching_help, "AlikedMatching.bruteforce_model_path"),
    )
    missing = [marker for output, marker in markers if marker not in output]
    if missing:
        return "COLMAP build is missing learned feature options: " + ", ".join(
            missing
        )
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
