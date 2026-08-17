from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from image3d_scenegraph.geometry.backends import get_backend_specs


VGGT_REPO_URL = "https://github.com/facebookresearch/vggt.git"
VGGT_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
DINO_REPO_URL = "https://github.com/facebookresearch/dinov2.git"
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
LIGHTGLUE_REPO_URL = "https://github.com/jytime/LightGlue.git"
LIGHTGLUE_REVISION = "2f23ca2ea9638cecad7f7220795210fc6b8353c3"
ALIKED_REVISION = "683d7c65197395c0b3f01ebe76e1084a27e73a65"
ALIKED_CHECKPOINT_URL = f"https://raw.githubusercontent.com/Shiaoming/ALIKED/{ALIKED_REVISION}/models/aliked-n16.pth"
DINO_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
VGGSFM_REVISION = "643a1eb2069dad9b4cff071960323542477272e3"
VGGSFM_TRACKER_SHA256 = "451f1d218be1ef661c81d6254cbc633a5190e7d6069fe51621cc29ba5d90a404"
VGGSFM_TRACKER_URL = f"https://huggingface.co/facebook/VGGSfM/resolve/{VGGSFM_REVISION}/vggsfm_v2_tracker.pt"
VGGT_MODEL_ID = "facebook/VGGT-1B"
VGGT_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_MODEL_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
VGGT_MIN_FREE_GB = 20.0
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up optional geometry backends.")
    parser.add_argument("--backend", choices=["vggt", "dust3r", "mast3r"], required=True)
    parser.add_argument("--install", action="store_true", help="Actually install/download files. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Bypass the free-space guard.")
    parser.add_argument("--min-free-gb", type=float, default=VGGT_MIN_FREE_GB)
    parser.add_argument("--model-id", default=VGGT_MODEL_ID)
    parser.add_argument("--torch-index-url", default=PYTORCH_CUDA_INDEX)
    args = parser.parse_args()

    project_root = Path.cwd()
    specs = {spec.backend_id: spec for spec in get_backend_specs(project_root)}
    spec = specs[args.backend]

    print(f"backend={spec.backend_id}")
    print(f"label={spec.label}")
    print(f"available={str(spec.available).lower()}")
    if spec.reason:
        print(f"reason={spec.reason}")
    print()

    if args.backend == "vggt":
        setup_vggt(
            project_root=project_root,
            install=args.install,
            force=args.force,
            min_free_gb=args.min_free_gb,
            model_id=args.model_id,
            torch_index_url=args.torch_index_url,
        )
        return

    print(f"{spec.label} setup is not implemented yet.")
    print("Expected local layout:")
    print(f"  external/{spec.backend_id}/")
    print(f"  checkpoints/{spec.backend_id}/")


def setup_vggt(
    *,
    project_root: Path,
    install: bool,
    force: bool,
    min_free_gb: float,
    model_id: str,
    torch_index_url: str,
) -> None:
    repo_dir = project_root / "external" / "vggt"
    dinov2_repo = project_root / "external" / "dinov2"
    lightglue_repo = project_root / "external" / "lightglue"
    checkpoint_dir = project_root / "checkpoints" / "vggt" / model_id.replace("/", "--")
    dependency_checkpoint_dir = project_root / "checkpoints" / "vggt"
    dinov2_checkpoint = dependency_checkpoint_dir / "dinov2_vitb14_reg4_pretrain.pth"
    tracker_checkpoint = dependency_checkpoint_dir / "vggsfm_v2_tracker.pt"
    aliked_checkpoint = (
        dependency_checkpoint_dir / "torch-hub" / "checkpoints" / "aliked-n16.pth"
    )
    venv_dir = project_root / ".venv"
    python_bin = venv_dir / "bin" / "python"

    print("VGGT setup plan:")
    print(f"  repo: {repo_dir} @ {VGGT_REVISION}")
    print(f"  dinov2_repo: {dinov2_repo} @ {DINO_REVISION}")
    print(f"  lightglue_repo: {lightglue_repo} @ {LIGHTGLUE_REVISION}")
    print(f"  venv: {venv_dir}")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  dinov2_checkpoint: {dinov2_checkpoint}")
    print(f"  tracker_checkpoint: {tracker_checkpoint}")
    print(f"  aliked_checkpoint: {aliked_checkpoint}")
    print(f"  model_id: {model_id}")
    print(f"  torch_index_url: {torch_index_url}")
    print("Expected local layout:")
    print("  external/vggt/")
    print("  .venv/")
    print("  checkpoints/vggt/facebook--VGGT-1B/")
    print()

    free_gb = get_free_gb(project_root)
    print(f"free_space={free_gb:.1f}GB")
    print("estimated_required_space=at least 12GB, recommended 20GB+")
    print("  checkpoint ~= 5.03GB")
    print("  repo + venv + Python deps depend on torch cache reuse; budget several GB more")
    print()

    if not install:
        print("dry_run=true")
        print("Add --install to perform clone, venv creation, dependency install, and checkpoint download.")
        return

    if free_gb < min_free_gb and not force:
        raise SystemExit(
            f"Refusing install: only {free_gb:.1f}GB free, require at least {min_free_gb:.1f}GB. "
            "Free disk space first or rerun with --force."
        )

    ensure_repo(repo_dir, VGGT_REPO_URL, VGGT_REVISION)
    ensure_repo(dinov2_repo, DINO_REPO_URL, DINO_REVISION)
    ensure_repo(lightglue_repo, LIGHTGLUE_REPO_URL, LIGHTGLUE_REVISION)
    ensure_venv(venv_dir)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--index-url",
            torch_index_url,
            "torch==2.3.1",
            "torchvision==0.18.1",
        ]
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "numpy==1.26.1",
            "Pillow",
            "huggingface_hub",
            "einops",
            "safetensors",
            "opencv-python",
            "scipy==1.15.3",
            "pycolmap==3.10.0",
            "hydra-core==1.3.2",
            "omegaconf==2.3.0",
        ]
    )
    run(["uv", "pip", "install", "--python", str(python_bin), "--no-deps", "-e", str(repo_dir)])
    run(["uv", "pip", "install", "--python", str(python_bin), "-e", str(lightglue_repo)])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(python_bin),
            "-m",
            "huggingface_hub.commands.huggingface_cli",
            "download",
            model_id,
            "--revision",
            VGGT_MODEL_REVISION,
            "model.safetensors",
            "config.json",
            "--local-dir",
            str(checkpoint_dir),
        ]
    )
    verify_sha256(checkpoint_dir / "model.safetensors", VGGT_MODEL_SHA256)
    download_file(DINO_CHECKPOINT_URL, dinov2_checkpoint)
    download_file(ALIKED_CHECKPOINT_URL, aliked_checkpoint)
    download_file(
        VGGSFM_TRACKER_URL,
        tracker_checkpoint,
        expected_sha256=VGGSFM_TRACKER_SHA256,
    )
    dependency_record = {
        "schema_version": 1,
        "vggt_revision": VGGT_REVISION,
        "vggt_model_revision": VGGT_MODEL_REVISION,
        "dinov2_revision": DINO_REVISION,
        "lightglue_revision": LIGHTGLUE_REVISION,
        "aliked_revision": ALIKED_REVISION,
        "vggt_checkpoint_sha256": sha256_file(checkpoint_dir / "model.safetensors"),
        "dinov2_checkpoint_sha256": sha256_file(dinov2_checkpoint),
        "aliked_checkpoint_sha256": sha256_file(aliked_checkpoint),
        "vggsfm_revision": VGGSFM_REVISION,
        "vggsfm_tracker_sha256": sha256_file(tracker_checkpoint),
        "research_only": True,
    }
    (dependency_checkpoint_dir / "ba-dependencies.json").write_text(
        json.dumps(dependency_record, indent=2) + "\n", encoding="utf-8"
    )
    print("VGGT setup complete.")


def ensure_repo(repo_dir: Path, url: str, revision: str) -> None:
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, str(repo_dir)])
    run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", revision])
    run(["git", "-C", str(repo_dir), "checkout", "--detach", revision])


def download_file(
    url: str, destination: Path, *, expected_sha256: str | None = None
) -> None:
    if destination.is_file():
        print(f"checkpoint_exists={destination}")
        if expected_sha256 is not None:
            verify_sha256(destination, expected_sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        if expected_sha256 is not None:
            verify_sha256(temporary, expected_sha256)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_venv(venv_dir: Path) -> None:
    if (venv_dir / "pyvenv.cfg").exists():
        print(f"venv_exists={venv_dir}")
        return
    run(["uv", "venv", str(venv_dir)])


def get_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        sys.exit(130)
