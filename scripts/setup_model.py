from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from image3d_scenegraph.geometry.backends import get_backend_specs


VGGT_REPO_URL = "https://github.com/facebookresearch/vggt.git"
VGGT_MODEL_ID = "facebook/VGGT-1B"
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
    checkpoint_dir = project_root / "checkpoints" / "vggt" / model_id.replace("/", "--")
    venv_dir = project_root / ".venv"
    python_bin = venv_dir / "bin" / "python"

    print("VGGT setup plan:")
    print(f"  repo: {repo_dir}")
    print(f"  venv: {venv_dir}")
    print(f"  checkpoint: {checkpoint_dir}")
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

    ensure_repo(repo_dir)
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
        ]
    )
    run(["uv", "pip", "install", "--python", str(python_bin), "--no-deps", "-e", str(repo_dir)])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(python_bin),
            "-m",
            "huggingface_hub.commands.huggingface_cli",
            "download",
            model_id,
            "model.safetensors",
            "config.json",
            "--local-dir",
            str(checkpoint_dir),
        ]
    )
    print("VGGT setup complete.")


def ensure_repo(repo_dir: Path) -> None:
    if (repo_dir / ".git").exists():
        print(f"repo_exists={repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", VGGT_REPO_URL, str(repo_dir)])


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
