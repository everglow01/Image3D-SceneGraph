#!/usr/bin/env python3
"""Set up one pinned external Gaussian trainer in an isolated environment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from image3d_scenegraph.gaussian.trainers import (
    GRAPHDECO_COMMIT,
    NERFSTUDIO_COMMIT,
    get_gaussian_trainer_specs,
)


REPOSITORIES = {
    "graphdeco": (
        "https://github.com/graphdeco-inria/gaussian-splatting.git",
        GRAPHDECO_COMMIT,
    ),
    "nerfstudio": (
        "https://github.com/nerfstudio-project/nerfstudio.git",
        NERFSTUDIO_COMMIT,
    ),
}
MIN_FREE_GB = 15.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", choices=["graphdeco", "nerfstudio"], required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--accept-research-license", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    repo = project_root / "external" / (
        "gaussian-splatting" if args.trainer == "graphdeco" else "nerfstudio"
    )
    url, revision = REPOSITORIES[args.trainer]
    free_gb = shutil.disk_usage(project_root).free / 1024**3
    cuda = _cuda_status(project_root)
    print(f"trainer={args.trainer}")
    print(f"repo={repo}")
    print(f"revision={revision}")
    print(f"free_space={free_gb:.1f}GB")
    print(f"cuda_available={str(cuda['available']).lower()}")
    if cuda["reason"]:
        print(f"cuda_reason={cuda['reason']}")
    if args.trainer == "graphdeco":
        print("license=Graphdeco research/evaluation only")
    print(f"install={str(args.install).lower()}")
    if not args.install:
        print("dry_run=true")
        return
    if args.trainer == "graphdeco" and not args.accept_research_license:
        raise SystemExit("Graphdeco setup requires --accept-research-license")
    if free_gb < args.min_free_gb and not args.force:
        raise SystemExit(
            f"refusing install: {free_gb:.1f}GB free, require {args.min_free_gb:.1f}GB"
        )
    if not cuda["available"] and not args.force:
        raise SystemExit(f"refusing install: {cuda['reason']}")

    _checkout(repo, url, revision, recursive=args.trainer == "graphdeco")
    venv = repo / ".venv"
    python = venv / "bin" / "python"
    if not python.exists():
        _run(["uv", "venv", "--python", "3.10", str(venv)])
    if args.trainer == "nerfstudio":
        _run(
            [
                "uv", "pip", "install", "--python", str(python),
                "torch==2.0.1", "torchvision==0.15.2",
                "--index-url", "https://download.pytorch.org/whl/cu117",
            ]
        )
        _run(["uv", "pip", "install", "--python", str(python), "-e", str(repo)])
        _run(["uv", "pip", "install", "--python", str(python), "tensorboard"])
    else:
        _run(
            [
                "uv", "pip", "install", "--python", str(python),
                "torch==2.0.1", "torchvision==0.15.2",
                "--index-url", "https://download.pytorch.org/whl/cu117",
            ]
        )
        _run(
            [
                "uv", "pip", "install", "--python", str(python),
                "plyfile", "tqdm", "tensorboard", "numpy<2",
            ]
        )
        for package in ("simple-knn", "diff-gaussian-rasterization", "fused-ssim"):
            _run([
                "uv", "pip", "install", "--no-build-isolation", "--python", str(python),
                str(repo / "submodules" / package),
            ])
    (repo / ".image3d-revision").write_text(revision + "\n", encoding="utf-8")
    probe = subprocess.run(
        [str(python), "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = {
        "trainer": args.trainer,
        "revision": revision,
        "python": str(python),
        "probe": probe.stdout.strip(),
        "status": [spec.to_dict() for spec in get_gaussian_trainer_specs(project_root)],
    }
    (repo / ".image3d-environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _cuda_status(project_root: Path) -> dict[str, object]:
    python = project_root / ".venv" / "bin" / "python"
    completed = subprocess.run(
        [str(python), "-c", "import torch; print(int(torch.cuda.is_available()))"],
        capture_output=True,
        text=True,
    )
    available = completed.returncode == 0 and completed.stdout.strip() == "1"
    reason = None if available else "NVIDIA driver/CUDA is unavailable to the project environment"
    return {"available": available, "reason": reason}


def _checkout(repo: Path, url: str, revision: str, *, recursive: bool) -> None:
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", url, str(repo)])
    _run(["git", "-C", str(repo), "fetch", "origin", revision])
    _run(["git", "-C", str(repo), "checkout", "--detach", revision])
    if recursive:
        _run(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"])


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
