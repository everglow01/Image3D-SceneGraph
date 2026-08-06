from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


COLMAP_REPOSITORY = "https://github.com/colmap/colmap.git"
COLMAP_TAG = "3.9.1"
DEFAULT_ROOT = Path("external/colmap-cuda")
UBUNTU_DEPENDENCIES = (
    "cmake libfreeimage-dev libmetis-dev libgoogle-glog-dev "
    "libceres-dev libsuitesparse-dev"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated CUDA-enabled COLMAP for Image3D-SceneGraph."
    )
    parser.add_argument("--install", action="store_true", help="Clone, build, and install. Default is dry-run.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 2:
        parser.error("--jobs must be 1 or 2")

    root = args.root.resolve()
    source_dir = root / "source"
    build_dir = root / "build"
    install_dir = root / "install"
    executable = install_dir / "bin" / "colmap"
    cmake = Path("/usr/bin/cmake")
    ninja = Path(shutil.which("ninja") or "ninja")
    gcc = Path("/usr/bin/gcc-11")
    gxx = Path("/usr/bin/g++-11")
    nvcc = Path("/usr/local/cuda/bin/nvcc")

    print("CUDA COLMAP setup plan:")
    print(f"  repository: {COLMAP_REPOSITORY}")
    print(f"  tag: {COLMAP_TAG}")
    print(f"  source: {source_dir}")
    print(f"  build: {build_dir}")
    print(f"  install: {install_dir}")
    print(f"  executable: {executable}")
    print(f"  CUDA architecture: 86")
    print(f"  cmake: {cmake}")
    print(f"  ninja: {ninja}")
    print(f"  C/C++ compiler: {gcc} / {gxx}")
    print(f"  CUDA compiler: {nvcc}")
    print(f"  Ubuntu packages: {UBUNTU_DEPENDENCIES}")
    print()

    if not args.install:
        print("dry_run=true")
        print("This script never installs system packages or runs sudo.")
        print("Install the listed packages, then add --install.")
        return

    required = [cmake, ninja, gcc, gxx, nvcc]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing build tools: " + ", ".join(missing))

    root.mkdir(parents=True, exist_ok=True)
    if not (source_dir / ".git").exists():
        run(
            [
                "git",
                "clone",
                "--branch",
                COLMAP_TAG,
                "--depth",
                "1",
                COLMAP_REPOSITORY,
                str(source_dir),
            ]
        )
    else:
        tag = capture(["git", "-C", str(source_dir), "describe", "--tags", "--exact-match"]).strip()
        if tag != COLMAP_TAG:
            raise SystemExit(f"Existing COLMAP source is {tag!r}, expected {COLMAP_TAG!r}")

    env = os.environ.copy()
    env.update(
        {
            "CC": str(gcc),
            "CXX": str(gxx),
            "CUDAHOSTCXX": str(gxx),
        }
    )
    configure = capture(
        [
            str(cmake),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_C_COMPILER={gcc}",
            f"-DCMAKE_CXX_COMPILER={gxx}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DCMAKE_CUDA_HOST_COMPILER={gxx}",
            "-DCMAKE_CUDA_ARCHITECTURES=86",
            "-DCUDA_ENABLED=ON",
            "-DGUI_ENABLED=OFF",
            "-DOPENGL_ENABLED=OFF",
            "-DCGAL_ENABLED=OFF",
            "-DTESTS_ENABLED=OFF",
        ],
        env=env,
    )
    print(configure, end="")
    if "Enabling CUDA support" not in configure:
        raise SystemExit("COLMAP configuration did not enable CUDA")

    run([str(cmake), "--build", str(build_dir), "--parallel", str(args.jobs)], env=env)
    run([str(cmake), "--install", str(build_dir)], env=env)
    if not executable.is_file():
        raise SystemExit(f"Build completed but COLMAP was not installed at {executable}")
    version = capture([str(executable), "-h"])
    if "with CUDA" not in version:
        raise SystemExit("Installed COLMAP does not report CUDA support")
    print(version.splitlines()[0])
    print(f"colmap={executable}")


def capture(command: list[str], *, env: dict[str, str] | None = None) -> str:
    print("+ " + " ".join(command))
    completed = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
    return completed.stdout + completed.stderr


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        sys.exit(130)
