from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


COLMAP_REPOSITORY = "https://github.com/colmap/colmap.git"
BASE_UBUNTU_DEPENDENCIES = (
    "cmake libfreeimage-dev libmetis-dev libgoogle-glog-dev "
    "libceres-dev libsuitesparse-dev"
)
LEARNED_UBUNTU_DEPENDENCIES = (
    "libopenimageio-dev openimageio-tools libopenexr-dev"
)


@dataclass(frozen=True)
class SetupProfile:
    tag: str
    root: Path
    cuda_root: Path
    cuda_architecture: str
    onnx: bool


PROFILES = {
    "learned": SetupProfile(
        tag="4.0.0",
        root=Path("external/colmap-4-cuda"),
        cuda_root=Path("/usr/local/cuda-12.2"),
        cuda_architecture="89",
        onnx=True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated CUDA-enabled COLMAP for Image3D-SceneGraph."
    )
    parser.add_argument("--install", action="store_true", help="Clone, build, and install. Default is dry-run.")
    parser.add_argument("--profile", choices=PROFILES, default="learned")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 2:
        parser.error("--jobs must be 1 or 2")

    profile = PROFILES[args.profile]
    root = (args.root or profile.root).resolve()
    source_dir = root / "source"
    build_dir = root / "build"
    install_dir = root / "install"
    executable = install_dir / "bin" / "colmap"
    cmake = Path("/usr/bin/cmake")
    ninja = Path(shutil.which("ninja") or "ninja")
    gcc = Path("/usr/bin/gcc-11")
    gxx = Path("/usr/bin/g++-11")
    nvcc = profile.cuda_root / "bin" / "nvcc"

    print("CUDA COLMAP setup plan:")
    print(f"  profile: {args.profile}")
    print(f"  repository: {COLMAP_REPOSITORY}")
    print(f"  tag: {profile.tag}")
    print(f"  source: {source_dir}")
    print(f"  build: {build_dir}")
    print(f"  install: {install_dir}")
    print(f"  executable: {executable}")
    print(f"  CUDA root: {profile.cuda_root}")
    print(f"  CUDA architecture: {profile.cuda_architecture}")
    print(f"  ONNX: {profile.onnx}")
    print(f"  cmake: {cmake}")
    print(f"  ninja: {ninja}")
    print(f"  C/C++ compiler: {gcc} / {gxx}")
    print(f"  CUDA compiler: {nvcc}")
    ubuntu_dependencies = BASE_UBUNTU_DEPENDENCIES
    if profile.onnx:
        ubuntu_dependencies += " " + LEARNED_UBUNTU_DEPENDENCIES
    print(f"  Ubuntu packages: {ubuntu_dependencies}")
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
                profile.tag,
                "--depth",
                "1",
                COLMAP_REPOSITORY,
                str(source_dir),
            ]
        )
    else:
        tag = capture(
            ["git", "-C", str(source_dir), "describe", "--tags", "--exact-match"]
        ).strip()
        if tag != profile.tag:
            raise SystemExit(
                f"Existing COLMAP source is {tag!r}, expected {profile.tag!r}"
            )

    env = os.environ.copy()
    env.update(
        {
            "CC": str(gcc),
            "CXX": str(gxx),
            "CUDAHOSTCXX": str(gxx),
            "CUDACXX": str(nvcc),
        }
    )
    configure_command = [
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
        f"-DCMAKE_CUDA_ARCHITECTURES={profile.cuda_architecture}",
        "-DCUDA_ENABLED=ON",
        "-DGUI_ENABLED=OFF",
        "-DOPENGL_ENABLED=OFF",
        "-DCGAL_ENABLED=OFF",
        "-DTESTS_ENABLED=OFF",
    ]
    if profile.onnx:
        configure_command.extend(
            (
                "-DONNX_ENABLED=ON",
                "-DFETCH_ONNX=ON",
                "-DDOWNLOAD_ENABLED=ON",
            )
        )
    configure = capture(configure_command, env=env)
    print(configure, end="")
    if "Enabling CUDA support" not in configure:
        raise SystemExit("COLMAP configuration did not enable CUDA")
    if profile.onnx and (
        "Configuring onnxruntime... done" not in configure
        or "Disabling ONNX support" in configure
    ):
        raise SystemExit("COLMAP configuration did not enable ONNX")

    run([str(cmake), "--build", str(build_dir), "--parallel", str(args.jobs)], env=env)
    run([str(cmake), "--install", str(build_dir)], env=env)
    if not executable.is_file():
        raise SystemExit(f"Build completed but COLMAP was not installed at {executable}")
    verify_install(executable, profile)
    print(f"colmap={executable}")


def verify_install(executable: Path, profile: SetupProfile) -> None:
    version = capture([str(executable), "-h"])
    if f"COLMAP {profile.tag}" not in version:
        raise SystemExit(f"Installed COLMAP does not report version {profile.tag}")
    if "with CUDA" not in version:
        raise SystemExit("Installed COLMAP does not report CUDA support")
    if profile.onnx:
        extractor_help = capture([str(executable), "feature_extractor", "-h"])
        matcher_help = capture([str(executable), "exhaustive_matcher", "-h"])
        importer_help = capture([str(executable), "matches_importer", "-h"])
        markers = (
            (extractor_help, "AlikedExtraction.max_num_features"),
            (matcher_help, "AlikedMatching.bruteforce_model_path"),
            (matcher_help, "SiftMatching.lightglue_model_path"),
            (matcher_help, "AlikedMatching.lightglue_model_path"),
            (importer_help, "AlikedMatching.bruteforce_model_path"),
            (importer_help, "SiftMatching.lightglue_model_path"),
            (importer_help, "AlikedMatching.lightglue_model_path"),
        )
        missing = [marker for output, marker in markers if marker not in output]
        if missing:
            raise SystemExit(
                "Installed COLMAP is missing learned feature options: "
                + ", ".join(missing)
            )
    print(" ".join(line.strip() for line in version.splitlines()[:2]))


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
