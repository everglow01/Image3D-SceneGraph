from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/ETH3D/multi-view-evaluation.git"
DEFAULT_SOURCE_DIR = Path("external/eth3d-multi-view-evaluation")
UBUNTU_DEPENDENCIES = "libboost-filesystem-dev libboost-system-dev libeigen3-dev libpcl-dev"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the optional ETH3D native point-cloud evaluator.")
    parser.add_argument("--install", action="store_true", help="Clone and build. Default is dry-run.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=0, help="Parallel build jobs. 0 uses the build tool default.")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    build_dir = (args.build_dir or source_dir / "build").resolve()
    evaluator = build_dir / "ETH3DMultiViewEvaluation"
    cmake = shutil.which("cmake")
    compiler = shutil.which("c++")

    print("ETH3D evaluator setup plan:")
    print(f"  repository: {REPO_URL}")
    print(f"  source: {source_dir}")
    print(f"  build: {build_dir}")
    print(f"  evaluator: {evaluator}")
    print(f"  cmake: {cmake or 'missing'}")
    print(f"  compiler: {compiler or 'missing'}")
    print("  native dependencies: Boost filesystem/system, Eigen3, PCL")
    print(f"  Ubuntu packages: {UBUNTU_DEPENDENCIES}")
    print()

    if not args.install:
        print("dry_run=true")
        print("This script never installs system packages or runs sudo.")
        print("Add --install after the native development packages are available.")
        return
    if cmake is None or compiler is None:
        raise SystemExit("CMake and a C++ compiler are required before building the ETH3D evaluator.")

    if not (source_dir / ".git").exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(source_dir)])
    else:
        print(f"repo_exists={source_dir}")

    configure_command = [
        cmake,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cmake_requires_legacy_policy(cmake):
        configure_command.append("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
    configure = subprocess.run(
        configure_command,
        text=True,
        capture_output=True,
    )
    if configure.returncode:
        print(configure.stdout, end="")
        print(configure.stderr, end="", file=sys.stderr)
        raise SystemExit(
            "ETH3D evaluator CMake configuration failed. On Ubuntu, ensure these packages are installed: "
            + UBUNTU_DEPENDENCIES
        )

    build_command = [cmake, "--build", str(build_dir), "--config", "Release"]
    if args.jobs > 0:
        build_command.extend(["--parallel", str(args.jobs)])
    run(build_command)
    if not evaluator.is_file():
        raise SystemExit(f"Build completed but evaluator was not found at {evaluator}")
    print(f"evaluator={evaluator}")


def cmake_requires_legacy_policy(cmake: str) -> bool:
    completed = subprocess.run([cmake, "--version"], check=True, capture_output=True, text=True)
    first_line = completed.stdout.splitlines()[0]
    version = first_line.rsplit(" ", 1)[-1]
    major = int(version.split(".", 1)[0])
    return major >= 4


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
