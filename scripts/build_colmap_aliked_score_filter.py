#!/usr/bin/env python3
"""Build an isolated ALIKED score-filter candidate from the installed COLMAP source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from image3d_scenegraph.geometry.colmap import sha256_file as digest
from setup_colmap_cuda import PROFILES, capture, run, verify_install


COLMAP_COMMIT = "8bac7b9aab8f1a86dac377538666b10d45544f47"
CANDIDATE_ROOT = "external/colmap-4-aliked-positive-v1"
PATCH_NAME = "colmap-4-aliked-positive-scores-v1.patch"
PATCHED_SOURCE = "src/colmap/feature/aliked.cc"


def copy_runtime_libraries(source_dir: Path, destination_dir: Path) -> dict[str, dict[str, str]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source in sorted(source_dir.glob("libonnxruntime*")):
        destination = destination_dir / source.name
        if source.is_symlink():
            target = os.readlink(source)
            if destination.is_symlink() and os.readlink(destination) == target:
                pass
            elif destination.exists() or destination.is_symlink():
                raise ValueError(f"refusing to replace runtime library: {destination}")
            else:
                destination.symlink_to(target)
            copied[source.name] = {"symlink": target}
        else:
            if destination.exists():
                if digest(destination) != digest(source):
                    raise ValueError(f"runtime library mismatch: {destination}")
            else:
                shutil.copy2(source, destination)
            copied[source.name] = {"sha256": digest(source)}
    if not copied or "libonnxruntime.so" not in copied:
        raise ValueError("installed ONNX Runtime libraries not found")
    return copied


def build(project: Path, *, resume_install: bool) -> None:
    original = project / "external/colmap-4-cuda"
    root = project / CANDIDATE_ROOT
    patch = project / "scripts/patches" / PATCH_NAME
    source = root / "source"
    runtime = original / "build/_deps/onnxruntime-build"
    baseline = original / "install/bin/colmap"
    baseline_sha = digest(baseline)
    if capture(["git", "-C", str(original / "source"), "rev-parse", "HEAD"]).strip() != COLMAP_COMMIT:
        raise ValueError("installed COLMAP source revision mismatch")
    if capture(["git", "-C", str(original / "source"), "status", "--porcelain"]).strip():
        raise ValueError("installed COLMAP source must be clean")
    if capture(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("project tracked files must be clean")
    if not (runtime / "share/onnxruntime/cmake/onnxruntimeConfig.cmake").is_file():
        raise ValueError("installed ONNX Runtime package not found")
    expected = {
        "profile": "aliked_positive_scores_v1", "colmap_commit": COLMAP_COMMIT,
        "patch_sha256": digest(patch), "baseline_binary_sha256": baseline_sha,
        "onnx_runtime_sha256": digest(runtime / "lib/libonnxruntime.so"),
        "score_policy": "finite_and_positive; retain positive scores below min_score",
        "production_default_changed": False,
    }
    if resume_install:
        record = json.loads((root / "build-request.json").read_text())
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("incomplete candidate build provenance mismatch")
        if (root / "build-record.json").exists():
            raise ValueError("candidate build is already complete")
        if capture(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() != COLMAP_COMMIT:
            raise ValueError("candidate source revision mismatch")
        if capture(["git", "-C", str(source), "status", "--porcelain"]).strip() != f"M {PATCHED_SOURCE}":
            raise ValueError("candidate source has unexpected changes")
        run(["git", "-C", str(source), "diff", "--check"])
        run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)])
    else:
        if root.exists():
            raise ValueError(f"refusing to overwrite candidate build: {root}")
        root.mkdir()
        record = {
            **expected,
            "project_commit": capture(["git", "-C", str(project), "rev-parse", "HEAD"]).strip(),
        }
        (root / "build-request.json").write_text(json.dumps(record, indent=2) + "\n")
        run(["git", "clone", "--local", str(original / "source"), str(source)])
        run(["git", "-C", str(source), "checkout", "--detach", COLMAP_COMMIT])
        run(["git", "-C", str(source), "apply", "--check", str(patch)])
        run(["git", "-C", str(source), "apply", str(patch)])
        config = [
            "/usr/bin/cmake", "-S", str(source), "-B", str(root / "build"), "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={root / 'install'}",
            "-DCMAKE_C_COMPILER=/usr/bin/gcc-11", "-DCMAKE_CXX_COMPILER=/usr/bin/g++-11",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc",
            "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11", "-DCMAKE_CUDA_ARCHITECTURES=89",
            "-DCUDA_ENABLED=ON", "-DGUI_ENABLED=OFF", "-DOPENGL_ENABLED=OFF",
            "-DCGAL_ENABLED=OFF", "-DTESTS_ENABLED=OFF", "-DONNX_ENABLED=ON",
            "-DFETCH_ONNX=OFF", "-DDOWNLOAD_ENABLED=OFF",
            f"-Donnxruntime_DIR={runtime / 'share/onnxruntime/cmake'}",
        ]
        run(config)
        run(["/usr/bin/cmake", "--build", str(root / "build"), "--parallel", "2"])
        run(["/usr/bin/cmake", "--install", str(root / "build")])
    binary = root / "install/bin/colmap"
    if not binary.is_file():
        raise ValueError("candidate binary was not installed")
    record["runtime_files"] = copy_runtime_libraries(
        original / "install/lib", root / "install/lib"
    )
    verify_install(binary, PROFILES["learned"])
    if digest(baseline) != baseline_sha:
        raise ValueError("baseline binary changed during candidate build")
    record["binary_sha256"] = digest(binary)
    record["source_diff"] = capture(
        ["git", "-C", str(source), "diff", "--", PATCHED_SOURCE]
    )
    with (root / "build-record.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--resume-install", action="store_true")
    args = parser.parse_args()
    build(args.project.resolve(), resume_install=args.resume_install)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
