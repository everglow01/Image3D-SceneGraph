#!/usr/bin/env python3
"""Prepare a research-only target-camera projection overlay without rebuilding CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

RENDERING_SHA256 = "6b5e4101035031afbb26bed7e496382c7502f9d8bc7cc9a432f290674a4d9133"
PROFILE = "camera_sizes_overlay_v1"
HELPER = '''

def _project_target_camera_sizes(projection, camera_sizes, means, covars, quats,
                                 scales, viewmats, Ks, width, height, **kwargs):
    if camera_sizes is None or len(set(camera_sizes)) == 1:
        return projection(means, covars, quats, scales, viewmats, Ks, width, height, **kwargs)
    if kwargs.get("packed") or kwargs.get("sparse_grad") or kwargs.get("camera_model") != "pinhole":
        raise ValueError("camera-size overlay requires dense pinhole projection")
    if len(camera_sizes) != len(viewmats) or len(Ks) != len(viewmats):
        raise ValueError("target camera sizes are misaligned")
    outputs = [projection(means, covars, quats, scales, viewmats[i:i+1], Ks[i:i+1],
                          target_width, target_height, **kwargs)
               for i, (target_width, target_height) in enumerate(camera_sizes)]
    result = []
    for values in zip(*outputs):
        if all(value is None for value in values):
            result.append(None)
        elif any(value is None for value in values):
            raise ValueError("inconsistent optional projection outputs")
        else:
            result.append(torch.cat(values, dim=0))
    return tuple(result)


PROJECT_CAMERA_SIZES_VERSION = 1
'''


def patch_rendering(source):
    if hashlib.sha256(source.encode()).hexdigest() != RENDERING_SHA256:
        raise ValueError("gsplat rendering.py does not match the audited original wheel")
    edits = [
        ("    # Implement the multi-GPU strategy proposed in\n",
         "    camera_sizes = None\n    # Implement the multi-GPU strategy proposed in\n"),
        ("        world_size = torch.distributed.get_world_size()\n",
         '''        world_size = torch.distributed.get_world_size()
        if C != 1 or packed or sparse_grad or with_ut or with_eval3d or camera_model != "pinhole":
            raise ValueError("camera-size overlay supports one dense classic pinhole camera per rank")
        widths = all_gather_int32(world_size, width, device=device)
        heights = all_gather_int32(world_size, height, device=device)
        camera_sizes = list(zip(widths, heights))
'''),
        ("        proj_results = fully_fused_projection(\n",
         "        proj_results = _project_target_camera_sizes(\n            fully_fused_projection, camera_sizes,\n"),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"camera-size patch anchor is not unique: {old!r}")
        source = source.replace(old, new, 1)
    return source + HELPER


def file_hashes(package):
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def prepare(output):
    spec = importlib.util.find_spec("gsplat")
    if spec is None or spec.origin is None:
        raise ValueError("installed gsplat is required")
    package = Path(spec.origin).resolve().parent
    original = file_hashes(package)
    patched = patch_rendering((package / "rendering.py").read_text())
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "gsplat"
    shutil.copytree(package, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "rendering.py").write_text(patched)
    record = {"profile": PROFILE, "source_package": str(package), "source_hashes": original,
              "files": file_hashes(destination), "installed_package_modified": False,
              "source_rendering_sha256": RENDERING_SHA256}
    if file_hashes(package) != original:
        raise ValueError("installed package changed during preparation")
    (output / "overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_overlay(output, actual_package):
    record = json.loads((output / "overlay.json").read_text())
    if record["profile"] != PROFILE or actual_package.resolve() != (output / "gsplat").resolve():
        raise ValueError("camera-size overlay identity/path mismatch")
    if file_hashes(actual_package) != record["files"]:
        raise ValueError("camera-size overlay files changed")
    if record["source_rendering_sha256"] != RENDERING_SHA256:
        raise ValueError("camera-size overlay base is not the audited package")
    if set(record["files"]) != set(record["source_hashes"]) or any(
        value != record["source_hashes"][name] for name, value in record["files"].items() if name != "rendering.py"
    ):
        raise ValueError("camera-size overlay changed more than rendering.py")
    return {"profile": PROFILE, "overlay_sha256": hashlib.sha256((output / "overlay.json").read_bytes()).hexdigest(),
            "rendering_sha256": record["files"]["rendering.py"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir)))


if __name__ == "__main__":
    main()
