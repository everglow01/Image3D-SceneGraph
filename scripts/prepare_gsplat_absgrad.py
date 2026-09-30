#!/usr/bin/env python3
"""Prepare an isolated, hash-pinned gsplat overlay for distributed AbsGrad."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


RENDERING_SHA256 = "6b5e4101035031afbb26bed7e496382c7502f9d8bc7cc9a432f290674a4d9133"


def patch_rendering(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != RENDERING_SHA256:
        raise ValueError("gsplat rendering.py does not match the audited 1.5.3 wheel")
    guard = '''    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."
'''
    replacement = '''    if absgrad and distributed:
        assert not packed and C == 1, "Project AbsGrad requires unpacked, one camera per rank."
        assert not with_ut and not with_eval3d, "Project AbsGrad requires classic 3DGS."
'''
    anchor = "            colors = reshape_view(C, colors, N_world)\n\n"
    capture = '''    if absgrad and distributed:
        assert colors.shape[-1] <= channel_chunk, "Project AbsGrad forbids channel chunking."
        meta["project_absgrad_means2d"] = means2d
        meta["project_absgrad_counts"] = N_world

'''
    if source.count(guard) != 1 or source.count(anchor) != 1:
        raise ValueError("gsplat AbsGrad patch anchors are not unique")
    return (
        source.replace(guard, replacement).replace(anchor, anchor + capture)
        + "\nPROJECT_DISTRIBUTED_ABSGRAD_VERSION = 1\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.find_spec("gsplat")
    if spec is None or spec.origin is None:
        raise SystemExit("The pinned gsplat wheel must already be installed")
    package = Path(spec.origin).resolve().parent
    source = package / "rendering.py"
    patched = patch_rendering(source.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "gsplat"
    shutil.copytree(package, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "rendering.py").write_text(patched, encoding="utf-8")
    record = {
        "profile": "distributed_absgrad_overlay_v1",
        "source_package": str(package),
        "source_rendering_sha256": RENDERING_SHA256,
        "patched_rendering_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "cuda_binaries": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.so"))
        },
        "installed_package_modified": False,
    }
    (args.output_dir / "overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
