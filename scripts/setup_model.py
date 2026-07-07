from __future__ import annotations

import argparse
from pathlib import Path

from image3d_scenegraph.geometry.backends import get_backend_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Show setup requirements for optional geometry backends.")
    parser.add_argument("--backend", choices=["vggt", "dust3r", "mast3r", "nerfstudio_3dgs"], required=True)
    args = parser.parse_args()

    specs = {spec.backend_id: spec for spec in get_backend_specs(Path.cwd())}
    spec = specs[args.backend]

    print(f"backend={spec.backend_id}")
    print(f"label={spec.label}")
    print(f"available={str(spec.available).lower()}")
    if spec.reason:
        print(f"reason={spec.reason}")
    print()

    if spec.backend_id == "nerfstudio_3dgs":
        print("Nerfstudio 3DGS is currently supported as an import path, not an automatic training backend.")
        print("Export an existing Nerfstudio splatfacto checkpoint, then run scripts/register_gaussian_splat.py.")
        return

    print("Expected local layout:")
    print(f"  external/{spec.backend_id}/")
    print(f"  checkpoints/{spec.backend_id}/")
    print()
    print("This script intentionally does not download model code or weights yet.")
    print("Next implementation step: add backend-specific clone/checkpoint commands after the target repo is chosen.")


if __name__ == "__main__":
    main()
