"""Import a canonical/INRIA Gaussian PLY into a project model snapshot (.pt).

Thin CLI wrapper around importer.import_inria_ply for experiment paths that
need a renderable/evaluable snapshot from a derivative PLY (e.g. the SOR
floater cleanup filter). Writes the destination .pt plus <destination>.import.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image3d_scenegraph.gaussian.importer import GaussianImportError, import_inria_ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Canonical 62-field binary little-endian Gaussian PLY.")
    parser.add_argument("--destination", required=True, type=Path, help="Model snapshot .pt to write.")
    args = parser.parse_args()
    try:
        record = import_inria_ply(args.source, args.destination)
    except (GaussianImportError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))
    print(f"model={record['model']}")
    print(f"gaussian_count={record['gaussian_count']}")
    print(f"model_sha256={record['model_sha256']}")


if __name__ == "__main__":
    main()
