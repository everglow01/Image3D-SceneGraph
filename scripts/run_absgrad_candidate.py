#!/usr/bin/env python3
"""Historical schema-1/2 candidate CLI; the matched-pair CLI owns new paired runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket

from image3d_scenegraph.gaussian.absgrad_experiment import (
    PROJECT_ROOT, execute, gate_template, preflight,
)
from image3d_scenegraph.gaussian.absgrad_resources import write_json
from image3d_scenegraph.gpu_lease import FileLease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    template = commands.add_parser("gate-template")
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--quality-exploration", action="store_true")
    for name in ("preflight", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--signed-experiment", type=Path, required=True)
        command.add_argument("--gate-contract", type=Path, required=True)
        command.add_argument("--gate-sha256", required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--quality-exploration", action="store_true")
    args = parser.parse_args()
    if args.action == "gate-template":
        write_json(args.output, gate_template(quality_exploration=args.quality_exploration))
        return
    if args.action == "execute":
        if socket.gethostname() != "i-94B8D131" or PROJECT_ROOT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
            raise ValueError("candidate execution is restricted to the authorized remote workspace")
        with FileLease(PROJECT_ROOT / "outputs/.gpu.lock") as lease:
            prepared = preflight(args.signed_experiment, args.gate_contract, args.gate_sha256, args.output_dir,
                                 quality_exploration=args.quality_exploration)
            execute(args.output_dir.resolve(), prepared, lease_fd=lease.fileno())
    else:
        preflight(args.signed_experiment, args.gate_contract, args.gate_sha256, args.output_dir,
                                 quality_exploration=args.quality_exploration)
        print("preflight_passed; no training or output directory created")


if __name__ == "__main__":
    main()
