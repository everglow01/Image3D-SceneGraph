#!/usr/bin/env python3
"""Same-code signed/absolute CLI; execution still requires a separately approved gate."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket

from image3d_scenegraph.gaussian.absgrad_experiment import (
    PROJECT_ROOT, execute_matched, matched_gate_template,
)
from image3d_scenegraph.gaussian.absgrad_resources import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-template", type=Path)
    parser.add_argument("--continue-recovered-signed", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--historical-signed", type=Path)
    parser.add_argument("--gate-contract", type=Path)
    parser.add_argument("--gate-sha256")
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    if args.gate_template:
        write_json(args.gate_template, matched_gate_template(recovered=args.continue_recovered_signed))
        return
    if socket.gethostname() != "i-94B8D131" or PROJECT_ROOT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
        raise ValueError("matched execution is restricted to the authorized remote repository")
    if not all((args.output_dir, args.historical_signed, args.gate_contract, args.gate_sha256, args.expected_revision)):
        parser.error("all execution identity arguments are required")
    execute_matched(args.output_dir, args.historical_signed, args.gate_contract, args.gate_sha256,
            expected_revision=args.expected_revision, recovered=args.continue_recovered_signed)


if __name__ == "__main__":
    main()
