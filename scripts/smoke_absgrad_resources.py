#!/usr/bin/env python3
"""Small CPU-only check of the transient service boundary, never a global OOM test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oom-child", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131":
        raise ValueError("resource smoke is restricted to the authorized remote host")
    group_name = Path("/proc/self/cgroup").read_text().strip().split("::")[1]
    expected = "/system.slice/image3d-absgrad-" + ("oom-smoke" if args.oom_child else "isolation-smoke") + "-20261009-v1.service"
    if group_name != expected:
        raise ValueError("smoke must run in its own transient service, not the panel cgroup")
    group = Path("/sys/fs/cgroup") / group_name.lstrip("/")
    maximum = 96 * 1024**2 if args.oom_child else 16 * 1024**3
    for name, value in (("memory.max", maximum), ("memory.swap.max", 0), ("memory.oom.group", 1)):
        if int((group / name).read_text()) != value:
            raise ValueError(f"unexpected smoke {name}")
    print(json.dumps({"cgroup": group_name, "memory_max": maximum, "swap_max": 0,
        "expected_result": "task_cgroup_oom_kill" if args.oom_child else "exit_0"}), flush=True)
    if args.oom_child:
        # ponytail: 128 MiB bounded allocation; the 96 MiB cgroup must kill only this smoke.
        payload = bytearray(128 * 1024**2)
        raise RuntimeError(f"cgroup OOM protection did not fire: allocated {len(payload)} bytes")


if __name__ == "__main__":
    main()
