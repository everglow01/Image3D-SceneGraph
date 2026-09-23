#!/usr/bin/env python3
"""Compare byte-identical export writers on one existing PLY; no model rendering."""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from image3d_scenegraph.gaussian.dataset import sha256_file
from image3d_scenegraph.gaussian.export import PLY_FIELDS, write_binary_ply, write_deterministic_zip
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu


def ply_layout(source: Path) -> tuple[bytes, int]:
    with source.open("rb") as handle:
        header = bytearray()
        while len(header) < 8192:
            line = handle.readline(8192 - len(header))
            if not line:
                break
            header.extend(line)
            if line == b"end_header\n":
                break
    lines = header.decode("ascii").splitlines()
    assert lines[0] == "ply" and lines[-1] == "end_header"
    assert "format binary_little_endian 1.0" in lines
    assert tuple(line.split()[-1] for line in lines if line.startswith("property ")) == PLY_FIELDS
    assert all(line.startswith("property float ") for line in lines if line.startswith("property "))
    count = int(next(line for line in lines if line.startswith("element vertex ")).split()[-1])
    assert 0 < count <= 3_000_000
    assert source.stat().st_size == len(header) + count * len(PLY_FIELDS) * 4
    return bytes(header), count


def trial(source: Path, output: Path, operation: str, writer: str) -> dict:
    started = time.perf_counter()
    if operation == "ply":
        header, count = ply_layout(source)
        rows = np.memmap(source, dtype="<f4", mode="r", offset=len(header),
                         shape=(count, len(PLY_FIELDS)))
        if writer == "previous":
            with output.open("xb") as handle:
                handle.write(header)
                handle.write(rows.astype("<f4", copy=False).tobytes(order="C"))
        else:
            write_binary_ply(output, rows)
    else:
        entries = {"gaussian/canonical.ply": source, "gaussian/scene.ply": source}
        if writer == "previous":
            with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name in sorted(entries):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, entries[name].read_bytes())
        else:
            write_deterministic_zip(output, entries)
    return {"operation": operation, "writer": writer,
            "seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--operation", choices=("ply", "zip"))
    parser.add_argument("--writer", choices=("previous", "streamed"))
    args = parser.parse_args()
    source, output = args.source.absolute(), args.output.absolute()
    assert source.is_file() and not source.is_symlink() and source.stat().st_size <= 1_073_741_824
    if args.operation:
        assert args.writer is not None and not output.exists()
        print(json.dumps(trial(source, output, args.operation, args.writer)))
        return
    assert args.writer is None and args.lease is not None
    assert not output.exists() and not output.is_relative_to(source.parent)
    with FileLease(args.lease):
        require_idle_gpu()
        output.mkdir(parents=True, exist_ok=False)
        report = {"status": "running", "source": str(source), "trials": [],
                  "scope": "CPU export IO only; existing SH3 PLY; ZIP contains two PLY roles only, not a complete export",
                  "test_rgb": "not_loaded", "timing_scope": "write and close; excludes hashing, model loading and fsync",
                  "order": "previous then streamed; OS page cache is not flushed; one trial each, not a cold-disk benchmark"}
        try:
            report["source_before_sha256"] = sha256_file(source)
            _, report["gaussian_count"] = ply_layout(source)
            for operation in ("ply", "zip"):
                hashes = []
                for writer in ("previous", "streamed"):
                    target = output / f"{writer}.{operation}"
                    result = subprocess.run([
                        sys.executable, str(Path(__file__).resolve()),
                        "--source", str(source), "--output", str(target),
                        "--operation", operation, "--writer", writer,
                    ], check=True, capture_output=True, text=True, timeout=600)
                    metrics = json.loads(result.stdout)
                    metrics["sha256"] = sha256_file(target)
                    report["trials"].append(metrics)
                    hashes.append(metrics["sha256"])
                    print(json.dumps(metrics), flush=True)
                assert hashes[0] == hashes[1], f"{operation} output bytes changed"
                if operation == "ply":
                    assert hashes[0] == report["source_before_sha256"], "PLY source representation changed"
            report["source_after_sha256"] = sha256_file(source)
            assert report["source_before_sha256"] == report["source_after_sha256"]
            report["status"] = "passed"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = str(exc)
            raise
        finally:
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
