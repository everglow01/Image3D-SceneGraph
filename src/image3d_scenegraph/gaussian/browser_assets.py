"""Optional immutable KSPLAT assets; never rewrite source exports or manifests."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from image3d_scenegraph.file_integrity import sha256_file

PROFILE = "ksplat_sh2_k2_v1"
RENDERER = "@mkkellogg/gaussian-splats-3d@0.4.7"
logger = logging.getLogger(__name__)


def contained_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or re.search(r"[\\?#%:]", relative)
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("invalid browser asset path")
    if root.is_symlink():
        raise ValueError("browser job directory must not be a symlink")
    path = root.resolve()
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("browser asset path must not contain symlinks")
    return path


def read_record(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("missing or oversized browser record")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("browser record must be an object")
    return value


def fingerprint(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("browser asset is not a regular file")
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def source_pairs(manifest: dict) -> list[tuple[str, str]]:
    if manifest.get("status") != "done":
        return []
    variants = manifest.get("gaussian_variants")
    if variants is not None:
        if not isinstance(variants, list) or not 1 <= len(variants) <= 4:
            return []
        return [
            (row.get("scene_splat"), row.get("export_metadata"))
            for row in variants if isinstance(row, dict)
        ]
    assets = manifest.get("assets", {})
    return [
        (assets.get("scene_splat"), assets.get("gaussian_export_metadata")),
        (assets.get("scene_splat_vggt_filtered"), assets.get("gaussian_vggt_filtered_export_metadata")),
    ]


def source_metadata(root: Path, source: str, metadata: str) -> tuple[Path, Path, dict]:
    ply = contained_path(root, source)
    meta_path = contained_path(root, metadata)
    meta = read_record(meta_path)
    count = meta.get("gaussian_count")
    if (
        not ply.is_file()
        or not re.fullmatch(r"[a-f0-9]{64}", str(meta.get("browser_sha256", "")))
        or type(count) is not int or not 1 <= count <= 3_000_000
        or meta.get("sh_degree") != 3
    ):
        raise ValueError("browser source requires a hash-bound SH3 export")
    return ply, meta_path, meta


def with_browser_assets(root: Path, manifest: dict) -> dict:
    """Discover complete source-bound assets without hashing large files on GET."""
    assets = []
    for source, metadata in source_pairs(manifest):
        record_path = None
        try:
            ply, meta_path, meta = source_metadata(root, source, metadata)
            folder = f"lifecycle/browser/{meta['browser_sha256']}"
            record_path = root / folder / "record.json"
            record = read_record(contained_path(root, f"{folder}/record.json"))
            output = contained_path(root, f"{folder}/scene.ksplat")
            if (
                record.get("schema_version") != 1
                or record.get("profile") != PROFILE
                or record.get("renderer") != RENDERER
                or record.get("sh_degree") != 2
                or record.get("compression_level") != 2
                or record.get("gaussian_count") != meta["gaussian_count"]
                or record.get("source") != {
                    "path": source, "sha256": meta["browser_sha256"], "stat": fingerprint(ply),
                }
                or record.get("metadata") != {"path": metadata, "sha256": sha256_file(meta_path)}
                or record["output"]["path"] != "scene.ksplat"
                or record["output"]["stat"] != fingerprint(output)
                or not 0 < output.stat().st_size <= 1_073_741_824
                or not re.fullmatch(r"[a-f0-9]{64}", str(record["output"]["sha256"]))
            ):
                raise ValueError("publication identity or source/output fingerprint mismatch")
            assets.append({
                "source": source, "path": f"{folder}/scene.ksplat",
                "sha256": record["output"]["sha256"], "bytes": output.stat().st_size,
                "gaussian_count": meta["gaussian_count"], "sh_degree": 2, "compression_level": 2,
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if record_path is not None and (record_path.exists() or record_path.is_symlink()):
                logger.warning("Ignoring invalid browser publication for %s/%s: %s", root.name, source, exc)
            continue
    # Ignore persisted/unverified declarations; only complete local publications are advertised.
    result = {key: value for key, value in manifest.items() if key != "gaussian_browser_assets"}
    if assets:
        result["gaussian_browser_assets"] = assets
    return result


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_browser_asset(root: Path, source: str, converter: Path) -> Path:
    manifest_path = contained_path(root, "manifest.json")
    manifest_hash = sha256_file(manifest_path)
    manifest = read_record(manifest_path)
    matches = [metadata for path, metadata in source_pairs(manifest) if path == source]
    if len(matches) != 1:
        raise ValueError("source must identify one completed Job variant")
    metadata = matches[0]
    ply, meta_path, meta = source_metadata(root, source, metadata)
    initial = fingerprint(ply)
    if initial["bytes"] > 1_073_741_824:
        raise ValueError("browser source exceeds 1 GiB")
    meta_hash = sha256_file(meta_path)
    source_hash = meta["browser_sha256"]
    if sha256_file(ply) != source_hash:
        raise ValueError("source PLY hash mismatch")
    with ply.open("rb") as handle:
        header = handle.read(8192).split(b"end_header\n", 1)[0]
    count = re.search(rb"^element vertex (\d+)$", header, re.MULTILINE)
    if not count or int(count[1]) != meta["gaussian_count"]:
        raise ValueError("source count disagrees with export metadata")

    parent = contained_path(root, "lifecycle/browser")
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = contained_path(root, "lifecycle/browser/.publish.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = contained_path(root, f"lifecycle/browser/{source_hash}")
        if destination.exists():
            raise FileExistsError(f"browser publication already exists: {destination}")
        # Failed staging remains separate from the discoverable hash directory for diagnosis.
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
        output = staging / "scene.ksplat"
        subprocess.run(
            ["node", "--max-old-space-size=6144", str(converter), str(ply), str(output), "2"],
            check=True, timeout=600,
        )
        if (
            fingerprint(ply) != initial or sha256_file(ply) != source_hash
            or sha256_file(meta_path) != meta_hash or sha256_file(manifest_path) != manifest_hash
        ):
            raise ValueError("source changed during conversion; publication refused")
        record = {
            "schema_version": 1, "profile": PROFILE, "renderer": RENDERER,
            "sh_degree": 2, "compression_level": 2, "gaussian_count": meta["gaussian_count"],
            "source": {"path": source, "sha256": source_hash, "stat": initial},
            "metadata": {"path": metadata, "sha256": meta_hash},
            "output": {"path": "scene.ksplat", "sha256": sha256_file(output), "stat": fingerprint(output)},
        }
        with (staging / "record.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(staging)
        # A single publisher lock protects the existence check and same-filesystem rename.
        staging.rename(destination)
        _sync_directory(parent)
        return destination
