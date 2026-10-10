"""Hash-bound geometry copies inside the normal Gaussian Job lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from image3d_scenegraph.execution import JobCancelled
from image3d_scenegraph.file_integrity import sha256_file

BUNDLE = "diagnostics/geometry_bundle.json"
RECEIPT = "diagnostics/geometry_reuse.json"
GEOMETRY_OPTIONS = (
    "sfm_feature_profile",
    "sfm_local_matcher",
    "sfm_pairing",
    "sfm_geometric_verification",
    "sfm_camera_calibration",
    "sfm_mapper",
    "gaussian_longest_edge",
    "sfm_capture_metadata",
)
REQUIRED = (
    "geometry/cameras.json",
    "geometry/points.ply",
    "colmap/undistorted/sparse_txt/cameras.txt",
    "colmap/undistorted/sparse_txt/images.txt",
    "colmap/undistorted/sparse_txt/points3D.txt",
    "diagnostics/colmap_timing.json",
    "diagnostics/sfm_pose_health.json",
    "diagnostics/sfm_pose_recovery.json",
    "diagnostics/sfm_camera_calibration.json",
    "diagnostics/sfm_frontend_contract.json",
)
OPTIONAL = ("diagnostics/rig_pairing.json", "diagnostics/capture_metadata.json")


def regular_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ValueError("unsafe geometry path")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("geometry source cannot contain symlinks")
    if not current.is_file():
        raise ValueError(f"missing geometry file: {relative}")
    return current


def signature(options: dict) -> dict:
    return {key: options.get(key) for key in GEOMETRY_OPTIONS}


def input_hashes(root: Path, assets: list[dict], cancel=None) -> dict:
    result = {}
    for asset in assets:
        if cancel is not None and cancel():
            raise JobCancelled("geometry reuse cancelled")
        name = asset["path"]
        if not name.startswith("input/images/") or name in result:
            raise ValueError("invalid geometry input identity")
        result[name] = sha256_file(regular_file(root, name))
    return result


def _allowed(name: str) -> bool:
    return name in REQUIRED or name in OPTIONAL or name.startswith("colmap/")


def load_bundle(root: Path, expected: str | None = None) -> dict:
    path = regular_file(root, BUNDLE)
    if expected is not None and sha256_file(path) != expected:
        raise ValueError("geometry bundle changed after enqueue")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("status") != "geometry_verified":
        raise ValueError("unsupported or incomplete geometry bundle")
    files = data.get("files")
    if not isinstance(files, dict) or not set(REQUIRED) <= files.keys():
        raise ValueError("geometry bundle is missing required files")
    for name, digest in files.items():
        if not _allowed(name) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError("invalid geometry bundle file")
        regular_file(root, name)
    return data


def source_bundle(
    output_root: Path, job_id: str, expected: str | None = None
) -> tuple[Path, dict]:
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise ValueError("invalid geometry source Job ID")
    root = output_root / job_id
    if root.is_symlink() or root.resolve().parent != output_root.resolve():
        raise ValueError("geometry source must belong to the same Job store")
    manifest = json.loads(regular_file(root, "manifest.json").read_text())
    if (
        manifest.get("job_id") != job_id
        or manifest.get("result_kind") not in (None, "job")
        or manifest.get("status") != "done"
        or manifest.get("mode") != "multi_image"
        or manifest.get("geometry_backend") != "project_3dgs"
        or manifest.get("output_type") != "gaussian_splat"
        or manifest.get("gaussian_geometry_source") != "colmap"
        or manifest.get("assets", {}).get("gaussian_geometry_bundle") != BUNDLE
    ):
        raise ValueError(
            "geometry source must be a completed standard Gaussian Job with a frozen bundle"
        )
    data = load_bundle(root, expected)
    if data.get("job_id") != job_id:
        raise ValueError("geometry bundle belongs to a different Job")
    return root, data


def freeze_geometry(
    root: Path,
    job_id: str,
    options: dict,
    assets: list[dict],
    contract: dict,
    cancel=None,
) -> dict:
    health = json.loads(
        regular_file(root, "diagnostics/sfm_pose_health.json").read_text()
    )
    if health.get("status") != "passed":
        raise ValueError("cannot freeze unhealthy geometry")
    names = set(REQUIRED)
    names.update(name for name in OPTIONAL if (root / name).exists())
    for path in (root / "colmap").rglob("*"):
        if path.is_symlink():
            raise ValueError("geometry source cannot contain symlinks")
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    files = {}
    for name in sorted(names):
        if cancel is not None and cancel():
            raise JobCancelled("geometry freeze cancelled")
        files[name] = sha256_file(regular_file(root, name))
    data = {
        "schema_version": 1,
        "status": "geometry_verified",
        "job_id": job_id,
        "geometry_options": signature(options),
        "inputs": input_hashes(root, assets, cancel),
        "splits": contract["splits"],
        "files": files,
    }
    with (root / BUNDLE).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True)
    return data


def copy_geometry(
    source: Path,
    target: Path,
    expected: str,
    options: dict,
    assets: list[dict],
    cancel=None,
) -> dict:
    data = load_bundle(source, expected)
    if data["geometry_options"] != signature(options):
        raise ValueError("geometry configuration differs from the source Job")
    if data["inputs"] != input_hashes(
        source, [{"path": name} for name in data["inputs"]], cancel
    ):
        raise ValueError("source Job input content changed after geometry was frozen")
    if data["inputs"] != input_hashes(target, assets, cancel):
        raise ValueError("geometry input content or paths differ from the source Job")
    for name, digest in data["files"].items():
        src = regular_file(source, name)
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with src.open("rb") as read, dst.open("xb") as write:
            for chunk in iter(lambda: read.read(8 * 1024 * 1024), b""):
                if cancel is not None and cancel():
                    raise JobCancelled("geometry copy cancelled")
                h.update(chunk)
                write.write(chunk)
        if h.hexdigest() != digest:
            raise ValueError(f"geometry source file changed: {name}")
    if sha256_file(regular_file(source, BUNDLE)) != expected:
        raise ValueError("geometry bundle changed during copy")
    receipt = {
        "schema_version": 1,
        "status": "verified_copy",
        "source_job_id": data["job_id"],
        "source_bundle_sha256": expected,
        "copied_files": data["files"],
        "geometry_recomputed": False,
        "model_reused": False,
    }
    with (target / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)
    return data
