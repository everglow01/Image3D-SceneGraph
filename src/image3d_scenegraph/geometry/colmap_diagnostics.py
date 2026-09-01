from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCHEMA_VERSION = 1
PROFILE_ID = "sfm_frontend_diagnostics_v1"
MAX_IMAGE_ID = 2_147_483_647
FEATURE_IMAGES_PER_SHARD = 32
PAIRS_PER_SHARD = 256
KEYPOINT_DECIMALS = 2


class ColmapDiagnosticsError(ValueError):
    """Raised when final COLMAP diagnostics cannot be exported safely."""


def export_colmap_diagnostics(
    *,
    job_dir: Path,
    database_path: Path,
    source_image_root: Path,
    dataset_contract_path: Path,
    output_dir: Path,
    matcher: str,
    colmap_build: str,
    video_selection_path: Path | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[Path, dict[str, int | str]]:
    root = job_dir.resolve()
    database = _contained(database_path, root, file=True, label="COLMAP database")
    image_root = _contained(source_image_root, root, file=False, label="source image root")
    contract_path = _contained(dataset_contract_path, root, file=True, label="dataset contract")
    final_output = output_dir.resolve()
    if final_output.exists():
        raise ColmapDiagnosticsError("SfM diagnostics output already exists")
    _relative(root, final_output)
    if matcher not in {"exhaustive", "sequential"}:
        raise ColmapDiagnosticsError(f"unsupported COLMAP matcher: {matcher}")

    contract = _read_json(contract_path, "dataset contract")
    registered, splits, normalized_from_world = _registered_images(contract)
    selection = _video_selection(video_selection_path, root)
    temporary = final_output.with_name(f".{final_output.name}.tmp")
    if temporary.exists():
        raise ColmapDiagnosticsError("temporary SfM diagnostics output already exists")
    temporary.mkdir(parents=True)

    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            _require_tables(connection)
            _check_cancelled(cancel_requested)
            images = _image_records(
                connection,
                root,
                image_root,
                registered,
                splits,
                normalized_from_world,
                selection,
                cancel_requested,
            )
            image_set_hash = _hash_json([image["frame_uid"] for image in images])
            run_id = _run_id(image_set_hash, matcher, colmap_build)
            temporary_run = temporary / "runs" / run_id
            final_run = final_output / "runs" / run_id
            feature_index, keypoint_counts = _write_feature_shards(
                connection, root, temporary_run, final_run, cancel_requested
            )
            for image in images:
                image["feature_count"] = keypoint_counts[
                    int(image["colmap_image_id"])
                ]
            pair_index, pair_counts = _write_pair_shards(
                connection,
                root,
                temporary_run,
                final_run,
                set(keypoint_counts),
                keypoint_counts,
                cancel_requested,
            )
        finally:
            connection.close()

        _write_gzip_json(
            temporary_run / "features" / "index.json.gz",
            {"schema_version": SCHEMA_VERSION, "images": feature_index},
        )
        _write_gzip_json(
            temporary_run / "pairs" / "index.json.gz",
            {"schema_version": SCHEMA_VERSION, "pairs": pair_index},
        )
        counts = {
            "images": len(images),
            "registered_images": sum(bool(image["registered"]) for image in images),
            "keypoints": sum(keypoint_counts.values()),
            **pair_counts,
        }
        run = {
            "run_id": run_id,
            "detector": {
                "name": "sift",
                "implementation": "colmap",
                "version": colmap_build,
                "keypoint_fields": ["x", "y"],
                "coordinate_precision_pixels": 10**-KEYPOINT_DECIMALS,
            },
            "matcher": {
                "name": matcher,
                "implementation": "colmap",
                "version": colmap_build,
            },
            "geometric_verification": {
                "implementation": "colmap",
                "version": colmap_build,
            },
            "image_set_hash": image_set_hash,
            "feature_index_path": _asset(root, final_run / "features" / "index.json.gz"),
            "pair_index_path": _asset(root, final_run / "pairs" / "index.json.gz"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "profile": PROFILE_ID,
            "coordinate_frame": "normalized",
            "camera_convention": "opencv",
            "camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "world_units": "arbitrary",
            "dataset_hash": str(contract["dataset_hash"]),
            "default_run_id": run_id,
            "runs": [run],
            "counts": counts,
            "images": images,
        }
        _write_json(temporary / "manifest.json", manifest)
        final_output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(final_output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    total_bytes = sum(path.stat().st_size for path in final_output.rglob("*") if path.is_file())
    return final_output / "manifest.json", {
        "sfm_diagnostics_status": "available",
        "sfm_diagnostics_image_count": counts["images"],
        "sfm_diagnostics_registered_image_count": counts["registered_images"],
        "sfm_diagnostics_keypoint_count": counts["keypoints"],
        "sfm_diagnostics_pair_count": counts["pairs"],
        "sfm_diagnostics_match_count": counts["candidate_matches"],
        "sfm_diagnostics_inlier_count": counts["inliers"],
        "sfm_diagnostics_bytes": total_bytes,
    }


def _image_records(
    connection: sqlite3.Connection,
    root: Path,
    image_root: Path,
    registered: dict[int, dict[str, Any]],
    splits: dict[int, str],
    normalized_from_world: np.ndarray,
    selection: dict[str, dict[str, Any]],
    cancel_requested: Callable[[], bool] | None,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT images.image_id, images.name, cameras.width, cameras.height
        FROM images JOIN cameras ON cameras.camera_id = images.camera_id
        ORDER BY images.image_id
        """
    ).fetchall()
    if not rows:
        raise ColmapDiagnosticsError("COLMAP database contains no images")
    result: list[dict[str, Any]] = []
    for row_number, (image_id, name, width, height) in enumerate(rows, start=1):
        if row_number % FEATURE_IMAGES_PER_SHARD == 1:
            _check_cancelled(cancel_requested)
        image_id, name = int(image_id), str(name)
        source = _contained(image_root / name, root, file=True, label="feature source image")
        relative = source.relative_to(root).as_posix()
        selected = selection.get(name)
        digest = (
            str(selected["sha256"])
            if selected is not None and _is_sha256(selected.get("sha256"))
            else _sha256_file(source)
        )
        record: dict[str, Any] = {
            "frame_uid": hashlib.sha256(f"{relative}\0{digest}".encode()).hexdigest(),
            "colmap_image_id": image_id,
            "name": name,
            "path": relative,
            "sha256": digest,
            "width": int(width),
            "height": int(height),
            "registered": image_id in registered,
            "split": splits.get(image_id),
        }
        if selected is not None:
            record["source_time_seconds"] = float(selected["time_seconds"])
            if selected.get("selection_reason") is not None:
                record["selection_reason"] = str(selected["selection_reason"])
        if image_id in registered:
            registered_image = registered[image_id]
            if Path(str(registered_image.get("path", ""))).name != name:
                raise ColmapDiagnosticsError(
                    f"final registered image {image_id} does not match the COLMAP database"
                )
            record.update(_normalized_camera(registered_image, normalized_from_world))
        result.append(record)
    return result


def _write_feature_shards(
    connection: sqlite3.Connection,
    root: Path,
    temporary_run: Path,
    final_run: Path,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    rows = connection.execute(
        """
        SELECT images.image_id, keypoints.rows, keypoints.cols, keypoints.data
        FROM images LEFT JOIN keypoints ON keypoints.image_id = images.image_id
        ORDER BY images.image_id
        """
    )
    output = temporary_run / "features"
    output.mkdir(parents=True)
    index: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    shard: dict[str, Any] = {}
    shard_number = 0

    def flush() -> None:
        nonlocal shard, shard_number
        if shard:
            _write_gzip_json(
                output / f"shard-{shard_number:05d}.json.gz",
                {"schema_version": SCHEMA_VERSION, "images": shard},
            )
            shard, shard_number = {}, shard_number + 1

    for row_number, (image_id, count, columns, blob) in enumerate(rows, start=1):
        if row_number % FEATURE_IMAGES_PER_SHARD == 1:
            _check_cancelled(cancel_requested)
        image_id, count, columns = int(image_id), int(count or 0), int(columns or 0)
        if count:
            if columns < 2 or blob is None:
                raise ColmapDiagnosticsError(f"invalid keypoints for image {image_id}")
            values = np.frombuffer(blob, dtype="<f4")
            if values.size != count * columns:
                raise ColmapDiagnosticsError(f"keypoint shape mismatch for image {image_id}")
            points = np.round(
                values.reshape(count, columns)[:, :2].astype(np.float64),
                KEYPOINT_DECIMALS,
            ).tolist()
        else:
            points = []
        counts[image_id] = count
        shard_name = f"shard-{shard_number:05d}.json.gz"
        shard[str(image_id)] = {"points": points}
        index.append(
            {
                "image_id": image_id,
                "feature_count": count,
                "detail_shard": _asset(root, final_run / "features" / shard_name),
            }
        )
        if row_number % FEATURE_IMAGES_PER_SHARD == 0:
            flush()
    flush()
    return index, counts


def _write_pair_shards(
    connection: sqlite3.Connection,
    root: Path,
    temporary_run: Path,
    final_run: Path,
    image_ids: set[int],
    keypoint_counts: dict[int, int],
    cancel_requested: Callable[[], bool] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = connection.execute(
        """
        WITH pair_ids AS (
          SELECT pair_id FROM matches UNION SELECT pair_id FROM two_view_geometries
        )
        SELECT pair_ids.pair_id,
               matches.rows, matches.cols, matches.data,
               two_view_geometries.rows, two_view_geometries.cols,
               two_view_geometries.data, two_view_geometries.config
        FROM pair_ids
        LEFT JOIN matches ON matches.pair_id = pair_ids.pair_id
        LEFT JOIN two_view_geometries ON two_view_geometries.pair_id = pair_ids.pair_id
        ORDER BY pair_ids.pair_id
        """
    )
    output = temporary_run / "pairs" / "shards"
    output.mkdir(parents=True)
    index: list[dict[str, Any]] = []
    shard: dict[str, Any] = {}
    shard_number = candidate_total = inlier_total = 0

    def flush() -> None:
        nonlocal shard, shard_number
        if shard:
            _write_gzip_json(
                output / f"shard-{shard_number:05d}.json.gz",
                {"schema_version": SCHEMA_VERSION, "pairs": shard},
            )
            shard, shard_number = {}, shard_number + 1

    for row_number, row in enumerate(rows, start=1):
        if row_number % PAIRS_PER_SHARD == 1:
            _check_cancelled(cancel_requested)
        pair_id = int(row[0])
        left_id, right_id = _pair_image_ids(pair_id)
        if left_id not in image_ids or right_id not in image_ids:
            raise ColmapDiagnosticsError(f"pair {pair_id} references a missing image")
        candidate = _decode_matches(row[1], row[2], row[3], pair_id, "candidate")
        verified = _decode_matches(row[4], row[5], row[6], pair_id, "verified")
        verified_set = {tuple(match) for match in verified}
        if not verified_set <= {tuple(match) for match in candidate}:
            raise ColmapDiagnosticsError(f"verified matches are not a subset for pair {pair_id}")
        _validate_match_indices(candidate, left_id, right_id, keypoint_counts, pair_id)
        inliers = [match for match in candidate if tuple(match) in verified_set]
        outliers = [match for match in candidate if tuple(match) not in verified_set]
        pair_key = f"{left_id}-{right_id}"
        shard_name = f"shard-{shard_number:05d}.json.gz"
        shard[pair_key] = {"inliers": inliers, "outliers": outliers}
        index.append(
            {
                "pair_key": pair_key,
                "image_ids": [left_id, right_id],
                "candidate_match_count": len(candidate),
                "inlier_count": len(inliers),
                "geometric_config": int(row[7] or 0),
                "detail_shard": _asset(root, final_run / "pairs" / "shards" / shard_name),
            }
        )
        candidate_total += len(candidate)
        inlier_total += len(inliers)
        if row_number % PAIRS_PER_SHARD == 0:
            flush()
    flush()
    return index, {
        "pairs": len(index),
        "candidate_matches": candidate_total,
        "inliers": inlier_total,
        "outliers": candidate_total - inlier_total,
    }


def _registered_images(
    contract: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, str], np.ndarray]:
    if contract.get("coordinate_system", {}).get("camera_convention") != "opencv":
        raise ColmapDiagnosticsError("dataset must use the OpenCV camera convention")
    images, split_payload = contract.get("images"), contract.get("splits")
    matrix = np.asarray(
        contract.get("normalization", {}).get("normalized_from_world"), dtype=np.float64
    )
    if not isinstance(images, list) or not isinstance(split_payload, dict):
        raise ColmapDiagnosticsError("dataset images or splits are invalid")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ColmapDiagnosticsError("dataset normalization is invalid")
    registered = {int(image["image_id"]): image for image in images}
    splits: dict[int, str] = {}
    for split in ("train", "validation", "test"):
        values = split_payload.get(split, [])
        if not isinstance(values, list):
            raise ColmapDiagnosticsError(f"dataset {split} split is invalid")
        for value in values:
            image_id = int(value)
            if image_id in splits:
                raise ColmapDiagnosticsError("dataset splits overlap")
            splits[image_id] = split
    if set(registered) != set(splits):
        raise ColmapDiagnosticsError("dataset splits do not cover registered images")
    return registered, splits, matrix


def _normalized_camera(
    image: dict[str, Any], normalized_from_world: np.ndarray
) -> dict[str, Any]:
    world_from_camera = np.asarray(image.get("world_from_camera"), dtype=np.float64)
    intrinsic = np.asarray(image.get("intrinsic"), dtype=np.float64)
    if world_from_camera.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ColmapDiagnosticsError("registered camera matrices are invalid")
    center = normalized_from_world @ world_from_camera[:, 3]
    rotation = normalized_from_world[:3, :3] @ world_from_camera[:3, :3]
    forward = _unit(rotation @ [0.0, 0.0, 1.0])
    up = _unit(rotation @ [0.0, -1.0, 0.0])
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    width, height = int(image.get("width", 0)), int(image.get("height", 0))
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
        raise ColmapDiagnosticsError("registered camera intrinsics are invalid")
    return {
        "center_normalized": center[:3].tolist(),
        "forward_normalized": forward.tolist(),
        "up_normalized": up.tolist(),
        "horizontal_fov_degrees": math.degrees(2 * math.atan(width / (2 * fx))),
        "vertical_fov_degrees": math.degrees(2 * math.atan(height / (2 * fy))),
    }


def _video_selection(path: Path | None, root: Path) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _read_json(_contained(path, root, file=True, label="video selection"), "video selection")
    if not isinstance(payload.get("selected"), list):
        raise ColmapDiagnosticsError("video selection has no selected frame list")
    result: dict[str, dict[str, Any]] = {}
    for item in payload["selected"]:
        if not isinstance(item, dict):
            raise ColmapDiagnosticsError("video selection frame is invalid")
        name = Path(str(item.get("path", ""))).name
        if not name or name in result:
            raise ColmapDiagnosticsError("video selection frame names are invalid")
        result[name] = item
    return result


def _decode_matches(rows: Any, columns: Any, blob: Any, pair_id: int, label: str) -> list[list[int]]:
    count, cols = int(rows or 0), int(columns or 0)
    if count == 0:
        return []
    if cols != 2 or blob is None:
        raise ColmapDiagnosticsError(f"invalid {label} matches for pair {pair_id}")
    values = np.frombuffer(blob, dtype="<u4")
    if values.size != count * cols:
        raise ColmapDiagnosticsError(f"{label} match shape mismatch for pair {pair_id}")
    return values.reshape(count, cols).astype(np.int64).tolist()


def _validate_match_indices(
    matches: list[list[int]],
    left_id: int,
    right_id: int,
    counts: dict[int, int],
    pair_id: int,
) -> None:
    if any(
        left < 0 or right < 0 or left >= counts[left_id] or right >= counts[right_id]
        for left, right in matches
    ):
        raise ColmapDiagnosticsError(f"pair {pair_id} has an invalid keypoint index")


def _pair_image_ids(pair_id: int) -> tuple[int, int]:
    right = pair_id % MAX_IMAGE_ID
    left = (pair_id - right) // MAX_IMAGE_ID
    if left <= 0 or right <= 0 or left >= right:
        raise ColmapDiagnosticsError(f"invalid COLMAP pair id: {pair_id}")
    return left, right


def _require_tables(connection: sqlite3.Connection) -> None:
    present = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {"cameras", "images", "keypoints", "matches", "two_view_geometries"} - present
    if missing:
        raise ColmapDiagnosticsError(f"COLMAP database is missing tables: {', '.join(sorted(missing))}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_bytes(value).decode() + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(_json_bytes(value), compresslevel=6, mtime=0))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _contained(path: Path, root: Path, *, file: bool, label: str) -> Path:
    candidate = path.resolve()
    _relative(root, candidate)
    exists = candidate.is_file() if file else candidate.is_dir()
    if not exists:
        raise ColmapDiagnosticsError(f"{label} is missing")
    return candidate


def _relative(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ColmapDiagnosticsError("SfM diagnostic path escapes the job directory") from exc


def _asset(root: Path, path: Path) -> str:
    return _relative(root, path.resolve()).as_posix()


def _unit(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ColmapDiagnosticsError("camera direction is degenerate")
    return value / norm


def _check_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ColmapDiagnosticsError("SfM diagnostics cancelled")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _run_id(image_set_hash: str, matcher: str, version: str) -> str:
    digest = _hash_json(
        {
            "detector": "sift",
            "implementation": "colmap",
            "version": version,
            "matcher": matcher,
            "image_set_hash": image_set_hash,
        }
    )
    return f"colmap-sift-{digest[:12]}"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColmapDiagnosticsError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ColmapDiagnosticsError(f"{label} must contain an object")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
