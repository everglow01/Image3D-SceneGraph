from __future__ import annotations

import json
import math
import sqlite3
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from image3d_scenegraph.geometry.colmap import ResolvedColmapCameraCalibration


CAMERA_DIAGNOSTICS_SCHEMA_VERSION = 1
CAMERA_DIAGNOSTICS_PROFILE = "sfm_camera_calibration_diagnostics_v1"
_MIN_FOCAL_RATIO = 0.1
_MAX_FOCAL_RATIO = 10.0
_MAX_EXTRA_PARAMETER = 1.0
_CAMERA_MODELS = {
    2: ("SIMPLE_RADIAL", 4),
    4: ("OPENCV", 8),
}


class CameraCalibrationError(ValueError):
    """Raised when camera grouping or calibration provenance is inconsistent."""


@dataclass(frozen=True)
class CameraExtractionBatch:
    image_names: tuple[str, ...]
    image_list_path: Path | None
    image_reader_options: tuple[str, ...]


@dataclass(frozen=True)
class CameraExtractionPlan:
    calibration: ResolvedColmapCameraCalibration
    groups: tuple[dict[str, Any], ...]
    batches: tuple[CameraExtractionBatch, ...]

    def record(self) -> dict[str, Any]:
        return {
            "profile": self.calibration.profile_id,
            "grouping_key_policy": self.calibration.grouping_key_policy,
            "planned_camera_count": len(self.groups),
            "groups": list(self.groups),
        }


def prepare_camera_extraction(
    calibration: ResolvedColmapCameraCalibration,
    image_root: Path,
    image_paths: list[Path],
    output_dir: Path,
) -> CameraExtractionPlan:
    root = image_root.resolve()
    names = _relative_image_names(root, image_paths)
    if calibration.sharing_policy == "single_camera":
        group = {
            "group_id": "camera-group-0000",
            "images": names,
            "metadata_status": "shared_profile",
            "evidence": None,
            "missing_fields": [],
        }
        return CameraExtractionPlan(
            calibration=calibration,
            groups=(group,),
            batches=(
                CameraExtractionBatch(
                    image_names=tuple(names),
                    image_list_path=None,
                    image_reader_options=calibration.image_reader_options,
                ),
            ),
        )
    if calibration.sharing_policy != "focal_aware_groups":
        raise CameraCalibrationError(
            f"unsupported camera sharing policy: {calibration.sharing_policy}"
        )

    metadata = [_camera_metadata(root / name, name) for name in names]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in metadata:
        if record["metadata_status"] == "reliable":
            key = (
                "shared",
                record["make"],
                record["model"],
                record["lens_model"],
                record["focal_length_mm"],
                record["focal_length_35mm"],
                record["width"],
                record["height"],
                record["orientation"],
            )
        else:
            key = ("singleton", record["image"])
        buckets.setdefault(key, []).append(record)

    grouped_records = sorted(
        buckets.values(),
        key=lambda records: tuple(record["image"] for record in records),
    )
    groups: list[dict[str, Any]] = []
    for index, records in enumerate(grouped_records):
        first = records[0]
        groups.append(
            {
                "group_id": f"camera-group-{index:04d}",
                "images": [record["image"] for record in records],
                "metadata_status": first["metadata_status"],
                "evidence": (
                    {
                        "make": first["make"],
                        "model": first["model"],
                        "lens_model": first["lens_model"],
                        "focal_length_mm": first["focal_length_mm"],
                        "focal_length_35mm": first["focal_length_35mm"],
                        "width": first["width"],
                        "height": first["height"],
                        "orientation": first["orientation"],
                    }
                    if first["metadata_status"] == "reliable"
                    else None
                ),
                "missing_fields": first["missing_fields"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    batches: list[CameraExtractionBatch] = []
    repeated = [group for group in groups if len(group["images"]) > 1]
    singletons = [group for group in groups if len(group["images"]) == 1]
    for batch_index, group in enumerate(repeated):
        path = output_dir / f"shared-{batch_index:04d}.txt"
        _write_image_list(path, group["images"])
        batches.append(
            CameraExtractionBatch(
                image_names=tuple(group["images"]),
                image_list_path=path,
                image_reader_options=(
                    *calibration.image_reader_options,
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.single_camera_per_image",
                    "0",
                ),
            )
        )
    if singletons:
        path = output_dir / "singletons.txt"
        singleton_names = [group["images"][0] for group in singletons]
        _write_image_list(path, singleton_names)
        batches.append(
            CameraExtractionBatch(
                image_names=tuple(singleton_names),
                image_list_path=path,
                image_reader_options=(
                    *calibration.image_reader_options,
                    "--ImageReader.single_camera",
                    "0",
                    "--ImageReader.single_camera_per_image",
                    "1",
                ),
            )
        )
    return CameraExtractionPlan(
        calibration=calibration,
        groups=tuple(groups),
        batches=tuple(batches),
    )


def build_camera_calibration_diagnostics(
    *,
    database_path: Path,
    final_camera_payload: dict[str, Any],
    points3d_path: Path,
    plan: CameraExtractionPlan,
    colmap_build: str,
) -> dict[str, Any]:
    database_cameras, database_images = _read_database(database_path)
    expected_names = {
        name for group in plan.groups for name in _string_list(group["images"])
    }
    if set(database_images) != expected_names:
        raise CameraCalibrationError(
            "COLMAP database images do not match the camera grouping plan"
        )
    group_camera_ids = _validate_group_partition(plan.groups, database_images)
    if len(database_cameras) != len(plan.groups):
        raise CameraCalibrationError(
            "COLMAP database camera count does not match the grouping plan"
        )

    expected_model = plan.calibration.camera_model
    if any(camera["model"] != expected_model for camera in database_cameras.values()):
        raise CameraCalibrationError(
            "COLMAP database camera model does not match the requested profile"
        )
    if (
        plan.calibration.sharing_policy == "single_camera"
        and len(database_cameras) != 1
    ):
        raise CameraCalibrationError("shared camera profile created multiple cameras")

    final_cameras_input = final_camera_payload.get("cameras")
    final_images_input = final_camera_payload.get("images")
    if not isinstance(final_cameras_input, list) or not isinstance(
        final_images_input, list
    ):
        raise CameraCalibrationError("final COLMAP camera payload is invalid")
    final_images: dict[str, int] = {}
    for image in final_images_input:
        if not isinstance(image, dict):
            raise CameraCalibrationError("final COLMAP image record is invalid")
        name = str(image.get("name", ""))
        camera_id = int(image.get("camera_id", 0))
        if not name or name in final_images or camera_id < 1:
            raise CameraCalibrationError("final COLMAP image assignment is invalid")
        if database_images.get(name) != camera_id:
            raise CameraCalibrationError(
                "final COLMAP image changed its database camera assignment"
            )
        final_images[name] = camera_id
    if not final_images:
        raise CameraCalibrationError("final COLMAP model has no registered images")

    registered_counts: dict[int, int] = {}
    for camera_id in final_images.values():
        registered_counts[camera_id] = registered_counts.get(camera_id, 0) + 1
    initial_counts: dict[int, int] = {}
    for camera_id in database_images.values():
        initial_counts[camera_id] = initial_counts.get(camera_id, 0) + 1

    final_cameras: dict[int, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    focal_ratios: list[float] = []
    for value in final_cameras_input:
        if not isinstance(value, dict):
            raise CameraCalibrationError("final COLMAP camera record is invalid")
        camera_id = int(value.get("camera_id", 0))
        if camera_id not in registered_counts or camera_id in final_cameras:
            raise CameraCalibrationError("final COLMAP camera IDs are inconsistent")
        initial_camera = database_cameras[camera_id]
        width = int(value.get("width", 0))
        height = int(value.get("height", 0))
        if (
            width != initial_camera["width"]
            or height != initial_camera["height"]
        ):
            raise CameraCalibrationError(
                "final COLMAP camera dimensions changed from the database"
            )
        camera = _named_camera_record(
            camera_id=camera_id,
            model=str(value.get("model", "")),
            width=width,
            height=height,
            params=value.get("params"),
            initial_image_count=initial_counts.get(camera_id, 0),
            registered_image_count=registered_counts[camera_id],
            prior_focal_length=bool(initial_camera["prior_focal_length"]),
            initial_params=initial_camera["params"],
        )
        if camera["model"] != expected_model:
            raise CameraCalibrationError(
                "final COLMAP camera model does not match the requested profile"
            )
        focal_ratios.extend(camera["focal_length_ratios"])
        warnings.extend(_camera_warnings(camera))
        final_cameras[camera_id] = camera
    if set(final_cameras) != set(registered_counts):
        raise CameraCalibrationError("final COLMAP cameras are incomplete")
    if (
        plan.calibration.sharing_policy == "single_camera"
        and len(final_cameras) != 1
    ):
        raise CameraCalibrationError(
            "shared camera profile produced multiple final cameras"
        )

    sparse = _sparse_summary(points3d_path)
    camera_group_ids = {
        camera_id: group_id for group_id, camera_id in group_camera_ids.items()
    }
    initial_groups = [
        {
            "group_id": camera_group_ids[camera_id],
            "camera_id": camera_id,
            "image_count": initial_counts.get(camera_id, 0),
            "model": camera["model"],
            "width": camera["width"],
            "height": camera["height"],
            "prior_focal_length": camera["prior_focal_length"],
        }
        for camera_id, camera in sorted(database_cameras.items())
    ]
    return {
        "schema_version": CAMERA_DIAGNOSTICS_SCHEMA_VERSION,
        "profile": CAMERA_DIAGNOSTICS_PROFILE,
        "colmap_build": colmap_build,
        "calibration": plan.calibration.provenance(),
        "grouping": plan.record(),
        "initial": {
            "image_count": len(database_images),
            "camera_count": len(database_cameras),
            "prior_focal_camera_count": sum(
                bool(camera["prior_focal_length"])
                for camera in database_cameras.values()
            ),
            "groups": initial_groups,
        },
        "final": {
            "registered_image_count": len(final_images),
            "camera_count": len(final_cameras),
            "unregistered_camera_count": len(database_cameras)
            - len(final_cameras),
            "median_focal_length_ratio": statistics.median(focal_ratios),
            "cameras": [final_cameras[key] for key in sorted(final_cameras)],
        },
        "sparse": sparse,
        "plausibility": {
            "policy": "colmap_mapper_defaults_v1",
            "focal_length_ratio_min": _MIN_FOCAL_RATIO,
            "focal_length_ratio_max": _MAX_FOCAL_RATIO,
            "max_extra_parameter": _MAX_EXTRA_PARAMETER,
            "warning_count": len(warnings),
            "warnings": warnings,
            "warnings_are_job_gates": False,
        },
    }


def camera_calibration_metrics(
    diagnostics: dict[str, Any],
) -> dict[str, int | float | str]:
    calibration = diagnostics["calibration"]
    initial = diagnostics["initial"]
    final = diagnostics["final"]
    sparse = diagnostics["sparse"]
    plausibility = diagnostics["plausibility"]
    metrics: dict[str, int | float | str] = {
        "sfm_camera_calibration_profile": str(calibration["profile"]),
        "sfm_camera_model": str(calibration["camera_model"]),
        "sfm_camera_planned_count": int(
            diagnostics["grouping"]["planned_camera_count"]
        ),
        "sfm_camera_initial_count": int(initial["camera_count"]),
        "sfm_camera_final_count": int(final["camera_count"]),
        "sfm_camera_prior_focal_count": int(initial["prior_focal_camera_count"]),
        "sfm_camera_warning_count": int(plausibility["warning_count"]),
        "sfm_camera_median_focal_length_ratio": float(
            final["median_focal_length_ratio"]
        ),
    }
    for source, target in (
        ("median_reprojection_error_pixels", "sfm_median_reprojection_error_pixels"),
        ("median_track_length", "sfm_median_track_length"),
    ):
        if sparse[source] is not None:
            metrics[target] = float(sparse[source])
    return metrics


def write_camera_calibration_diagnostics(
    path: Path,
    diagnostics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_image_names(root: Path, image_paths: list[Path]) -> list[str]:
    if not image_paths:
        raise CameraCalibrationError("camera calibration requires at least one image")
    names: list[str] = []
    for image_path in image_paths:
        try:
            name = image_path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise CameraCalibrationError(
                f"camera calibration image escapes its root: {image_path}"
            ) from exc
        if (
            not name
            or "\n" in name
            or "\r" in name
            or name in names
        ):
            raise CameraCalibrationError(
                "camera calibration image names must be unique safe lines"
            )
        names.append(name)
    return sorted(names)


def _camera_metadata(path: Path, name: str) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            try:
                exif = image.getexif()
            except (OSError, SyntaxError, TypeError, ValueError):
                exif = {}
                exif_invalid = True
            else:
                exif_invalid = False
    except (OSError, SyntaxError, ValueError) as exc:
        raise CameraCalibrationError(f"cannot inspect camera metadata for {name}") from exc

    make = _normalized_text(exif.get(271))
    model = _normalized_text(exif.get(272))
    lens_model = _normalized_text(exif.get(42036))
    focal_length = _positive_number(exif.get(37386))
    focal_length_35mm = _positive_number(exif.get(41989))
    orientation = exif.get(274, 1)
    try:
        orientation = 0 if isinstance(orientation, bool) else int(orientation)
    except (TypeError, ValueError):
        orientation = 0
    missing = []
    if make is None:
        missing.append("make")
    if model is None:
        missing.append("model")
    if focal_length is None and focal_length_35mm is None:
        missing.append("focal_length")
    if orientation not in range(1, 9):
        missing.append("orientation")
    status = "reliable" if not missing and not exif_invalid else "insufficient"
    return {
        "image": name,
        "width": int(width),
        "height": int(height),
        "orientation": orientation if orientation in range(1, 9) else None,
        "make": make,
        "model": model,
        "lens_model": lens_model,
        "focal_length_mm": focal_length,
        "focal_length_35mm": focal_length_35mm,
        "metadata_status": status,
        "missing_fields": ["invalid_exif"] if exif_invalid else missing,
    }


def _normalized_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).casefold()
    return normalized or None


def _positive_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return round(number, 6)


def _write_image_list(path: Path, names: list[str]) -> None:
    if not names:
        raise CameraCalibrationError("camera extraction batch is empty")
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _read_database(
    database_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    try:
        connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True)
        try:
            camera_rows = connection.execute(
                "SELECT camera_id, model, width, height, params, prior_focal_length "
                "FROM cameras ORDER BY camera_id"
            ).fetchall()
            image_rows = connection.execute(
                "SELECT name, camera_id FROM images ORDER BY image_id"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise CameraCalibrationError("cannot read COLMAP camera database") from exc
    if not camera_rows or not image_rows:
        raise CameraCalibrationError("COLMAP camera database is empty")

    cameras: dict[int, dict[str, Any]] = {}
    for camera_id, model_id, width, height, blob, prior in camera_rows:
        camera_id, model_id = int(camera_id), int(model_id)
        if model_id not in _CAMERA_MODELS or camera_id in cameras:
            raise CameraCalibrationError("COLMAP database camera model is unsupported")
        model, parameter_count = _CAMERA_MODELS[model_id]
        if blob is None or len(blob) != parameter_count * 8:
            raise CameraCalibrationError("COLMAP database camera parameters are invalid")
        params = list(struct.unpack(f"<{parameter_count}d", blob))
        focal_count = 1 if model == "SIMPLE_RADIAL" else 2
        if (
            int(width) < 1
            or int(height) < 1
            or not all(math.isfinite(value) for value in params)
            or min(params[:focal_count]) <= 0
        ):
            raise CameraCalibrationError(
                "COLMAP database camera parameters are invalid"
            )
        cameras[camera_id] = {
            "model": model,
            "width": int(width),
            "height": int(height),
            "params": params,
            "prior_focal_length": bool(prior),
        }
    images: dict[str, int] = {}
    for name, camera_id in image_rows:
        name, camera_id = str(name), int(camera_id)
        if not name or name in images or camera_id not in cameras:
            raise CameraCalibrationError("COLMAP database image assignment is invalid")
        images[name] = camera_id
    return cameras, images


def _validate_group_partition(
    groups: tuple[dict[str, Any], ...], database_images: dict[str, int]
) -> dict[str, int]:
    camera_to_group: dict[int, str] = {}
    group_to_camera: dict[str, int] = {}
    for group in groups:
        group_id = str(group["group_id"])
        names = _string_list(group["images"])
        camera_ids = {database_images[name] for name in names}
        if len(camera_ids) != 1:
            raise CameraCalibrationError(
                f"camera group {group_id} was split in the COLMAP database"
            )
        camera_id = next(iter(camera_ids))
        if camera_id in camera_to_group:
            raise CameraCalibrationError(
                f"camera groups {camera_to_group[camera_id]} and {group_id} were merged"
            )
        camera_to_group[camera_id] = group_id
        group_to_camera[group_id] = camera_id
    if len(camera_to_group) != len(groups):
        raise CameraCalibrationError("COLMAP camera group count is inconsistent")
    return group_to_camera


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CameraCalibrationError("camera group image list is invalid")
    return value


def _named_camera_record(
    *,
    camera_id: int,
    model: str,
    width: int,
    height: int,
    params: Any,
    initial_image_count: int,
    registered_image_count: int,
    prior_focal_length: bool,
    initial_params: Any,
) -> dict[str, Any]:
    expected = {"SIMPLE_RADIAL": 4, "OPENCV": 8}
    if model not in expected or width < 1 or height < 1:
        raise CameraCalibrationError("final COLMAP camera model is unsupported")
    if not isinstance(params, list) or len(params) != expected[model]:
        raise CameraCalibrationError("final COLMAP camera parameters are invalid")
    values = [float(value) for value in params]
    if not all(math.isfinite(value) for value in values):
        raise CameraCalibrationError("final COLMAP camera parameters are non-finite")
    if model == "SIMPLE_RADIAL":
        focal = {"f": values[0]}
        principal = {"cx": values[1], "cy": values[2]}
        distortion = {"k": values[3]}
        focal_values = values[:1]
    else:
        focal = {"fx": values[0], "fy": values[1]}
        principal = {"cx": values[2], "cy": values[3]}
        distortion = {
            "k1": values[4],
            "k2": values[5],
            "p1": values[6],
            "p2": values[7],
        }
        focal_values = values[:2]
    if min(focal_values) <= 0:
        raise CameraCalibrationError("final COLMAP focal length is not positive")
    scale = max(width, height)
    ratios = [value / scale for value in focal_values]
    initial_focal = None
    relative_change = None
    if isinstance(initial_params, list) and initial_params:
        initial_values = [float(value) for value in initial_params]
        initial_focal_values = initial_values[: 1 if model == "SIMPLE_RADIAL" else 2]
        if min(initial_focal_values) > 0:
            initial_focal = statistics.fmean(initial_focal_values)
            relative_change = (
                statistics.fmean(focal_values) - initial_focal
            ) / initial_focal
            if not math.isfinite(relative_change):
                raise CameraCalibrationError(
                    "COLMAP focal length change is non-finite"
                )
    return {
        "camera_id": camera_id,
        "model": model,
        "width": width,
        "height": height,
        "initial_image_count": initial_image_count,
        "registered_image_count": registered_image_count,
        "registration_rate": registered_image_count / initial_image_count,
        "prior_focal_length": prior_focal_length,
        "focal_lengths_pixels": focal,
        "principal_point_pixels": principal,
        "distortion": distortion,
        "focal_length_ratios": ratios,
        "initial_mean_focal_length_pixels": initial_focal,
        "relative_focal_length_change": relative_change,
    }


def _camera_warnings(camera: dict[str, Any]) -> list[dict[str, Any]]:
    camera_id = int(camera["camera_id"])
    warnings = []
    if any(
        ratio < _MIN_FOCAL_RATIO or ratio > _MAX_FOCAL_RATIO
        for ratio in camera["focal_length_ratios"]
    ):
        warnings.append({"code": "focal_length_ratio_out_of_range", "camera_id": camera_id})
    if max(abs(float(value)) for value in camera["distortion"].values()) > _MAX_EXTRA_PARAMETER:
        warnings.append({"code": "extra_parameter_out_of_range", "camera_id": camera_id})
    principal = camera["principal_point_pixels"]
    if not (
        0 <= float(principal["cx"]) <= int(camera["width"])
        and 0 <= float(principal["cy"]) <= int(camera["height"])
    ):
        warnings.append({"code": "principal_point_outside_image", "camera_id": camera_id})
    return warnings


def _sparse_summary(path: Path) -> dict[str, int | float | None]:
    errors: list[float] = []
    track_lengths: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or (len(parts) - 8) % 2:
            raise CameraCalibrationError("COLMAP points3D text row is invalid")
        error = float(parts[7])
        if not math.isfinite(error) or error < 0:
            raise CameraCalibrationError("COLMAP reprojection error is invalid")
        errors.append(error)
        track_lengths.append((len(parts) - 8) // 2)
    return {
        "point_count": len(errors),
        "observation_count": sum(track_lengths),
        "mean_reprojection_error_pixels": statistics.fmean(errors) if errors else None,
        "median_reprojection_error_pixels": statistics.median(errors) if errors else None,
        "mean_track_length": statistics.fmean(track_lengths) if track_lengths else None,
        "median_track_length": statistics.median(track_lengths) if track_lengths else None,
    }
