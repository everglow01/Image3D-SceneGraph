#!/usr/bin/env python3
"""Derive an immutable Gaussian replay after explicitly excluding bad poses."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.dataset import (
    camera_normalization,
    contract_hash,
    validate_contract,
    with_initialization,
    write_contract,
)
from image3d_scenegraph.gaussian.initialization import (
    sparse_initialization,
    write_initialization,
)
from image3d_scenegraph.gaussian.replay import build_replay_bundle


PROFILE_ID = "gaussian_pose_repair_v1"
MIN_TRACK_LENGTH = 3


def derive_gaussian_pose_repair(
    *,
    dataset_contract: Path,
    dataset_root: Path,
    points_path: Path,
    excluded_image_ids: set[int],
    output_dir: Path,
    max_initial_points: int = 1_000_000,
) -> dict[str, Any]:
    if not excluded_image_ids:
        raise ValueError("at least one explicit image ID must be excluded")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"refusing to overwrite existing output: {output_dir}")
    if max_initial_points < 1:
        raise ValueError("max initial points must be positive")

    source_root = dataset_root.resolve()
    contract = json.loads(dataset_contract.read_text(encoding="utf-8"))
    validate_contract(contract, source_root)
    excluded = {str(value) for value in excluded_image_ids}
    known_ids = {str(image["image_id"]) for image in contract["images"]}
    missing = sorted(excluded - known_ids)
    if missing:
        raise ValueError(f"excluded image IDs are not in the dataset: {missing}")

    camera_relative = _safe_relative(contract["source"]["camera_path"])
    camera_source = source_root / camera_relative
    expected_camera_hash = str(contract["source"].get("camera_sha256", ""))
    if expected_camera_hash and sha256_file(camera_source) != expected_camera_hash:
        raise ValueError("source camera hash does not match the dataset contract")
    camera_payload = json.loads(camera_source.read_text(encoding="utf-8"))
    source_images = camera_payload.get("images")
    source_cameras = camera_payload.get("cameras")
    if not isinstance(source_images, list) or not isinstance(source_cameras, list):
        raise ValueError("camera source must contain images and cameras arrays")
    source_ids = {str(image.get("image_id")) for image in source_images}
    if not excluded <= source_ids:
        raise ValueError("excluded image IDs are missing from the camera source")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        preparation = temporary / "preparation"
        filtered_camera = _filtered_camera_payload(camera_payload, excluded)
        filtered_camera_path = preparation / camera_relative
        filtered_camera_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_camera_path.write_text(
            json.dumps(filtered_camera, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        derived = copy.deepcopy(contract)
        repair_key = hashlib.sha256(
            (
                str(contract["dataset_hash"])
                + ":"
                + ",".join(sorted(excluded))
            ).encode()
        ).hexdigest()[:12]
        derived["dataset_id"] = (
            f"{contract['dataset_id']}-pose-repair-{repair_key}"
        )
        derived["images"] = [
            image
            for image in derived["images"]
            if str(image["image_id"]) not in excluded
        ]
        for split in ("train", "validation", "test"):
            derived["splits"][split] = [
                image_id
                for image_id in derived["splits"][split]
                if str(image_id) not in excluded
            ]
        if not derived["splits"]["train"]:
            raise ValueError("pose repair leaves no Train image")
        derived["normalization"] = camera_normalization(derived["images"])
        derived["source"]["camera_sha256"] = sha256_file(filtered_camera_path)
        derived["initialization"] = {
            "coordinate_frame": "world",
            "asset": None,
            "sha256": None,
        }
        derived["dataset_hash"] = contract_hash(derived)

        for image in derived["images"]:
            relative = _safe_relative(image["path"])
            _link_file(source_root / relative, preparation / relative)

        filtered_points_path = preparation / "repair" / "points3D.txt"
        point_counts = filter_sparse_tracks(
            points_path,
            filtered_points_path,
            excluded_image_ids=excluded_image_ids,
            min_track_length=MIN_TRACK_LENGTH,
        )
        initialized = sparse_initialization(
            filtered_points_path,
            derived["normalization"]["normalized_from_world"],
            max_points=max_initial_points,
            min_track_length=MIN_TRACK_LENGTH,
        )
        initialization_path = preparation / "initialization" / "sparse.npz"
        initialization_diagnostics = initialization_path.with_suffix(".json")
        write_initialization(
            initialization_path, initialization_diagnostics, initialized
        )
        derived = with_initialization(
            derived,
            asset="initialization/sparse.npz",
            asset_sha256=sha256_file(initialization_path),
        )
        derived_path = preparation / "dataset.json"
        write_contract(derived_path, derived)
        validate_contract(derived, preparation)

        replay_record = build_replay_bundle(
            contract=derived,
            dataset_path=derived_path,
            dataset_root=preparation,
            initialization_path=initialization_path,
            diagnostics_path=initialization_diagnostics,
            replay_root=temporary / "replay",
        )
        excluded_records = []
        split_by_id = {
            str(image_id): split
            for split, image_ids in contract["splits"].items()
            for image_id in image_ids
        }
        image_by_id = {
            str(image["image_id"]): image for image in contract["images"]
        }
        source_by_id = {
            str(image["image_id"]): image for image in source_images
        }
        for image_id in sorted(excluded, key=int):
            excluded_records.append(
                {
                    "image_id": image_id,
                    "path": str(image_by_id[image_id]["path"]),
                    "split": split_by_id[image_id],
                    "camera_id": int(source_by_id[image_id]["camera_id"]),
                }
            )
        repair_record = {
            "schema_version": 1,
            "profile": PROFILE_ID,
            "status": "repaired_derivative",
            "training_started": False,
            "parent": {
                "dataset_id": str(contract["dataset_id"]),
                "dataset_hash": str(contract["dataset_hash"]),
                "dataset_sha256": sha256_file(dataset_contract),
                "camera_sha256": sha256_file(camera_source),
                "points3d_sha256": sha256_file(points_path),
            },
            "policy": {
                "selection": "explicit_image_ids_only",
                "split_policy": "preserve_remaining_assignments",
                "normalization": "camera_center_max_radius_v1_recomputed",
                "point_coordinates": "preserved_parent_sparse_no_bundle_adjustment",
                "minimum_track_length": MIN_TRACK_LENGTH,
                "test_rgb_loaded": False,
                "original_arm_status": "inconclusive",
            },
            "excluded_images": excluded_records,
            "counts": {
                "images_before": len(contract["images"]),
                "images_after": len(derived["images"]),
                **point_counts,
                "initialization_accepted": len(initialized.points),
            },
            "derived": {
                "dataset_id": str(derived["dataset_id"]),
                "dataset_hash": str(derived["dataset_hash"]),
                "dataset_sha256": sha256_file(derived_path),
                "camera_sha256": sha256_file(filtered_camera_path),
                "points3d_sha256": sha256_file(filtered_points_path),
                "initialization_sha256": str(
                    derived["initialization"]["sha256"]
                ),
                "replay_dataset_hash": str(replay_record["dataset_hash"]),
                "replay_record_sha256": sha256_file(
                    temporary / "replay" / "replay.json"
                ),
            },
        }
        (temporary / "repair.json").write_text(
            json.dumps(repair_record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return repair_record


def filter_sparse_tracks(
    source: Path,
    destination: Path,
    *,
    excluded_image_ids: set[int],
    min_track_length: int,
) -> dict[str, int]:
    if min_track_length < 1:
        raise ValueError("minimum track length must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_points = 0
    output_points = 0
    input_observations = 0
    removed_observations = 0
    removed_points = 0
    output_observations = 0
    with source.open("r", encoding="utf-8") as source_handle, destination.open(
        "w", encoding="utf-8"
    ) as destination_handle:
        destination_handle.write(
            "# Repaired POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        )
        for line in source_handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or (len(parts) - 8) % 2:
                raise ValueError("invalid COLMAP points3D row")
            input_points += 1
            observations = [
                parts[index : index + 2]
                for index in range(8, len(parts), 2)
            ]
            input_observations += len(observations)
            retained = [
                observation
                for observation in observations
                if int(observation[0]) not in excluded_image_ids
            ]
            removed_observations += len(observations) - len(retained)
            if len(retained) < min_track_length:
                removed_points += 1
                continue
            output_points += 1
            output_observations += len(retained)
            destination_handle.write(
                " ".join(
                    parts[:8] + [value for pair in retained for value in pair]
                )
                + "\n"
            )
    if output_points == 0:
        raise ValueError("pose repair leaves no sparse point with sufficient track support")
    return {
        "points_before": input_points,
        "points_removed_track_support": removed_points,
        "points_after_track_repair": output_points,
        "observations_before": input_observations,
        "observations_removed": removed_observations,
        "observations_removed_with_dropped_points": (
            input_observations - removed_observations - output_observations
        ),
        "observations_after": output_observations,
    }


def _filtered_camera_payload(
    payload: dict[str, Any], excluded_image_ids: set[str]
) -> dict[str, Any]:
    filtered = copy.deepcopy(payload)
    filtered["images"] = [
        image
        for image in filtered["images"]
        if str(image.get("image_id")) not in excluded_image_ids
    ]
    used_cameras = {int(image["camera_id"]) for image in filtered["images"]}
    filtered["cameras"] = [
        camera
        for camera in filtered["cameras"]
        if int(camera["camera_id"]) in used_cameras
    ]
    return filtered


def _link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing repair source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_relative(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not str(value):
        raise ValueError("repair paths must be project-relative")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument(
        "--exclude-image-id", required=True, action="append", type=int
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-initial-points", type=int, default=1_000_000)
    args = parser.parse_args()
    record = derive_gaussian_pose_repair(
        dataset_contract=args.dataset_contract,
        dataset_root=args.dataset_root,
        points_path=args.points,
        excluded_image_ids=set(args.exclude_image_id),
        output_dir=args.output_dir,
        max_initial_points=args.max_initial_points,
    )
    print(f"repair_status={record['status']}")
    print(f"repair_record={args.output_dir / 'repair.json'}")
    print(f"replay_root={args.output_dir / 'replay'}")
    print("training_started=false")


if __name__ == "__main__":
    main()
