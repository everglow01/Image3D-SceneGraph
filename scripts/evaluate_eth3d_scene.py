from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from align_pointcloud import write_binary_ply
from analyze_pointcloud import read_ply_points_and_colors
from geometry_utils import (
    decompose_similarity_transform,
    estimate_similarity_transform_ransac,
    transform_points,
)
from run_colmap_sparse import parse_colmap_cameras, parse_colmap_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_LABELS = {
    "Tolerances": "tolerances",
    "Completenesses": "completeness",
    "Accuracies": "accuracy",
    "F1-scores": "f1",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align a reconstruction from camera centers and run the official ETH3D evaluator."
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--reconstruction-dir", required=True, type=Path)
    parser.add_argument("--evaluator-bin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-ransac-threshold", type=float, default=0.05)
    parser.add_argument("--camera-ransac-iterations", type=int, default=1000)
    parser.add_argument("--min-camera-inliers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-visualizations", action="store_true")
    args = parser.parse_args()

    try:
        result = evaluate_scene(
            benchmark_path=args.benchmark,
            reconstruction_dir=args.reconstruction_dir,
            evaluator_bin=args.evaluator_bin,
            output_dir=args.output_dir,
            ransac_threshold=args.camera_ransac_threshold,
            ransac_iterations=args.camera_ransac_iterations,
            min_camera_inliers=args.min_camera_inliers,
            seed=args.seed,
            write_visualizations=args.write_visualizations,
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"wrote {args.output_dir / 'result.json'}")
    for metric in result["metrics"]:
        print(
            f"tolerance={metric['tolerance']:.6g} "
            f"accuracy={metric['accuracy']:.6g} "
            f"completeness={metric['completeness']:.6g} "
            f"f1={metric['f1']:.6g}"
        )


def evaluate_scene(
    *,
    benchmark_path: Path,
    reconstruction_dir: Path,
    evaluator_bin: Path,
    output_dir: Path,
    ransac_threshold: float,
    ransac_iterations: int,
    min_camera_inliers: int,
    seed: int,
    write_visualizations: bool = False,
) -> dict[str, Any]:
    benchmark_path = benchmark_path.resolve()
    reconstruction_dir = reconstruction_dir.resolve()
    evaluator_bin = evaluator_bin.resolve()
    output_dir = output_dir.resolve()
    ensure_empty_output_dir(output_dir)

    benchmark = load_benchmark(benchmark_path)
    dataset_root = (PROJECT_ROOT / benchmark["dataset_root"]).resolve()
    inputs = benchmark["inputs"]
    image_dir = dataset_root / inputs["image_dir"]
    reference_cameras_path = dataset_root / inputs["reference_cameras"]
    reference_images_path = dataset_root / inputs["reference_images"]
    ground_truth_mlp = dataset_root / inputs["ground_truth_mlp"]
    reconstruction_ply = reconstruction_dir / "geometry" / "points.ply"
    reconstruction_cameras_path = reconstruction_dir / "geometry" / "cameras.json"

    validate_benchmark_assets(
        benchmark,
        image_dir=image_dir,
        reference_cameras_path=reference_cameras_path,
        reference_images_path=reference_images_path,
        ground_truth_mlp=ground_truth_mlp,
    )
    for path in (reconstruction_ply, reconstruction_cameras_path, evaluator_bin):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input does not exist: {path}")
    if not os.access(evaluator_bin, os.X_OK):
        raise PermissionError(f"ETH3D evaluator is not executable: {evaluator_bin}")

    estimated_images, estimated_camera_ids = load_reconstruction_images(reconstruction_cameras_path)
    reference_cameras = parse_colmap_cameras(reference_cameras_path)
    reference_images = parse_colmap_images(reference_images_path)
    validate_image_camera_ids(reference_images, {item["camera_id"] for item in reference_cameras}, "reference")
    validate_image_camera_ids(estimated_images, estimated_camera_ids, "reconstruction")

    names, estimated_centers, reference_centers = match_camera_centers(estimated_images, reference_images)
    alignment = estimate_similarity_transform_ransac(
        estimated_centers,
        reference_centers,
        threshold=ransac_threshold,
        iterations=ransac_iterations,
        min_inliers=min_camera_inliers,
        seed=seed,
    )
    scale, rotation, translation = decompose_similarity_transform(alignment.transform)

    points, colors = read_ply_points_and_colors(reconstruction_ply)
    if len(points) == 0:
        raise ValueError(f"Reconstruction point cloud is empty: {reconstruction_ply}")
    aligned_points = transform_points(points, alignment.transform)

    alignment_dir = output_dir / "alignment"
    official_dir = output_dir / "official"
    alignment_dir.mkdir(parents=True)
    official_dir.mkdir(parents=True)
    aligned_ply = alignment_dir / "reconstruction_aligned.ply"
    write_binary_ply(aligned_ply, aligned_points.astype(np.float32), colors)

    inlier_names = [name for name, inlier in zip(names, alignment.inliers, strict=True) if inlier]
    outlier_names = [name for name, inlier in zip(names, alignment.inliers, strict=True) if not inlier]
    sim3_payload = {
        "schema_version": 1,
        "transform_direction": benchmark["evaluation"]["transform_direction"],
        "alignment_method": benchmark["evaluation"]["alignment"],
        "source_coordinate_system": "reconstruction_colmap_world",
        "target_coordinate_system": "eth3d_reference_colmap_world",
        "matched_image_count": len(names),
        "matched_image_names": names,
        "ransac": {
            "seed": seed,
            "iterations": alignment.iterations,
            "inlier_threshold_reference_units": alignment.threshold,
            "min_inliers": min_camera_inliers,
            "inlier_count": int(alignment.inliers.sum()),
            "inlier_image_names": inlier_names,
            "outlier_image_names": outlier_names,
        },
        "residuals_reference_units": summarize_residuals(alignment.residuals),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "matrix_4x4": alignment.transform.tolist(),
    }
    write_json(alignment_dir / "sim3.json", sim3_payload)

    tolerances = [float(value) for value in benchmark["evaluation"]["tolerances_meters"]]
    command = build_evaluator_command(
        evaluator_bin=evaluator_bin,
        reconstruction_ply=aligned_ply,
        ground_truth_mlp=ground_truth_mlp,
        tolerances=tolerances,
        official_dir=official_dir,
        write_visualizations=write_visualizations,
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    stdout_path = output_dir / "evaluator.stdout.log"
    stderr_path = output_dir / "evaluator.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    result_payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark["benchmark_id"],
        "scene_id": benchmark["scene_id"],
        "split": benchmark["split"],
        "status": "failed" if completed.returncode else "succeeded",
        "inputs": {
            "benchmark": str(benchmark_path),
            "reconstruction_ply": str(reconstruction_ply),
            "reconstruction_ply_sha256": sha256_file(reconstruction_ply),
            "reconstruction_cameras": str(reconstruction_cameras_path),
            "reference_images": str(reference_images_path),
            "ground_truth_mlp": str(ground_truth_mlp),
        },
        "alignment": {
            "diagnostics": "alignment/sim3.json",
            "aligned_ply": "alignment/reconstruction_aligned.ply",
            "aligned_ply_sha256": sha256_file(aligned_ply),
        },
        "evaluator": {
            "binary": str(evaluator_bin),
            "argv": command,
            "return_code": completed.returncode,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        },
        "metrics": [],
    }
    if completed.returncode:
        write_json(output_dir / "result.json", result_payload)
        raise RuntimeError(
            f"ETH3D evaluator failed with exit code {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )

    result_payload["metrics"] = parse_evaluator_output(completed.stdout)
    expected_tolerances = np.asarray(tolerances, dtype=np.float64)
    actual_tolerances = np.asarray(
        [row["tolerance"] for row in result_payload["metrics"]], dtype=np.float64
    )
    if not np.allclose(actual_tolerances, expected_tolerances, rtol=0, atol=1e-9):
        raise RuntimeError(
            f"ETH3D evaluator returned tolerances {actual_tolerances.tolist()}, "
            f"expected {expected_tolerances.tolist()}"
        )
    write_json(output_dir / "result.json", result_payload)
    return result_payload


def load_benchmark(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark definition does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema_version")
    required = {"benchmark_id", "scene_id", "split", "dataset_root", "inputs", "evaluation", "protocol"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Benchmark definition is missing fields: {', '.join(missing)}")
    protocol = payload["protocol"]
    if (
        protocol.get("reconstruction_inputs") != ["rgb_images"]
        or protocol.get("reference_cameras_evaluation_only") is not True
        or protocol.get("ground_truth_scan_evaluation_only") is not True
        or protocol.get("scan_based_registration_forbidden") is not True
        or protocol.get("metric_scale_recovery_claimed") is not False
    ):
        raise ValueError("Benchmark protocol does not preserve image-only reconstruction boundaries")
    return payload


def validate_benchmark_assets(
    benchmark: dict[str, Any],
    *,
    image_dir: Path,
    reference_cameras_path: Path,
    reference_images_path: Path,
    ground_truth_mlp: Path,
) -> None:
    for path in (image_dir, reference_cameras_path, reference_images_path, ground_truth_mlp):
        if not path.exists():
            raise FileNotFoundError(f"ETH3D benchmark asset does not exist: {path}")
    expected_names = benchmark["inputs"]["expected_image_names"]
    actual_names = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    if actual_names != sorted(expected_names):
        raise ValueError(f"ETH3D image set differs from the frozen manifest: {actual_names}")
    validate_mlp_scan_paths(ground_truth_mlp)


def validate_mlp_scan_paths(mlp_path: Path) -> None:
    root = ET.parse(mlp_path).getroot()
    filenames = [element.attrib.get("filename") for element in root.iter("MLMesh")]
    if not filenames or any(not name for name in filenames):
        raise ValueError(f"MeshLab project contains no valid scan paths: {mlp_path}")
    missing = [name for name in filenames if not (mlp_path.parent / str(name)).is_file()]
    if missing:
        raise FileNotFoundError(f"MeshLab project references missing scans: {missing}")


def load_reconstruction_images(path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coordinate_system") != "colmap_world":
        raise ValueError(
            f"Only colmap_world reconstruction cameras are supported; found {payload.get('coordinate_system')!r}"
        )
    images = payload.get("images")
    cameras = payload.get("cameras")
    if not isinstance(images, list) or not isinstance(cameras, list):
        raise ValueError("Reconstruction cameras.json must contain cameras and images arrays")
    return images, {int(item["camera_id"]) for item in cameras}


def validate_image_camera_ids(images: list[dict[str, Any]], camera_ids: set[int], label: str) -> None:
    missing = sorted({int(image["camera_id"]) for image in images} - camera_ids)
    if missing:
        raise ValueError(f"{label} images reference missing camera IDs: {missing}")


def match_camera_centers(
    estimated_images: list[dict[str, Any]],
    reference_images: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    estimated = index_images_by_basename(estimated_images, "reconstruction")
    reference = index_images_by_basename(reference_images, "reference")
    names = sorted(estimated.keys() & reference.keys())
    if len(names) < 3:
        raise ValueError(f"Only {len(names)} camera names match; at least three are required")
    estimated_centers = np.stack([colmap_camera_center(estimated[name]) for name in names])
    reference_centers = np.stack([colmap_camera_center(reference[name]) for name in names])
    return names, estimated_centers, reference_centers


def index_images_by_basename(images: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for image in images:
        name = Path(str(image["name"])).name
        if name in indexed:
            raise ValueError(f"Duplicate {label} image basename: {name}")
        indexed[name] = image
    return indexed


def colmap_camera_center(image: dict[str, Any]) -> np.ndarray:
    rotation = qvec_to_rotmat(np.asarray(image["qvec"], dtype=np.float64))
    translation = np.asarray(image["tvec"], dtype=np.float64)
    return -(rotation.T @ translation)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    if qvec.shape != (4,) or not np.isfinite(qvec).all():
        raise ValueError("COLMAP qvec must contain four finite values")
    norm = float(np.linalg.norm(qvec))
    if norm <= 1e-12:
        raise ValueError("COLMAP qvec cannot be zero")
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def build_evaluator_command(
    *,
    evaluator_bin: Path,
    reconstruction_ply: Path,
    ground_truth_mlp: Path,
    tolerances: list[float],
    official_dir: Path,
    write_visualizations: bool,
) -> list[str]:
    command = [
        str(evaluator_bin),
        "--reconstruction_ply_path",
        str(reconstruction_ply),
        "--ground_truth_mlp_path",
        str(ground_truth_mlp),
        "--tolerances",
        ",".join(f"{value:g}" for value in tolerances),
    ]
    if write_visualizations:
        command.extend(
            [
                "--accuracy_cloud_output_path",
                str(official_dir / "accuracy"),
                "--completeness_cloud_output_path",
                str(official_dir / "completeness"),
            ]
        )
    return command


def parse_evaluator_output(stdout: str) -> list[dict[str, float]]:
    values: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        for label, key in RESULT_LABELS.items():
            match = re.match(rf"^{re.escape(label)}\s*:\s*(.*)$", stripped)
            if match:
                tokens = [token for token in re.split(r"[\s,]+", match.group(1).strip()) if token]
                try:
                    values[key] = [float(token) for token in tokens]
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value in ETH3D {label} output") from exc
    missing = [key for key in RESULT_LABELS.values() if key not in values]
    if missing:
        raise ValueError(f"ETH3D evaluator output is missing labelled rows: {missing}")
    lengths = {len(row) for row in values.values()}
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("ETH3D evaluator output rows have inconsistent or empty lengths")
    return [
        {
            "tolerance": values["tolerances"][index],
            "completeness": values["completeness"][index],
            "accuracy": values["accuracy"][index],
            "f1": values["f1"][index],
        }
        for index in range(len(values["tolerances"]))
    ]


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Evaluation output directory must be absent or empty: {path}")


def summarize_residuals(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
