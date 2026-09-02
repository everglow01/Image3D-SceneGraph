from __future__ import annotations

import argparse
import hashlib
import resource
import shutil
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import open3d as o3d
import torch

from image3d_scenegraph.geometry.colmap import (
    COLMAP_FEATURE_PROFILE_IDS,
    COLMAP_LEGACY_MATCHER_IDS,
    COLMAP_LOCAL_MATCHER_IDS,
    COLMAP_PAIRING_IDS,
    ColmapFeatureError,
    ResolvedColmapFeatureProfile,
    ResolvedColmapLocalMatcher,
    ResolvedColmapPairing,
    resolve_colmap_executable,
    resolve_colmap_feature_profile,
    resolve_colmap_local_matcher,
    resolve_colmap_pairing,
)
from image3d_scenegraph.geometry.grouping import (
    ColmapImage,
    CovisibilityEdge,
    build_covisibility_graph,
    build_scale_disagreement_diagnostics,
    build_vggt_group_diagnostics,
    build_vggt_group_selection,
    parse_colmap_images_with_points,
    parse_colmap_points3d,
    qvec_to_rotmat,
)
from run_colmap_sparse import build_camera_payload, discover_images, parse_colmap_cameras, run_command
from run_vggt_pointcloud import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_VGGT_REPO_DIR,
    infer_image_group,
    load_padded_rgb_images,
    load_vggt_model,
    select_device,
    select_dtype,
    validate_local_vggt,
    write_json,
    write_ply,
)


@dataclass(frozen=True)
class DepthScaleEstimate:
    scale: float
    observation_count: int
    log_mad: float


@dataclass(frozen=True)
class DepthScaleGraphResult:
    scales: dict[int, float]
    image_records: dict[int, dict[str, Any]]
    component_count: int
    anchored_image_count: int
    fallback_image_count: int
    pair_constraint_count: int
    objective_history: list[float]
    edge_records: list[dict[str, Any]]


@dataclass(frozen=True)
class VggtImageTransform:
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int


@dataclass(frozen=True)
class VggtWindowPredictionCapture:
    records: list[dict[str, Any]]
    write_seconds: float
    bytes_written: int


@dataclass(frozen=True)
class FusionCamera:
    model: str
    intrinsic: np.ndarray
    radial_distortion: tuple[float, ...]


@dataclass(frozen=True)
class FusionFrame:
    image_path: Path
    colmap_image: ColmapImage
    camera: FusionCamera
    depth: np.ndarray
    confidence: np.ndarray
    colors: np.ndarray
    scale: float
    image_shape: tuple[int, int]
    original_size: tuple[int, int]
    source_group_index: int = -1
    source_group_position: int = -1
    source_window_role: str = "unknown"
    scale_observations: int = 0
    scale_log_mad: float = float("nan")
    overlap_disagreement: np.ndarray | None = None


@dataclass(frozen=True)
class SupportPointDiagnostics:
    source_image_index: np.ndarray
    source_u: np.ndarray
    source_v: np.ndarray
    confidence: np.ndarray
    visible_counts: np.ndarray
    support_counts: np.ndarray
    contradicted_counts: np.ndarray
    occluded_counts: np.ndarray
    not_observed_counts: np.ndarray
    mean_relative_error: np.ndarray
    overlap_disagreement: np.ndarray


@dataclass(frozen=True)
class ConsistencyFilterResult:
    points: np.ndarray
    colors: np.ndarray
    candidate_points: int
    accepted_points: int
    rejected_points: int
    unverified_points: int
    supported_points: int
    occluded_only_points: int
    not_observed_only_points: int
    contradicted_only_points: int
    supported_and_contradicted_points: int
    residual_samples: np.ndarray
    image_records: list[dict[str, Any]]
    multi_visible_points: int
    policy_rejected_supported_points: int
    point_diagnostics: SupportPointDiagnostics | None = None


@dataclass(frozen=True)
class PointBudgetResult:
    points: np.ndarray
    colors: np.ndarray
    policy: str
    input_points: int
    output_points: int
    applied: bool
    spatial_quantization_bits: int | None
    occupied_spatial_codes: int | None
    selected_indices: np.ndarray | None


@dataclass(frozen=True)
class TsdfParameters:
    voxel_length: float
    sdf_trunc: float
    depth_trunc: float
    full_diagonal: float
    robust_diagonal: float


@dataclass(frozen=True)
class CrossViewValidation:
    accepted: np.ndarray
    support_counts: np.ndarray
    visible_counts: np.ndarray
    contradicted_counts: np.ndarray
    occluded_counts: np.ndarray
    not_observed_counts: np.ndarray
    mean_relative_error: np.ndarray


def prepare_colmap_text_model(
    *,
    source_dir: Path | None,
    text_dir: Path,
    run_pipeline: Any,
) -> tuple[list[str], str]:
    if source_dir is None:
        return run_pipeline(), "reconstructed"
    source_dir = source_dir.resolve()
    required = [source_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    if not all(path.is_file() for path in required):
        raise ValueError(f"--colmap-model-dir is missing a COLMAP text model: {source_dir}")
    for path in required:
        shutil.copy2(path, text_dir / path.name)
    return [f"colmap_model_source={source_dir}"], "reused_text_model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse VGGT dense depth with COLMAP global camera poses.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_VGGT_REPO_DIR)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--matcher", choices=COLMAP_LEGACY_MATCHER_IDS, default=None
    )
    parser.add_argument("--pairing", choices=COLMAP_PAIRING_IDS)
    parser.add_argument(
        "--feature-profile",
        choices=COLMAP_FEATURE_PROFILE_IDS,
        default="sift_v1",
    )
    parser.add_argument(
        "--local-matcher",
        choices=COLMAP_LOCAL_MATCHER_IDS,
        default="bruteforce",
    )
    parser.add_argument(
        "--colmap-model-dir",
        type=Path,
        help="Reuse an existing COLMAP text model instead of rerunning sparse reconstruction.",
    )
    parser.add_argument("--colmap-single-camera", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mapper-abs-pose-min-num-inliers", type=int, default=30)
    parser.add_argument("--mapper-abs-pose-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--vggt-batch-size", type=int, default=4)
    parser.add_argument("--vggt-overlap-size", type=int, default=2)
    parser.add_argument("--vggt-grouping", choices=["covisibility", "sequential"], default="sequential")
    parser.add_argument(
        "--vggt-frames-chunk-size",
        type=int,
        default=None,
        help="Process VGGT depth-head frames in chunks of this size to lower peak GPU memory.",
    )
    parser.add_argument(
        "--retain-vggt-window-predictions",
        action="store_true",
        help="Save every VGGT window depth/confidence prediction for overlap diagnostics.",
    )
    parser.add_argument("--fusion-mode", choices=["tsdf", "points"], default="points")
    parser.add_argument("--tsdf-voxel-length", type=float, default=0.0)
    parser.add_argument("--tsdf-sdf-trunc", type=float, default=0.0)
    parser.add_argument("--tsdf-depth-trunc", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument(
        "--point-budget-policy",
        choices=["random", "spatial_balanced"],
        default="random",
    )
    parser.add_argument("--factorial-output-dir", type=Path)
    parser.add_argument("--point-budget-sensitivity-output-dir", type=Path)
    parser.add_argument("--conf-percentile", type=float, default=50.0)
    parser.add_argument(
        "--confidence-threshold-scope", choices=["global", "per_frame"], default="global"
    )
    parser.add_argument("--confidence-comparison-ply", type=Path)
    parser.add_argument("--min-scale-observations", type=int, default=20)
    parser.add_argument("--depth-scale-mode", choices=["per_frame", "global_graph"], default="per_frame")
    parser.add_argument("--scale-graph-iterations", type=int, default=5)
    parser.add_argument("--scale-graph-pair-weight", type=float, default=1.0)
    parser.add_argument("--scale-graph-huber-delta", type=float, default=0.05)
    parser.add_argument("--scale-graph-max-pairs-per-edge", type=int, default=256)
    parser.add_argument("--consistency-neighbors", type=int, default=6)
    parser.add_argument("--consistency-min-shared-points", type=int, default=20)
    parser.add_argument("--consistency-relative-threshold", type=float, default=0.08)
    parser.add_argument("--consistency-min-relative-threshold", type=float, default=0.02)
    parser.add_argument("--consistency-stride", type=int, default=1)
    parser.add_argument(
        "--consistency-support-policy",
        choices=["any_support", "adaptive_two"],
        default="any_support",
    )
    parser.add_argument("--support-policy-comparison-ply", type=Path)
    parser.add_argument("--joint-comparison-ply", type=Path)
    parser.add_argument(
        "--support-diagnostics-output",
        type=Path,
        help="Write per-final-point support provenance as a compressed NPZ sidecar.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pairing is not None and args.matcher is not None:
        parser.error("--pairing and legacy --matcher cannot be combined")
    if args.pairing == "sequential_loop":
        parser.error("COLMAP+VGGT multi-image geometry does not support sequential_loop")
    legacy_matcher = args.matcher
    if args.pairing is None and legacy_matcher is None:
        legacy_matcher = "exhaustive"

    if args.mapper_abs_pose_min_num_inliers <= 0:
        raise SystemExit("--mapper-abs-pose-min-num-inliers must be positive")
    if not 0 < args.mapper_abs_pose_min_inlier_ratio <= 1:
        raise SystemExit("--mapper-abs-pose-min-inlier-ratio must be between 0 and 1")
    if not 0 <= args.conf_percentile < 100:
        raise SystemExit("--conf-percentile must be between 0 and 100")
    if args.consistency_neighbors <= 0:
        raise SystemExit("--consistency-neighbors must be positive")
    if args.consistency_min_shared_points <= 0:
        raise SystemExit("--consistency-min-shared-points must be positive")
    if not 0 < args.consistency_relative_threshold < 1:
        raise SystemExit("--consistency-relative-threshold must be between 0 and 1")
    if not 0 < args.consistency_min_relative_threshold <= args.consistency_relative_threshold:
        raise SystemExit(
            "--consistency-min-relative-threshold must be positive and no larger than "
            "--consistency-relative-threshold"
        )
    if args.consistency_stride <= 0:
        raise SystemExit("--consistency-stride must be positive")
    if args.scale_graph_iterations <= 0:
        raise SystemExit("--scale-graph-iterations must be positive")
    if args.scale_graph_pair_weight < 0:
        raise SystemExit("--scale-graph-pair-weight must be non-negative")
    if args.scale_graph_huber_delta <= 0:
        raise SystemExit("--scale-graph-huber-delta must be positive")
    if args.scale_graph_max_pairs_per_edge <= 0:
        raise SystemExit("--scale-graph-max-pairs-per-edge must be positive")
    if args.vggt_frames_chunk_size is not None and args.vggt_frames_chunk_size <= 0:
        raise SystemExit("--vggt-frames-chunk-size must be positive")
    if args.support_diagnostics_output is not None and args.fusion_mode != "points":
        raise SystemExit("--support-diagnostics-output requires --fusion-mode points")

    started_at = time.perf_counter()
    colmap_path = resolve_colmap_executable()
    if colmap_path is None:
        raise SystemExit(
            "COLMAP executable not found. Run `uv run python scripts/setup_colmap_cuda.py --install` "
            "or install COLMAP on PATH."
        )
    colmap = str(colmap_path)
    try:
        feature_profile = resolve_colmap_feature_profile(args.feature_profile)
        local_matcher = resolve_colmap_local_matcher(
            feature_profile, args.local_matcher
        )
        pairing = (
            resolve_colmap_pairing(feature_profile, args.pairing)
            if args.pairing is not None
            else ResolvedColmapPairing(
                profile_id=str(legacy_matcher),
                command=f"{legacy_matcher}_matcher",
                pairing_options=(),
            )
        )
    except ColmapFeatureError as exc:
        raise SystemExit(str(exc)) from exc

    image_paths = discover_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    validate_local_vggt(args.repo_dir, args.checkpoint_dir)
    sys.path.insert(0, str(args.repo_dir.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = args.output_dir
    geometry_dir = output_dir / "geometry"
    diagnostics_dir = output_dir / "diagnostics"
    logs_dir = output_dir / "logs"
    work_dir = output_dir / "colmap_vggt"
    sparse_dir = work_dir / "sparse"
    text_dir = work_dir / "sparse_txt"
    database_path = work_dir / "database.db"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    colmap_started_at = time.perf_counter()
    try:
        colmap_logs, colmap_source = prepare_colmap_text_model(
            source_dir=args.colmap_model_dir,
            text_dir=text_dir,
            run_pipeline=lambda: run_colmap_pipeline(
                colmap=colmap,
                image_dir=args.image_dir,
                database_path=database_path,
                sparse_dir=sparse_dir,
                text_dir=text_dir,
                pairing=pairing,
                feature_profile=feature_profile,
                local_matcher=local_matcher,
                single_camera=bool(args.colmap_single_camera),
                mapper_abs_pose_min_num_inliers=args.mapper_abs_pose_min_num_inliers,
                mapper_abs_pose_min_inlier_ratio=args.mapper_abs_pose_min_inlier_ratio,
            ),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    colmap_seconds = time.perf_counter() - colmap_started_at

    colmap_images = parse_colmap_images_with_points(text_dir / "images.txt")
    points3d = parse_colmap_points3d(text_dir / "points3D.txt")
    colmap_cameras = {
        camera["camera_id"]: camera for camera in parse_colmap_cameras(text_dir / "cameras.txt")
    }
    registered_by_name = {image.name: image for image in colmap_images}
    registered_paths = [path for path in image_paths if path.name in registered_by_name]
    if not registered_paths:
        raise RuntimeError("COLMAP did not register any input images")

    device = select_device(args.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dtype = select_dtype(device, args.precision)
    model = load_vggt_model(
        model_cls=VGGT,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        dtype=dtype,
        enable_point=False,
    )
    model.eval()

    group_selection = build_vggt_group_selection(
        registered_paths=registered_paths,
        registered_by_name=registered_by_name,
        grouping=args.vggt_grouping,
        batch_size=args.vggt_batch_size,
        overlap_size=args.vggt_overlap_size,
    )
    vggt_groups = group_selection.groups
    vggt_groups_path = diagnostics_dir / "vggt_groups.json"
    write_json(
        vggt_groups_path,
        build_vggt_group_diagnostics(
            groups=vggt_groups,
            registered_by_name=registered_by_name,
            grouping=args.vggt_grouping,
            batch_size=args.vggt_batch_size,
            requested_overlap_size=args.vggt_overlap_size,
            selection_records=group_selection.records,
        ),
    )
    vggt_started_at = time.perf_counter()
    depth_items, window_prediction_capture = run_vggt_depth_batches(
        model=model,
        groups=vggt_groups,
        load_and_preprocess_images=load_and_preprocess_images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        device=device,
        dtype=dtype,
        frames_chunk_size=args.vggt_frames_chunk_size,
        capture_dir=(
            diagnostics_dir / "vggt_window_predictions"
            if args.retain_vggt_window_predictions
            else None
        ),
        selection_records=group_selection.records,
        registered_by_name=registered_by_name,
        points3d=points3d,
        min_scale_observations=args.min_scale_observations,
        retain_point_diagnostics=args.support_diagnostics_output is not None,
    )
    vggt_seconds = time.perf_counter() - vggt_started_at

    scale_estimates: dict[str, DepthScaleEstimate] = {}
    for image_path, item in depth_items.items():
        colmap_image = registered_by_name[image_path.name]
        estimate = estimate_depth_scale(
            colmap_image=colmap_image,
            points3d=points3d,
            depth=item["depth"],
            image_shape=item["image_shape"],
            original_size=item["original_size"],
            min_observations=args.min_scale_observations,
        )
        if estimate is not None:
            scale_estimates[image_path.name] = estimate
    if not scale_estimates:
        raise RuntimeError("Could not estimate VGGT-to-COLMAP depth scale from sparse observations")
    fallback_scale = float(np.median([estimate.scale for estimate in scale_estimates.values()]))
    consistency_relative_threshold = derive_consistency_relative_threshold(
        scale_estimates.values(),
        min_threshold=args.consistency_min_relative_threshold,
        max_threshold=args.consistency_relative_threshold,
    )

    used_scales: list[float] = []
    fusion_camera_records: list[dict[str, Any]] = []
    fusion_frames: list[FusionFrame] = []
    for image_path, item in depth_items.items():
        colmap_image = registered_by_name[image_path.name]
        colmap_camera = colmap_cameras.get(colmap_image.camera_id)
        if colmap_camera is None:
            raise RuntimeError(f"COLMAP camera {colmap_image.camera_id} missing for {image_path.name}")
        fusion_camera = build_fusion_camera(
            colmap_camera=colmap_camera,
            original_size=item["original_size"],
            image_shape=item["image_shape"],
        )
        estimate = scale_estimates.get(image_path.name)
        scale = estimate.scale if estimate is not None else fallback_scale
        used_scales.append(scale)
        fusion_frames.append(
            FusionFrame(
                image_path=image_path,
                colmap_image=colmap_image,
                camera=fusion_camera,
                depth=item["depth"],
                confidence=item["confidence"],
                colors=item["colors"],
                scale=scale,
                image_shape=item["image_shape"],
                original_size=item["original_size"],
                source_group_index=int(item["source_group_index"]),
                source_group_position=int(item["source_group_position"]),
                source_window_role=str(item["source_window_role"]),
                scale_observations=(estimate.observation_count if estimate is not None else 0),
                scale_log_mad=(estimate.log_mad if estimate is not None else float("nan")),
                overlap_disagreement=item.get("overlap_disagreement"),
            )
        )
        fusion_camera_records.append(
            build_fusion_camera_record(
                image_name=image_path.name,
                camera_id=colmap_image.camera_id,
                camera=fusion_camera,
                vggt_intrinsic=item["intrinsic"],
                scale_estimate=estimate,
                fallback_scale=fallback_scale,
            )
        )

    scale_disagreement_path = diagnostics_dir / "scale_disagreement.json"
    write_json(
        scale_disagreement_path,
        build_scale_disagreement_diagnostics(
            colmap_images=colmap_images,
            groups=vggt_groups,
            scales_by_name={name: estimate.scale for name, estimate in scale_estimates.items()},
        ),
    )
    window_predictions_path = diagnostics_dir / "vggt_window_predictions.json"
    if window_prediction_capture is not None:
        prediction_counts: dict[str, int] = {}
        for record in window_prediction_capture.records:
            image_name = str(record["image"])
            prediction_counts[image_name] = prediction_counts.get(image_name, 0) + 1
        write_json(
            window_predictions_path,
            {
                "schema_version": 1,
                "fusion_policy": "first_wins",
                "capture_enabled": True,
                "grouping": args.vggt_grouping,
                "batch_size": args.vggt_batch_size,
                "requested_overlap_size": args.vggt_overlap_size,
                "frames_chunk_size": args.vggt_frames_chunk_size,
                "group_count": len(vggt_groups),
                "registered_image_count": len(registered_paths),
                "prediction_count": len(window_prediction_capture.records),
                "unique_image_count": len(prediction_counts),
                "overlap_image_count": sum(count > 1 for count in prediction_counts.values()),
                "max_predictions_per_image": max(prediction_counts.values(), default=0),
                "bytes_written": window_prediction_capture.bytes_written,
                "write_seconds": window_prediction_capture.write_seconds,
                "max_resident_set_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "predictions": window_prediction_capture.records,
            },
        )

    covisibility_graph = build_covisibility_graph(
        [frame.colmap_image for frame in fusion_frames],
        max_neighbors=args.consistency_neighbors,
        min_shared_points=args.consistency_min_shared_points,
    )
    scale_graph_result: DepthScaleGraphResult | None = None
    if args.depth_scale_mode == "global_graph":
        scale_graph_result = optimize_depth_scale_graph(
            frames=fusion_frames,
            covisibility_graph=covisibility_graph,
            scale_estimates=scale_estimates,
            fallback_scale=fallback_scale,
            confidence_threshold=compute_global_confidence_threshold(fusion_frames, args.conf_percentile),
            relative_threshold=consistency_relative_threshold,
            iterations=args.scale_graph_iterations,
            pair_weight=args.scale_graph_pair_weight,
            huber_delta=args.scale_graph_huber_delta,
            max_pairs_per_edge=args.scale_graph_max_pairs_per_edge,
        )
        fusion_frames = [
            replace(frame, scale=scale_graph_result.scales[frame.colmap_image.image_id])
            for frame in fusion_frames
        ]
        used_scales = [frame.scale for frame in fusion_frames]
        records_by_id = scale_graph_result.image_records
        for record, frame in zip(fusion_camera_records, fusion_frames, strict=True):
            graph_record = records_by_id[frame.colmap_image.image_id]
            record["depth_scale_initial"] = graph_record["initial_scale"]
            record["depth_scale"] = frame.scale
            record["depth_scale_graph_component"] = graph_record["component"]
            record["depth_scale_graph_anchor"] = graph_record["anchored"]
            record["depth_scale_graph_fallback"] = graph_record["fallback"]

    tsdf_params: dict[str, float | int] | None = None
    effective_confidence_threshold_scope = (
        "per_frame" if args.fusion_mode == "tsdf" else args.confidence_threshold_scope
    )
    if args.fusion_mode == "tsdf":
        tsdf_parameters = derive_tsdf_parameters(
            points3d,
            fusion_frames,
            voxel_length=args.tsdf_voxel_length,
            sdf_trunc=args.tsdf_sdf_trunc,
            depth_trunc=args.tsdf_depth_trunc,
        )
        tsdf_points, tsdf_colors, tsdf_stats = fuse_frames_tsdf(
            fusion_frames,
            confidence_percentile=args.conf_percentile,
            voxel_length=tsdf_parameters.voxel_length,
            sdf_trunc=tsdf_parameters.sdf_trunc,
            depth_trunc=tsdf_parameters.depth_trunc,
        )
        validate_tsdf_output(tsdf_stats)
        point_budget_result = apply_point_budget(
            tsdf_points,
            tsdf_colors,
            args.max_points,
            args.seed,
            policy=args.point_budget_policy,
        )
        flat_points = point_budget_result.points
        flat_colors = point_budget_result.colors
        if len(flat_points) == 0:
            raise RuntimeError("TSDF fusion produced no dense points")
        tsdf_params = {
            "voxel_length": tsdf_parameters.voxel_length,
            "sdf_trunc": tsdf_parameters.sdf_trunc,
            "depth_trunc": tsdf_parameters.depth_trunc,
            "full_sparse_diagonal": tsdf_parameters.full_diagonal,
            "robust_sparse_diagonal": tsdf_parameters.robust_diagonal,
            "integrated_frames": tsdf_stats["integrated_frames"],
        }
        consistency_summary = None
        consistency_payload = {
            "fusion_mode": "tsdf",
            "confidence_percentile": args.conf_percentile,
            "relative_threshold": consistency_relative_threshold,
            "voxel_length": tsdf_parameters.voxel_length,
            "sdf_trunc": tsdf_parameters.sdf_trunc,
            "depth_trunc": tsdf_parameters.depth_trunc,
            "full_sparse_diagonal": tsdf_parameters.full_diagonal,
            "robust_sparse_diagonal": tsdf_parameters.robust_diagonal,
            "integrated_frames": tsdf_stats["integrated_frames"],
            "confidence_threshold_min": tsdf_stats["confidence_threshold_min"],
            "confidence_threshold_median": tsdf_stats["confidence_threshold_median"],
            "confidence_threshold_max": tsdf_stats["confidence_threshold_max"],
            "fused_points": tsdf_stats["num_points"],
            "output_points": int(len(flat_points)),
        }
    else:
        confidence_thresholds = compute_confidence_thresholds(
            fusion_frames,
            args.conf_percentile,
            scope=args.confidence_threshold_scope,
        )
        confidence_threshold_values = np.asarray(list(confidence_thresholds.values()), dtype=np.float64)
        confidence_threshold = float(np.median(confidence_threshold_values))
        filtered = filter_points_by_cross_view_consistency(
            fusion_frames,
            covisibility_graph=covisibility_graph,
            confidence_thresholds=confidence_thresholds,
            relative_threshold=consistency_relative_threshold,
            support_policy=args.consistency_support_policy,
            stride=args.consistency_stride,
            retain_point_diagnostics=args.support_diagnostics_output is not None,
        )
        comparison_summary: dict[str, Any] | None = None
        comparison_summaries: list[dict[str, Any]] = []
        if args.confidence_comparison_ply is not None:
            comparison_scope = (
                "global" if args.confidence_threshold_scope == "per_frame" else "per_frame"
            )
            comparison_thresholds = compute_confidence_thresholds(
                fusion_frames,
                args.conf_percentile,
                scope=comparison_scope,
            )
            comparison_filtered = filter_points_by_cross_view_consistency(
                fusion_frames,
                covisibility_graph=covisibility_graph,
                confidence_thresholds=comparison_thresholds,
                relative_threshold=consistency_relative_threshold,
                support_policy=args.consistency_support_policy,
                stride=args.consistency_stride,
            )
            comparison_points, comparison_colors = cap_points(
                comparison_filtered.points,
                comparison_filtered.colors,
                args.max_points,
                args.seed,
                policy=args.point_budget_policy,
            )
            args.confidence_comparison_ply.parent.mkdir(parents=True, exist_ok=True)
            write_ply(args.confidence_comparison_ply, comparison_points, comparison_colors)
            comparison_summary = {
                "kind": "confidence_threshold_scope",
                "scope": comparison_scope,
                "path": str(args.confidence_comparison_ply),
                "candidate_points": comparison_filtered.candidate_points,
                "accepted_points": comparison_filtered.accepted_points,
                "output_points": int(len(comparison_points)),
            }
            comparison_summaries.append(comparison_summary)
        if args.support_policy_comparison_ply is not None:
            comparison_policy = (
                "any_support"
                if args.consistency_support_policy == "adaptive_two"
                else "adaptive_two"
            )
            comparison_filtered = filter_points_by_cross_view_consistency(
                fusion_frames,
                covisibility_graph=covisibility_graph,
                confidence_thresholds=confidence_thresholds,
                relative_threshold=consistency_relative_threshold,
                support_policy=comparison_policy,
                stride=args.consistency_stride,
            )
            comparison_points, comparison_colors = cap_points(
                comparison_filtered.points,
                comparison_filtered.colors,
                args.max_points,
                args.seed,
                policy=args.point_budget_policy,
            )
            args.support_policy_comparison_ply.parent.mkdir(parents=True, exist_ok=True)
            write_ply(
                args.support_policy_comparison_ply,
                comparison_points,
                comparison_colors,
            )
            comparison_summary = {
                "kind": "support_policy",
                "support_policy": comparison_policy,
                "path": str(args.support_policy_comparison_ply),
                "candidate_points": comparison_filtered.candidate_points,
                "accepted_points": comparison_filtered.accepted_points,
                "multi_visible_points": comparison_filtered.multi_visible_points,
                "policy_rejected_supported_points": (
                    comparison_filtered.policy_rejected_supported_points
                ),
                "output_points": int(len(comparison_points)),
            }
            comparison_summaries.append(comparison_summary)
        if args.joint_comparison_ply is not None:
            comparison_scope = (
                "global" if args.confidence_threshold_scope == "per_frame" else "per_frame"
            )
            comparison_policy = (
                "any_support"
                if args.consistency_support_policy == "adaptive_two"
                else "adaptive_two"
            )
            comparison_thresholds = compute_confidence_thresholds(
                fusion_frames,
                args.conf_percentile,
                scope=comparison_scope,
            )
            comparison_filtered = filter_points_by_cross_view_consistency(
                fusion_frames,
                covisibility_graph=covisibility_graph,
                confidence_thresholds=comparison_thresholds,
                relative_threshold=consistency_relative_threshold,
                support_policy=comparison_policy,
                stride=args.consistency_stride,
            )
            comparison_points, comparison_colors = cap_points(
                comparison_filtered.points,
                comparison_filtered.colors,
                args.max_points,
                args.seed,
                policy=args.point_budget_policy,
            )
            args.joint_comparison_ply.parent.mkdir(parents=True, exist_ok=True)
            write_ply(args.joint_comparison_ply, comparison_points, comparison_colors)
            comparison_summary = {
                "kind": "joint_confidence_support",
                "confidence_threshold_scope": comparison_scope,
                "support_policy": comparison_policy,
                "path": str(args.joint_comparison_ply),
                "candidate_points": comparison_filtered.candidate_points,
                "accepted_points": comparison_filtered.accepted_points,
                "multi_visible_points": comparison_filtered.multi_visible_points,
                "policy_rejected_supported_points": (
                    comparison_filtered.policy_rejected_supported_points
                ),
                "output_points": int(len(comparison_points)),
            }
            comparison_summaries.append(comparison_summary)
        factorial_summaries: list[dict[str, Any]] = []
        if args.factorial_output_dir is not None:
            args.factorial_output_dir.mkdir(parents=True, exist_ok=True)
            cameras_payload = build_camera_payload(text_dir)
            for scope, support_policy in product(
                ("global", "per_frame"),
                ("any_support", "adaptive_two"),
            ):
                if (
                    scope == args.confidence_threshold_scope
                    and support_policy == args.consistency_support_policy
                ):
                    arm_filtered = filtered
                else:
                    arm_thresholds = compute_confidence_thresholds(
                        fusion_frames,
                        args.conf_percentile,
                        scope=scope,
                    )
                    arm_filtered = filter_points_by_cross_view_consistency(
                        fusion_frames,
                        covisibility_graph=covisibility_graph,
                        confidence_thresholds=arm_thresholds,
                        relative_threshold=consistency_relative_threshold,
                        support_policy=support_policy,
                        stride=args.consistency_stride,
                    )
                for budget_policy in ("random", "spatial_balanced"):
                    budget_result = apply_point_budget(
                        arm_filtered.points,
                        arm_filtered.colors,
                        args.max_points,
                        args.seed,
                        policy=budget_policy,
                    )
                    arm_name = factorial_arm_name(scope, support_policy, budget_policy)
                    arm_dir = args.factorial_output_dir / arm_name
                    geometry_output_dir = arm_dir / "geometry"
                    geometry_output_dir.mkdir(parents=True, exist_ok=True)
                    points_path = geometry_output_dir / "points.ply"
                    cameras_path = geometry_output_dir / "cameras.json"
                    write_ply(points_path, budget_result.points, budget_result.colors)
                    write_json(cameras_path, cameras_payload)
                    factorial_summaries.append(
                        {
                            "arm": arm_name,
                            "confidence_threshold_scope": scope,
                            "support_policy": support_policy,
                            "point_budget_policy": budget_policy,
                            "path": str(points_path),
                            "candidate_points": arm_filtered.candidate_points,
                            "accepted_points": arm_filtered.accepted_points,
                            **point_budget_diagnostics(budget_result),
                        }
                    )
        sensitivity_summaries: list[dict[str, Any]] = []
        if args.point_budget_sensitivity_output_dir is not None:
            args.point_budget_sensitivity_output_dir.mkdir(parents=True, exist_ok=True)
            cameras_payload = build_camera_payload(text_dir)
            sensitivity_budget = max(1, args.max_points // 2)
            for budget_policy in ("random", "spatial_balanced"):
                budget_result = apply_point_budget(
                    filtered.points,
                    filtered.colors,
                    sensitivity_budget,
                    args.seed,
                    policy=budget_policy,
                )
                arm_dir = args.point_budget_sensitivity_output_dir / budget_policy
                geometry_output_dir = arm_dir / "geometry"
                geometry_output_dir.mkdir(parents=True, exist_ok=True)
                points_path = geometry_output_dir / "points.ply"
                cameras_path = geometry_output_dir / "cameras.json"
                write_ply(points_path, budget_result.points, budget_result.colors)
                write_json(cameras_path, cameras_payload)
                sensitivity_summaries.append(
                    {
                        "point_budget_policy": budget_policy,
                        "max_points": sensitivity_budget,
                        "path": str(points_path),
                        **point_budget_diagnostics(budget_result),
                    }
                )
        point_budget_result = apply_point_budget(
            filtered.points,
            filtered.colors,
            args.max_points,
            args.seed,
            policy=args.point_budget_policy,
        )
        flat_points = point_budget_result.points
        flat_colors = point_budget_result.colors
        if len(flat_points) == 0:
            raise RuntimeError("Cross-view consistency rejected every dense point")
        consistency_summary = {
            "candidate_points": filtered.candidate_points,
            "accepted_points": filtered.accepted_points,
            "rejected_points": filtered.rejected_points,
            "unverified_points": filtered.unverified_points,
            "supported_points": filtered.supported_points,
            "multi_visible_points": filtered.multi_visible_points,
            "policy_rejected_supported_points": filtered.policy_rejected_supported_points,
            "acceptance_rate": filtered.accepted_points / max(filtered.candidate_points, 1),
            "residual_p50": percentile_or_zero(filtered.residual_samples, 50),
            "residual_p90": percentile_or_zero(filtered.residual_samples, 90),
        }
        consistency_payload = {
            "fusion_mode": "points",
            **build_consistency_payload(
                filtered,
                confidence_thresholds=confidence_thresholds,
                confidence_percentile=args.conf_percentile,
                confidence_threshold_scope=args.confidence_threshold_scope,
                support_policy=args.consistency_support_policy,
                relative_threshold=consistency_relative_threshold,
                stride=args.consistency_stride,
            ),
        }

    write_ply(geometry_dir / "points.ply", flat_points, flat_colors)
    support_diagnostics_summary = None
    if args.support_diagnostics_output is not None:
        if filtered.point_diagnostics is None:
            raise RuntimeError("Per-point support diagnostics were not retained")
        support_diagnostics_summary = write_support_point_diagnostics(
            args.support_diagnostics_output,
            diagnostics=filtered.point_diagnostics,
            selected_indices=point_budget_result.selected_indices,
            frames=fusion_frames,
            expected_point_count=len(flat_points),
            source_index_path=(
                window_predictions_path if window_prediction_capture is not None else None
            ),
        )
    write_json(geometry_dir / "cameras.json", build_camera_payload(text_dir))
    visibility_graph_path = diagnostics_dir / "visibility_graph.json"
    consistency_path = diagnostics_dir / "consistency.json"
    fusion_diagnostics_path = diagnostics_dir / "fusion.json"
    scale_graph_path = diagnostics_dir / "depth_scale_graph.json"
    if scale_graph_result is not None:
        write_json(
            scale_graph_path,
            {
                "mode": args.depth_scale_mode,
                "iterations": args.scale_graph_iterations,
                "pair_weight": args.scale_graph_pair_weight,
                "huber_delta": args.scale_graph_huber_delta,
                "max_pairs_per_edge": args.scale_graph_max_pairs_per_edge,
                "component_count": scale_graph_result.component_count,
                "anchored_image_count": scale_graph_result.anchored_image_count,
                "fallback_image_count": scale_graph_result.fallback_image_count,
                "pair_constraint_count": scale_graph_result.pair_constraint_count,
                "objective_history": scale_graph_result.objective_history,
                "images": [
                    scale_graph_result.image_records[frame.colmap_image.image_id]
                    for frame in fusion_frames
                ],
                "edges": scale_graph_result.edge_records,
            },
        )
    write_json(
        visibility_graph_path,
        build_visibility_graph_payload(fusion_frames, covisibility_graph),
    )
    write_json(consistency_path, consistency_payload)
    write_json(
        fusion_diagnostics_path,
        {
            "intrinsics_source": "colmap",
            "depth_source": "vggt",
            "fusion_mode": args.fusion_mode,
            "depth_scale_mode": args.depth_scale_mode,
            "depth_scale_graph": (
                scale_graph_path.relative_to(output_dir).as_posix()
                if scale_graph_result is not None
                else None
            ),
            "registered_images": len(registered_paths),
            "scale_fallback": fallback_scale,
            "camera_models": sorted({record["colmap_model"] for record in fusion_camera_records}),
            "tsdf": tsdf_params,
            "point_budget": point_budget_diagnostics(point_budget_result),
            "vggt_window_predictions": (
                window_predictions_path.relative_to(output_dir).as_posix()
                if window_prediction_capture is not None
                else None
            ),
            "factorial_outputs": factorial_summaries if args.fusion_mode == "points" else [],
            "point_budget_sensitivity_outputs": (
                sensitivity_summaries if args.fusion_mode == "points" else []
            ),
            "support_point_diagnostics": support_diagnostics_summary,
            "cross_view_filter": (
                {
                    "confidence_threshold": confidence_threshold,
                    "confidence_threshold_scope": args.confidence_threshold_scope,
                    "confidence_percentile": args.conf_percentile,
                    "confidence_threshold_min": float(np.min(confidence_threshold_values)),
                    "confidence_threshold_median": confidence_threshold,
                    "confidence_threshold_max": float(np.max(confidence_threshold_values)),
                    "support_policy": args.consistency_support_policy,
                    "neighbors": args.consistency_neighbors,
                    "min_shared_points": args.consistency_min_shared_points,
                    "relative_threshold": consistency_relative_threshold,
                    "relative_threshold_cap": args.consistency_relative_threshold,
                    "min_relative_threshold": args.consistency_min_relative_threshold,
                    "stride": args.consistency_stride,
                    "comparison": comparison_summary,
                    "comparisons": comparison_summaries,
                }
                if consistency_summary is not None
                else None
            ),
            "images": fusion_camera_records,
        },
    )

    consistency_log_lines: list[str]
    if consistency_summary is not None:
        consistency_log_lines = [
            f"consistency_confidence_threshold={confidence_threshold:.6f}",
            f"consistency_confidence_threshold_scope={effective_confidence_threshold_scope}",
            f"consistency_confidence_percentile={args.conf_percentile:.6f}",
            f"consistency_confidence_threshold_min={float(np.min(confidence_threshold_values)):.6f}",
            f"consistency_confidence_threshold_median={confidence_threshold:.6f}",
            f"consistency_confidence_threshold_max={float(np.max(confidence_threshold_values)):.6f}",
            f"consistency_candidates={consistency_summary['candidate_points']}",
            f"consistency_accepted={consistency_summary['accepted_points']}",
            f"consistency_rejected={consistency_summary['rejected_points']}",
            f"consistency_unverified={consistency_summary['unverified_points']}",
            f"consistency_supported={consistency_summary['supported_points']}",
            f"consistency_multi_visible={consistency_summary['multi_visible_points']}",
            f"consistency_policy_rejected_supported={consistency_summary['policy_rejected_supported_points']}",
            f"consistency_support_policy={args.consistency_support_policy}",
            f"consistency_acceptance_rate={consistency_summary['acceptance_rate']:.6f}",
            f"consistency_relative_threshold={consistency_relative_threshold:.6f}",
            f"consistency_residual_p50={consistency_summary['residual_p50']:.6f}",
            f"consistency_residual_p90={consistency_summary['residual_p90']:.6f}",
            f"consistency_stride={args.consistency_stride}",
        ]
    else:
        consistency_log_lines = ["consistency_status=not_run_tsdf"]

    effective_overlap_size = (
        args.vggt_overlap_size
        if args.vggt_grouping == "covisibility" and args.vggt_batch_size < len(registered_paths)
        else 0
    )
    elapsed_seconds = time.perf_counter() - started_at
    gpu_peak_memory_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    log_lines = [
        "backend=colmap_vggt",
        f"num_images={len(image_paths)}",
        f"registered_images={len(registered_paths)}",
        f"num_points={len(flat_points)}",
        f"colmap_points={len(points3d)}",
        f"scaled_images={len(scale_estimates)}",
        f"scale_median={float(np.median(used_scales)):.6f}",
        f"scale_min={float(np.min(used_scales)):.6f}",
        f"scale_max={float(np.max(used_scales)):.6f}",
        f"scale_observations_median={float(np.median([estimate.observation_count for estimate in scale_estimates.values()])):.3f}",
        f"scale_log_mad_median={float(np.median([estimate.log_mad for estimate in scale_estimates.values()])):.6f}",
        f"depth_scale_mode={args.depth_scale_mode}",
        f"scale_graph_components={scale_graph_result.component_count if scale_graph_result is not None else 0}",
        f"scale_graph_anchored_images={scale_graph_result.anchored_image_count if scale_graph_result is not None else 0}",
        f"scale_graph_fallback_images={scale_graph_result.fallback_image_count if scale_graph_result is not None else 0}",
        f"scale_graph_pair_constraints={scale_graph_result.pair_constraint_count if scale_graph_result is not None else 0}",
        "fusion_intrinsics=colmap",
        f"fusion_mode={args.fusion_mode}",
        f"fusion_diagnostics={fusion_diagnostics_path.relative_to(output_dir).as_posix()}",
        f"visibility_graph={visibility_graph_path.relative_to(output_dir).as_posix()}",
        f"vggt_group_diagnostics={vggt_groups_path.relative_to(output_dir).as_posix()}",
        f"scale_disagreement_diagnostics={scale_disagreement_path.relative_to(output_dir).as_posix()}",
        f"consistency_diagnostics={consistency_path.relative_to(output_dir).as_posix()}",
        *(
            [f"depth_scale_graph_diagnostics={scale_graph_path.relative_to(output_dir).as_posix()}"]
            if scale_graph_result is not None
            else []
        ),
        *consistency_log_lines,
        f"sfm_feature_profile={feature_profile.profile_id}",
        f"sfm_feature_extractor={feature_profile.extractor}",
        f"sfm_feature_descriptor={feature_profile.descriptor}",
        f"sfm_local_matcher_profile={local_matcher.profile_id}",
        f"sfm_local_matcher={local_matcher.name}",
        f"sfm_feature_max_features={feature_profile.max_features}",
        f"sfm_feature_extractor_model_sha256={feature_profile.extractor_model_sha256 or 'none'}",
        f"sfm_local_matcher_model_sha256={local_matcher.model_sha256 or 'none'}",
        f"sfm_pairing={pairing.profile_id}",
        f"sfm_pairing_command={pairing.command}",
        f"sfm_pairing_vocab_tree_sha256={pairing.vocab_tree_sha256 or 'none'}",
        "sfm_mapper=incremental",
        f"matcher={pairing.command.removesuffix('_matcher')}",
        f"colmap_executable={colmap}",
        f"colmap_source={colmap_source}",
        f"vggt_batch_size={args.vggt_batch_size}",
        f"vggt_overlap_size={args.vggt_overlap_size}",
        f"overlap_size={effective_overlap_size}",
        f"vggt_grouping={args.vggt_grouping}",
        f"vggt_frames_chunk_size={args.vggt_frames_chunk_size}",
        f"vggt_window_prediction_capture={str(window_prediction_capture is not None).lower()}",
        f"vggt_window_prediction_count={len(window_prediction_capture.records) if window_prediction_capture is not None else 0}",
        f"vggt_window_prediction_bytes={window_prediction_capture.bytes_written if window_prediction_capture is not None else 0}",
        f"vggt_window_prediction_write_seconds={window_prediction_capture.write_seconds if window_prediction_capture is not None else 0.0:.3f}",
        *(
            [f"vggt_window_prediction_diagnostics={window_predictions_path.relative_to(output_dir).as_posix()}"]
            if window_prediction_capture is not None
            else []
        ),
        f"num_groups={len(vggt_groups)}",
        f"conf_percentile={args.conf_percentile}",
        f"confidence_threshold_scope={effective_confidence_threshold_scope}",
        f"max_points={args.max_points}",
        f"point_budget_policy={point_budget_result.policy}",
        f"point_budget_input_points={point_budget_result.input_points}",
        f"point_budget_output_points={point_budget_result.output_points}",
        f"point_budget_applied={str(point_budget_result.applied).lower()}",
        f"point_budget_quantization_bits={point_budget_result.spatial_quantization_bits or 0}",
        f"point_budget_occupied_codes={point_budget_result.occupied_spatial_codes or 0}",
        f"factorial_output_count={len(factorial_summaries) if args.fusion_mode == 'points' else 0}",
        f"point_budget_sensitivity_output_count={len(sensitivity_summaries) if args.fusion_mode == 'points' else 0}",
        f"support_point_diagnostics={args.support_diagnostics_output if args.support_diagnostics_output is not None else ''}",
        f"support_point_diagnostics_bytes={support_diagnostics_summary['file_bytes'] if support_diagnostics_summary is not None else 0}",
        f"support_point_diagnostics_write_seconds={support_diagnostics_summary['write_seconds'] if support_diagnostics_summary is not None else 0.0:.3f}",
        f"colmap_seconds={colmap_seconds:.3f}",
        f"vggt_seconds={vggt_seconds:.3f}",
        f"gpu_peak_memory_bytes={gpu_peak_memory_bytes}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        *colmap_logs,
    ]
    if tsdf_params is not None:
        log_lines.extend(
            [
                f"tsdf_voxel_length={float(tsdf_params['voxel_length']):.8f}",
                f"tsdf_sdf_trunc={float(tsdf_params['sdf_trunc']):.8f}",
                f"tsdf_depth_trunc={float(tsdf_params['depth_trunc']):.6f}",
                f"tsdf_full_sparse_diagonal={float(tsdf_params['full_sparse_diagonal']):.6f}",
                f"tsdf_robust_sparse_diagonal={float(tsdf_params['robust_sparse_diagonal']):.6f}",
                f"integrated_frames={int(tsdf_params['integrated_frames'])}",
            ]
        )
    (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {geometry_dir / 'points.ply'}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
    print(f"wrote {fusion_diagnostics_path}")
    print(f"wrote {visibility_graph_path}")
    print(f"wrote {vggt_groups_path}")
    print(f"wrote {scale_disagreement_path}")
    print(f"wrote {consistency_path}")
    print(f"registered_images={len(registered_paths)}")
    print(f"scaled_images={len(scale_estimates)}")
    print(f"num_points={len(flat_points)}")


def run_colmap_pipeline(
    *,
    colmap: str,
    image_dir: Path,
    database_path: Path,
    sparse_dir: Path,
    text_dir: Path,
    pairing: ResolvedColmapPairing,
    feature_profile: ResolvedColmapFeatureProfile,
    local_matcher: ResolvedColmapLocalMatcher,
    single_camera: bool,
    mapper_abs_pose_min_num_inliers: int,
    mapper_abs_pose_min_inlier_ratio: float,
) -> list[str]:
    commands = [
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            str(int(single_camera)),
            "--FeatureExtraction.use_gpu",
            "1",
            "--FeatureExtraction.gpu_index",
            "0",
            *feature_profile.extraction_options,
        ],
        [
            colmap,
            pairing.command,
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            "1",
            "--FeatureMatching.gpu_index",
            "0",
            *local_matcher.matching_options,
            *pairing.pairing_options,
        ],
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.abs_pose_min_num_inliers",
            str(mapper_abs_pose_min_num_inliers),
            "--Mapper.abs_pose_min_inlier_ratio",
            str(mapper_abs_pose_min_inlier_ratio),
        ],
    ]
    command_logs: list[str] = []
    for stage, command in zip(
        ("feature_extraction", "feature_matching", "mapping"),
        commands,
        strict=True,
    ):
        stage_started_at = time.perf_counter()
        command_logs.append(run_command(command))
        command_logs.append(
            f"colmap_{stage}_seconds={time.perf_counter() - stage_started_at:.3f}"
        )
    model_dir, selection_logs = convert_best_sparse_model(colmap, sparse_dir, text_dir)
    command_logs.extend(selection_logs)
    command_logs.append(f"selected_model={model_dir}")
    return command_logs


def convert_best_sparse_model(colmap: str, sparse_dir: Path, text_dir: Path) -> tuple[Path, list[str]]:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse models")

    logs: list[str] = []
    best_model: Path | None = None
    best_count = -1
    for candidate in candidates:
        candidate_text_dir = sparse_dir.parent / f"sparse_txt_{candidate.name}"
        candidate_text_dir.mkdir(parents=True, exist_ok=True)
        logs.append(
            run_command(
                [
                    colmap,
                    "model_converter",
                    "--input_path",
                    str(candidate),
                    "--output_path",
                    str(candidate_text_dir),
                    "--output_type",
                    "TXT",
                ]
            )
        )
        registered_images = count_colmap_text_images(candidate_text_dir / "images.txt")
        logs.append(f"candidate_model={candidate} registered_images={registered_images}")
        if registered_images > best_count:
            best_model = candidate
            best_count = registered_images

    if best_model is None:
        raise RuntimeError("Could not select a COLMAP sparse model")
    text_dir.mkdir(parents=True, exist_ok=True)
    logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(best_model),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )
    return best_model, logs


def count_colmap_text_images(path: Path) -> int:
    data_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return len(data_lines) // 2


def vggt_window_member_role(
    *,
    position: int,
    selection_record: Mapping[str, Any] | None,
) -> str:
    if position == 0:
        return "reference"
    if selection_record is None:
        return "fresh"
    overlap_end = 1 + int(selection_record["selected_overlap_size"])
    fresh_end = overlap_end + int(selection_record["selected_fresh_size"])
    if position < overlap_end:
        return "overlap"
    if position < fresh_end:
        return "fresh"
    return "fallback"


def run_vggt_depth_batches(
    *,
    model: torch.nn.Module,
    groups: list[list[Path]],
    load_and_preprocess_images: Any,
    pose_encoding_to_extri_intri: Any,
    device: str,
    dtype: torch.dtype,
    frames_chunk_size: int | None = None,
    capture_dir: Path | None = None,
    selection_records: list[dict[str, Any]] | None = None,
    registered_by_name: Mapping[str, ColmapImage] | None = None,
    points3d: dict[int, np.ndarray] | None = None,
    min_scale_observations: int = 20,
    retain_point_diagnostics: bool = False,
) -> tuple[dict[Path, dict[str, Any]], VggtWindowPredictionCapture | None]:
    if (capture_dir is not None or retain_point_diagnostics) and (
        registered_by_name is None or points3d is None
    ):
        raise ValueError(
            "capturing VGGT diagnostics requires COLMAP images and points"
        )
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=False)

    selection_by_index = {
        int(record["group_index"]): record for record in selection_records or []
    }
    results: dict[Path, dict[str, Any]] = {}
    overlap_disagreement_by_path: dict[Path, np.ndarray] = {}
    capture_records: list[dict[str, Any]] = []
    capture_write_seconds = 0.0
    capture_bytes = 0
    for group_index, batch_paths in enumerate(groups):
        if not batch_paths:
            continue
        prediction_np, _seconds = infer_image_group(
            model=model,
            image_paths=batch_paths,
            load_and_preprocess_images=load_and_preprocess_images,
            pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
            device=device,
            dtype=dtype,
            frames_chunk_size=frames_chunk_size,
        )
        colors = load_padded_rgb_images(batch_paths, prediction_np["images"].shape[-2:])
        image_shape = tuple(int(value) for value in prediction_np["images"].shape[-2:])
        selection_record = selection_by_index.get(group_index)
        for position, image_path in enumerate(batch_paths):
            depth = np.squeeze(prediction_np["depth"][position]).astype(np.float32)
            confidence = np.squeeze(prediction_np["depth_conf"][position]).astype(np.float32)
            intrinsic = prediction_np["intrinsic"][position].astype(np.float32)
            original_size = read_image_size(image_path)
            first_prediction = image_path not in results
            if retain_point_diagnostics:
                scale_estimate = estimate_depth_scale(
                    colmap_image=registered_by_name[image_path.name],
                    points3d=points3d,
                    depth=depth,
                    image_shape=image_shape,
                    original_size=original_size,
                    min_observations=min_scale_observations,
                )
            else:
                scale_estimate = None

            if capture_dir is not None:
                colmap_image = registered_by_name[image_path.name]
                transform = build_vggt_image_transform(original_size, image_shape)
                if scale_estimate is None:
                    scale_estimate = estimate_depth_scale(
                        colmap_image=colmap_image,
                        points3d=points3d,
                        depth=depth,
                        image_shape=image_shape,
                        original_size=original_size,
                        min_observations=min_scale_observations,
                    )
                filename = (
                    f"group_{group_index:04d}_position_{position:02d}_"
                    f"image_{colmap_image.image_id}.npz"
                )
                prediction_path = capture_dir / filename
                write_started_at = time.perf_counter()
                np.savez_compressed(
                    prediction_path,
                    depth=depth,
                    confidence=confidence,
                    intrinsic=intrinsic,
                )
                capture_write_seconds += time.perf_counter() - write_started_at
                file_bytes = prediction_path.stat().st_size
                capture_bytes += file_bytes
                capture_records.append(
                    {
                        "image": image_path.name,
                        "image_path": image_path.as_posix(),
                        "image_id": colmap_image.image_id,
                        "group_index": group_index,
                        "group_position": position,
                        "role": vggt_window_member_role(
                            position=position,
                            selection_record=selection_record,
                        ),
                        "group_selection": (
                            {
                                key: selection_record[key]
                                for key in (
                                    "reference",
                                    "reference_selection",
                                    "requested_overlap_size",
                                    "selected_overlap_size",
                                    "selected_fresh_size",
                                    "selected_fallback_size",
                                    "fallback",
                                )
                            }
                            if selection_record is not None
                            else None
                        ),
                        "selected_for_first_wins": first_prediction,
                        "prediction_file": filename,
                        "file_bytes": file_bytes,
                        "depth_dtype": str(depth.dtype),
                        "confidence_dtype": str(confidence.dtype),
                        "intrinsic": intrinsic.tolist(),
                        "image_shape": list(image_shape),
                        "original_size": list(original_size),
                        "canvas_transform": {
                            "scale_x": transform.scale_x,
                            "scale_y": transform.scale_y,
                            "pad_left": transform.pad_left,
                            "pad_top": transform.pad_top,
                            "pad_right": image_shape[1] - transform.pad_left - transform.resized_width,
                            "pad_bottom": image_shape[0] - transform.pad_top - transform.resized_height,
                            "resized_width": transform.resized_width,
                            "resized_height": transform.resized_height,
                        },
                        "sparse_scale_anchor": (
                            {
                                "scale": scale_estimate.scale,
                                "observation_count": scale_estimate.observation_count,
                                "log_mad": scale_estimate.log_mad,
                            }
                            if scale_estimate is not None
                            else None
                        ),
                    }
                )

            if not first_prediction:
                if retain_point_diagnostics:
                    first_item = results[image_path]
                    if scale_estimate is not None and first_item["source_scale_estimate"] is not None:
                        first_scaled = first_item["depth"] * first_item["source_scale_estimate"].scale
                        current_scaled = depth * scale_estimate.scale
                        valid = (
                            np.isfinite(first_scaled)
                            & (first_scaled > 0)
                            & np.isfinite(current_scaled)
                            & (current_scaled > 0)
                        )
                        disagreement = np.full(depth.shape, np.nan, dtype=np.float32)
                        disagreement[valid] = np.abs(
                            np.log(first_scaled[valid]) - np.log(current_scaled[valid])
                        )
                        previous = overlap_disagreement_by_path.get(image_path)
                        overlap_disagreement_by_path[image_path] = (
                            disagreement
                            if previous is None
                            else np.fmax(previous, disagreement)
                        )
                continue
            results[image_path] = {
                "depth": depth,
                "confidence": confidence,
                "intrinsic": intrinsic,
                "colors": colors[position],
                "image_shape": image_shape,
                "original_size": original_size,
                "source_group_index": group_index,
                "source_group_position": position,
                "source_window_role": vggt_window_member_role(
                    position=position,
                    selection_record=selection_record,
                ),
                "source_scale_estimate": scale_estimate,
            }
        if device == "cuda":
            torch.cuda.empty_cache()

    if retain_point_diagnostics:
        for image_path, item in results.items():
            item["overlap_disagreement"] = overlap_disagreement_by_path.get(image_path)

    capture = None
    if capture_dir is not None:
        capture = VggtWindowPredictionCapture(
            records=capture_records,
            write_seconds=capture_write_seconds,
            bytes_written=capture_bytes,
        )
    return results, capture


def estimate_depth_scale(
    *,
    colmap_image: ColmapImage,
    points3d: dict[int, np.ndarray],
    depth: np.ndarray,
    image_shape: tuple[int, int],
    original_size: tuple[int, int],
    min_observations: int,
) -> DepthScaleEstimate | None:
    ratios: list[float] = []
    rotation = qvec_to_rotmat(colmap_image.qvec)
    for x, y, point3d_id in colmap_image.observations:
        point = points3d.get(point3d_id)
        if point is None:
            continue
        colmap_depth = float((rotation @ point + colmap_image.tvec)[2])
        if colmap_depth <= 0:
            continue
        u, v = map_original_pixel_to_vggt(float(x), float(y), original_size, image_shape)
        ui = int(round(u))
        vi = int(round(v))
        if vi < 0 or ui < 0 or vi >= depth.shape[0] or ui >= depth.shape[1]:
            continue
        vggt_depth = float(depth[vi, ui])
        if np.isfinite(vggt_depth) and vggt_depth > 1e-6:
            ratios.append(colmap_depth / vggt_depth)
    if len(ratios) < min_observations:
        return None
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    median_log_ratio = float(np.median(log_ratios))
    return DepthScaleEstimate(
        scale=float(np.exp(median_log_ratio)),
        observation_count=len(ratios),
        log_mad=float(np.median(np.abs(log_ratios - median_log_ratio))),
    )


def optimize_depth_scale_graph(
    *,
    frames: list[FusionFrame],
    covisibility_graph: dict[int, list[CovisibilityEdge]],
    scale_estimates: dict[str, DepthScaleEstimate],
    fallback_scale: float,
    confidence_threshold: float,
    relative_threshold: float,
    iterations: int,
    pair_weight: float,
    huber_delta: float,
    max_pairs_per_edge: int,
) -> DepthScaleGraphResult:
    frame_by_id = {frame.colmap_image.image_id: frame for frame in frames}
    image_ids = sorted(frame_by_id)
    index_by_id = {image_id: index for index, image_id in enumerate(image_ids)}
    initial_scales = {
        frame.colmap_image.image_id: frame.scale
        for frame in frames
    }
    components = scale_graph_components(image_ids, covisibility_graph)
    component_by_id = {
        image_id: component_index
        for component_index, component in enumerate(components)
        for image_id in component
    }
    anchored_ids = {
        frame.colmap_image.image_id
        for frame in frames
        if frame.image_path.name in scale_estimates
    }
    fallback_ids: set[int] = set()
    for component in components:
        if not (set(component) & anchored_ids):
            fallback_ids.update(component)

    log_scales = np.log(np.asarray([initial_scales[image_id] for image_id in image_ids], dtype=np.float64))
    objective_history: list[float] = []
    edge_records: list[dict[str, Any]] = []
    pair_constraint_count = 0
    for _iteration in range(iterations):
        residuals: list[float] = []
        jacobians: list[np.ndarray] = []
        weights: list[float] = []
        for component in components:
            if set(component) & anchored_ids:
                continue
            image_index = index_by_id[min(component)]
            residual = log_scales[image_index] - np.log(fallback_scale)
            row = np.zeros(len(image_ids), dtype=np.float64)
            row[image_index] = 1.0
            residuals.append(residual)
            jacobians.append(row)
            weights.append(1.0)
        for frame in frames:
            estimate = scale_estimates.get(frame.image_path.name)
            if estimate is None:
                continue
            image_index = index_by_id[frame.colmap_image.image_id]
            target = float(np.log(estimate.scale))
            residual = log_scales[image_index] - target
            row = np.zeros(len(image_ids), dtype=np.float64)
            row[image_index] = 1.0
            base_weight = min(4.0, 1.0 + np.log10(max(estimate.observation_count, 1)))
            base_weight /= 1.0 + 10.0 * max(estimate.log_mad, 0.0)
            residuals.append(residual)
            jacobians.append(row)
            weights.append(base_weight * huber_weight(residual, huber_delta))

        iteration_edges: list[dict[str, Any]] = []
        if pair_weight > 0:
            for source_id in image_ids:
                source = frame_by_id[source_id]
                source_index = index_by_id[source_id]
                for edge in covisibility_graph.get(source_id, []):
                    target = frame_by_id.get(edge.target_image_id)
                    if target is None:
                        continue
                    target_index = index_by_id[target.colmap_image.image_id]
                    samples = select_scale_graph_source_pixels(
                        source,
                        confidence_threshold=confidence_threshold,
                        max_samples=max_pairs_per_edge,
                    )
                    if len(samples) == 0:
                        continue
                    sample_u = samples[:, 0]
                    sample_v = samples[:, 1]
                    residual, valid, occluded = evaluate_scale_graph_pair(
                        source=source,
                        target=target,
                        source_log_scale=log_scales[source_index],
                        target_log_scale=log_scales[target_index],
                        u=sample_u,
                        v=sample_v,
                        relative_threshold=relative_threshold,
                        confidence_threshold=confidence_threshold,
                        detect_occlusions=_iteration > 0,
                    )
                    accepted = valid & ~occluded
                    edge_record = {
                        "source_image_id": source_id,
                        "target_image_id": target.colmap_image.image_id,
                        "shared_points": edge.shared_points,
                        "sample_count": int(len(samples)),
                        "valid_count": int(valid.sum()),
                        "occluded_count": int(occluded.sum()),
                        "accepted_count": int(accepted.sum()),
                    }
                    iteration_edges.append(edge_record)
                    if not accepted.any():
                        continue
                    epsilon = 1e-3
                    perturbed, perturbed_valid, perturbed_occluded = evaluate_scale_graph_pair(
                        source=source,
                        target=target,
                        source_log_scale=log_scales[source_index] + epsilon,
                        target_log_scale=log_scales[target_index],
                        u=sample_u,
                        v=sample_v,
                        relative_threshold=relative_threshold,
                        confidence_threshold=confidence_threshold,
                        detect_occlusions=_iteration > 0,
                    )
                    derivative_source = np.zeros(len(samples), dtype=np.float64)
                    derivative_mask = accepted & perturbed_valid & ~perturbed_occluded
                    derivative_source[derivative_mask] = (
                        perturbed[derivative_mask] - residual[derivative_mask]
                    ) / epsilon
                    edge_weight = pair_weight / max(int(accepted.sum()), 1)
                    for sample_index in np.flatnonzero(accepted):
                        row = np.zeros(len(image_ids), dtype=np.float64)
                        row[source_index] = derivative_source[sample_index]
                        row[target_index] = 1.0
                        sample_residual = float(residual[sample_index])
                        residuals.append(sample_residual)
                        jacobians.append(row)
                        weights.append(edge_weight * huber_weight(sample_residual, huber_delta))
        edge_records = iteration_edges
        pair_constraint_count = sum(record["accepted_count"] for record in edge_records)
        if not residuals:
            raise RuntimeError("Depth-scale graph has no constraints")
        residual_array = np.asarray(residuals, dtype=np.float64)
        jacobian_array = np.stack(jacobians)
        weight_array = np.asarray(weights, dtype=np.float64)
        objective_history.append(float(np.sum(weight_array * huber_loss(residual_array, huber_delta))))
        normal = jacobian_array.T @ (weight_array[:, None] * jacobian_array)
        gradient = jacobian_array.T @ (weight_array * residual_array)
        normal.flat[:: len(image_ids) + 1] += 1e-6
        delta = np.linalg.solve(normal, -gradient)
        log_scales += delta
        if float(np.max(np.abs(delta))) < 1e-5:
            break

    scales = {image_id: float(np.exp(log_scales[index_by_id[image_id]])) for image_id in image_ids}
    image_records = {
        image_id: {
            "image": frame_by_id[image_id].image_path.name,
            "component": component_by_id[image_id],
            "anchored": image_id in anchored_ids,
            "fallback": image_id in fallback_ids,
            "initial_scale": initial_scales[image_id],
            "optimized_scale": scales[image_id],
        }
        for image_id in image_ids
    }
    return DepthScaleGraphResult(
        scales=scales,
        image_records=image_records,
        component_count=len(components),
        anchored_image_count=len(anchored_ids),
        fallback_image_count=len(fallback_ids),
        pair_constraint_count=pair_constraint_count,
        objective_history=objective_history,
        edge_records=edge_records,
    )


def scale_graph_components(
    image_ids: list[int],
    graph: dict[int, list[CovisibilityEdge]],
) -> list[list[int]]:
    adjacent = {image_id: set() for image_id in image_ids}
    for source_id, edges in graph.items():
        for edge in edges:
            if source_id in adjacent and edge.target_image_id in adjacent:
                adjacent[source_id].add(edge.target_image_id)
                adjacent[edge.target_image_id].add(source_id)
    components: list[list[int]] = []
    unseen = set(image_ids)
    while unseen:
        pending = [min(unseen)]
        component: list[int] = []
        while pending:
            image_id = pending.pop()
            if image_id not in unseen:
                continue
            unseen.remove(image_id)
            component.append(image_id)
            pending.extend(sorted(adjacent[image_id] & unseen, reverse=True))
        components.append(sorted(component))
    return components


def select_scale_graph_source_pixels(
    frame: FusionFrame,
    *,
    confidence_threshold: float,
    max_samples: int,
) -> np.ndarray:
    valid = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
    valid &= np.isfinite(frame.depth) & (frame.depth > 1e-6)
    valid &= np.isfinite(frame.confidence) & (frame.confidence >= confidence_threshold)
    flat_indices = np.flatnonzero(valid)
    if len(flat_indices) == 0:
        return np.empty((0, 2), dtype=np.float32)
    selection = flat_indices[np.linspace(0, len(flat_indices) - 1, min(len(flat_indices), max_samples), dtype=np.int64)]
    height, width = frame.depth.shape
    return np.stack([selection % width, selection // width], axis=1).astype(np.float32)


def evaluate_scale_graph_pair(
    *,
    source: FusionFrame,
    target: FusionFrame,
    source_log_scale: float,
    target_log_scale: float,
    u: np.ndarray,
    v: np.ndarray,
    relative_threshold: float,
    confidence_threshold: float,
    detect_occlusions: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_depth = source.depth[v.astype(np.int32), u.astype(np.int32)] * np.exp(source_log_scale)
    world_points = unproject_depth_pixels_with_colmap_pose(
        depth=source_depth,
        u=u,
        v=v,
        camera=source.camera,
        qvec=source.colmap_image.qvec,
        tvec=source.colmap_image.tvec,
    )
    target_u, target_v, projected_depth = project_world_points_to_depth_canvas(
        world_points,
        camera=target.camera,
        qvec=target.colmap_image.qvec,
        tvec=target.colmap_image.tvec,
    )
    observed_depth, in_bounds = sample_bilinear(target.depth, target_u, target_v)
    observed_confidence, _ = sample_bilinear(target.confidence, target_u, target_v)
    valid = in_bounds & valid_depth_canvas_coordinates(target_u, target_v, target.original_size, target.image_shape)
    valid &= np.isfinite(projected_depth) & (projected_depth > 1e-6)
    valid &= np.isfinite(observed_depth) & (observed_depth > 1e-6)
    valid &= np.isfinite(observed_confidence) & (observed_confidence >= confidence_threshold)
    scaled_observed = observed_depth * np.exp(target_log_scale)
    occluded = valid & (scaled_observed < projected_depth * (1 - relative_threshold)) if detect_occlusions else np.zeros(len(valid), dtype=bool)
    residual = np.log(np.maximum(scaled_observed, 1e-6)) - np.log(np.maximum(projected_depth, 1e-6))
    return residual.astype(np.float64), valid, occluded


def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residuals)
    return np.where(absolute <= delta, 0.5 * residuals * residuals, delta * (absolute - 0.5 * delta))


def huber_weight(residual: float, delta: float) -> float:
    absolute = abs(residual)
    return 1.0 if absolute <= delta else delta / absolute


def build_fusion_camera(
    *,
    colmap_camera: dict[str, Any],
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> FusionCamera:
    source_width = int(colmap_camera["width"])
    source_height = int(colmap_camera["height"])
    if (source_width, source_height) != original_size:
        raise RuntimeError(
            "COLMAP camera size does not match the input image for dense fusion: "
            f"camera={source_width}x{source_height}, image={original_size[0]}x{original_size[1]}"
        )

    model = str(colmap_camera["model"])
    params = [float(value) for value in colmap_camera["params"]]
    if model == "SIMPLE_PINHOLE" and len(params) == 3:
        fx = fy = params[0]
        cx, cy = params[1:]
        radial_distortion: tuple[float, ...] = ()
    elif model == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        radial_distortion = ()
    elif model == "SIMPLE_RADIAL" and len(params) == 4:
        fx = fy = params[0]
        cx, cy = params[1:3]
        radial_distortion = (params[3],)
    elif model == "RADIAL" and len(params) == 5:
        fx = fy = params[0]
        cx, cy = params[1:3]
        radial_distortion = (params[3], params[4])
    else:
        raise RuntimeError(
            f"Unsupported COLMAP camera model '{model}' for dense fusion. "
            "Supported models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL."
        )

    transform = build_vggt_image_transform(original_size, image_shape)
    intrinsic = np.array(
        [
            [fx * transform.scale_x, 0.0, cx * transform.scale_x + transform.pad_left],
            [0.0, fy * transform.scale_y, cy * transform.scale_y + transform.pad_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return FusionCamera(model=model, intrinsic=intrinsic, radial_distortion=radial_distortion)


def build_fusion_camera_record(
    *,
    image_name: str,
    camera_id: int,
    camera: FusionCamera,
    vggt_intrinsic: np.ndarray,
    scale_estimate: DepthScaleEstimate | None,
    fallback_scale: float,
) -> dict[str, Any]:
    vggt_fx = float(vggt_intrinsic[0, 0])
    vggt_fy = float(vggt_intrinsic[1, 1])
    return {
        "image": image_name,
        "camera_id": camera_id,
        "colmap_model": camera.model,
        "fusion_intrinsic": camera.intrinsic.tolist(),
        "radial_distortion": list(camera.radial_distortion),
        "vggt_intrinsic": vggt_intrinsic.tolist(),
        "vggt_to_colmap_focal_ratio": {
            "fx": vggt_fx / float(camera.intrinsic[0, 0]),
            "fy": vggt_fy / float(camera.intrinsic[1, 1]),
        },
        "depth_scale": scale_estimate.scale if scale_estimate is not None else fallback_scale,
        "scale_observations": scale_estimate.observation_count if scale_estimate is not None else 0,
        "scale_log_mad": scale_estimate.log_mad if scale_estimate is not None else None,
        "scale_source": "sparse_colmap" if scale_estimate is not None else "median_fallback",
    }


def unproject_depth_with_colmap_pose(
    *,
    depth: np.ndarray,
    camera: FusionCamera,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    world_points = unproject_depth_pixels_with_colmap_pose(
        depth=depth.reshape(-1),
        u=xx.reshape(-1),
        v=yy.reshape(-1),
        camera=camera,
        qvec=qvec,
        tvec=tvec,
    )
    return world_points.reshape(height, width, 3).astype(np.float32)


def unproject_depth_pixels_with_colmap_pose(
    *,
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    camera: FusionCamera,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    fx = camera.intrinsic[0, 0]
    fy = camera.intrinsic[1, 1]
    cx = camera.intrinsic[0, 2]
    cy = camera.intrinsic[1, 2]
    z = depth.astype(np.float32)
    normalized_x, normalized_y = undistort_radial_coordinates(
        (u.astype(np.float32) - cx) / fx,
        (v.astype(np.float32) - cy) / fy,
        camera.radial_distortion,
    )
    camera_points = np.stack([normalized_x * z, normalized_y * z, z], axis=-1)
    rotation = qvec_to_rotmat(qvec).astype(np.float32)
    world_points = (camera_points.reshape(-1, 3) - tvec.astype(np.float32)) @ rotation
    return world_points.astype(np.float32)


def map_original_pixel_to_vggt(
    x: float,
    y: float,
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    transform = build_vggt_image_transform(original_size, image_shape)
    return x * transform.scale_x + transform.pad_left, y * transform.scale_y + transform.pad_top


def build_vggt_image_transform(
    original_size: tuple[int, int], image_shape: tuple[int, int]
) -> VggtImageTransform:
    original_width, original_height = original_size
    target_height, target_width = image_shape
    target_size = max(target_height, target_width)
    if original_width >= original_height:
        new_width = target_size
        new_height = round(original_height * (new_width / original_width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(original_width * (new_height / original_height) / 14) * 14
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    return VggtImageTransform(
        scale_x=new_width / original_width,
        scale_y=new_height / original_height,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=new_width,
        resized_height=new_height,
    )


def undistort_radial_coordinates(
    distorted_x: np.ndarray,
    distorted_y: np.ndarray,
    radial_distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not radial_distortion:
        return distorted_x, distorted_y

    undistorted_x = distorted_x.copy()
    undistorted_y = distorted_y.copy()
    for _ in range(8):
        radius_squared = undistorted_x * undistorted_x + undistorted_y * undistorted_y
        radial = np.ones_like(radius_squared)
        radius_power = radius_squared
        for coefficient in radial_distortion:
            radial += coefficient * radius_power
            radius_power *= radius_squared
        undistorted_x = distorted_x / radial
        undistorted_y = distorted_y / radial
    return undistorted_x, undistorted_y


def derive_tsdf_parameters(
    points3d: dict[int, np.ndarray],
    frames: list[FusionFrame],
    *,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
) -> TsdfParameters:
    """Resolve TSDF parameters in COLMAP's arbitrary world units.

    Auto voxel sizing uses a percentile-clipped sparse extent so a few bad
    COLMAP points cannot make the output hundreds of times sparser. Other
    non-positive arguments are derived from that voxel and the depth maps.
    """
    full_diagonal = 0.0
    robust_diagonal = 0.0
    if points3d:
        coords = np.stack(list(points3d.values()))
        full_extent = coords.max(axis=0) - coords.min(axis=0)
        full_diagonal = float(np.linalg.norm(full_extent))
        robust_min = np.percentile(coords, 0.5, axis=0)
        robust_max = np.percentile(coords, 99.5, axis=0)
        robust_diagonal = float(np.linalg.norm(robust_max - robust_min))
    scene_diagonal = robust_diagonal if robust_diagonal > 0 else full_diagonal
    if scene_diagonal <= 0:
        scene_diagonal = 1.0

    if voxel_length <= 0:
        voxel_length = max(scene_diagonal / 1024.0, 1e-6)
    if sdf_trunc <= 0:
        sdf_trunc = 5.0 * voxel_length
    if depth_trunc <= 0:
        far_samples: list[float] = []
        for frame in frames:
            valid = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
            valid &= np.isfinite(frame.depth) & (frame.depth > 0)
            if valid.any():
                far_samples.append(float(np.percentile(frame.depth[valid] * frame.scale, 99)))
        depth_trunc = 1.5 * (float(np.median(far_samples)) if far_samples else scene_diagonal)
    return TsdfParameters(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        full_diagonal=full_diagonal,
        robust_diagonal=robust_diagonal,
    )


def undistort_to_pinhole(
    depth: np.ndarray, color: np.ndarray, camera: FusionCamera
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-remap a distorted depth/color pair onto the linear pinhole model.

    Returns the inputs unchanged for pinhole cameras. For radial models it uses
    the existing analytic distortion to find each pinhole pixel's source pixel
    and samples nearest-neighbour (avoids blending across depth discontinuities),
    leaving the ``fx, fy, cx, cy`` intrinsic distortion-free for TSDF integrate.
    """
    if not camera.radial_distortion:
        return depth, color

    height, width = depth.shape
    fx = float(camera.intrinsic[0, 0])
    fy = float(camera.intrinsic[1, 1])
    cx = float(camera.intrinsic[0, 2])
    cy = float(camera.intrinsic[1, 2])
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    normalized_x = (xx.astype(np.float32) - cx) / fx
    normalized_y = (yy.astype(np.float32) - cy) / fy
    distorted_x, distorted_y = distort_radial_coordinates(
        normalized_x, normalized_y, camera.radial_distortion
    )
    source_u = np.round(distorted_x * fx + cx).astype(np.int64)
    source_v = np.round(distorted_y * fy + cy).astype(np.int64)
    in_bounds = (source_u >= 0) & (source_u < width) & (source_v >= 0) & (source_v < height)
    clipped_u = np.clip(source_u, 0, width - 1)
    clipped_v = np.clip(source_v, 0, height - 1)

    out_depth = np.where(in_bounds, depth[clipped_v, clipped_u], 0.0).astype(np.float32)
    sampled_color = color[clipped_v, clipped_u]
    out_color = np.zeros_like(color)
    out_color[in_bounds] = sampled_color[in_bounds]
    return out_depth, out_color


def fuse_frames_tsdf(
    frames: list[FusionFrame],
    *,
    confidence_percentile: float,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse per-frame VGGT depth into one TSDF volume in COLMAP's global frame.

    Confidence is thresholded per frame because independently inferred VGGT
    windows do not share a calibrated absolute confidence scale.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    integrated_valid_pixels = 0
    confidence_thresholds: list[float] = []
    for frame in frames:
        metric_depth = frame.depth.astype(np.float32) * float(frame.scale)
        valid_canvas = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
        confidence_valid = valid_canvas & np.isfinite(frame.confidence) & (frame.confidence > 0)
        if not confidence_valid.any():
            continue
        confidence_threshold = float(np.percentile(frame.confidence[confidence_valid], confidence_percentile))
        valid = valid_canvas & np.isfinite(metric_depth) & (metric_depth > 0)
        valid &= np.isfinite(frame.confidence) & (frame.confidence >= confidence_threshold)
        if not valid.any():
            continue
        metric_depth = np.where(valid, metric_depth, 0.0).astype(np.float32)
        depth_map, color_map = undistort_to_pinhole(
            metric_depth, np.ascontiguousarray(frame.colors, dtype=np.uint8), frame.camera
        )

        height, width = depth_map.shape
        color_image = o3d.geometry.Image(np.ascontiguousarray(color_map, dtype=np.uint8))
        depth_image = o3d.geometry.Image(np.ascontiguousarray(depth_map, dtype=np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(frame.camera.intrinsic[0, 0]),
            float(frame.camera.intrinsic[1, 1]),
            float(frame.camera.intrinsic[0, 2]),
            float(frame.camera.intrinsic[1, 2]),
        )
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = qvec_to_rotmat(frame.colmap_image.qvec)
        extrinsic[:3, 3] = frame.colmap_image.tvec
        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1
        integrated_valid_pixels += int(valid.sum())
        confidence_thresholds.append(confidence_threshold)

    if integrated == 0:
        raise RuntimeError("TSDF fusion integrated no frames after confidence filtering")

    cloud = volume.extract_point_cloud()
    points = np.asarray(cloud.points, dtype=np.float32)
    raw_colors = np.asarray(cloud.colors)
    if raw_colors.size:
        colors = np.clip(np.round(raw_colors * 255.0), 0, 255).astype(np.uint8)
    else:
        colors = np.zeros((len(points), 3), dtype=np.uint8)
    stats = {
        "input_frames": len(frames),
        "integrated_frames": integrated,
        "integrated_valid_pixels": integrated_valid_pixels,
        "num_points": int(len(points)),
        "voxel_length": float(voxel_length),
        "sdf_trunc": float(sdf_trunc),
        "depth_trunc": float(depth_trunc),
        "confidence_threshold_min": float(np.min(confidence_thresholds)),
        "confidence_threshold_median": float(np.median(confidence_thresholds)),
        "confidence_threshold_max": float(np.max(confidence_thresholds)),
    }
    return points, colors, stats


def validate_tsdf_output(stats: dict[str, Any]) -> None:
    input_frames = int(stats["input_frames"])
    integrated_frames = int(stats["integrated_frames"])
    num_points = int(stats["num_points"])
    if integrated_frames < max(1, int(np.ceil(input_frames * 0.9))):
        raise RuntimeError(
            "TSDF fusion skipped too many frames: "
            f"integrated={integrated_frames}, input={input_frames}"
        )
    if num_points < max(10_000, integrated_frames * 500):
        raise RuntimeError(
            "TSDF fusion output is implausibly sparse: "
            f"points={num_points}, integrated_frames={integrated_frames}. "
            "Use --fusion-mode points or reduce --tsdf-voxel-length."
        )


def compute_confidence_thresholds(
    frames: list[FusionFrame], percentile: float, *, scope: str
) -> dict[int, float]:
    if scope == "per_frame":
        return compute_frame_confidence_thresholds(frames, percentile)
    global_threshold = compute_global_confidence_threshold(frames, percentile)
    return {
        frame.colmap_image.image_id: global_threshold
        for frame in frames
    }


def compute_frame_confidence_thresholds(
    frames: list[FusionFrame], percentile: float
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for frame in frames:
        valid_canvas = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
        valid = valid_canvas & np.isfinite(frame.confidence) & (frame.confidence > 0)
        if not valid.any():
            raise RuntimeError(
                f"No positive VGGT depth confidence values were available for {frame.image_path.name}"
            )
        thresholds[frame.colmap_image.image_id] = float(
            np.percentile(frame.confidence[valid], percentile)
        )
    return thresholds


def compute_global_confidence_threshold(frames: list[FusionFrame], percentile: float) -> float:
    confidence_parts: list[np.ndarray] = []
    for frame in frames:
        valid_canvas = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
        valid = valid_canvas & np.isfinite(frame.confidence) & (frame.confidence > 0)
        if valid.any():
            confidence_parts.append(frame.confidence[valid])
    if not confidence_parts:
        raise RuntimeError("No positive VGGT depth confidence values were available")
    return float(np.percentile(np.concatenate(confidence_parts), percentile))


def derive_consistency_relative_threshold(
    scale_estimates: Any,
    *,
    min_threshold: float,
    max_threshold: float,
) -> float:
    log_mads = np.asarray([estimate.log_mad for estimate in scale_estimates], dtype=np.float64)
    robust_threshold = float(np.expm1(2 * np.median(log_mads)))
    return float(np.clip(robust_threshold, min_threshold, max_threshold))


def build_visibility_graph_payload(
    frames: list[FusionFrame],
    graph: dict[int, list[CovisibilityEdge]],
) -> dict[str, Any]:
    frame_by_id = {frame.colmap_image.image_id: frame for frame in frames}
    return {
        "image_count": len(frames),
        "directed_edge_count": sum(len(edges) for edges in graph.values()),
        "images": [
            {
                "image": frame.image_path.name,
                "image_id": frame.colmap_image.image_id,
                "neighbors": [
                    {
                        "image": frame_by_id[edge.target_image_id].image_path.name,
                        "image_id": edge.target_image_id,
                        "shared_points": edge.shared_points,
                        "baseline": edge.baseline,
                    }
                    for edge in graph.get(frame.colmap_image.image_id, [])
                ],
            }
            for frame in frames
        ],
    }


def filter_points_by_cross_view_consistency(
    frames: list[FusionFrame],
    *,
    covisibility_graph: dict[int, list[CovisibilityEdge]],
    confidence_thresholds: Mapping[int, float],
    relative_threshold: float,
    support_policy: str = "any_support",
    stride: int,
    retain_point_diagnostics: bool = False,
) -> ConsistencyFilterResult:
    frame_by_id = {frame.colmap_image.image_id: frame for frame in frames}
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "source_image_index",
            "source_u",
            "source_v",
            "confidence",
            "visible_counts",
            "support_counts",
            "contradicted_counts",
            "occluded_counts",
            "not_observed_counts",
            "mean_relative_error",
            "overlap_disagreement",
        )
    }
    residual_samples: list[np.ndarray] = []
    image_records: list[dict[str, Any]] = []
    candidate_points = 0
    accepted_points = 0
    rejected_points = 0
    unverified_points = 0
    supported_points = 0
    occluded_only_points = 0
    not_observed_only_points = 0
    contradicted_only_points = 0
    supported_and_contradicted_points = 0
    multi_visible_points = 0
    policy_rejected_supported_points = 0

    for frame in frames:
        confidence_threshold = confidence_thresholds[frame.colmap_image.image_id]
        valid = valid_depth_canvas_mask(frame.original_size, frame.image_shape)
        valid &= np.isfinite(frame.depth) & (frame.depth > 0)
        valid &= np.isfinite(frame.confidence) & (frame.confidence >= confidence_threshold)
        if stride > 1:
            sampled = np.zeros_like(valid)
            sampled[::stride, ::stride] = True
            valid &= sampled

        v, u = np.nonzero(valid)
        source_depth = frame.depth[v, u] * frame.scale
        source_points = unproject_depth_pixels_with_colmap_pose(
            depth=source_depth,
            u=u,
            v=v,
            camera=frame.camera,
            qvec=frame.colmap_image.qvec,
            tvec=frame.colmap_image.tvec,
        )
        neighbors = [
            frame_by_id[edge.target_image_id]
            for edge in covisibility_graph.get(frame.colmap_image.image_id, [])
        ]
        validation = validate_cross_view_consistency(
            source_points,
            neighbors=neighbors,
            confidence_thresholds=confidence_thresholds,
            relative_threshold=relative_threshold,
            support_policy=support_policy,
        )
        accepted = validation.accepted
        supported = accepted & (validation.support_counts > 0)
        unverified = accepted & (validation.visible_counts == 0)
        occluded_only = (validation.visible_counts == 0) & (validation.occluded_counts > 0)
        not_observed_only = (validation.visible_counts == 0) & (validation.occluded_counts == 0)
        contradicted_only = (validation.support_counts == 0) & (
            validation.contradicted_counts > 0
        )
        supported_and_contradicted = (validation.support_counts > 0) & (
            validation.contradicted_counts > 0
        )
        rejected = ~accepted
        multi_visible = validation.visible_counts >= 2
        policy_rejected_supported = rejected & (validation.support_counts > 0)
        accepted_residuals = validation.mean_relative_error[supported]

        point_parts.append(source_points[accepted])
        color_parts.append(frame.colors[v[accepted], u[accepted]])
        if retain_point_diagnostics:
            overlap_disagreement = (
                frame.overlap_disagreement[v, u]
                if frame.overlap_disagreement is not None
                else np.full(len(u), np.nan, dtype=np.float32)
            )
            diagnostic_parts["source_image_index"].append(
                np.full(int(accepted.sum()), len(image_records), dtype=np.int32)
            )
            diagnostic_parts["source_u"].append(u[accepted].astype(np.uint16))
            diagnostic_parts["source_v"].append(v[accepted].astype(np.uint16))
            diagnostic_parts["confidence"].append(frame.confidence[v[accepted], u[accepted]].astype(np.float32))
            diagnostic_parts["visible_counts"].append(validation.visible_counts[accepted].astype(np.uint16))
            diagnostic_parts["support_counts"].append(validation.support_counts[accepted].astype(np.uint16))
            diagnostic_parts["contradicted_counts"].append(
                validation.contradicted_counts[accepted].astype(np.uint16)
            )
            diagnostic_parts["occluded_counts"].append(validation.occluded_counts[accepted].astype(np.uint16))
            diagnostic_parts["not_observed_counts"].append(
                validation.not_observed_counts[accepted].astype(np.uint16)
            )
            diagnostic_parts["mean_relative_error"].append(validation.mean_relative_error[accepted].astype(np.float32))
            diagnostic_parts["overlap_disagreement"].append(overlap_disagreement[accepted].astype(np.float32))
        residual_samples.append(diagnostic_sample(accepted_residuals, 1_000))
        candidate_points += len(source_points)
        accepted_points += int(accepted.sum())
        rejected_points += int(rejected.sum())
        unverified_points += int(unverified.sum())
        supported_points += int(supported.sum())
        occluded_only_points += int(occluded_only.sum())
        not_observed_only_points += int(not_observed_only.sum())
        contradicted_only_points += int(contradicted_only.sum())
        supported_and_contradicted_points += int(supported_and_contradicted.sum())
        multi_visible_points += int(multi_visible.sum())
        policy_rejected_supported_points += int(policy_rejected_supported.sum())
        image_records.append(
            {
                "image": frame.image_path.name,
                "confidence_threshold": confidence_threshold,
                "candidate_points": int(len(source_points)),
                "accepted_points": int(accepted.sum()),
                "rejected_points": int(rejected.sum()),
                "unverified_points": int(unverified.sum()),
                "supported_points": int(supported.sum()),
                "occluded_only_points": int(occluded_only.sum()),
                "not_observed_only_points": int(not_observed_only.sum()),
                "contradicted_only_points": int(contradicted_only.sum()),
                "supported_and_contradicted_points": int(
                    supported_and_contradicted.sum()
                ),
                "multi_visible_points": int(multi_visible.sum()),
                "policy_rejected_supported_points": int(policy_rejected_supported.sum()),
                "median_relative_error": percentile_or_zero(accepted_residuals, 50),
                "p90_relative_error": percentile_or_zero(accepted_residuals, 90),
            }
        )

    if not point_parts:
        raise RuntimeError("No dense points were available after confidence filtering")
    return ConsistencyFilterResult(
        points=np.concatenate(point_parts, axis=0).astype(np.float32),
        colors=np.concatenate(color_parts, axis=0).astype(np.uint8),
        candidate_points=candidate_points,
        accepted_points=accepted_points,
        rejected_points=rejected_points,
        unverified_points=unverified_points,
        supported_points=supported_points,
        occluded_only_points=occluded_only_points,
        not_observed_only_points=not_observed_only_points,
        contradicted_only_points=contradicted_only_points,
        supported_and_contradicted_points=supported_and_contradicted_points,
        residual_samples=np.concatenate(residual_samples) if residual_samples else np.empty(0, dtype=np.float32),
        image_records=image_records,
        multi_visible_points=multi_visible_points,
        policy_rejected_supported_points=policy_rejected_supported_points,
        point_diagnostics=(
            SupportPointDiagnostics(
                **{
                    name: np.concatenate(parts)
                    for name, parts in diagnostic_parts.items()
                }
            )
            if retain_point_diagnostics
            else None
        ),
    )


def validate_cross_view_consistency(
    source_points: np.ndarray,
    *,
    neighbors: list[FusionFrame],
    confidence_thresholds: Mapping[int, float],
    relative_threshold: float,
    support_policy: str = "any_support",
) -> CrossViewValidation:
    point_count = len(source_points)
    support_counts = np.zeros(point_count, dtype=np.int16)
    visible_counts = np.zeros(point_count, dtype=np.int16)
    contradicted_counts = np.zeros(point_count, dtype=np.int16)
    occluded_counts = np.zeros(point_count, dtype=np.int16)
    not_observed_counts = np.zeros(point_count, dtype=np.int16)
    residual_sums = np.zeros(point_count, dtype=np.float32)

    for neighbor in neighbors:
        u, v, projected_depth = project_world_points_to_depth_canvas(
            source_points,
            camera=neighbor.camera,
            qvec=neighbor.colmap_image.qvec,
            tvec=neighbor.colmap_image.tvec,
        )
        observed_depth, in_bounds = sample_bilinear(neighbor.depth, u, v)
        observed_confidence, _ = sample_bilinear(neighbor.confidence, u, v)
        valid = in_bounds & valid_depth_canvas_coordinates(u, v, neighbor.original_size, neighbor.image_shape)
        valid &= np.isfinite(projected_depth) & (projected_depth > 0)
        valid &= np.isfinite(observed_depth) & (observed_depth > 0)
        valid &= np.isfinite(observed_confidence) & (
            observed_confidence >= confidence_thresholds[neighbor.colmap_image.image_id]
        )
        observed_depth *= neighbor.scale
        relative_error = np.abs(observed_depth - projected_depth) / np.maximum(
            np.maximum(observed_depth, projected_depth), 1e-6
        )
        occluded = valid & (observed_depth < projected_depth * (1 - relative_threshold))
        visible = valid & ~occluded
        supported = visible & (relative_error <= relative_threshold)
        contradicted = visible & ~supported
        not_observed = ~valid
        support_counts += supported
        visible_counts += visible
        contradicted_counts += contradicted
        occluded_counts += occluded
        not_observed_counts += not_observed
        residual_sums += np.where(visible, relative_error, 0.0)

    accepted = apply_support_policy(
        support_counts,
        visible_counts,
        policy=support_policy,
    )
    mean_relative_error = np.divide(
        residual_sums,
        visible_counts,
        out=np.full(point_count, np.nan, dtype=np.float32),
        where=visible_counts > 0,
    )
    return CrossViewValidation(
        accepted=accepted,
        support_counts=support_counts,
        visible_counts=visible_counts,
        contradicted_counts=contradicted_counts,
        occluded_counts=occluded_counts,
        not_observed_counts=not_observed_counts,
        mean_relative_error=mean_relative_error,
    )


def apply_support_policy(
    support_counts: np.ndarray,
    visible_counts: np.ndarray,
    *,
    policy: str,
) -> np.ndarray:
    if policy == "any_support":
        return (support_counts > 0) | (visible_counts == 0)
    if policy == "adaptive_two":
        required_supports = np.minimum(visible_counts, 2)
        return support_counts >= required_supports
    if policy == "contradiction_free":
        return (visible_counts == 0) | (
            (support_counts > 0) & (support_counts == visible_counts)
        )
    raise ValueError(f"Unknown cross-view support policy: {policy}")


def project_world_points_to_depth_canvas(
    world_points: np.ndarray,
    *,
    camera: FusionCamera,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = qvec_to_rotmat(qvec).astype(np.float32)
    camera_points = world_points @ rotation.T + tvec.astype(np.float32)
    depth = camera_points[:, 2]
    safe_depth = np.where(depth > 1e-6, depth, 1.0)
    distorted_x, distorted_y = distort_radial_coordinates(
        camera_points[:, 0] / safe_depth,
        camera_points[:, 1] / safe_depth,
        camera.radial_distortion,
    )
    u = camera.intrinsic[0, 0] * distorted_x + camera.intrinsic[0, 2]
    v = camera.intrinsic[1, 1] * distorted_y + camera.intrinsic[1, 2]
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32)


def distort_radial_coordinates(
    undistorted_x: np.ndarray,
    undistorted_y: np.ndarray,
    radial_distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not radial_distortion:
        return undistorted_x, undistorted_y
    radius_squared = undistorted_x * undistorted_x + undistorted_y * undistorted_y
    radial = np.ones_like(radius_squared)
    radius_power = radius_squared
    for coefficient in radial_distortion:
        radial += coefficient * radius_power
        radius_power *= radius_squared
    return undistorted_x * radial, undistorted_y * radial


def sample_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    in_bounds = (u >= 0) & (v >= 0) & (u <= width - 1) & (v <= height - 1)
    clipped_u = np.clip(u, 0, width - 1)
    clipped_v = np.clip(v, 0, height - 1)
    u0 = np.floor(clipped_u).astype(np.int32)
    v0 = np.floor(clipped_v).astype(np.int32)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    du = clipped_u - u0
    dv = clipped_v - v0
    top = image[v0, u0] * (1 - du) + image[v0, u1] * du
    bottom = image[v1, u0] * (1 - du) + image[v1, u1] * du
    return (top * (1 - dv) + bottom * dv).astype(np.float32), in_bounds


def valid_depth_canvas_mask(
    original_size: tuple[int, int], image_shape: tuple[int, int]
) -> np.ndarray:
    transform = build_vggt_image_transform(original_size, image_shape)
    valid = np.zeros(image_shape, dtype=bool)
    valid[
        transform.pad_top : transform.pad_top + transform.resized_height,
        transform.pad_left : transform.pad_left + transform.resized_width,
    ] = True
    return valid


def valid_depth_canvas_coordinates(
    u: np.ndarray,
    v: np.ndarray,
    original_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    transform = build_vggt_image_transform(original_size, image_shape)
    return (
        (u >= transform.pad_left)
        & (u <= transform.pad_left + transform.resized_width - 1)
        & (v >= transform.pad_top)
        & (v <= transform.pad_top + transform.resized_height - 1)
    )


def factorial_arm_name(
    confidence_scope: str,
    support_policy: str,
    point_budget_policy: str,
) -> str:
    phase_parts = []
    if confidence_scope == "per_frame":
        phase_parts.append("phase1")
    if support_policy == "adaptive_two":
        phase_parts.append("phase2")
    if point_budget_policy == "spatial_balanced":
        phase_parts.append("phase3")
    return "_".join(phase_parts) if phase_parts else "baseline"


def point_budget_diagnostics(result: PointBudgetResult) -> dict[str, Any]:
    return {
        "policy": result.policy,
        "input_points": result.input_points,
        "output_points": result.output_points,
        "applied": result.applied,
        "spatial_quantization_bits": result.spatial_quantization_bits,
        "occupied_spatial_codes": result.occupied_spatial_codes,
    }


def apply_point_budget(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
    *,
    policy: str = "random",
) -> PointBudgetResult:
    if len(points) != len(colors):
        raise ValueError("Point-budget points and colors must have matching lengths")
    if policy not in {"random", "spatial_balanced"}:
        raise ValueError(f"Unknown point-budget policy: {policy}")
    input_points = len(points)
    if max_points <= 0 or input_points <= max_points:
        return PointBudgetResult(
            points=points,
            colors=colors,
            policy=policy,
            input_points=input_points,
            output_points=input_points,
            applied=False,
            spatial_quantization_bits=None,
            occupied_spatial_codes=None,
            selected_indices=None,
        )
    if policy == "random":
        rng = np.random.default_rng(seed)
        selected = rng.choice(input_points, size=max_points, replace=False)
        selected.sort()
        quantization_bits = None
        occupied_codes = None
    else:
        selected, quantization_bits, occupied_codes = morton_stratified_indices(
            points,
            max_points,
        )
    return PointBudgetResult(
        points=points[selected],
        colors=colors[selected],
        policy=policy,
        input_points=input_points,
        output_points=len(selected),
        applied=True,
        spatial_quantization_bits=quantization_bits,
        occupied_spatial_codes=occupied_codes,
        selected_indices=selected,
    )


def morton_stratified_indices(
    points: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, int, int]:
    """Select midpoints of equal-mass strata in deterministic Morton order."""
    if max_points <= 0 or len(points) <= max_points:
        raise ValueError("Spatial balancing requires 0 < max_points < len(points)")
    if not np.isfinite(points).all():
        raise ValueError("Point-budget input contains non-finite coordinates")

    quantization_bits = 21
    quantization_max = (1 << quantization_bits) - 1
    lower = points.min(axis=0).astype(np.float64)
    extent = points.max(axis=0).astype(np.float64) - lower
    side = float(np.max(extent))
    if side > 0:
        quantized = np.floor(
            (points.astype(np.float64) - lower) * (quantization_max / side)
        ).astype(np.uint64)
        np.minimum(quantized, quantization_max, out=quantized)
    else:
        quantized = np.zeros(points.shape, dtype=np.uint64)

    morton_codes = (
        spread_morton_bits(quantized[:, 0])
        | (spread_morton_bits(quantized[:, 1]) << np.uint64(1))
        | (spread_morton_bits(quantized[:, 2]) << np.uint64(2))
    )
    order = np.argsort(morton_codes, kind="stable")
    ranks = (
        (2 * np.arange(max_points, dtype=np.int64) + 1) * len(points)
        // (2 * max_points)
    )
    selected = np.sort(order[ranks].astype(np.int64))
    occupied_codes = int(np.count_nonzero(np.diff(morton_codes[order])) + 1)
    return selected, quantization_bits, occupied_codes


def spread_morton_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True) & np.uint64(0x1FFFFF)
    values = (values | (values << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    values = (values | (values << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    values = (values | (values << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    values = (values | (values << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    values = (values | (values << np.uint64(2))) & np.uint64(0x1249249249249249)
    return values


def cap_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
    *,
    policy: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    result = apply_point_budget(
        points,
        colors,
        max_points,
        seed,
        policy=policy,
    )
    return result.points, result.colors


def write_support_point_diagnostics(
    path: Path,
    *,
    diagnostics: SupportPointDiagnostics,
    selected_indices: np.ndarray | None,
    frames: list[FusionFrame],
    expected_point_count: int,
    source_index_path: Path | None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    arrays = {
        name: (
            np.asarray(getattr(diagnostics, name))
            if selected_indices is None
            else np.asarray(getattr(diagnostics, name))[selected_indices]
        )
        for name in SupportPointDiagnostics.__dataclass_fields__
    }
    point_count = expected_point_count
    if any(len(array) != point_count for array in arrays.values()):
        raise ValueError("support diagnostic attributes do not match final points")
    role_codes = {"unknown": 0, "reference": 1, "overlap": 2, "fresh": 3, "fallback": 4}
    image_records = []
    for index, frame in enumerate(frames):
        image_records.append(
            {
                "source_image_index": index,
                "image": frame.image_path.name,
                "image_id": frame.colmap_image.image_id,
                "group_index": frame.source_group_index,
                "group_position": frame.source_group_position,
                "window_role": frame.source_window_role,
                "window_role_code": role_codes.get(frame.source_window_role, 0),
                "scale": frame.scale,
                "scale_observations": frame.scale_observations,
                "scale_log_mad": frame.scale_log_mad if np.isfinite(frame.scale_log_mad) else None,
            }
        )
    source_indices = arrays["source_image_index"]
    visible_counts = arrays["visible_counts"]
    support_counts = arrays["support_counts"]
    contradicted_counts = arrays["contradicted_counts"]
    occluded_counts = arrays["occluded_counts"]
    if not np.array_equal(visible_counts, support_counts + contradicted_counts):
        raise ValueError("visible counts do not equal support plus contradiction counts")
    visibility_state_counts = {
        "support_strata": {
            "0": int(np.count_nonzero(support_counts == 0)),
            "1": int(np.count_nonzero(support_counts == 1)),
            "2_plus": int(np.count_nonzero(support_counts >= 2)),
        },
        "supported_points": int(np.count_nonzero(support_counts > 0)),
        "occluded_only_points": int(
            np.count_nonzero((visible_counts == 0) & (occluded_counts > 0))
        ),
        "not_observed_only_points": int(
            np.count_nonzero((visible_counts == 0) & (occluded_counts == 0))
        ),
        "contradicted_only_points": int(
            np.count_nonzero((support_counts == 0) & (contradicted_counts > 0))
        ),
        "supported_and_contradicted_points": int(
            np.count_nonzero((support_counts > 0) & (contradicted_counts > 0))
        ),
    }
    if source_indices.size and (
        int(source_indices.min()) < 0 or int(source_indices.max()) >= len(frames)
    ):
        raise ValueError("support diagnostic source image index is out of range")
    source_group_index = np.asarray(
        [frames[index].source_group_index for index in source_indices], dtype=np.int32
    )
    source_group_position = np.asarray(
        [frames[index].source_group_position for index in source_indices], dtype=np.int16
    )
    source_window_role = np.asarray(
        [role_codes.get(frames[index].source_window_role, 0) for index in source_indices],
        dtype=np.uint8,
    )
    scale_observations = np.asarray(
        [frames[index].scale_observations for index in source_indices], dtype=np.int32
    )
    scale_log_mad = np.asarray(
        [frames[index].scale_log_mad for index in source_indices], dtype=np.float32
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        source_group_index=source_group_index,
        source_group_position=source_group_position,
        source_window_role=source_window_role,
        scale_observations=scale_observations,
        scale_log_mad=scale_log_mad,
    )
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    file_bytes = path.stat().st_size
    write_seconds = time.perf_counter() - started_at
    index_path = path.with_suffix(".json")
    payload = {
        "schema_version": 1,
        "point_order": "exactly matches geometry/points.ply vertex order",
        "point_count": point_count,
        "sidecar": path.name,
        "sidecar_sha256": data_hash,
        "file_bytes": file_bytes,
        "write_seconds": write_seconds,
        "source_prediction_index": (
            source_index_path.as_posix() if source_index_path is not None else None
        ),
        "arrays": {
            "source_image_index": "int32[N] index into images",
            "source_u/source_v": "uint16[N] source VGGT canvas pixel",
            "confidence": "float32[N] source depth confidence",
            "visible_counts/support_counts/contradicted_counts/occluded_counts/not_observed_counts": (
                "uint16[N] cross-view counts; visible = support + contradicted and "
                "support + contradicted + occluded + not_observed = checked neighbors"
            ),
            "mean_relative_error": "float32[N], NaN when visible_count is zero",
            "scale_observations/scale_log_mad": "per-source sparse scale quality repeated per point",
            "source_group_index/source_group_position/source_window_role": "first-wins window provenance",
            "overlap_disagreement": "maximum sparse-anchored absolute log-depth disagreement against another retained prediction; NaN without overlap",
        },
        "window_role_codes": role_codes,
        "visibility_state_counts": visibility_state_counts,
        "images": image_records,
    }
    write_json(index_path, payload)
    return {
        "sidecar": path.as_posix(),
        "index": index_path.as_posix(),
        "point_count": point_count,
        "file_bytes": file_bytes,
        "write_seconds": write_seconds,
        "sha256": data_hash,
    }


def diagnostic_sample(values: np.ndarray, max_samples: int) -> np.ndarray:
    if len(values) <= max_samples:
        return values.astype(np.float32)
    stride = max(1, len(values) // max_samples)
    return values[::stride][:max_samples].astype(np.float32)


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def build_consistency_payload(
    filtered: ConsistencyFilterResult,
    *,
    confidence_thresholds: Mapping[int, float],
    confidence_percentile: float,
    confidence_threshold_scope: str = "per_frame",
    support_policy: str = "any_support",
    relative_threshold: float,
    stride: int,
) -> dict[str, Any]:
    threshold_values = np.asarray(list(confidence_thresholds.values()), dtype=np.float64)
    threshold_median = float(np.median(threshold_values))
    return {
        "confidence_threshold": threshold_median,
        "confidence_threshold_scope": confidence_threshold_scope,
        "confidence_percentile": confidence_percentile,
        "confidence_threshold_min": float(np.min(threshold_values)),
        "confidence_threshold_median": threshold_median,
        "confidence_threshold_max": float(np.max(threshold_values)),
        "support_policy": support_policy,
        "relative_threshold": relative_threshold,
        "visibility_state_semantics": {
            "supported": "reliable non-occluded neighbor depth within relative_threshold",
            "contradicted": "reliable non-occluded neighbor depth outside relative_threshold",
            "occluded": "reliable neighbor depth in front by more than relative_threshold",
            "not_observed": "neighbor supplied no reliable observation for the candidate",
        },
        "stride": stride,
        "candidate_points": filtered.candidate_points,
        "accepted_points": filtered.accepted_points,
        "rejected_points": filtered.rejected_points,
        "unverified_points": filtered.unverified_points,
        "supported_points": filtered.supported_points,
        "occluded_only_points": filtered.occluded_only_points,
        "not_observed_only_points": filtered.not_observed_only_points,
        "contradicted_only_points": filtered.contradicted_only_points,
        "supported_and_contradicted_points": filtered.supported_and_contradicted_points,
        "multi_visible_points": filtered.multi_visible_points,
        "policy_rejected_supported_points": filtered.policy_rejected_supported_points,
        "acceptance_rate": filtered.accepted_points / max(filtered.candidate_points, 1),
        "residual_p50": percentile_or_zero(filtered.residual_samples, 50),
        "residual_p90": percentile_or_zero(filtered.residual_samples, 90),
        "images": filtered.image_records,
    }


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


if __name__ == "__main__":
    main()
