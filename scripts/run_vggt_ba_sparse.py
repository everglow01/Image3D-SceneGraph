from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.geometry.camera_calibration import (
    build_camera_calibration_diagnostics,
    camera_calibration_metrics,
    prepare_camera_extraction,
    write_camera_calibration_diagnostics,
)
from image3d_scenegraph.geometry.colmap import (
    COLMAP_CAMERA_CALIBRATION_IDS,
    COLMAP_FEATURE_PROFILE_IDS,
    COLMAP_GEOMETRIC_VERIFICATION_IDS,
    COLMAP_LEGACY_MATCHER_IDS,
    COLMAP_LOCAL_MATCHER_IDS,
    COLMAP_PAIRING_IDS,
    ColmapFeatureError,
    colmap_frontend_provenance,
    resolve_colmap_camera_calibration,
    resolve_colmap_executable,
    resolve_colmap_feature_profile,
    resolve_colmap_geometric_verification,
    resolve_colmap_local_matcher,
    resolve_colmap_pairing,
)
from image3d_scenegraph.geometry.video_recovery import (
    recover_video_registration,
    sequential_overlap,
    v2_mapper_options,
)
from image3d_scenegraph.geometry.vggt_ba import (
    MIN_RELIABLE_CAMERAS,
    MIN_RELIABLE_CAMERA_RATE,
    MIN_SUPPORTED_OBSERVATIONS,
    MIN_TEMPORAL_COVERAGE,
    VGGT_BA_FALLBACK_REASONS,
    VggtBaError,
    bridge_windows,
    classify_frame_support,
    count_frame_inliers,
    estimate_window_edge,
    merge_window_cameras,
    optimize_window_graph,
    read_colmap_database_image_ids,
    recovery_windows,
    select_reliable_component,
    sequential_windows,
    supported_image_ids,
    write_initial_colmap_model,
    write_json,
)
from image3d_scenegraph.video.keyframes import V2_PROFILE_ID
from scripts.run_colmap_sparse import (
    build_camera_payload,
    colmap_version,
    find_largest_sparse_model,
    read_ply_vertex_count,
    read_sparse_model_counts,
)
from scripts.run_vggt_pointcloud import load_vggt_model, select_dtype


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROFILE_ID = "vggt_ba_standard_v1"
WINDOW_SIZE = 8
WINDOW_OVERLAP = 4
QUERY_FRAME_COUNT = 5
MAX_QUERY_POINTS = 2048
MAX_BRIDGES = 16


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a global COLMAP reconstruction from batched VGGT cameras and BA."
    )
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", type=Path, default=Path("external/vggt"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/vggt/facebook--VGGT-1B"),
    )
    parser.add_argument("--dinov2-repo", type=Path, default=Path("external/dinov2"))
    parser.add_argument("--lightglue-repo", type=Path, default=Path("external/lightglue"))
    parser.add_argument(
        "--dinov2-checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/dinov2_vitb14_reg4_pretrain.pth"),
    )
    parser.add_argument(
        "--aliked-checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/torch-hub/checkpoints/aliked-n16.pth"),
    )
    parser.add_argument(
        "--tracker-checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/vggsfm_v2_tracker.pt"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max-image-size", type=int, default=1280)
    parser.add_argument("--num-threads", type=int, default=8)
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
        "--geometric-verification",
        choices=COLMAP_GEOMETRIC_VERIFICATION_IDS,
        default="default_v1",
    )
    parser.add_argument(
        "--camera-calibration",
        choices=COLMAP_CAMERA_CALIBRATION_IDS,
        default="shared_opencv_v1",
    )
    parser.add_argument("--vocab-tree-path", type=Path)
    parser.add_argument("--video-source", type=Path)
    parser.add_argument("--video-selection", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.max_image_size < 1 or args.num_threads < 1:
        parser.error("--max-image-size and --num-threads must be positive")
    if args.pairing is not None and args.matcher is not None:
        parser.error("--pairing and legacy --matcher cannot be combined")
    if args.pairing is not None and args.vocab_tree_path is not None:
        parser.error("--pairing resolves its vocabulary tree; omit --vocab-tree-path")
    if args.pairing == "vocab_tree":
        parser.error("VGGT-BA video geometry does not support vocab_tree pairing")
    if args.camera_calibration == "auto_grouped_simple_radial_v1":
        parser.error("VGGT-BA video geometry does not support auto-grouped cameras")
    legacy_matcher = args.matcher
    if args.pairing is None and legacy_matcher is None:
        legacy_matcher = "exhaustive"
    if legacy_matcher == "sequential" and (
        args.vocab_tree_path is None or not args.vocab_tree_path.is_file()
    ):
        parser.error("--vocab-tree-path must be an existing file for sequential matching")
    if args.vocab_tree_path is not None and args.feature_profile != "sift_v1":
        parser.error(
            "the legacy --vocab-tree-path is SIFT-only; "
            "use --pairing with the descriptor-specific vocabulary tree"
        )
    if (args.video_source is None) != (args.video_selection is None):
        parser.error("--video-source and --video-selection must be provided together")
    video_selection: dict[str, Any] | None = None
    if args.video_source is not None and args.video_selection is not None:
        if not args.video_source.is_file():
            raise SystemExit(f"Video source is missing: {args.video_source}")
        try:
            video_selection = json.loads(
                args.video_selection.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit("Cannot read video selection metadata") from exc

    started_at = time.perf_counter()
    image_paths = discover_images(args.image_dir)
    if len(image_paths) < 12:
        raise SystemExit("VGGT-BA Gaussian geometry requires at least 12 images")
    validate_runtime(args)
    colmap_path = resolve_colmap_executable()
    if colmap_path is None:
        raise SystemExit("COLMAP executable not found")
    colmap = str(colmap_path)
    colmap_build = colmap_version(colmap)
    try:
        feature_profile = resolve_colmap_feature_profile(args.feature_profile)
        local_matcher = resolve_colmap_local_matcher(
            feature_profile, args.local_matcher
        )
        geometric_verification = resolve_colmap_geometric_verification(
            args.geometric_verification
        )
        camera_calibration = resolve_colmap_camera_calibration(
            args.camera_calibration
        )
        if args.pairing is not None:
            pairing = resolve_colmap_pairing(feature_profile, args.pairing)
            pairing_profile = pairing.profile_id
            pairing_command = pairing.command
            pairing_options = pairing.pairing_options
            vocab_tree_path = pairing.vocab_tree_path
            vocab_tree_sha256 = pairing.vocab_tree_sha256
        else:
            pairing_command = f"{legacy_matcher}_matcher"
            pairing_options = ()
            vocab_tree_path = args.vocab_tree_path
            vocab_tree_sha256 = (
                sha256_file(vocab_tree_path)
                if vocab_tree_path is not None
                else None
            )
            pairing_profile = (
                "sequential_loop"
                if legacy_matcher == "sequential"
                else str(legacy_matcher)
            )
    except ColmapFeatureError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = args.output_dir
    work_dir = output_dir / "vggt_ba"
    windows_dir = work_dir / "windows"
    diagnostics_dir = output_dir / "diagnostics"
    geometry_dir = output_dir / "geometry"
    colmap_dir = output_dir / "colmap"
    for path in (windows_dir, diagnostics_dir, geometry_dir, colmap_dir):
        path.mkdir(parents=True, exist_ok=True)
    camera_plan = prepare_camera_extraction(
        camera_calibration,
        args.image_dir,
        image_paths,
        work_dir / "camera_groups",
    )

    write_progress(args.progress_file, "vggt_ba_descriptors")
    runtime = load_runtime(args)
    descriptors = compute_dino_descriptors(
        image_paths,
        runtime["dino_model"],
        runtime["device"],
        runtime["torch"],
    )
    del runtime["dino_model"]
    if runtime["device"].type == "cuda":
        runtime["torch"].cuda.empty_cache()
    bases = sequential_windows(
        len(image_paths), window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP
    )
    bridges = bridge_windows(
        descriptors,
        bases,
        window_size=WINDOW_SIZE,
        minimum_index_gap=WINDOW_SIZE * 2,
        maximum_bridges=MAX_BRIDGES,
    )
    write_json(
        work_dir / "profile.json",
        {
            "profile": PROFILE_ID,
            "window_size": WINDOW_SIZE,
            "overlap": WINDOW_OVERLAP,
            "query_frame_num": QUERY_FRAME_COUNT,
            "max_query_points": MAX_QUERY_POINTS,
            "keypoint_extractor": "aliked",
            "tracker_keypoint_extractor": "aliked",
            "colmap_feature": colmap_frontend_provenance(
                feature_profile, local_matcher
            ),
            "colmap_pairing": pairing_profile,
            "colmap_pairing_command": pairing_command,
            "colmap_vocab_tree_sha256": vocab_tree_sha256,
            "seed": args.seed,
            "base_window_count": len(bases),
            "bridge_candidate_count": len(bridges),
        },
    )

    write_progress(args.progress_file, "vggt_ba_windows")
    windows: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    window_records: list[dict[str, Any]] = []
    processed_specs = []
    for spec in bases:
        cameras, record = process_window(
            spec,
            image_paths,
            descriptors,
            runtime,
            windows_dir / spec.window_id,
            args,
        )
        processed_specs.append(spec)
        window_records.append(record)
        if cameras:
            windows[spec.window_id] = cameras
        print(
            f"window={spec.window_id} kind={spec.kind} status={record['status']} "
            f"images={len(spec.image_indices)}",
            flush=True,
        )

    reliable_by_window = {
        window_id: set(cameras) for window_id, cameras in windows.items()
    }
    usable_bases = [
        spec for spec in bases if len(reliable_by_window.get(spec.window_id, set())) >= 3
    ]
    forced_recovery_pairs = set()
    for left, right in zip(usable_bases, usable_bases[1:], strict=False):
        left_cameras = windows[left.window_id]
        right_cameras = windows[right.window_id]
        if len(set(left_cameras) & set(right_cameras)) < 3:
            continue
        try:
            estimate_window_edge(
                left.window_id,
                right.window_id,
                {index: camera["extrinsic"] for index, camera in left_cameras.items()},
                {index: camera["extrinsic"] for index, camera in right_cameras.items()},
            )
        except VggtBaError:
            forced_recovery_pairs.add((left.window_id, right.window_id))
    recoveries = recovery_windows(
        bases,
        reliable_by_window,
        window_size=WINDOW_SIZE,
        forced_pairs=forced_recovery_pairs,
    )
    if recoveries:
        write_progress(args.progress_file, "vggt_ba_recovery")
    for spec in [*recoveries, *bridges]:
        cameras, record = process_window(
            spec,
            image_paths,
            descriptors,
            runtime,
            windows_dir / spec.window_id,
            args,
        )
        processed_specs.append(spec)
        window_records.append(record)
        if cameras:
            windows[spec.window_id] = cameras
        print(
            f"window={spec.window_id} kind={spec.kind} status={record['status']} "
            f"images={len(spec.image_indices)}",
            flush=True,
        )

    write_progress(args.progress_file, "vggt_ba_pose_graph")
    edges = []
    edge_rejections = []
    window_ids = list(windows)
    for left_index, source in enumerate(window_ids):
        for target in window_ids[left_index + 1 :]:
            if len(set(windows[source]) & set(windows[target])) < 3:
                continue
            try:
                edges.append(
                    estimate_window_edge(
                        source,
                        target,
                        {
                            index: camera["extrinsic"]
                            for index, camera in windows[source].items()
                        },
                        {
                            index: camera["extrinsic"]
                            for index, camera in windows[target].items()
                        },
                    )
                )
            except VggtBaError as exc:
                edge_rejections.append(
                    {"source": source, "target": target, "reason": str(exc)}
                )
    selected_ids, component_metrics = select_reliable_component(
        windows, edges, len(image_paths)
    )
    selected_windows = {
        window_id: windows[window_id] for window_id in selected_ids
    }
    selected_edges = [
        edge
        for edge in edges
        if edge.source in selected_windows and edge.target in selected_windows
    ]
    fallback_reason = None
    initial_record = None
    merged: dict[int, dict[str, np.ndarray]] = {}
    merge_metrics: dict[str, Any] = {"camera_count": 0}
    if selected_ids:
        try:
            transforms, graph_metrics = optimize_window_graph(
                selected_ids, selected_edges
            )
        except VggtBaError as exc:
            message = str(exc)
            if not (
                message.startswith("window pose graph optimization failed:")
                or message == "window pose graph residual increased"
            ):
                raise
            graph_metrics = {
                "connected": True,
                "edge_count": len(selected_edges),
                "status": "unusable_after_recovery",
                "reason": message,
            }
            fallback_reason = "vggt_graph_unusable_after_recovery"
        else:
            merged, merge_metrics = merge_window_cameras(
                selected_windows, transforms
            )
    else:
        graph_metrics = {
            "connected": False,
            "edge_count": len(edges),
            "status": "unusable_after_recovery",
        }
        fallback_reason = "vggt_graph_unusable_after_recovery"

    kinds = {spec.window_id: spec.kind for spec in processed_specs}
    loop_edges = [
        edge
        for edge in selected_edges
        if kinds.get(edge.source) == "bridge" or kinds.get(edge.target) == "bridge"
    ]
    graph_payload = {
        "schema_version": 2,
        "profile": PROFILE_ID,
        "windows": window_records,
        "recovery_attempt_count": len(recoveries),
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "shared_indices": list(edge.shared_indices),
                "target_from_source": edge.target_from_source.tolist(),
                "center_residual_p90": edge.center_residual_p90,
                "rotation_residual_p90_degrees": edge.rotation_residual_p90_degrees,
            }
            for edge in edges
        ],
        "rejected_edges": edge_rejections,
        "component_selection": component_metrics,
        "selected_window_ids": selected_ids,
        "graph": graph_metrics,
        "merge": merge_metrics,
        "verified_nonlocal_edge_count": len(loop_edges),
        "trajectory_status": (
            "closed_graph_verified" if loop_edges else "open_trajectory_unverified"
        ),
    }
    write_json(work_dir / "window_graph.json", graph_payload)

    image_names = [path.name for path in image_paths]
    image_sizes = {
        index: Image.open(path).size for index, path in enumerate(image_paths)
    }

    write_progress(args.progress_file, "vggt_ba_feature_extraction")
    command_logs: list[str] = []
    colmap_stage_elapsed_seconds: dict[str, float] = {}
    database_path = colmap_dir / "database.db"
    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(args.image_dir),
        *camera_plan.batches[0].image_reader_options,
        "--FeatureExtraction.use_gpu",
        "1",
        "--FeatureExtraction.num_threads",
        str(args.num_threads),
        *feature_profile.extraction_options,
    ]
    feature_started_at = time.perf_counter()
    command_logs.append(run_command(feature_command))
    colmap_stage_elapsed_seconds["feature_extraction"] = (
        time.perf_counter() - feature_started_at
    )
    write_progress(args.progress_file, "vggt_ba_feature_matching")
    matcher_command = [
        colmap,
        pairing_command,
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        "1",
        "--FeatureMatching.num_threads",
        str(args.num_threads),
        *local_matcher.matching_options,
        *geometric_verification.matching_options,
        *pairing_options,
    ]
    if legacy_matcher == "sequential":
        matcher_command.extend(
            [
                "--SequentialMatching.loop_detection",
                "1",
                "--SequentialMatching.vocab_tree_path",
                str(vocab_tree_path),
            ]
        )
    sequential_overlap_value: int | None = None
    if (
        pairing_command == "sequential_matcher"
        and video_selection is not None
        and video_selection.get("profile") == V2_PROFILE_ID
    ):
        sequential_overlap_value = sequential_overlap(video_selection)
        matcher_command.extend(
            ["--SequentialMatching.overlap", str(sequential_overlap_value)]
        )
    matcher_started_at = time.perf_counter()
    command_logs.append(run_command(matcher_command))
    colmap_stage_elapsed_seconds["feature_matching"] = (
        time.perf_counter() - matcher_started_at
    )
    image_ids_by_name = read_colmap_database_image_ids(database_path)
    image_index_by_name = {
        name: index for index, name in enumerate(image_names)
    }
    if set(image_ids_by_name) != set(image_index_by_name):
        raise RuntimeError(
            "COLMAP database image names do not match the selected inputs"
        )
    image_index_by_id = {
        image_id: image_index_by_name[name]
        for name, image_id in image_ids_by_name.items()
    }
    final_model: Path | None = None
    recovery_requested = (
        video_selection is not None
        and video_selection.get("profile") == V2_PROFILE_ID
        and args.video_source is not None
        and args.video_selection is not None
    )
    seeded_registration = None
    if fallback_reason is None:
        initial_record = write_initial_colmap_model(
            work_dir / "initial_model",
            image_names,
            merged,
            image_sizes,
            image_ids_by_name=image_ids_by_name,
            camera_model=camera_calibration.camera_model,
        )
        write_progress(args.progress_file, "vggt_ba_global_triangulation")
        triangulated_dir = work_dir / "triangulated"
        triangulated_dir.mkdir()
        command_logs.append(
            run_command(
                [
                    colmap,
                    "point_triangulator",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(args.image_dir),
                    "--input_path",
                    str(work_dir / "initial_model"),
                    "--output_path",
                    str(triangulated_dir),
                    "--clear_points",
                    "1",
                    "--refine_intrinsics",
                    "1",
                    "--Mapper.num_threads",
                    str(args.num_threads),
                    "--Mapper.ba_global_function_tolerance",
                    "0.000001",
                ]
            )
        )
        _triangulated_images, triangulated_points = read_sparse_model_counts(
            triangulated_dir
        )
        if triangulated_points == 0:
            fallback_reason = "vggt_seed_geometry_insufficient"
        else:
            write_progress(args.progress_file, "vggt_ba_image_registration")
            registered_dir = work_dir / "registered_model"
            registered_dir.mkdir()
            command_logs.append(
                run_command(
                    [
                        colmap,
                        "image_registrator",
                        "--database_path",
                        str(database_path),
                        "--input_path",
                        str(triangulated_dir),
                        "--output_path",
                        str(registered_dir),
                    ]
                )
            )
            write_progress(
                args.progress_file, "vggt_ba_global_bundle_adjustment"
            )
            bundled_dir = work_dir / "global_model"
            bundled_dir.mkdir()
            command_logs.append(
                run_command(
                    [
                        colmap,
                        "bundle_adjuster",
                        "--input_path",
                        str(registered_dir),
                        "--output_path",
                        str(bundled_dir),
                        "--BundleAdjustment.refine_focal_length",
                        "1",
                        "--BundleAdjustment.refine_principal_point",
                        "0",
                        "--BundleAdjustment.refine_extra_params",
                        "1",
                        "--BundleAdjustmentCeres.function_tolerance",
                        "0.000001",
                    ]
                )
            )
            seeded_text = work_dir / "global_model_txt"
            seeded_registration = inspect_model_registration(
                colmap,
                bundled_dir,
                seeded_text,
                image_index_by_id,
                len(image_paths),
                command_logs,
            )
            if seeded_registration["usable"]:
                final_model = bundled_dir
            else:
                fallback_reason = "vggt_registration_gate_failed"

    fallback_applied = fallback_reason is not None
    effective_source = "vggt_ba"
    fallback_registration = None
    if fallback_applied:
        if fallback_reason not in VGGT_BA_FALLBACK_REASONS:
            raise RuntimeError(
                f"unclassified VGGT-BA fallback reason: {fallback_reason}"
            )
        write_progress(args.progress_file, "colmap_fallback_mapping")
        fallback_sparse = work_dir / "fallback_sparse"
        fallback_sparse.mkdir()
        fallback_mapper_command = build_colmap_fallback_mapper_command(
            colmap=colmap,
            database_path=database_path,
            image_dir=args.image_dir,
            output_path=fallback_sparse,
            num_threads=args.num_threads,
            video_selection=video_selection,
        )
        command_logs.append(run_command(fallback_mapper_command))
        final_model, _registered_images, _sparse_points = find_largest_sparse_model(
            fallback_sparse
        )
        fallback_registration = inspect_model_registration(
            colmap,
            final_model,
            work_dir / "fallback_model_txt",
            image_index_by_id,
            len(image_paths),
            command_logs,
        )
        if not fallback_registration["usable"] and not recovery_requested:
            raise RuntimeError(
                "ordinary COLMAP fallback failed the final registration gates: "
                f"{fallback_registration}"
            )
        effective_source = "colmap"

    if final_model is None:
        raise RuntimeError("geometry state machine did not produce a final COLMAP model")

    final_model, recovery_diagnostics, recovered_fallback_registration = (
        apply_video_registration_recovery(
            colmap=colmap,
            database_path=database_path,
            image_dir=args.image_dir,
            final_model=final_model,
            video_selection=video_selection,
            video_source=args.video_source,
            selection_path=args.video_selection,
            diagnostics_dir=diagnostics_dir,
            work_dir=work_dir,
            num_threads=args.num_threads,
            progress_file=args.progress_file,
            fallback_applied=fallback_applied,
            command_logs=command_logs,
            feature_extraction_options=feature_profile.extraction_options,
            local_matching_options=local_matcher.matching_options,
            geometric_verification_options=(
                geometric_verification.matching_options
            ),
            sfm_feature_profile=feature_profile.profile_id,
            sfm_local_matcher=local_matcher.name,
            sfm_geometric_verification=geometric_verification.profile_id,
            initial_sfm_pairing=pairing_profile,
        )
    )
    if recovered_fallback_registration is not None:
        fallback_registration = recovered_fallback_registration

    camera_plan = prepare_camera_extraction(
        camera_calibration,
        args.image_dir,
        discover_images(args.image_dir),
        work_dir / "camera_groups",
    )
    raw_text_dir = work_dir / "final_raw_txt"
    raw_text_dir.mkdir(parents=True, exist_ok=False)
    raw_conversion_started = time.perf_counter()
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(final_model),
                "--output_path",
                str(raw_text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )
    colmap_stage_elapsed_seconds["raw_model_conversion"] = (
        time.perf_counter() - raw_conversion_started
    )
    camera_diagnostics = build_camera_calibration_diagnostics(
        database_path=database_path,
        final_camera_payload=build_camera_payload(raw_text_dir),
        points3d_path=raw_text_dir / "points3D.txt",
        plan=camera_plan,
        colmap_build=colmap_build,
    )
    camera_diagnostics_path = (
        diagnostics_dir / "sfm_camera_calibration.json"
    )
    write_camera_calibration_diagnostics(
        camera_diagnostics_path, camera_diagnostics
    )
    camera_metrics = camera_calibration_metrics(camera_diagnostics)

    write_progress(args.progress_file, "colmap_undistortion")
    undistorted_dir = colmap_dir / "undistorted"
    command_logs.append(
        run_command(
            [
                colmap,
                "image_undistorter",
                "--image_path",
                str(args.image_dir),
                "--input_path",
                str(final_model),
                "--output_path",
                str(undistorted_dir),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(args.max_image_size),
            ]
        )
    )
    sparse_source = undistorted_dir / "sparse"
    sparse_text = undistorted_dir / "sparse_txt"
    sparse_text.mkdir()
    points_ply = geometry_dir / "points.ply"
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(sparse_source),
                "--output_path",
                str(sparse_text),
                "--output_type",
                "TXT",
            ]
        )
    )
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(sparse_source),
                "--output_path",
                str(points_ply),
                "--output_type",
                "PLY",
            ]
        )
    )
    camera_payload = build_camera_payload(sparse_text)
    supported = supported_image_ids(
        sparse_text / "images.txt",
        minimum_observations=MIN_SUPPORTED_OBSERVATIONS,
    )
    camera_payload["images"] = [
        image for image in camera_payload["images"] if int(image["image_id"]) in supported
    ]
    if len(camera_payload["images"]) < 12:
        raise RuntimeError(
            f"final COLMAP model has only {len(camera_payload['images'])} geometrically supported cameras"
        )
    write_json(geometry_dir / "cameras.json", camera_payload)

    graph_payload["effective_geometry_source"] = effective_source
    graph_payload["fallback_applied"] = fallback_applied
    graph_payload["fallback_reason"] = fallback_reason
    graph_payload["video_registration_recovery_method"] = (
        "incremental_colmap" if recovery_requested else None
    )
    graph_payload["video_registration_recovery_status"] = (
        recovery_diagnostics["status"]
        if recovery_diagnostics is not None
        else "not_requested"
    )
    write_json(work_dir / "window_graph.json", graph_payload)
    elapsed = time.perf_counter() - started_at
    final_input_count = len(image_paths)
    if recovery_requested:
        final_selection = json.loads(args.video_selection.read_text(encoding="utf-8"))
        final_input_count = len(final_selection["selected"])
    diagnostics = {
        "schema_version": 2,
        "profile": PROFILE_ID,
        "geometry_source": "vggt_ba",
        "effective_geometry_source": effective_source,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
        "colmap_feature": colmap_frontend_provenance(
            feature_profile, local_matcher
        ),
        "colmap_pairing": pairing_profile,
        "colmap_pairing_command": pairing_command,
        "colmap_vocab_tree_sha256": vocab_tree_sha256,
        "colmap_geometric_verification": geometric_verification.provenance(),
        "colmap_camera_calibration": camera_diagnostics["calibration"],
        "camera_calibration_diagnostics": camera_diagnostics_path.relative_to(
            output_dir
        ).as_posix(),
        "colmap_mapper": "incremental",
        "colmap_matcher": pairing_command.removesuffix("_matcher"),
        "colmap_stage_elapsed_seconds": colmap_stage_elapsed_seconds,
        "sequential_overlap": sequential_overlap_value,
        "input_count": final_input_count,
        "initial_input_count": len(image_paths),
        "supported_camera_count": len(camera_payload["images"]),
        "supported_camera_rate": len(camera_payload["images"]) / final_input_count,
        "video_registration_recovery_method": (
            "incremental_colmap" if recovery_requested else None
        ),
        "video_registration_recovery": recovery_diagnostics,
        "point_count": read_ply_vertex_count(points_ply),
        "initial_camera": initial_record,
        "seeded_registration": seeded_registration,
        "fallback_registration": fallback_registration,
        "window_graph": graph_payload,
        "colmap_executable": colmap,
        "colmap_build": colmap_build,
        "elapsed_seconds": elapsed,
        "dependencies": dependency_record(args),
    }
    write_json(diagnostics_dir / "vggt_ba.json", diagnostics)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs" / "vggt_ba.log").write_text(
        "\n".join(
            [
                "geometry_source=vggt_ba",
                f"effective_geometry_source={effective_source}",
                f"fallback_applied={str(fallback_applied).lower()}",
                f"fallback_reason={fallback_reason or 'none'}",
                f"sfm_feature_profile={feature_profile.profile_id}",
                f"sfm_feature_extractor={feature_profile.extractor}",
                f"sfm_feature_descriptor={feature_profile.descriptor}",
                f"sfm_local_matcher_profile={local_matcher.profile_id}",
                f"sfm_local_matcher={local_matcher.name}",
                f"sfm_feature_max_features={feature_profile.max_features}",
                f"sfm_feature_extractor_model_sha256={feature_profile.extractor_model_sha256 or 'none'}",
                f"sfm_local_matcher_model_sha256={local_matcher.model_sha256 or 'none'}",
                f"sfm_pairing={pairing_profile}",
                f"sfm_pairing_command={pairing_command}",
                f"sfm_pairing_vocab_tree_sha256={vocab_tree_sha256 or 'none'}",
                f"sfm_geometric_verification_profile={geometric_verification.profile_id}",
                f"sfm_geometric_verification_guided_matching={str(geometric_verification.guided_matching).lower()}",
                *[
                    f"{key}={value}"
                    for key, value in camera_metrics.items()
                ],
                f"sfm_camera_calibration_diagnostics={camera_diagnostics_path.relative_to(output_dir).as_posix()}",
                "sfm_mapper=incremental",
                f"colmap_feature_extraction_seconds={colmap_stage_elapsed_seconds['feature_extraction']:.3f}",
                f"colmap_feature_matching_seconds={colmap_stage_elapsed_seconds['feature_matching']:.3f}",
                f"colmap_matcher={pairing_command.removesuffix('_matcher')}",
                f"sequential_overlap={sequential_overlap_value if sequential_overlap_value is not None else 'default'}",
                f"video_registration_recovery_method={'incremental_colmap' if recovery_requested else 'none'}",
                f"video_registration_recovery_status={recovery_diagnostics['status'] if recovery_diagnostics is not None else 'not_requested'}",
                f"video_registration_recovery_rounds={len(recovery_diagnostics['rounds']) if recovery_diagnostics is not None else 0}",
                f"profile={PROFILE_ID}",
                f"input_count={final_input_count}",
                f"initial_input_count={len(image_paths)}",
                f"supported_camera_count={len(camera_payload['images'])}",
                f"point_count={diagnostics['point_count']}",
                f"trajectory_status={graph_payload['trajectory_status']}",
                f"elapsed_seconds={elapsed:.3f}",
                *command_logs,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"supported_cameras={len(camera_payload['images'])}")
    print(f"num_points={diagnostics['point_count']}")
    print(f"effective_geometry_source={effective_source}")
    print(f"fallback_reason={fallback_reason or 'none'}")
    print(f"trajectory_status={graph_payload['trajectory_status']}")


def build_colmap_fallback_mapper_command(
    *,
    colmap: str,
    database_path: Path,
    image_dir: Path,
    output_path: Path,
    num_threads: int,
    video_selection: dict[str, Any] | None,
) -> list[str]:
    return [
        colmap,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(output_path),
        "--Mapper.num_threads",
        str(num_threads),
        "--Mapper.ba_global_function_tolerance",
        "0.000001",
        *v2_mapper_options(video_selection),
    ]


def apply_video_registration_recovery(
    *,
    colmap: str,
    database_path: Path,
    image_dir: Path,
    final_model: Path,
    video_selection: dict[str, Any] | None,
    video_source: Path | None,
    selection_path: Path | None,
    diagnostics_dir: Path,
    work_dir: Path,
    num_threads: int,
    progress_file: Path | None,
    fallback_applied: bool,
    command_logs: list[str],
    feature_extraction_options: tuple[str, ...] = (),
    local_matching_options: tuple[str, ...] = (),
    geometric_verification_options: tuple[str, ...] = (),
    sfm_feature_profile: str = "sift_v1",
    sfm_local_matcher: str = "SIFT_BRUTEFORCE",
    sfm_geometric_verification: str = "default_v1",
    initial_sfm_pairing: str = "exhaustive",
) -> tuple[Path, dict[str, Any] | None, dict[str, Any] | None]:
    if (
        video_selection is None
        or video_selection.get("profile") != V2_PROFILE_ID
        or video_source is None
        or selection_path is None
    ):
        return final_model, None, None
    recovered_model, diagnostics, recovery_logs = recover_video_registration(
        colmap=colmap,
        database_path=database_path,
        image_dir=image_dir,
        initial_model=final_model,
        selection_path=selection_path,
        video_source=video_source,
        diagnostics_path=diagnostics_dir / "video_registration_recovery.json",
        use_gpu=True,
        gpu_index=None,
        num_threads=num_threads,
        feature_extraction_options=feature_extraction_options,
        local_matching_options=local_matching_options,
        geometric_verification_options=geometric_verification_options,
        sfm_feature_profile=sfm_feature_profile,
        sfm_local_matcher=sfm_local_matcher,
        sfm_geometric_verification=sfm_geometric_verification,
        initial_sfm_pairing=initial_sfm_pairing,
        progress=lambda stage: write_progress(progress_file, stage),
    )
    command_logs.extend(recovery_logs)
    if not fallback_applied:
        return recovered_model, diagnostics, None

    final_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ordered_names = [
        Path(str(item["path"])).name
        for item in sorted(
            final_selection["selected"],
            key=lambda item: (float(item["time_seconds"]), int(item["pts"])),
        )
    ]
    final_ids_by_name = read_colmap_database_image_ids(database_path)
    final_index_by_id = {
        final_ids_by_name[name]: index for index, name in enumerate(ordered_names)
    }
    registration = inspect_model_registration(
        colmap,
        recovered_model,
        work_dir / "fallback_final_model_txt",
        final_index_by_id,
        len(ordered_names),
        command_logs,
    )
    if not registration["usable"]:
        raise RuntimeError(
            "ordinary COLMAP fallback failed the final registration gates "
            "after incremental recovery: "
            f"{registration}"
        )
    return recovered_model, diagnostics, registration


def discover_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_model_registration(
    colmap: str,
    model_dir: Path,
    text_dir: Path,
    image_index_by_id: dict[int, int],
    image_count: int,
    command_logs: list[str],
) -> dict[str, Any]:
    text_dir.mkdir()
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )
    supported = supported_image_ids(
        text_dir / "images.txt",
        minimum_observations=MIN_SUPPORTED_OBSERVATIONS,
    )
    unknown = supported - set(image_index_by_id)
    if unknown:
        raise RuntimeError(
            f"COLMAP model references unknown database image IDs: {sorted(unknown)}"
        )
    indices = sorted(image_index_by_id[image_id] for image_id in supported)
    camera_rate = len(indices) / image_count
    temporal_coverage = (
        (indices[-1] - indices[0] + 1) / image_count if indices else 0.0
    )
    return {
        "supported_camera_count": len(indices),
        "supported_camera_rate": camera_rate,
        "temporal_coverage": temporal_coverage,
        "first_image_index": indices[0] if indices else None,
        "last_image_index": indices[-1] if indices else None,
        "usable": (
            len(indices) >= MIN_RELIABLE_CAMERAS
            and camera_rate >= MIN_RELIABLE_CAMERA_RATE
            and temporal_coverage >= MIN_TEMPORAL_COVERAGE
        ),
    }


def validate_runtime(args: argparse.Namespace) -> None:
    required = [
        args.repo_dir / "vggt",
        args.checkpoint_dir / "model.safetensors",
        args.dinov2_repo / "hubconf.py",
        args.lightglue_repo / ".git",
        args.dinov2_checkpoint,
        args.aliked_checkpoint,
        args.tracker_checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("VGGT-BA runtime is incomplete: " + "; ".join(missing))
    if (
        args.aliked_checkpoint.name != "aliked-n16.pth"
        or args.aliked_checkpoint.parent.name != "checkpoints"
    ):
        raise SystemExit(
            "ALIKED checkpoint must use the local torch-hub/checkpoints/aliked-n16.pth layout"
        )
    try:
        import lightglue  # noqa: F401
        import pycolmap  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"VGGT-BA dependency is missing: {exc}") from exc


def load_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import random

    import torch
    import torch.nn.functional as functional

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = select_dtype(args.device, args.precision)
    torch.hub.set_dir(str(args.aliked_checkpoint.parent.parent.resolve()))
    sys.path.insert(0, str(args.repo_dir.resolve()))
    from vggt.dependency.vggsfm_utils import (
        build_vggsfm_tracker,
        initialize_feature_extractors,
    )
    from vggt.models.vggt import VGGT

    vggt_model = load_vggt_model(
        model_cls=VGGT,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dtype=dtype,
        enable_point=False,
    ).eval()
    tracker = build_vggsfm_tracker(str(args.tracker_checkpoint)).to(device).eval()
    extractors = initialize_feature_extractors(
        MAX_QUERY_POINTS, extractor_method="aliked", device=device
    )
    dino_model = torch.hub.load(
        str(args.dinov2_repo.resolve()),
        "dinov2_vitb14_reg",
        source="local",
        pretrained=False,
    )
    payload = torch.load(args.dinov2_checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }
    missing, unexpected = dino_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv2 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    dino_model = dino_model.to(device).eval()
    return {
        "torch": torch,
        "functional": functional,
        "device": device,
        "dtype": dtype,
        "vggt_model": vggt_model,
        "tracker": tracker,
        "extractors": extractors,
        "dino_model": dino_model,
    }


def compute_dino_descriptors(
    image_paths: list[Path], model: Any, device: Any, torch: Any
) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    descriptors = []
    for start in range(0, len(image_paths), 16):
        arrays = []
        for path in image_paths[start : start + 16]:
            image = Image.open(path).convert("RGB").resize((336, 336), Image.Resampling.BICUBIC)
            arrays.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        batch = torch.from_numpy(np.stack(arrays)).to(device)
        with torch.no_grad():
            features = model.forward_features((batch - mean) / std)
            cls = features["x_norm_clstoken"] if isinstance(features, dict) else features
        descriptors.append(cls.detach().float().cpu().numpy())
    return np.concatenate(descriptors, axis=0)


def process_window(
    spec: Any,
    image_paths: list[Path],
    descriptors: np.ndarray,
    runtime: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    import pycolmap
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
    from vggt.dependency.projection import project_3D_points_np
    from vggt.dependency.track_predict import _forward_on_query
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    torch = runtime["torch"]
    functional = runtime["functional"]
    paths = [image_paths[index] for index in spec.image_indices]
    images, original_coords = load_and_preprocess_images_square(
        [str(path) for path in paths], 1024
    )
    images = images.to(runtime["device"])
    images_518 = functional.interpolate(
        images, size=(518, 518), mode="bilinear", align_corners=False
    )
    autocast = (
        torch.cuda.amp.autocast(dtype=runtime["dtype"])
        if runtime["device"].type == "cuda"
        else nullcontext()
    )
    started = time.perf_counter()
    with torch.no_grad(), autocast:
        predictions = runtime["vggt_model"](images_518[None])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images_518.shape[-2:]
    )
    points_3d = unproject_depth_map_to_point_map(
        predictions["depth"].squeeze(0).detach().float().cpu().numpy(),
        extrinsic.squeeze(0).detach().float().cpu().numpy(),
        intrinsic.squeeze(0).detach().float().cpu().numpy(),
    )
    confidence = predictions["depth_conf"].squeeze(0).detach().float().cpu().numpy()
    extrinsic_np = extrinsic.squeeze(0).detach().float().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().float().cpu().numpy()
    del predictions, images_518

    tracker_autocast = (
        torch.cuda.amp.autocast(dtype=runtime["dtype"])
        if runtime["device"].type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), tracker_autocast:
        fmaps = runtime["tracker"].process_images_to_fmaps(images)
        query_indices = choose_queries(
            descriptors[list(spec.image_indices)], QUERY_FRAME_COUNT
        )
        tracks = []
        visibility = []
        point_parts = []
        color_parts = []
        for query_index in query_indices:
            track, visible, _conf, point, color = _forward_on_query(
                query_index,
                images,
                confidence,
                points_3d,
                fmaps,
                runtime["extractors"],
                runtime["tracker"],
                163840,
                True,
                runtime["device"],
            )
            tracks.append(track)
            visibility.append(visible)
            point_parts.append(point)
            color_parts.append(color)
    tracks_np = np.concatenate(tracks, axis=1)
    visibility_np = np.concatenate(visibility, axis=1)
    points_np = np.concatenate(point_parts, axis=0)
    colors_np = np.concatenate(color_parts, axis=0)
    intrinsic_np[:, :2, :] *= 1024 / 518
    projected_points, _projected_cameras = project_3D_points_np(
        points_np,
        extrinsic_np,
        intrinsic_np,
    )
    reprojection_errors = np.linalg.norm(projected_points - tracks_np, axis=-1)
    visibility_mask = visibility_np > 0.2
    reprojection_mask = reprojection_errors < 8.0
    mask = np.logical_and(visibility_mask, reprojection_mask)
    visibility_counts = count_frame_inliers(visibility_mask)
    reprojection_counts = count_frame_inliers(reprojection_mask)
    inlier_counts = count_frame_inliers(mask)
    strong_local, weak_local = classify_frame_support(inlier_counts)
    strong_global = [spec.image_indices[index] for index in strong_local]
    weak_global = [spec.image_indices[index] for index in weak_local]
    print(
        f"window={spec.window_id} inliers_per_frame={inlier_counts}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "window_id": spec.window_id,
        "kind": spec.kind,
        "image_indices": list(spec.image_indices),
        "image_names": [path.name for path in paths],
        "query_indices": query_indices,
        "visibility_per_frame": visibility_counts,
        "reprojection_pass_per_frame": reprojection_counts,
        "inliers_per_frame": inlier_counts,
        "strong_image_indices": strong_global,
        "weak_image_indices": weak_global,
        "minimum_inliers_per_frame": MIN_SUPPORTED_OBSERVATIONS,
        "track_count": int(tracks_np.shape[1]),
    }
    if len(strong_local) < 3:
        record.update(
            {
                "status": "rejected",
                "reason": "fewer_than_three_reliable_cameras",
                "point_count": 0,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(output_dir / "window.json", record)
        del images, fmaps
        if runtime["device"].type == "cuda":
            torch.cuda.empty_cache()
        return {}, record

    strong_array = np.asarray(strong_local, dtype=np.int64)
    reconstruction, _valid = batch_np_matrix_to_pycolmap(
        points_np,
        extrinsic_np[strong_array],
        intrinsic_np[strong_array],
        tracks_np[strong_array],
        np.array([1024, 1024]),
        masks=mask[strong_array],
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
        min_inlier_per_frame=0,
        points_rgb=colors_np,
    )
    if reconstruction is None or len(reconstruction.points3D) == 0:
        record.update(
            {
                "status": "rejected",
                "reason": "no_shared_reliable_track_geometry",
                "point_count": 0,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(output_dir / "window.json", record)
        del images, fmaps
        if runtime["device"].type == "cuda":
            torch.cuda.empty_cache()
        return {}, record
    before = reconstruction.compute_mean_reprojection_error()
    if not np.isfinite(before):
        raise RuntimeError(
            f"{spec.window_id} local BA input reprojection is non-finite"
        )
    options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, options)
    after = reconstruction.compute_mean_reprojection_error()
    if not np.isfinite(after):
        raise RuntimeError(f"{spec.window_id} local BA produced non-finite reprojection")
    if after > before * 1.05 + 1e-6:
        record.update(
            {
                "status": "rejected",
                "reason": "local_ba_reprojection_increased",
                "point_count": len(reconstruction.points3D),
                "mean_reprojection_before": float(before),
                "mean_reprojection_after": float(after),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(output_dir / "window.json", record)
        del images, fmaps
        if runtime["device"].type == "cuda":
            torch.cuda.empty_cache()
        return {}, record
    reconstruction = restore_original_coordinates(
        reconstruction,
        [paths[index].name for index in strong_local],
        original_coords.detach().cpu().numpy()[strong_array],
        1024,
    )
    reconstruction.write(output_dir)
    cameras: dict[int, dict[str, np.ndarray]] = {}
    for ba_index, local_index in enumerate(strong_local):
        global_index = spec.image_indices[local_index]
        image = reconstruction.images[ba_index + 1]
        camera = reconstruction.cameras[image.camera_id]
        cameras[global_index] = {
            "extrinsic": np.asarray(
                image.cam_from_world.matrix(), dtype=np.float64
            )[:3, :4],
            "intrinsic": np.asarray(camera.calibration_matrix(), dtype=np.float64),
        }
    elapsed = time.perf_counter() - started
    record.update(
        {
            "status": "accepted",
            "reason": None,
            "point_count": len(reconstruction.points3D),
            "mean_reprojection_before": float(before),
            "mean_reprojection_after": float(after),
            "elapsed_seconds": elapsed,
        }
    )
    write_json(output_dir / "window.json", record)
    del images, fmaps
    if runtime["device"].type == "cuda":
        torch.cuda.empty_cache()
    return cameras, record


def choose_queries(descriptors: np.ndarray, count: int) -> list[int]:
    values = np.asarray(descriptors, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    chosen = [0]
    while len(chosen) < min(count, len(values)):
        similarity = values @ values[chosen].T
        minimum_distance = (1.0 - similarity).min(axis=1)
        minimum_distance[chosen] = -1
        chosen.append(int(np.argmax(minimum_distance)))
    return chosen


def restore_original_coordinates(
    reconstruction: Any,
    image_names: list[str],
    original_coords: np.ndarray,
    image_size: int,
) -> Any:
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        image.name = image_names[image_id - 1]
        real_size = original_coords[image_id - 1, -2:]
        ratio = max(real_size) / image_size
        params = camera.params.copy() * ratio
        params[-2:] = real_size / 2
        camera.params = params
        camera.width = int(real_size[0])
        camera.height = int(real_size[1])
        top_left = original_coords[image_id - 1, :2]
        for point in image.points2D:
            point.xy = (point.xy - top_left) * ratio
    return reconstruction


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return (
        "command="
        + " ".join(command)
        + "\nstdout="
        + completed.stdout.strip()
        + "\nstderr="
        + completed.stderr.strip()
    )


def write_progress(path: Path | None, stage: str) -> None:
    if path is not None:
        write_json(path, {"stage": stage})


def dependency_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vggt_revision": git_revision(args.repo_dir),
        "vggt_checkpoint_sha256": sha256_file(args.checkpoint_dir / "model.safetensors"),
        "dinov2_revision": git_revision(args.dinov2_repo),
        "dinov2_checkpoint_sha256": sha256_file(args.dinov2_checkpoint),
        "lightglue_revision": git_revision(args.lightglue_repo),
        "aliked_checkpoint_sha256": sha256_file(args.aliked_checkpoint),
        "vggsfm_tracker_sha256": sha256_file(args.tracker_checkpoint),
        "runtime_network_access": False,
        "research_only": True,
    }


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    try:
        main()
    except (VggtBaError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
