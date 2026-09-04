from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

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
    sha256_file,
)
from image3d_scenegraph.geometry.sfm_pose_health import (
    build_sfm_pose_health_from_text,
    require_sfm_pose_health,
    selected_timestamps_from_payload,
    write_sfm_pose_health,
)
from image3d_scenegraph.geometry.video_recovery import (
    expand_v2_initial_registration,
    recover_video_registration,
    sequential_overlap,
    v2_mapper_options,
    v2_mapper_seed_image_names,
)
from image3d_scenegraph.video.keyframes import V2_PROFILE_ID
from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT,
    MIN_VIDEO_REGISTRATION_RATE,
    MIN_VIDEO_TEMPORAL_COVERAGE,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP sparse SfM and export a point cloud.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    )
    parser.add_argument(
        "--single-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-index", type=parse_gpu_indices)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--max-image-size", type=int)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--gaussian-baseline", action="store_true")
    parser.add_argument("--vocab-tree-path", type=Path)
    parser.add_argument("--video-source", type=Path)
    parser.add_argument("--video-selection", type=Path)
    args = parser.parse_args()
    if args.camera_calibration is not None and args.single_camera is not None:
        parser.error("--camera-calibration cannot be combined with --single-camera")
    legacy_single_camera = (
        True if args.single_camera is None else bool(args.single_camera)
    )
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be at least 1")
    if args.max_image_size is not None and args.max_image_size < 1:
        parser.error("--max-image-size must be at least 1")
    if args.pairing is not None and args.matcher is not None:
        parser.error("--pairing and legacy --matcher cannot be combined")
    if args.pairing is not None and args.vocab_tree_path is not None:
        parser.error("--pairing resolves its vocabulary tree; omit --vocab-tree-path")
    legacy_matcher = args.matcher
    if args.pairing is None and legacy_matcher is None:
        legacy_matcher = "sequential"
    if (
        args.gaussian_baseline
        and legacy_matcher == "sequential"
        and args.vocab_tree_path is None
    ):
        raise SystemExit(
            "Gaussian baseline sequential matching requires --vocab-tree-path for loop closure"
        )
    if args.vocab_tree_path is not None and args.feature_profile != "sift_v1":
        raise SystemExit(
            "the legacy --vocab-tree-path is SIFT-only; "
            "use --pairing with the descriptor-specific vocabulary tree"
        )
    if (args.video_source is None) != (args.video_selection is None):
        parser.error("--video-source and --video-selection must be provided together")
    video_selection: dict[str, Any] | None = None
    initial_video_selection_sha256: str | None = None
    if args.video_source is not None and args.video_selection is not None:
        if not args.video_source.is_file():
            raise SystemExit(f"Video source is missing: {args.video_source}")
        try:
            initial_video_selection_sha256 = sha256_file(args.video_selection)
            video_selection = json.loads(
                args.video_selection.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit("Cannot read video selection metadata") from exc

    started_at = time.perf_counter()
    colmap_path = resolve_colmap_executable()
    if colmap_path is None:
        raise SystemExit(
            "COLMAP executable not found. Run `uv run python scripts/setup_colmap_cuda.py --install` "
            "or install COLMAP on PATH."
        )
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
        camera_calibration = (
            resolve_colmap_camera_calibration(args.camera_calibration)
            if args.camera_calibration is not None
            else None
        )
        if (
            camera_calibration is not None
            and camera_calibration.sharing_policy != "single_camera"
            and video_selection is not None
        ):
            raise ColmapFeatureError(
                "auto-grouped camera calibration is not supported for video"
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
                if vocab_tree_path is not None and vocab_tree_path.is_file()
                else None
            )
            pairing_profile = (
                "sequential_loop"
                if legacy_matcher == "sequential" and vocab_tree_path is not None
                else str(legacy_matcher)
            )
    except ColmapFeatureError as exc:
        raise SystemExit(str(exc)) from exc

    image_paths = discover_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    output_dir = args.output_dir
    geometry_dir = output_dir / "geometry"
    logs_dir = output_dir / "logs"
    work_dir = output_dir / "colmap"
    sparse_dir = work_dir / "sparse"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    camera_plan = (
        prepare_camera_extraction(
            camera_calibration,
            args.image_dir,
            image_paths,
            work_dir / "camera_groups",
        )
        if camera_calibration is not None
        else None
    )
    mapper_seed_names: list[str] = []
    mapper_seed_path: Path | None = None
    mapper_seed_option: str | None = None
    if video_selection is not None and video_selection.get("profile") == V2_PROFILE_ID:
        mapper_seed_names = v2_mapper_seed_image_names(video_selection)
        discovered_names = {path.name for path in image_paths}
        if not set(mapper_seed_names) <= discovered_names:
            raise SystemExit("v2 Mapper seed images are missing from the image directory")
        mapper_seed_path = work_dir / "v2-mapper-seed.txt"
        mapper_seed_path.write_text(
            "\n".join(mapper_seed_names) + "\n",
            encoding="utf-8",
        )
        mapper_seed_option = mapper_image_list_option(colmap)
    all_video_timestamps = (
        selected_timestamps_from_payload(video_selection)
        if video_selection is not None
        else None
    )
    mapper_pose_timestamps = all_video_timestamps
    if all_video_timestamps is not None and mapper_seed_names:
        mapper_pose_timestamps = {
            name: all_video_timestamps[name] for name in mapper_seed_names
        }

    def build_feature_command(
        image_list_path: Path | None,
        image_reader_options: tuple[str, ...],
    ) -> list[str]:
        command = [
            colmap,
            "feature_extractor",
            "--database_path",
            str(work_dir / "database.db"),
            "--image_path",
            str(args.image_dir),
        ]
        if image_list_path is not None:
            command.extend(("--image_list_path", str(image_list_path)))
        command.extend(
            (
                *image_reader_options,
                "--FeatureExtraction.use_gpu",
                "1" if args.use_gpu else "0",
                *feature_profile.extraction_options,
            )
        )
        if args.gpu_index is not None:
            command.extend(("--FeatureExtraction.gpu_index", args.gpu_index))
        if args.num_threads is not None:
            command.extend(
                ("--FeatureExtraction.num_threads", str(args.num_threads))
            )
        return command

    if camera_plan is not None:
        feature_commands = [
            build_feature_command(
                batch.image_list_path,
                batch.image_reader_options,
            )
            for batch in camera_plan.batches
        ]
    else:
        legacy_reader_options = [
            "--ImageReader.single_camera",
            "1" if legacy_single_camera else "0",
        ]
        if args.gaussian_baseline:
            legacy_reader_options.extend(("--ImageReader.camera_model", "OPENCV"))
        feature_commands = [
            build_feature_command(None, tuple(legacy_reader_options))
        ]
    mapper_command = [
        colmap,
        "mapper",
        "--database_path",
        str(work_dir / "database.db"),
        "--image_path",
        str(args.image_dir),
        "--output_path",
        str(sparse_dir),
        "--default_random_seed",
        "0",
    ]
    if args.gaussian_baseline:
        mapper_command.extend(("--Mapper.ba_global_function_tolerance", "0.000001"))
    mapper_command.extend(v2_mapper_options(video_selection))
    if mapper_seed_path is not None and mapper_seed_option is not None:
        mapper_command.extend((mapper_seed_option, str(mapper_seed_path)))
    if args.num_threads is not None:
        mapper_command.extend(("--Mapper.num_threads", str(args.num_threads)))
    matcher_command = [
        colmap,
        pairing_command,
        "--database_path",
        str(work_dir / "database.db"),
        "--default_random_seed",
        "0",
        "--FeatureMatching.use_gpu",
        "1" if args.use_gpu else "0",
        *local_matcher.matching_options,
        *geometric_verification.matching_options,
        *pairing_options,
    ]
    if legacy_matcher == "sequential" and vocab_tree_path is not None:
        matcher_command.extend(
            (
                "--SequentialMatching.loop_detection",
                "1",
                "--SequentialMatching.vocab_tree_path",
                str(vocab_tree_path),
            )
        )
    sequential_overlap_value: int | None = None
    if pairing_command == "sequential_matcher" and video_selection is not None:
        if video_selection.get("profile") == V2_PROFILE_ID:
            sequential_overlap_value = sequential_overlap(video_selection)
            matcher_command.extend(
                ("--SequentialMatching.overlap", str(sequential_overlap_value))
            )
    if args.gpu_index is not None:
        matcher_command.extend(("--FeatureMatching.gpu_index", args.gpu_index))
    if args.num_threads is not None:
        matcher_command.extend(("--FeatureMatching.num_threads", str(args.num_threads)))
    frontend_contract_path = output_dir / "diagnostics" / "sfm_frontend_contract.json"
    frontend_contract = {
        "schema_version": 1,
        "profile": "sfm_frontend_contract_v1",
        "colmap_build": colmap_build,
        "feature": colmap_frontend_provenance(feature_profile, local_matcher),
        "pairing": pairing_profile,
        "geometric_verification": geometric_verification.provenance(),
        "camera_calibration": (
            camera_plan.calibration.provenance()
            if camera_plan is not None
            else None
        ),
        "requested_mapper": "incremental",
        "colmap_random_seed": 0,
        "video_profile": (
            str(video_selection.get("profile"))
            if video_selection is not None
            else None
        ),
        "initial_video_selection_sha256": initial_video_selection_sha256,
        "v2_mapper_options": v2_mapper_options(video_selection),
        "v2_mapper_seed_count": len(mapper_seed_names),
        "test_rgb_loaded": False,
    }
    write_json(frontend_contract_path, frontend_contract)

    commands = [
        *[
            ("feature_extraction", "colmap_feature_extraction", command)
            for command in feature_commands
        ],
        ("feature_matching", "colmap_feature_matching", matcher_command),
        ("mapping", "colmap_mapping", mapper_command),
    ]
    command_logs = []
    stage_elapsed_seconds: dict[str, float] = {}
    for timing_stage, progress_stage, command in commands:
        write_progress(args.progress_file, progress_stage)
        command_started_at = time.perf_counter()
        command_logs.append(run_command(command))
        stage_elapsed_seconds[timing_stage] = stage_elapsed_seconds.get(
            timing_stage, 0.0
        ) + (time.perf_counter() - command_started_at)

    database_path = work_dir / "database.db"
    pose_recovery: dict[str, Any] | None = None
    if args.gaussian_baseline:
        (
            model_dir,
            registered_images,
            sparse_points,
            database_path,
            pose_recovery,
        ) = select_or_recover_sparse_model(
            colmap=colmap,
            sparse_dir=sparse_dir,
            database_path=database_path,
            image_dir=args.image_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            selected_timestamps=mapper_pose_timestamps,
            mapper_seed_path=mapper_seed_path,
            use_gpu=args.use_gpu,
            gpu_index=args.gpu_index,
            num_threads=args.num_threads,
            command_logs=command_logs,
        )
    else:
        model_dir, registered_images, sparse_points = find_largest_sparse_model(
            sparse_dir
        )
    initial_registered_images = registered_images
    initial_sparse_points = sparse_points
    expansion_diagnostics: dict[str, Any] | None = None
    if video_selection is not None and video_selection.get("profile") == V2_PROFILE_ID:
        expansion_started_at = time.perf_counter()
        model_dir, expansion_diagnostics, expansion_logs = (
            expand_v2_initial_registration(
                colmap=colmap,
                database_path=database_path,
                image_dir=args.image_dir,
                initial_model=model_dir,
                selection=video_selection,
                work_dir=work_dir / "v2_initial_expansion",
                diagnostics_path=output_dir
                / "diagnostics"
                / "video_initial_registration_expansion.json",
                num_threads=args.num_threads,
                progress=lambda stage: write_progress(args.progress_file, stage),
            )
        )
        stage_elapsed_seconds["initial_registration_expansion"] = (
            time.perf_counter() - expansion_started_at
        )
        command_logs.extend(expansion_logs)
        registered_images, sparse_points = read_sparse_model_counts(model_dir)
    recovery_diagnostics: dict[str, Any] | None = None
    if (
        video_selection is not None
        and video_selection.get("profile") == V2_PROFILE_ID
        and args.video_source is not None
        and args.video_selection is not None
    ):
        recovery_started_at = time.perf_counter()
        model_dir, recovery_diagnostics, recovery_logs = recover_video_registration(
            colmap=colmap,
            database_path=database_path,
            image_dir=args.image_dir,
            initial_model=model_dir,
            selection_path=args.video_selection,
            video_source=args.video_source,
            diagnostics_path=output_dir
            / "diagnostics"
            / "video_registration_recovery.json",
            use_gpu=args.use_gpu,
            gpu_index=args.gpu_index,
            num_threads=args.num_threads,
            feature_extraction_options=feature_profile.extraction_options,
            local_matching_options=local_matcher.matching_options,
            geometric_verification_options=(
                geometric_verification.matching_options
            ),
            sfm_feature_profile=feature_profile.profile_id,
            sfm_local_matcher=local_matcher.name,
            sfm_geometric_verification=geometric_verification.profile_id,
            initial_sfm_pairing=pairing_profile,
            progress=lambda stage: write_progress(args.progress_file, stage),
            force_final_bundle_adjustment=bool(
                expansion_diagnostics
                and expansion_diagnostics.get("accepted_pass_count", 0)
            ),
        )
        stage_elapsed_seconds["registration_recovery"] = (
            time.perf_counter() - recovery_started_at
        )
        command_logs.extend(recovery_logs)
        registered_images, sparse_points = read_sparse_model_counts(model_dir)
    camera_diagnostics: dict[str, Any] | None = None
    camera_metrics: dict[str, int | float | str] = {}
    camera_diagnostics_path = (
        output_dir / "diagnostics" / "sfm_camera_calibration.json"
    )
    raw_text_dir: Path | None = None
    if camera_plan is not None or args.gaussian_baseline:
        raw_text_dir = work_dir / "sparse_raw_txt"
        raw_text_dir.mkdir(parents=True, exist_ok=False)
        conversion_started_at = time.perf_counter()
        command_logs.append(
            run_command(
                [
                    colmap,
                    "model_converter",
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(raw_text_dir),
                    "--output_type",
                    "TXT",
                ]
            )
        )
        stage_elapsed_seconds["raw_model_conversion"] = (
            time.perf_counter() - conversion_started_at
        )
    pose_health: dict[str, Any] | None = None
    pose_health_path = output_dir / "diagnostics" / "sfm_pose_health.json"
    if args.gaussian_baseline:
        if raw_text_dir is None:
            raise RuntimeError("Gaussian SfM pose health has no raw text model")
        pose_health = build_sfm_pose_health_from_text(
            model_dir=raw_text_dir,
            selected_timestamps=all_video_timestamps,
            database_path=database_path,
        )
        write_sfm_pose_health(pose_health_path, pose_health)
        require_sfm_pose_health(pose_health)
    if camera_plan is not None:
        if raw_text_dir is None:
            raise RuntimeError("camera calibration has no raw text model")
        if camera_plan.calibration.sharing_policy == "single_camera":
            camera_plan = prepare_camera_extraction(
                camera_plan.calibration,
                args.image_dir,
                discover_images(args.image_dir),
                work_dir / "camera_groups",
            )
        camera_diagnostics = build_camera_calibration_diagnostics(
            database_path=database_path,
            final_camera_payload=build_camera_payload(raw_text_dir),
            points3d_path=raw_text_dir / "points3D.txt",
            plan=camera_plan,
            colmap_build=colmap_build,
        )
        write_camera_calibration_diagnostics(
            camera_diagnostics_path, camera_diagnostics
        )
        camera_metrics = camera_calibration_metrics(camera_diagnostics)
    model_source = model_dir
    text_dir = work_dir / "sparse_txt"
    training_image_dir = args.image_dir
    if args.gaussian_baseline:
        undistorted_dir = work_dir / "undistorted"
        write_progress(args.progress_file, "colmap_undistortion")
        undistort_command = [
            colmap,
            "image_undistorter",
            "--image_path",
            str(args.image_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(undistorted_dir),
            "--output_type",
            "COLMAP",
        ]
        if args.max_image_size is not None:
            undistort_command.extend(("--max_image_size", str(args.max_image_size)))
        undistortion_started_at = time.perf_counter()
        command_logs.append(run_command(undistort_command))
        stage_elapsed_seconds["undistortion"] = (
            time.perf_counter() - undistortion_started_at
        )
        model_source = undistorted_dir / "sparse"
        text_dir = undistorted_dir / "sparse_txt"
        training_image_dir = undistorted_dir / "images"
    text_dir.mkdir(parents=True, exist_ok=True)

    point_cloud_path = geometry_dir / "points.ply"
    point_conversion_started_at = time.perf_counter()
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_source),
                "--output_path",
                str(point_cloud_path),
                "--output_type",
                "PLY",
            ]
        )
    )
    stage_elapsed_seconds["point_cloud_conversion"] = (
        time.perf_counter() - point_conversion_started_at
    )
    text_conversion_started_at = time.perf_counter()
    command_logs.append(
        run_command(
            [
                colmap,
                "model_converter",
                "--input_path",
                str(model_source),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
    )
    stage_elapsed_seconds["text_conversion"] = (
        time.perf_counter() - text_conversion_started_at
    )

    camera_payload = build_camera_payload(text_dir)
    if args.gaussian_baseline:
        camera_models = {camera["model"] for camera in camera_payload["cameras"]}
        if not camera_models <= {"SIMPLE_PINHOLE", "PINHOLE"}:
            raise RuntimeError(
                f"COLMAP undistortion produced unsupported camera models: {sorted(camera_models)}"
            )
        if len(camera_payload["images"]) < 12:
            raise RuntimeError("Gaussian baseline requires at least 12 registered images")
    write_json(geometry_dir / "cameras.json", camera_payload)

    num_points = read_ply_vertex_count(point_cloud_path)
    final_input_count = len(image_paths)
    if args.video_selection is not None:
        final_selection = json.loads(args.video_selection.read_text(encoding="utf-8"))
        final_input_count = len(final_selection["selected"])
    elapsed_seconds = time.perf_counter() - started_at
    timing_path = output_dir / "diagnostics" / "colmap_timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_payload = {
        "schema_version": 1,
        "profile": "colmap_timing_v1",
        "colmap_executable": colmap,
        "colmap_build": colmap_build,
        "feature": colmap_frontend_provenance(feature_profile, local_matcher),
        "pairing": pairing_profile,
        "pairing_command": pairing_command,
        "vocab_tree_sha256": vocab_tree_sha256,
        "geometric_verification": geometric_verification.provenance(),
        "camera_calibration": (
            camera_diagnostics["calibration"]
            if camera_diagnostics is not None
            else None
        ),
        "mapper": (
            str(pose_recovery["effective_mapper"])
            if pose_recovery is not None
            else "incremental"
        ),
        "requested_mapper": "incremental",
        "colmap_random_seed": 0,
        "effective_database_path": database_path.resolve()
        .relative_to(output_dir.resolve())
        .as_posix(),
        "source_database_sha256": (
            pose_recovery["source_database_sha256"]
            if pose_recovery is not None
            else sha256_file(database_path) if database_path.is_file() else None
        ),
        "effective_database_sha256": (
            pose_recovery["effective_database_sha256"]
            if pose_recovery is not None
            else sha256_file(database_path) if database_path.is_file() else None
        ),
        "sfm_pose_health_path": (
            pose_health_path.resolve().relative_to(output_dir.resolve()).as_posix()
            if pose_health is not None
            else None
        ),
        "sfm_pose_recovery_path": (
            (output_dir / "diagnostics" / "sfm_pose_recovery.json")
            .resolve()
            .relative_to(output_dir.resolve())
            .as_posix()
            if pose_recovery is not None
            else None
        ),
        "matcher": pairing_command.removesuffix("_matcher"),
        "video_profile": (
            str(video_selection.get("profile"))
            if video_selection is not None
            else None
        ),
        "initial_video_selection_sha256": initial_video_selection_sha256,
        "sfm_frontend_contract_path": frontend_contract_path.resolve()
        .relative_to(output_dir.resolve())
        .as_posix(),
        "stage_elapsed_seconds": stage_elapsed_seconds,
        "total_elapsed_seconds": elapsed_seconds,
        "v2_mapper_options": v2_mapper_options(video_selection),
        "v2_mapper_seed_count": len(mapper_seed_names),
        "initial_registration_expansion_status": (
            expansion_diagnostics.get("status")
            if expansion_diagnostics is not None
            else None
        ),
    }
    write_json(timing_path, timing_payload)
    log_lines = [
        "backend=colmap",
        f"num_images={final_input_count}",
        f"initial_input_count={len(image_paths)}",
        f"registered_images={len(camera_payload['images'])}",
        f"registration_ratio={len(camera_payload['images']) / final_input_count:.6f}",
        f"num_points={num_points}",
        f"selected_sparse_model={model_dir.name}",
        f"selected_sparse_registered_images={registered_images}",
        f"selected_sparse_points={sparse_points}",
        f"initial_sparse_registered_images={initial_registered_images}",
        f"initial_sparse_points={initial_sparse_points}",
        f"v2_mapper_seed_count={len(mapper_seed_names)}",
        f"video_initial_registration_expansion_status={expansion_diagnostics['status'] if expansion_diagnostics is not None else 'not_requested'}",
        f"video_initial_registration_expansion_passes={expansion_diagnostics['accepted_pass_count'] if expansion_diagnostics is not None else 0}",
        f"video_registration_recovery_status={recovery_diagnostics['status'] if recovery_diagnostics is not None else 'not_requested'}",
        f"video_registration_recovery_rounds={len(recovery_diagnostics['rounds']) if recovery_diagnostics is not None else 0}",
        f"training_image_dir={training_image_dir}",
        f"camera_models={','.join(sorted(camera['model'] for camera in camera_payload['cameras']))}",
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
        *(
            [
                f"{key}={value}"
                for key, value in camera_metrics.items()
            ]
            if camera_diagnostics is not None
            else ["sfm_camera_calibration_profile=legacy_cli"]
        ),
        f"sfm_mapper={pose_recovery['effective_mapper'] if pose_recovery is not None else 'incremental'}",
        f"sfm_pose_health_status={pose_health['status'] if pose_health is not None else 'not_run'}",
        f"sfm_pose_health_reason_codes={','.join(pose_health['reason_codes']) if pose_health is not None else ''}",
        f"sfm_pose_recovery_status={pose_recovery['status'] if pose_recovery is not None else 'not_run'}",
        f"sfm_pose_recovery_applied={str(bool(pose_recovery and pose_recovery['recovery_applied'])).lower()}",
        f"sfm_pose_recovery_removed_camera_count={len(pose_recovery.get('selected', {}).get('excluded_image_ids', [])) if pose_recovery is not None else 0}",
        f"effective_colmap_database={database_path}",
        f"matcher={pairing_command.removesuffix('_matcher')}",
        f"sequential_overlap={sequential_overlap_value if sequential_overlap_value is not None else 'default'}",
        f"vocab_tree={vocab_tree_path if vocab_tree_path is not None else 'none'}",
        f"single_camera={legacy_single_camera if camera_plan is None else camera_plan.calibration.sharing_policy == 'single_camera'}",
        f"colmap_executable={colmap}",
        f"colmap_build={colmap_build}",
        f"use_gpu={args.use_gpu}",
        f"gpu_index={args.gpu_index if args.gpu_index is not None else 'all_visible'}",
        f"num_threads={args.num_threads if args.num_threads is not None else 'auto'}",
        f"max_image_size={args.max_image_size if args.max_image_size is not None else 'original'}",
        f"gaussian_baseline={args.gaussian_baseline}",
        *(
            [
                "sfm_camera_calibration_diagnostics="
                + camera_diagnostics_path.relative_to(output_dir).as_posix()
            ]
            if camera_diagnostics is not None
            else []
        ),
        f"stage_elapsed_seconds={json.dumps(stage_elapsed_seconds, sort_keys=True)}",
        f"timing_diagnostics={timing_path}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        *command_logs,
    ]
    (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {point_cloud_path}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
    if camera_diagnostics is not None:
        print(f"wrote {camera_diagnostics_path}")
    if pose_health is not None:
        print(f"wrote {pose_health_path}")
        print(f"sfm_pose_health_status={pose_health['status']}")
        print(f"sfm_pose_recovery_status={pose_recovery['status']}")
    print(f"registered_images={len(camera_payload['images'])}")
    print(f"num_points={num_points}")


def parse_gpu_indices(value: str) -> str:
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated non-negative integers")
    return ",".join(str(int(part)) for part in parts)


def discover_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def write_progress(path: Path | None, stage: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"stage": stage}) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def colmap_version(colmap: str) -> str:
    completed = subprocess.run(
        [colmap, "-h"], check=True, capture_output=True, text=True
    )
    output = completed.stdout + completed.stderr
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " ".join(lines[:2]) or "unknown"


def mapper_image_list_option(colmap: str) -> str:
    completed = subprocess.run(
        [colmap, "mapper", "-h"], check=True, capture_output=True, text=True
    )
    output = completed.stdout + completed.stderr
    for option in ("--Mapper.image_list_path", "--image_list_path"):
        if option in output:
            return option
    raise RuntimeError("COLMAP mapper does not support an image list option")


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "command="
            + " ".join(command)
            + f"\nreturncode={exc.returncode}"
            + "\nstdout="
            + str(exc.stdout or "").strip()
            + "\nstderr="
            + str(exc.stderr or "").strip()
        ) from exc
    return "command=" + " ".join(command) + "\nstdout=" + completed.stdout.strip() + "\nstderr=" + completed.stderr.strip()


def select_or_recover_sparse_model(
    *,
    colmap: str,
    sparse_dir: Path,
    database_path: Path,
    image_dir: Path,
    work_dir: Path,
    output_dir: Path,
    selected_timestamps: dict[str, float] | None,
    mapper_seed_path: Path | None,
    use_gpu: bool,
    gpu_index: str | None,
    num_threads: int | None,
    command_logs: list[str],
) -> tuple[Path, int, int, Path, dict[str, Any]]:
    recovery_path = output_dir / "diagnostics" / "sfm_pose_recovery.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse models")
    record: dict[str, Any] = {
        "schema_version": 1,
        "profile": "sfm_pose_recovery_v1",
        "status": "not_needed",
        "requested_mapper": "incremental",
        "effective_mapper": None,
        "recovery_applied": False,
        "primary_candidates": [],
        "recovery_candidates": [],
        "selected": None,
        "source_database_sha256": (
            sha256_file(database_path) if database_path.is_file() else None
        ),
        "test_rgb_loaded": False,
    }
    healthy: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda path: path.name):
        evaluated = _evaluate_pose_candidate(
            colmap=colmap,
            model_dir=candidate,
            text_dir=work_dir / "pose_health" / f"primary-{candidate.name}",
            database_path=database_path,
            selected_timestamps=selected_timestamps,
            command_logs=command_logs,
            kind="incremental",
        )
        record["primary_candidates"].append(evaluated)
        if evaluated["accepted"]:
            healthy.append(evaluated)
    effective_database = database_path
    if not healthy:
        record["status"] = "recovery_attempted"
        global_candidate = _try_global_pose_recovery(
            colmap=colmap,
            database_path=database_path,
            image_dir=image_dir,
            recovery_dir=work_dir / "pose_recovery" / "global",
            selected_timestamps=selected_timestamps,
            mapper_seed_path=mapper_seed_path,
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            num_threads=num_threads,
            command_logs=command_logs,
        )
        record["recovery_candidates"].append(global_candidate)
        if global_candidate.get("accepted"):
            healthy.append(global_candidate)
        else:
            repairable_primary = [
                candidate
                for candidate in record["primary_candidates"]
                if candidate.get("pose_health", {})
                .get("automatic_repair", {})
                .get("eligible")
                is True
            ]
            primary = max(
                repairable_primary or record["primary_candidates"],
                key=lambda value: (
                    int(value["registered_count"]),
                    int(value["point_count"]),
                    str(value["model_path"]),
                ),
            )
            core_candidate = _try_core_pose_repair(
                colmap=colmap,
                primary=primary,
                database_path=database_path,
                recovery_dir=work_dir / "pose_recovery" / "core",
                selected_timestamps=selected_timestamps,
                use_gpu=use_gpu,
                gpu_index=gpu_index,
                command_logs=command_logs,
            )
            record["recovery_candidates"].append(core_candidate)
            if core_candidate.get("accepted"):
                healthy.append(core_candidate)
    if not healthy:
        record["status"] = "failed"
        record["reason"] = "no_healthy_sfm_pose_candidate"
        _relativize_pose_recovery_paths(record, output_dir)
        write_json(recovery_path, record)
        reasons = sorted(
            {
                str(reason)
                for candidate in (
                    record["primary_candidates"] + record["recovery_candidates"]
                )
                for reason in (
                    candidate.get("gate_reason_codes", [])
                    or candidate.get("pose_health", {}).get("reason_codes", [])
                )
            }
        )
        raise RuntimeError(
            "SfM pose recovery produced no healthy model: "
            + (",".join(reasons) or "no_candidate_completed")
        )
    selected = max(
        healthy,
        key=lambda value: (
            int(value["registered_count"]),
            float(
                (value.get("registration_timeline") or {}).get(
                    "temporal_coverage", -1.0
                )
            ),
            int(value["point_count"]),
            value["kind"] == "incremental",
            str(value["model_path"]),
        ),
    )
    if selected["kind"] != "incremental":
        record["status"] = "recovered"
        record["recovery_applied"] = True
    record["effective_mapper"] = selected["kind"]
    record["selected"] = {
        **{
            key: selected[key]
            for key in (
                "kind",
                "model_path",
                "database_path",
                "registered_count",
                "point_count",
            )
        },
        **(
            {"excluded_image_ids": selected["excluded_image_ids"]}
            if "excluded_image_ids" in selected
            else {}
        ),
    }
    selected_model = Path(selected["model_path"])
    effective_database = Path(selected["database_path"])
    record["effective_database_sha256"] = (
        record["source_database_sha256"]
        if effective_database.resolve() == database_path.resolve()
        else sha256_file(effective_database)
        if effective_database.is_file()
        else None
    )
    _relativize_pose_recovery_paths(record, output_dir)
    write_json(recovery_path, record)
    return (
        selected_model,
        int(selected["registered_count"]),
        int(selected["point_count"]),
        effective_database,
        record,
    )


def _evaluate_pose_candidate(
    *,
    colmap: str,
    model_dir: Path,
    text_dir: Path,
    database_path: Path,
    selected_timestamps: dict[str, float] | None,
    command_logs: list[str],
    kind: str,
) -> dict[str, Any]:
    try:
        return _evaluate_pose_candidate_unchecked(
            colmap=colmap,
            model_dir=model_dir,
            text_dir=text_dir,
            database_path=database_path,
            selected_timestamps=selected_timestamps,
            command_logs=command_logs,
            kind=kind,
        )
    except Exception as exc:
        try:
            registered, points = read_sparse_model_counts(model_dir)
        except Exception:
            registered = points = 0
        return {
            "kind": kind,
            "status": "failed",
            "accepted": False,
            "gate_reason_codes": ["candidate_evaluation_failed"],
            "reason": "candidate_evaluation_failed",
            "error_type": exc.__class__.__name__,
            "model_path": str(model_dir),
            "database_path": str(database_path),
            "registered_count": registered,
            "point_count": points,
        }


def _evaluate_pose_candidate_unchecked(
    *,
    colmap: str,
    model_dir: Path,
    text_dir: Path,
    database_path: Path,
    selected_timestamps: dict[str, float] | None,
    command_logs: list[str],
    kind: str,
) -> dict[str, Any]:
    text_dir.mkdir(parents=True, exist_ok=True)
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
    registered, points = read_sparse_model_counts(model_dir)
    model_files_sha256 = {
        name: sha256_file(model_dir / name)
        for name in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
        )
        if (model_dir / name).is_file()
    }
    health = build_sfm_pose_health_from_text(
        model_dir=text_dir,
        selected_timestamps=selected_timestamps,
        database_path=database_path,
    )
    timeline = (
        health.get("temporal", {}).get("registration_timeline")
        if selected_timestamps is not None
        else None
    )
    gate_reasons = list(health.get("reason_codes", []))
    if selected_timestamps is not None and timeline is None:
        gate_reasons.append("video_registration_timeline_missing")
    if registered < MIN_VIDEO_REGISTERED_COUNT:
        gate_reasons.append("registered_count_below_gate")
    if timeline is not None:
        if float(timeline["registration_rate"]) < MIN_VIDEO_REGISTRATION_RATE:
            gate_reasons.append("registration_rate_below_gate")
        if float(timeline["temporal_coverage"]) < MIN_VIDEO_TEMPORAL_COVERAGE:
            gate_reasons.append("temporal_coverage_below_gate")
    accepted = not gate_reasons
    return {
        "kind": kind,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "gate_reason_codes": gate_reasons,
        "model_path": str(model_dir),
        "database_path": str(database_path),
        "registered_count": registered,
        "point_count": points,
        "model_files_sha256": model_files_sha256,
        "registration_timeline": timeline,
        "pose_health": health,
    }


def _try_global_pose_recovery(
    *,
    colmap: str,
    database_path: Path,
    image_dir: Path,
    recovery_dir: Path,
    selected_timestamps: dict[str, float] | None,
    mapper_seed_path: Path | None,
    use_gpu: bool,
    gpu_index: str | None,
    num_threads: int | None,
    command_logs: list[str],
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "kind": "global_recovery_v1",
        "status": "failed",
        "accepted": False,
    }
    try:
        recovery_dir.mkdir(parents=True, exist_ok=False)
        copied_database = recovery_dir / "database.db"
        _backup_sqlite_database(database_path, copied_database)
        command_logs.append(
            run_command(
                [
                    colmap,
                    "view_graph_calibrator",
                    "--database_path",
                    str(copied_database),
                    "--default_random_seed",
                    "0",
                ]
            )
        )
        output = recovery_dir / "model"
        output.mkdir()
        command = [
            colmap,
            "global_mapper",
            "--database_path",
            str(copied_database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(output),
            "--default_random_seed",
            "0",
            "--GlobalMapper.gp_use_gpu",
            "1" if use_gpu else "0",
            "--GlobalMapper.ba_ceres_use_gpu",
            "1" if use_gpu else "0",
        ]
        if mapper_seed_path is not None:
            command.extend(
                ("--GlobalMapper.image_list_path", str(mapper_seed_path))
            )
        if num_threads is not None:
            command.extend(("--GlobalMapper.num_threads", str(num_threads)))
        if use_gpu and gpu_index is not None:
            first_gpu = gpu_index.split(",")[0]
            command.extend(
                (
                    "--GlobalMapper.gp_gpu_index",
                    first_gpu,
                    "--GlobalMapper.ba_ceres_gpu_index",
                    first_gpu,
                )
            )
        command_logs.append(run_command(command))
        evaluated = [
            _evaluate_pose_candidate(
                colmap=colmap,
                model_dir=model,
                text_dir=recovery_dir / "text" / model.name,
                database_path=copied_database,
                selected_timestamps=selected_timestamps,
                command_logs=command_logs,
                kind="global_recovery_v1",
            )
            for model in _model_candidates(output)
        ]
        if not evaluated:
            candidate["reason"] = "global_mapper_produced_no_model"
            return candidate
        return max(
            evaluated,
            key=lambda value: (
                bool(value["accepted"]),
                int(value["registered_count"]),
                float(
                    (value.get("registration_timeline") or {}).get(
                        "temporal_coverage", -1.0
                    )
                ),
                int(value["point_count"]),
            ),
        )
    except Exception as exc:
        candidate["reason"] = "global_recovery_failed"
        candidate["error_type"] = exc.__class__.__name__
        return candidate


def _try_core_pose_repair(
    *,
    colmap: str,
    primary: dict[str, Any],
    database_path: Path,
    recovery_dir: Path,
    selected_timestamps: dict[str, float] | None,
    use_gpu: bool,
    gpu_index: str | None,
    command_logs: list[str],
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "kind": "incremental_core_repair_v1",
        "status": "not_eligible",
        "accepted": False,
    }
    eligibility = primary["pose_health"].get("automatic_repair", {})
    candidate["eligibility"] = eligibility
    if eligibility.get("eligible") is not True:
        candidate["reason"] = str(eligibility.get("reason", "not_eligible"))
        return candidate
    try:
        recovery_dir.mkdir(parents=True, exist_ok=False)
        image_ids_path = recovery_dir / "excluded-image-ids.txt"
        image_ids = [int(value) for value in eligibility["excluded_image_ids"]]
        image_ids_path.write_text(
            "".join(f"{value}\n" for value in image_ids), encoding="utf-8"
        )
        deleted = recovery_dir / "deleted"
        filtered = recovery_dir / "filtered"
        adjusted = recovery_dir / "adjusted"
        for path in (deleted, filtered, adjusted):
            path.mkdir()
        command_logs.append(
            run_command(
                [
                    colmap,
                    "image_deleter",
                    "--input_path",
                    str(primary["model_path"]),
                    "--output_path",
                    str(deleted),
                    "--image_ids_path",
                    str(image_ids_path),
                ]
            )
        )
        command_logs.append(
            run_command(
                [
                    colmap,
                    "point_filtering",
                    "--input_path",
                    str(deleted),
                    "--output_path",
                    str(filtered),
                    "--min_track_len",
                    "3",
                ]
            )
        )
        command = [
            colmap,
            "bundle_adjuster",
            "--input_path",
            str(filtered),
            "--output_path",
            str(adjusted),
            "--default_random_seed",
            "0",
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--BundleAdjustmentCeres.function_tolerance",
            "0.000001",
        ]
        if use_gpu:
            command.extend(("--BundleAdjustmentCeres.use_gpu", "1"))
            if gpu_index is not None:
                command.extend(
                    ("--BundleAdjustmentCeres.gpu_index", gpu_index.split(",")[0])
                )
        command_logs.append(run_command(command))
        evaluated = _evaluate_pose_candidate(
            colmap=colmap,
            model_dir=adjusted,
            text_dir=recovery_dir / "text",
            database_path=database_path,
            selected_timestamps=selected_timestamps,
            command_logs=command_logs,
            kind="incremental_core_repair_v1",
        )
        evaluated["excluded_image_ids"] = image_ids
        return evaluated
    except Exception as exc:
        candidate["status"] = "failed"
        candidate["reason"] = "core_repair_failed"
        candidate["error_type"] = exc.__class__.__name__
        return candidate


def _relativize_pose_recovery_paths(
    record: dict[str, Any], output_dir: Path
) -> None:
    root = output_dir.resolve()
    candidates = [
        *record.get("primary_candidates", []),
        *record.get("recovery_candidates", []),
    ]
    if isinstance(record.get("selected"), dict):
        candidates.append(record["selected"])
    for candidate in candidates:
        for key in ("model_path", "database_path"):
            if key not in candidate:
                continue
            path = Path(str(candidate[key])).resolve()
            try:
                candidate[key] = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise RuntimeError(
                    f"SfM pose recovery {key} escapes the output directory"
                ) from exc


def _backup_sqlite_database(source: Path, destination: Path) -> None:
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as source_db:
        with sqlite3.connect(destination) as destination_db:
            source_db.backup(destination_db)


def _model_candidates(root: Path) -> list[Path]:
    if all((root / name).is_file() for name in ("images.bin", "points3D.bin")):
        return [root]
    if all((root / name).is_file() for name in ("images.txt", "points3D.txt")):
        return [root]
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and (
                all((path / name).is_file() for name in ("images.bin", "points3D.bin"))
                or all((path / name).is_file() for name in ("images.txt", "points3D.txt"))
            )
        ),
        key=lambda path: path.name,
    )


def find_largest_sparse_model(sparse_dir: Path) -> tuple[Path, int, int]:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse models")
    ranked = [(*read_sparse_model_counts(path), path) for path in candidates]
    registered_images, points, selected = max(ranked, key=lambda item: (item[0], item[1], item[2].name))
    return selected, registered_images, points


def read_sparse_model_counts(model_dir: Path) -> tuple[int, int]:
    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"
    if images_bin.is_file() and points_bin.is_file():
        return _binary_count(images_bin), _binary_count(points_bin)
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"
    if images_txt.is_file() and points_txt.is_file():
        image_rows = [
            line
            for line in images_txt.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        point_rows = [
            line
            for line in points_txt.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        return len(image_rows) // 2, len(point_rows)
    raise RuntimeError(f"COLMAP sparse model is incomplete: {model_dir}")


def _binary_count(path: Path) -> int:
    with path.open("rb") as handle:
        payload = handle.read(8)
    if len(payload) != 8:
        raise RuntimeError(f"COLMAP binary model file is truncated: {path}")
    return int(struct.unpack("<Q", payload)[0])


def build_camera_payload(text_dir: Path) -> dict[str, Any]:
    cameras = parse_colmap_cameras(text_dir / "cameras.txt")
    images = parse_colmap_images(text_dir / "images.txt")
    return {
        "coordinate_system": "colmap_world",
        "cameras": cameras,
        "images": images,
    }


def parse_colmap_cameras(path: Path) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras.append(
            {
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(value) for value in parts[4:]],
            }
        )
    return cameras


def parse_colmap_images(path: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    data_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for index in range(0, len(data_lines), 2):
        parts = data_lines[index].split(maxsplit=9)
        images.append(
            {
                "image_id": int(parts[0]),
                "qvec": [float(value) for value in parts[1:5]],
                "tvec": [float(value) for value in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        )
    return images


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as file:
        for raw_line in file:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
