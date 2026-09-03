from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

from image3d_scenegraph.geometry.colmap import (
    COLMAP_FEATURE_PROFILE_IDS,
    COLMAP_GEOMETRIC_VERIFICATION_IDS,
    COLMAP_LEGACY_MATCHER_IDS,
    COLMAP_LOCAL_MATCHER_IDS,
    COLMAP_PAIRING_IDS,
    ColmapFeatureError,
    colmap_frontend_provenance,
    resolve_colmap_executable,
    resolve_colmap_feature_profile,
    resolve_colmap_geometric_verification,
    resolve_colmap_local_matcher,
    resolve_colmap_pairing,
    sha256_file,
)
from image3d_scenegraph.geometry.video_recovery import (
    expand_v2_initial_registration,
    recover_video_registration,
    sequential_overlap,
    v2_mapper_options,
    v2_mapper_seed_image_names,
)
from image3d_scenegraph.video.keyframes import V2_PROFILE_ID


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
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)
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

    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(work_dir / "database.db"),
        "--image_path",
        str(args.image_dir),
        "--ImageReader.single_camera",
        "1" if args.single_camera else "0",
        "--FeatureExtraction.use_gpu",
        "1" if args.use_gpu else "0",
        *feature_profile.extraction_options,
    ]
    if args.gpu_index is not None:
        feature_command.extend(("--FeatureExtraction.gpu_index", args.gpu_index))
    if args.gaussian_baseline:
        feature_command.extend(("--ImageReader.camera_model", "OPENCV"))
    if args.num_threads is not None:
        feature_command.extend(("--FeatureExtraction.num_threads", str(args.num_threads)))
    mapper_command = [
        colmap,
        "mapper",
        "--database_path",
        str(work_dir / "database.db"),
        "--image_path",
        str(args.image_dir),
        "--output_path",
        str(sparse_dir),
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
    commands = [
        ("feature_extraction", "colmap_feature_extraction", feature_command),
        ("feature_matching", "colmap_feature_matching", matcher_command),
        ("mapping", "colmap_mapping", mapper_command),
    ]
    command_logs = []
    stage_elapsed_seconds: dict[str, float] = {}
    for timing_stage, progress_stage, command in commands:
        write_progress(args.progress_file, progress_stage)
        command_started_at = time.perf_counter()
        command_logs.append(run_command(command))
        stage_elapsed_seconds[timing_stage] = time.perf_counter() - command_started_at

    model_dir, registered_images, sparse_points = find_largest_sparse_model(sparse_dir)
    initial_registered_images = registered_images
    initial_sparse_points = sparse_points
    expansion_diagnostics: dict[str, Any] | None = None
    if video_selection is not None and video_selection.get("profile") == V2_PROFILE_ID:
        expansion_started_at = time.perf_counter()
        model_dir, expansion_diagnostics, expansion_logs = (
            expand_v2_initial_registration(
                colmap=colmap,
                database_path=work_dir / "database.db",
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
            database_path=work_dir / "database.db",
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
        "mapper": "incremental",
        "matcher": pairing_command.removesuffix("_matcher"),
        "video_profile": (
            str(video_selection.get("profile"))
            if video_selection is not None
            else None
        ),
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
        "sfm_mapper=incremental",
        f"matcher={pairing_command.removesuffix('_matcher')}",
        f"sequential_overlap={sequential_overlap_value if sequential_overlap_value is not None else 'default'}",
        f"vocab_tree={vocab_tree_path if vocab_tree_path is not None else 'none'}",
        f"single_camera={args.single_camera}",
        f"colmap_executable={colmap}",
        f"colmap_build={colmap_build}",
        f"use_gpu={args.use_gpu}",
        f"gpu_index={args.gpu_index if args.gpu_index is not None else 'all_visible'}",
        f"num_threads={args.num_threads if args.num_threads is not None else 'auto'}",
        f"max_image_size={args.max_image_size if args.max_image_size is not None else 'original'}",
        f"gaussian_baseline={args.gaussian_baseline}",
        f"stage_elapsed_seconds={json.dumps(stage_elapsed_seconds, sort_keys=True)}",
        f"timing_diagnostics={timing_path}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        *command_logs,
    ]
    (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"wrote {point_cloud_path}")
    print(f"wrote {geometry_dir / 'cameras.json'}")
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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
