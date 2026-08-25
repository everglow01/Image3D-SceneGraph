from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from image3d_scenegraph.video.registration import (
    MIN_VIDEO_REGISTERED_COUNT,
    MIN_VIDEO_REGISTRATION_RATE,
    MIN_VIDEO_TEMPORAL_COVERAGE,
    analyze_registration_timeline,
)


VALID_GEOMETRY_BACKENDS = {
    "mock",
    "vggt",
    "colmap",
    "colmap_vggt",
    "dust3r",
    "mast3r",
    "project_3dgs",
}
VALID_OUTPUT_TYPES = {"point_cloud", "mesh", "gaussian_splat"}


class ReconstructionError(ValueError):
    """Raised when a reconstruction request cannot be served."""


@dataclass(frozen=True)
class ReconstructionContext:
    job_id: str
    job_dir: Path
    mode: str
    input_assets: list[dict[str, Any]]
    options: dict[str, int | float | str]
    cancel_requested: Callable[[], bool] | None = None
    progress_callback: Callable[[str, float], None] | None = None


@dataclass(frozen=True)
class ReconstructionResult:
    stage: str
    assets: dict[str, str]
    metrics: dict[str, int | float | str | bool]
    log_lines: list[str]


class ReconstructionAdapter(Protocol):
    backend: str
    output_type: str

    def run(self, context: ReconstructionContext) -> ReconstructionResult:
        """Write reconstruction assets under the job directory and return metadata."""


class ProjectGaussianAdapter:
    backend = "project_3dgs"
    output_type = "gaussian_splat"

    @staticmethod
    def _colmap_progress_callback(
        context: ReconstructionContext, path: Path
    ) -> Callable[[], None]:
        progress_by_stage = {
            "colmap_feature_extraction": 0.16,
            "colmap_feature_matching": 0.20,
            "colmap_mapping": 0.26,
            "video_registration_recovery_round_1": 0.28,
            "video_registration_recovery_round_2": 0.30,
            "colmap_undistortion": 0.31,
        }
        last_stage: str | None = None

        def update() -> None:
            nonlocal last_stage
            try:
                stage = str(json.loads(path.read_text(encoding="utf-8"))["stage"])
            except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
                return
            if stage == last_stage or stage not in progress_by_stage:
                return
            last_stage = stage
            _adapter_progress(context, stage, progress_by_stage[stage])

        return update

    @staticmethod
    def _video_progress_callback(
        context: ReconstructionContext, path: Path
    ) -> Callable[[], None]:
        progress_by_stage = {
            "video_probing": 0.06,
            "video_frame_scoring": 0.08,
            "video_frame_extraction": 0.12,
        }
        last_stage: str | None = None

        def update() -> None:
            nonlocal last_stage
            try:
                stage = str(json.loads(path.read_text(encoding="utf-8"))["stage"])
            except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
                return
            if stage == last_stage or stage not in progress_by_stage:
                return
            last_stage = stage
            _adapter_progress(context, stage, progress_by_stage[stage])

        return update

    @staticmethod
    def _vggt_ba_progress_callback(
        context: ReconstructionContext, path: Path
    ) -> Callable[[], None]:
        progress_by_stage = {
            "vggt_ba_descriptors": 0.16,
            "vggt_ba_windows": 0.18,
            "vggt_ba_recovery": 0.20,
            "vggt_ba_pose_graph": 0.22,
            "vggt_ba_feature_extraction": 0.25,
            "vggt_ba_feature_matching": 0.27,
            "vggt_ba_global_triangulation": 0.285,
            "vggt_ba_image_registration": 0.295,
            "vggt_ba_global_bundle_adjustment": 0.30,
            "colmap_fallback_mapping": 0.30,
            "video_registration_recovery_round_1": 0.303,
            "video_registration_recovery_round_2": 0.307,
            "colmap_undistortion": 0.31,
        }
        last_stage: str | None = None

        def update() -> None:
            nonlocal last_stage
            try:
                stage = str(json.loads(path.read_text(encoding="utf-8"))["stage"])
            except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
                return
            if stage == last_stage or stage not in progress_by_stage:
                return
            last_stage = stage
            _adapter_progress(context, stage, progress_by_stage[stage])

        return update

    def run(self, context: ReconstructionContext) -> ReconstructionResult:
        if context.mode not in {"multi_image", "video"}:
            raise ReconstructionError("project 3DGS training requires multi_image or video input")
        project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
        video_script = project_root / "scripts" / "extract_video_keyframes.py"
        colmap_script = project_root / "scripts" / "run_colmap_sparse.py"
        vggt_ba_script = project_root / "scripts" / "run_vggt_ba_sparse.py"
        trainer_script = project_root / "scripts" / "run_gaussian_training.py"
        evaluator_script = project_root / "scripts" / "evaluate_gaussian.py"
        exporter_script = project_root / "scripts" / "export_gaussian.py"
        filter_script = project_root / "scripts" / "filter_gaussian_vggt.py"
        sor_filter_script = project_root / "scripts" / "filter_gaussian_sor.py"
        required_scripts = [trainer_script, evaluator_script, exporter_script]
        postprocess = str(context.options.get("gaussian_postprocess", "none"))
        if postprocess not in {"none", "vggt_visibility_v1"}:
            raise ReconstructionError(
                f"unsupported Gaussian postprocess: {postprocess}"
            )
        if postprocess == "vggt_visibility_v1":
            required_scripts.append(filter_script)
        sor_filter = _choice_option(
            context,
            "gaussian_sor_filter",
            "IMAGE3D_GAUSSIAN_SOR_FILTER",
            "on",
            {"on", "off"},
        )
        if sor_filter == "on":
            required_scripts.append(sor_filter_script)
        geometry_source = str(
            context.options.get("gaussian_geometry_source", "colmap")
        )
        if geometry_source == "colmap":
            required_scripts.append(colmap_script)
        elif geometry_source == "vggt_ba":
            if context.mode != "video":
                raise ReconstructionError(
                    "vggt_ba Gaussian geometry currently requires video mode"
                )
            required_scripts.append(vggt_ba_script)
        else:
            raise ReconstructionError(
                f"unsupported Gaussian geometry source: {geometry_source}"
            )
        if context.mode == "video":
            required_scripts.append(video_script)
        if not all(path.is_file() for path in required_scripts):
            raise ReconstructionError("project 3DGS runner scripts are missing")
        config_path = context.job_dir / "gaussian_config.json"
        dataset_path = context.job_dir / "dataset.json"
        gaussian_longest_edge = int(context.options.get("gaussian_longest_edge", 1280))
        image_dir = context.job_dir / "input" / "images"
        video_assets: dict[str, str] = {}
        video_metrics: dict[str, int | float | str | bool] = {}
        video_selection: dict[str, Any] | None = None
        video_selection_path: Path | None = None
        video_source_path: Path | None = None
        video_recovery_log_lines: list[str] = []
        video_profile = str(
            context.options.get("video_keyframe_profile", "standard_v1")
        )
        if video_profile not in {"standard_v1", "standard_v2"}:
            raise ReconstructionError(
                f"unsupported video keyframe profile: {video_profile}"
            )
        if context.mode == "video":
            if len(context.input_assets) != 1:
                raise ReconstructionError("video mode requires exactly one persisted input")
            video_source_path = context.job_dir / str(context.input_assets[0]["path"])
            video_progress_path = context.job_dir / "frames" / "progress.json"
            command_video = [
                os.environ.get("IMAGE3D_PYTHON", sys.executable),
                str(video_script),
                "--input",
                str(video_source_path),
                "--output-dir",
                str(context.job_dir),
                "--profile",
                video_profile,
                "--longest-edge",
                str(gaussian_longest_edge),
                "--rotation",
                str(context.options.get("video_rotation", "auto")),
                "--progress-file",
                str(video_progress_path),
            ]
            _adapter_progress(context, "video_probing", 0.06)
            _run_adapter_command(
                command_video,
                context,
                project_root,
                env=None,
                poll_callback=self._video_progress_callback(context, video_progress_path),
            )
            video_selection_path = context.job_dir / "frames" / "selection.json"
            probe_path = context.job_dir / "diagnostics" / "video_probe.json"
            contact_sheet_path = context.job_dir / "diagnostics" / "video_keyframes.jpg"
            keyframe_timing_path = (
                context.job_dir / "diagnostics" / "video_keyframe_timing.json"
            )
            if not all(
                path.is_file()
                for path in (
                    video_selection_path,
                    probe_path,
                    contact_sheet_path,
                    keyframe_timing_path,
                )
            ):
                raise ReconstructionError("video keyframe extraction did not produce complete diagnostics")
            video_selection = json.loads(
                video_selection_path.read_text(encoding="utf-8")
            )
            expected_profile = f"video_keyframes_{video_profile}"
            if video_selection.get("profile") != expected_profile:
                raise ReconstructionError(
                    "video keyframe extraction returned a different profile than requested"
                )
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            try:
                keyframe_timing = json.loads(
                    keyframe_timing_path.read_text(encoding="utf-8")
                )
                if (
                    keyframe_timing.get("schema_version") != 1
                    or keyframe_timing.get("profile")
                    != "video_keyframe_timing_v1"
                    or keyframe_timing.get("video_profile")
                    != video_selection.get("profile")
                ):
                    raise ValueError("unsupported video keyframe timing schema")
                keyframe_elapsed_seconds = float(
                    keyframe_timing["elapsed_seconds"]
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ReconstructionError(
                    "video keyframe timing diagnostics are invalid"
                ) from exc
            image_dir = context.job_dir / "frames" / "selected"
            video_assets = {
                "video_probe": probe_path.relative_to(context.job_dir).as_posix(),
                "video_frame_selection": video_selection_path.relative_to(
                    context.job_dir
                ).as_posix(),
                "video_keyframe_contact_sheet": contact_sheet_path.relative_to(context.job_dir).as_posix(),
                "video_keyframe_timing": keyframe_timing_path.relative_to(
                    context.job_dir
                ).as_posix(),
            }
            video_metrics = {
                "video_profile": str(video_selection["profile"]),
                "video_keyframe_elapsed_seconds": keyframe_elapsed_seconds,
                "video_duration_seconds": float(video_selection["duration_seconds"]),
                "video_orientation": str(probe["orientation"]),
                "video_rotation_degrees": int(probe["rotation"]["applied_degrees"]),
                "video_source_width": int(probe["source_width"]),
                "video_source_height": int(probe["source_height"]),
                "video_display_width": int(probe["display_width"]),
                "video_display_height": int(probe["display_height"]),
                "video_candidate_count": int(video_selection["candidate_count"]),
                "video_selected_count": int(video_selection["selected_count"]),
                "video_rejection_counts": json.dumps(
                    video_selection.get("rejection_counts", {}), sort_keys=True
                ),
                "video_initial_selected_count": int(
                    video_selection["selected_count"]
                ),
            }
            if video_profile == "standard_v2":
                video_metrics.update(
                    video_base_selected_count=int(
                        video_selection["base_selected_count"]
                    ),
                    video_adaptive_selected_count=int(
                        video_selection["adaptive_selected_count"]
                    ),
                    video_recovery_selected_count=0,
                )
        sparse_dir = context.job_dir / "colmap" / "undistorted" / "sparse_txt"
        points_path = sparse_dir / "points3D.txt"
        default_colmap_threads = min(8, max(1, (os.cpu_count() or 1) // 2))
        try:
            colmap_threads = int(
                os.environ.get("IMAGE3D_COLMAP_NUM_THREADS", default_colmap_threads)
            )
        except ValueError as exc:
            raise ReconstructionError("IMAGE3D_COLMAP_NUM_THREADS must be an integer") from exc
        if colmap_threads < 1:
            raise ReconstructionError("IMAGE3D_COLMAP_NUM_THREADS must be at least 1")
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        geometry_assets: dict[str, str] = {}
        geometry_metrics: dict[str, int | float | str | bool] = {
            "gaussian_geometry_source": geometry_source,
            "gaussian_geometry_effective_source": geometry_source,
            "gaussian_geometry_fallback_applied": False,
        }
        colmap_matcher = _choice_option(
            context,
            "colmap_matcher",
            "IMAGE3D_GAUSSIAN_COLMAP_MATCHER",
            "exhaustive",
            {"exhaustive", "sequential"},
        )
        geometry_metrics["colmap_matcher"] = colmap_matcher
        matcher_args = ["--matcher", colmap_matcher]
        if colmap_matcher == "sequential":
            from image3d_scenegraph.geometry.colmap import resolve_colmap_vocab_tree

            vocab_tree = resolve_colmap_vocab_tree(project_root)
            if vocab_tree is None:
                raise ReconstructionError(
                    "sequential COLMAP matching requires the vocab tree; run "
                    "`uv run python scripts/setup_colmap_vocab_tree.py --install` "
                    "or set IMAGE3D_COLMAP_VOCAB_TREE"
                )
            matcher_args.extend(("--vocab-tree-path", str(vocab_tree)))
        video_geometry_args: list[str] = []
        if context.mode == "video" and video_profile == "standard_v2":
            if video_source_path is None or video_selection_path is None:
                raise ReconstructionError("standard_v2 video metadata is unavailable")
            video_geometry_args = [
                "--video-source",
                str(video_source_path),
                "--video-selection",
                str(video_selection_path),
            ]
        if geometry_source == "colmap":
            progress_path = context.job_dir / "colmap" / "progress.json"
            command_geometry = [
                os.environ.get("IMAGE3D_PYTHON", sys.executable),
                str(colmap_script),
                "--image-dir",
                str(image_dir),
                "--output-dir",
                str(context.job_dir),
                *video_geometry_args,
                *matcher_args,
                "--gaussian-baseline",
                "--use-gpu",
                "--num-threads",
                str(colmap_threads),
                "--max-image-size",
                str(gaussian_longest_edge),
                "--progress-file",
                str(progress_path),
            ]
            progress_callback = self._colmap_progress_callback(context, progress_path)
        else:
            external_root = Path(
                os.environ.get("IMAGE3D_EXTERNAL_ROOT", project_root / "external")
            )
            checkpoint_root = Path(
                os.environ.get("IMAGE3D_CHECKPOINT_ROOT", project_root / "checkpoints")
            )
            progress_path = context.job_dir / "vggt_ba" / "progress.json"
            command_geometry = [
                os.environ.get("IMAGE3D_PYTHON", sys.executable),
                str(vggt_ba_script),
                "--image-dir",
                str(image_dir),
                "--output-dir",
                str(context.job_dir),
                *video_geometry_args,
                *matcher_args,
                "--repo-dir",
                str(external_root / "vggt"),
                "--checkpoint-dir",
                str(checkpoint_root / "vggt" / "facebook--VGGT-1B"),
                "--dinov2-repo",
                str(external_root / "dinov2"),
                "--lightglue-repo",
                str(external_root / "lightglue"),
                "--dinov2-checkpoint",
                os.environ.get(
                    "IMAGE3D_DINOV2_CHECKPOINT",
                    str(checkpoint_root / "vggt" / "dinov2_vitb14_reg4_pretrain.pth"),
                ),
                "--tracker-checkpoint",
                os.environ.get(
                    "IMAGE3D_VGGSFM_TRACKER_CHECKPOINT",
                    str(checkpoint_root / "vggt" / "vggsfm_v2_tracker.pt"),
                ),
                "--aliked-checkpoint",
                os.environ.get(
                    "IMAGE3D_ALIKED_CHECKPOINT",
                    str(
                        checkpoint_root
                        / "vggt"
                        / "torch-hub"
                        / "checkpoints"
                        / "aliked-n16.pth"
                    ),
                ),
                "--max-image-size",
                str(gaussian_longest_edge),
                "--num-threads",
                str(colmap_threads),
                "--progress-file",
                str(progress_path),
            ]
            progress_callback = self._vggt_ba_progress_callback(context, progress_path)
        _adapter_progress(context, "geometry_reconstruction", 0.15)
        _run_adapter_command(
            command_geometry,
            context,
            project_root,
            env=(None if geometry_source == "colmap" else env),
            poll_callback=progress_callback,
        )
        cameras_path = context.job_dir / "geometry" / "cameras.json"
        if not cameras_path.is_file() or not points_path.is_file():
            raise ReconstructionError(
                f"{geometry_source} did not produce project 3DGS camera/sparse inputs"
            )
        if geometry_source == "colmap" and video_profile == "standard_v2":
            colmap_timing_path = (
                context.job_dir / "diagnostics" / "colmap_timing.json"
            )
            try:
                colmap_timing = json.loads(
                    colmap_timing_path.read_text(encoding="utf-8")
                )
                if (
                    colmap_timing.get("schema_version") != 1
                    or colmap_timing.get("profile") != "colmap_timing_v1"
                    or not isinstance(
                        colmap_timing.get("stage_elapsed_seconds"), dict
                    )
                ):
                    raise ValueError("unsupported COLMAP timing schema")
                colmap_timing_total = float(
                    colmap_timing["total_elapsed_seconds"]
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ReconstructionError(
                    "COLMAP geometry timing diagnostics are missing or invalid"
                ) from exc
            geometry_assets["colmap_timing"] = colmap_timing_path.relative_to(
                context.job_dir
            ).as_posix()
            geometry_metrics.update(
                colmap_geometry_elapsed_seconds=colmap_timing_total,
                colmap_geometry_stage_elapsed_seconds=json.dumps(
                    colmap_timing["stage_elapsed_seconds"], sort_keys=True
                ),
            )
        if video_selection_path is not None:
            try:
                final_video_selection = json.loads(
                    video_selection_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ReconstructionError(
                    "cannot read final video keyframe selection"
                ) from exc
            if final_video_selection.get("profile") != f"video_keyframes_{video_profile}":
                raise ReconstructionError(
                    "final video keyframe selection profile changed during geometry"
                )
            video_selection = final_video_selection
            video_metrics["video_selected_count"] = int(
                final_video_selection["selected_count"]
            )
            if video_profile == "standard_v2":
                video_metrics["video_recovery_selected_count"] = int(
                    final_video_selection.get("recovery_selected_count", 0)
                )
                recovery_path = (
                    context.job_dir
                    / "diagnostics"
                    / "video_registration_recovery.json"
                )
                recovery_metrics, video_recovery_log_lines = (
                    _read_video_registration_recovery(
                        final_video_selection,
                        recovery_path,
                    )
                )
                video_assets["video_registration_recovery"] = (
                    recovery_path.relative_to(context.job_dir).as_posix()
                )
                video_metrics.update(recovery_metrics)
        if geometry_source == "vggt_ba":
            diagnostics_path = context.job_dir / "diagnostics" / "vggt_ba.json"
            graph_path = context.job_dir / "vggt_ba" / "window_graph.json"
            if not diagnostics_path.is_file() or not graph_path.is_file():
                raise ReconstructionError("VGGT-BA geometry diagnostics are incomplete")
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            graph = diagnostics["window_graph"]
            effective_geometry_source = str(
                diagnostics.get("effective_geometry_source", "vggt_ba")
            )
            fallback_applied = bool(diagnostics.get("fallback_applied", False))
            fallback_reason = diagnostics.get("fallback_reason")
            allowed_fallback_reasons = {
                "vggt_graph_unusable_after_recovery",
                "vggt_seed_geometry_insufficient",
                "vggt_registration_gate_failed",
            }
            if effective_geometry_source not in {"vggt_ba", "colmap"}:
                raise ReconstructionError(
                    "VGGT-BA diagnostics contain an invalid effective geometry source"
                )
            if fallback_applied != (effective_geometry_source == "colmap"):
                raise ReconstructionError(
                    "VGGT-BA diagnostics contain inconsistent fallback state"
                )
            if fallback_applied and fallback_reason not in allowed_fallback_reasons:
                raise ReconstructionError(
                    "VGGT-BA diagnostics contain an unclassified fallback reason"
                )
            if not fallback_applied and fallback_reason is not None:
                raise ReconstructionError(
                    "VGGT-BA diagnostics contain a fallback reason without fallback"
                )
            geometry_assets = {
                "vggt_ba_diagnostics": diagnostics_path.relative_to(
                    context.job_dir
                ).as_posix(),
                "vggt_ba_window_graph": graph_path.relative_to(
                    context.job_dir
                ).as_posix(),
            }
            geometry_metrics.update(
                gaussian_geometry_effective_source=effective_geometry_source,
                gaussian_geometry_fallback_applied=fallback_applied,
                vggt_ba_profile=str(diagnostics["profile"]),
                vggt_ba_supported_camera_count=int(
                    diagnostics["supported_camera_count"]
                ),
                vggt_ba_point_count=int(diagnostics["point_count"]),
                vggt_ba_trajectory_status=str(graph["trajectory_status"]),
                vggt_ba_verified_nonlocal_edge_count=int(
                    graph["verified_nonlocal_edge_count"]
                ),
                vggt_ba_elapsed_seconds=float(diagnostics["elapsed_seconds"]),
            )
            if fallback_applied:
                geometry_metrics["gaussian_geometry_fallback_reason"] = str(
                    fallback_reason
                )
        temporal_timestamps: dict[str, float] | None = None
        registration_log_lines: list[str] = []
        if video_selection is not None:
            registration_path = context.job_dir / "diagnostics" / "video_registration.json"
            (
                temporal_timestamps,
                registration_metrics,
                gap_violations,
            ) = _write_video_registration_diagnostics(
                video_selection, cameras_path, registration_path
            )
            video_assets["video_registration_diagnostics"] = registration_path.relative_to(
                context.job_dir
            ).as_posix()
            video_metrics.update(registration_metrics)
            if gap_violations:
                maximum_violation = max(
                    violation["seconds"] for violation in gap_violations
                )
                intervals = ",".join(
                    f"{violation['start_seconds']:.1f}-{violation['end_seconds']:.1f}"
                    for violation in gap_violations
                )
                registration_log_lines.append(
                    f"video_registration_gap_warning={len(gap_violations)} "
                    f"max={maximum_violation:.2f}s intervals={intervals}"
                )
        try:
            from image3d_scenegraph.gaussian.dataset import build_colmap_contract, write_contract
        except ImportError as exc:
            raise ReconstructionError("project Gaussian dataset module is unavailable") from exc
        contract = build_colmap_contract(
            dataset_id=context.job_id,
            dataset_root=context.job_dir,
            image_root="colmap/undistorted/images",
            cameras_path="geometry/cameras.json",
            temporal_timestamps=temporal_timestamps,
        )
        write_contract(dataset_path, contract)
        training_points_path = points_path
        if geometry_metrics["gaussian_geometry_effective_source"] == "vggt_ba":
            try:
                from image3d_scenegraph.geometry.vggt_ba import (
                    filter_train_supported_points,
                    write_json as write_vggt_json,
                )

                filtered_points_path = (
                    context.job_dir / "vggt_ba" / "train_points3D.txt"
                )
                filter_diagnostics = filter_train_supported_points(
                    points_path,
                    sparse_dir / "images.txt",
                    context.job_dir / "colmap" / "undistorted" / "images",
                    {int(image_id) for image_id in contract["splits"]["train"]},
                    filtered_points_path,
                    minimum_train_observations=2,
                )
                filter_diagnostics_path = (
                    context.job_dir / "diagnostics" / "vggt_ba_initialization.json"
                )
                write_vggt_json(filter_diagnostics_path, filter_diagnostics)
            except (OSError, ValueError) as exc:
                raise ReconstructionError(
                    f"VGGT-BA Train-supported initialization failed: {exc}"
                ) from exc
            training_points_path = filtered_points_path
            geometry_assets["vggt_ba_initialization_diagnostics"] = (
                filter_diagnostics_path.relative_to(context.job_dir).as_posix()
            )
            geometry_metrics["vggt_ba_train_supported_point_count"] = int(
                filter_diagnostics["counts"]["accepted"]
            )
            geometry_metrics["vggt_ba_heldout_only_point_count"] = int(
                filter_diagnostics["counts"]["heldout_only_rejected"]
            )
        config_record = context.options.get("gaussian_config_record")
        if not isinstance(config_record, str):
            raise ReconstructionError("project 3DGS requires a resolved Gaussian config record")
        trainer_id = str(context.options.get("gaussian_trainer", "graphdeco"))
        try:
            from image3d_scenegraph.gaussian.trainers import (
                get_gaussian_trainer_specs,
                validate_trainer_id,
            )

            validate_trainer_id(trainer_id)
            trainer_spec = next(
                spec
                for spec in get_gaussian_trainer_specs(project_root)
                if spec.trainer_id == trainer_id
            )
            if not trainer_spec.available:
                raise ReconstructionError(
                    f"{trainer_spec.label} is unavailable: {trainer_spec.reason}"
                )
        except ImportError as exc:
            raise ReconstructionError(str(exc)) from exc
        config_path.write_text(config_record + "\n", encoding="utf-8")
        training_dir = context.job_dir / "gaussian"
        command_train = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(trainer_script),
            "--dataset-contract",
            str(dataset_path),
            "--dataset-root",
            str(context.job_dir),
            "--run-dir",
            str(training_dir),
            "--trainer",
            trainer_id,
            "--initialization",
            "sparse",
            "--points",
            str(training_points_path),
            "--resolved-config-json",
            str(config_path),
            "--max-initial-points",
            str(context.options.get("gaussian_max_initial_points", 1_000_000)),
        ]
        if trainer_id == "project":
            command_train.append("--distributed")
        _adapter_progress(context, "gaussian_training", 0.35)
        completed = _run_adapter_command(command_train, context, project_root, env=env)
        result_candidates = sorted(training_dir.glob("attempts/*/artifacts/result.json"))
        if not result_candidates:
            raise ReconstructionError("project Gaussian trainer did not produce complete results")
        result_path = result_candidates[-1]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        model_path = training_dir / str(result["model_path"])
        progress_path = training_dir / str(result["progress_path"])
        attempt_id = result_path.parents[1].name
        effective_dataset_path = training_dir / "preparation" / attempt_id / "dataset.json"
        effective_config_path = training_dir / "preparation" / attempt_id / "effective_config.json"
        if not all(
            path.is_file()
            for path in (model_path, progress_path, effective_dataset_path, effective_config_path)
        ):
            raise ReconstructionError("project Gaussian trainer result references missing assets")

        (
            model_path,
            sor_metrics,
            sor_status,
            sor_reason,
            sor_log_lines,
            sor_record_path,
            sor_mask_path,
        ) = _try_apply_sor_filter(
            context=context,
            sor_filter=sor_filter,
            project_root=project_root,
            filter_script=sor_filter_script,
            training_dir=training_dir,
            attempt_id=attempt_id,
            model_path=model_path,
            env=env,
        )
        sor_assets = (
            {
                "gaussian_sor_filter_record": sor_record_path.relative_to(context.job_dir).as_posix(),
                "gaussian_sor_filter_mask": sor_mask_path.relative_to(context.job_dir).as_posix(),
            }
            if sor_record_path is not None and sor_mask_path is not None
            else {}
        )
        evaluation_dir = training_dir / "evaluation" / attempt_id / "validation"
        command_evaluate = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(evaluator_script),
            "--dataset-contract",
            str(effective_dataset_path),
            "--dataset-root",
            str(context.job_dir),
            "--model",
            str(model_path),
            "--resolved-config-json",
            str(effective_config_path),
            "--split",
            "validation",
            "--output-dir",
            str(evaluation_dir),
            "--progress",
            str(progress_path),
        ]
        _adapter_progress(context, "gaussian_validation", 0.72)
        _run_adapter_command(command_evaluate, context, project_root, env=env)
        evaluation_path = evaluation_dir / "evaluation.json"
        test_assets: dict[str, str] = {}
        required_test_paths: tuple[Path, ...] = ()
        test_status = "not_run"
        test_reason = "frontend_validation_only_visual_comparison"
        if _automatic_test_evaluation_enabled(trainer_id):
            frozen_candidate_path = training_dir / "evaluation" / attempt_id / "frozen-candidate.json"
            test_evaluation_dir = training_dir / "evaluation" / attempt_id / "test"
            command_test = [
                os.environ.get("IMAGE3D_PYTHON", sys.executable),
                str(evaluator_script),
                "--dataset-contract",
                str(effective_dataset_path),
                "--dataset-root",
                str(context.job_dir),
                "--model",
                str(model_path),
                "--resolved-config-json",
                str(effective_config_path),
                "--split",
                "test",
                "--output-dir",
                str(test_evaluation_dir),
                "--progress",
                str(progress_path),
                "--frozen-candidate",
                str(frozen_candidate_path),
                "--freeze-candidate-id",
                f"{context.job_id}-{attempt_id}",
            ]
            _adapter_progress(context, "gaussian_test_evaluation", 0.80)
            _run_adapter_command(command_test, context, project_root, env=env)
            test_evaluation_path = test_evaluation_dir / "evaluation.json"
            test_consumption_path = frozen_candidate_path.with_name(
                f"{frozen_candidate_path.stem}.test-consumed.json"
            )
            required_test_paths = (
                test_evaluation_path,
                frozen_candidate_path,
                test_consumption_path,
            )
            test_assets = {
                "gaussian_test_evaluation": test_evaluation_path.relative_to(
                    context.job_dir
                ).as_posix(),
                "gaussian_test_decision": test_consumption_path.relative_to(
                    context.job_dir
                ).as_posix(),
            }
            test_status = "complete"
            test_reason = "frozen_candidate_test_evaluation_complete"
        export_dir = training_dir / "export" / attempt_id
        command_export = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(exporter_script),
            "--model",
            str(model_path),
            "--dataset-contract",
            str(effective_dataset_path),
            "--resolved-config-json",
            str(effective_config_path),
            "--evaluation",
            str(evaluation_path),
            "--output-dir",
            str(export_dir),
            "--checkpoint-hash",
            str(result["final_checkpoint_hash"]),
        ]
        if sor_record_path is not None and sor_mask_path is not None:
            command_export += [
                "--postprocess-record",
                str(sor_record_path),
                "--postprocess-mask",
                str(sor_mask_path),
            ]
        _adapter_progress(context, "gaussian_export", 0.86)
        _run_adapter_command(command_export, context, project_root, env=env)
        export_metadata_path = export_dir / "export.json"
        scene_splat_path = export_dir / "scene.ply"
        canonical_path = export_dir / "canonical.ply"
        camera_path = export_dir / "camera_path.json"
        bundle_path = export_dir / "result.zip"
        if not all(
            path.is_file()
            for path in (
                evaluation_path,
                *required_test_paths,
                export_metadata_path,
                scene_splat_path,
                canonical_path,
                camera_path,
                bundle_path,
            )
        ):
            raise ReconstructionError("project Gaussian export is incomplete")
        (
            postprocess_assets,
            postprocess_metrics,
            postprocess_status,
            postprocess_reason,
            postprocess_log_lines,
        ) = _try_build_vggt_filtered_variant(
            context=context,
            postprocess=postprocess,
            project_root=project_root,
            filter_script=filter_script,
            evaluator_script=evaluator_script,
            exporter_script=exporter_script,
            env=env,
            training_dir=training_dir,
            attempt_id=attempt_id,
            model_path=model_path,
            effective_dataset_path=effective_dataset_path,
            effective_config_path=effective_config_path,
            progress_path=progress_path,
            sparse_dir=sparse_dir,
            checkpoint_hash=str(result["final_checkpoint_hash"]),
        )
        log_lines = [
            "geometry_backend=project_3dgs",
            "output_type=gaussian_splat",
            "adapter=ProjectGaussianAdapter",
            f"gaussian_trainer={trainer_id}",
            f"gaussian_geometry_source={geometry_source}",
            f"gaussian_geometry_effective_source={geometry_metrics['gaussian_geometry_effective_source']}",
            f"gaussian_geometry_fallback_applied={str(geometry_metrics['gaussian_geometry_fallback_applied']).lower()}",
            *(
                [
                    "gaussian_geometry_fallback_reason="
                    + str(geometry_metrics["gaussian_geometry_fallback_reason"])
                ]
                if "gaussian_geometry_fallback_reason" in geometry_metrics
                else []
            ),
            f"gaussian_test_status={test_status}",
            f"gaussian_test_reason={test_reason}",
            f"gaussian_postprocess={postprocess}",
            f"gaussian_postprocess_status={postprocess_status}",
            *(
                [f"gaussian_postprocess_reason={postprocess_reason}"]
                if postprocess_reason
                else []
            ),
            f"gaussian_sor_filter={sor_filter}",
            f"gaussian_sor_filter_status={sor_status}",
            *(
                [f"gaussian_sor_filter_reason={sor_reason}"]
                if sor_reason
                else []
            ),
            *video_recovery_log_lines,
            *registration_log_lines,
            f"trainer={' '.join(command_train)}",
            *sor_log_lines,
            *postprocess_log_lines,
        ]
        if completed.stdout.strip():
            log_lines.append(f"stdout={completed.stdout.strip()}")
        return ReconstructionResult(
            stage="gaussian_export",
            assets={
                **video_assets,
                **geometry_assets,
                "gaussian_model": model_path.relative_to(context.job_dir).as_posix(),
                "gaussian_training_result": result_path.relative_to(context.job_dir).as_posix(),
                "gaussian_progress": progress_path.relative_to(context.job_dir).as_posix(),
                "gaussian_dataset": effective_dataset_path.relative_to(context.job_dir).as_posix(),
                "gaussian_effective_config": effective_config_path.relative_to(context.job_dir).as_posix(),
                "gaussian_evaluation": evaluation_path.relative_to(context.job_dir).as_posix(),
                **test_assets,
                "gaussian_export_metadata": export_metadata_path.relative_to(context.job_dir).as_posix(),
                "gaussian_canonical": canonical_path.relative_to(context.job_dir).as_posix(),
                "scene_splat": scene_splat_path.relative_to(context.job_dir).as_posix(),
                "gaussian_camera_path": camera_path.relative_to(context.job_dir).as_posix(),
                "gaussian_bundle": bundle_path.relative_to(context.job_dir).as_posix(),
                **sor_assets,
                **postprocess_assets,
            },
            metrics={
                **video_metrics,
                **geometry_metrics,
                "gaussian_count": int(result["gaussian_count"]),
                "gaussian_trainer": trainer_id,
                "gaussian_initial_loss": float(result["initial_loss"]),
                "gaussian_final_loss": float(result["final_loss"]),
                "gaussian_peak_allocated_bytes": int(result["peak_allocated_bytes"]),
                "gaussian_peak_reserved_bytes": int(result["peak_reserved_bytes"]),
                "gaussian_training_seconds": float(result["elapsed_seconds"]),
                "gaussian_world_size": int(result.get("world_size", 1)),
                "gaussian_test_status": test_status,
                "gaussian_test_reason": test_reason,
                "gaussian_postprocess": postprocess,
                "gaussian_postprocess_status": postprocess_status,
                **(
                    {"gaussian_postprocess_reason": postprocess_reason}
                    if postprocess_reason
                    else {}
                ),
                "gaussian_sor_filter": sor_filter,
                "gaussian_sor_filter_status": sor_status,
                **(
                    {"gaussian_sor_filter_reason": sor_reason}
                    if sor_reason
                    else {}
                ),
                **sor_metrics,
                **postprocess_metrics,
            },
            log_lines=log_lines,
        )


def _try_build_vggt_filtered_variant(
    *,
    context: ReconstructionContext,
    postprocess: str,
    project_root: Path,
    filter_script: Path,
    evaluator_script: Path,
    exporter_script: Path,
    env: dict[str, str],
    training_dir: Path,
    attempt_id: str,
    model_path: Path,
    effective_dataset_path: Path,
    effective_config_path: Path,
    progress_path: Path,
    sparse_dir: Path,
    checkpoint_hash: str,
) -> tuple[
    dict[str, str],
    dict[str, int | float | str | bool],
    str,
    str | None,
    list[str],
]:
    if postprocess == "none":
        return {}, {}, "not_requested", None, []
    if postprocess != "vggt_visibility_v1":
        raise ReconstructionError(f"unsupported Gaussian postprocess: {postprocess}")
    postprocess_dir = training_dir / "postprocess" / attempt_id
    filtered_evaluation_dir = (
        training_dir / "evaluation" / attempt_id / "validation-vggt-filtered"
    )
    filtered_export_dir = training_dir / "export" / f"{attempt_id}-vggt-filtered"
    external_root = Path(
        os.environ.get("IMAGE3D_EXTERNAL_ROOT", project_root / "external")
    )
    checkpoint_root = Path(
        os.environ.get("IMAGE3D_CHECKPOINT_ROOT", project_root / "checkpoints")
    )
    command_filter = [
        os.environ.get("IMAGE3D_PYTHON", sys.executable),
        str(filter_script),
        "--dataset-contract",
        str(effective_dataset_path),
        "--dataset-root",
        str(context.job_dir),
        "--model",
        str(model_path),
        "--colmap-model-dir",
        str(sparse_dir),
        "--output-dir",
        str(postprocess_dir),
        "--repo-dir",
        str(external_root / "vggt"),
        "--checkpoint-dir",
        str(checkpoint_root / "vggt" / "facebook--VGGT-1B"),
    ]
    try:
        _adapter_progress(context, "gaussian_vggt_postprocess", 0.89)
        completed = _run_adapter_command(
            command_filter, context, project_root, env=env
        )
        result_path = postprocess_dir / "result.json"
        diagnostics_path = postprocess_dir / "diagnostics.json"
        mask_path = postprocess_dir / "filter-mask.npz"
        if not all(path.is_file() for path in (result_path, diagnostics_path, mask_path)):
            raise ReconstructionError("VGGT Gaussian postprocess output is incomplete")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        filtered_model_path = Path(str(result["filtered_model"]))
        if not filtered_model_path.is_absolute():
            filtered_model_path = (project_root / filtered_model_path).resolve()
        if not filtered_model_path.is_file():
            raise ReconstructionError("VGGT filtered model is missing")

        command_evaluate = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(evaluator_script),
            "--dataset-contract",
            str(effective_dataset_path),
            "--dataset-root",
            str(context.job_dir),
            "--model",
            str(filtered_model_path),
            "--resolved-config-json",
            str(effective_config_path),
            "--split",
            "validation",
            "--output-dir",
            str(filtered_evaluation_dir),
            "--progress",
            str(progress_path),
        ]
        _adapter_progress(context, "gaussian_vggt_filtered_validation", 0.91)
        _run_adapter_command(command_evaluate, context, project_root, env=env)
        filtered_evaluation_path = filtered_evaluation_dir / "evaluation.json"
        command_export = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(exporter_script),
            "--model",
            str(filtered_model_path),
            "--dataset-contract",
            str(effective_dataset_path),
            "--resolved-config-json",
            str(effective_config_path),
            "--evaluation",
            str(filtered_evaluation_path),
            "--output-dir",
            str(filtered_export_dir),
            "--checkpoint-hash",
            checkpoint_hash,
            "--postprocess-record",
            str(diagnostics_path),
            "--postprocess-mask",
            str(mask_path),
        ]
        _adapter_progress(context, "gaussian_vggt_filtered_export", 0.93)
        _run_adapter_command(command_export, context, project_root, env=env)
        filtered_export_metadata = filtered_export_dir / "export.json"
        filtered_scene = filtered_export_dir / "scene.ply"
        filtered_canonical = filtered_export_dir / "canonical.ply"
        filtered_bundle = filtered_export_dir / "result.zip"
        required = (
            filtered_model_path,
            filtered_evaluation_path,
            filtered_export_metadata,
            filtered_scene,
            filtered_canonical,
            filtered_bundle,
        )
        if not all(path.is_file() for path in required):
            raise ReconstructionError("VGGT filtered Gaussian export is incomplete")
        evaluation = json.loads(filtered_evaluation_path.read_text(encoding="utf-8"))
        assets = {
            "gaussian_vggt_filtered_model": filtered_model_path.relative_to(
                context.job_dir.resolve()
            ).as_posix(),
            "gaussian_vggt_filter_diagnostics": diagnostics_path.relative_to(
                context.job_dir
            ).as_posix(),
            "gaussian_vggt_filter_mask": mask_path.relative_to(
                context.job_dir
            ).as_posix(),
            "gaussian_vggt_filtered_evaluation": filtered_evaluation_path.relative_to(
                context.job_dir
            ).as_posix(),
            "gaussian_vggt_filtered_export_metadata": filtered_export_metadata.relative_to(
                context.job_dir
            ).as_posix(),
            "gaussian_vggt_filtered_canonical": filtered_canonical.relative_to(
                context.job_dir
            ).as_posix(),
            "scene_splat_vggt_filtered": filtered_scene.relative_to(
                context.job_dir
            ).as_posix(),
            "gaussian_vggt_filtered_bundle": filtered_bundle.relative_to(
                context.job_dir
            ).as_posix(),
        }
        metrics: dict[str, int | float | str | bool] = {
            "gaussian_vggt_filter_input_count": int(result["input_count"]),
            "gaussian_vggt_filter_kept_count": int(result["kept_count"]),
            "gaussian_vggt_filter_removed_count": int(result["removed_count"]),
            "gaussian_vggt_filtered_validation_psnr": float(
                evaluation["psnr"]["mean"]
            ),
            "gaussian_vggt_filtered_validation_ssim": float(
                evaluation["ssim"]["mean"]
            ),
        }
        log_lines = [
            f"gaussian_postprocess_command={' '.join(command_filter)}",
            *([f"gaussian_postprocess_stdout={completed.stdout.strip()}"] if completed.stdout.strip() else []),
        ]
        return assets, metrics, "available", None, log_lines
    except (KeyError, OSError, ValueError, ReconstructionError) as exc:
        if context.cancel_requested is not None and context.cancel_requested():
            raise
        return {}, {}, "unavailable", str(exc), [f"gaussian_postprocess_error={exc}"]


def _try_apply_sor_filter(
    *,
    context: ReconstructionContext,
    sor_filter: str,
    project_root: Path,
    filter_script: Path,
    training_dir: Path,
    attempt_id: str,
    model_path: Path,
    env: dict[str, str] | None,
) -> tuple[Path, dict[str, int | float | str | bool], str, str | None, list[str], Path | None, Path | None]:
    """In-place conservative-band SOR cleanup of the selected model snapshot.

    On success the returned model_path replaces the training snapshot for
    evaluation/export; on any failure the original model continues unchanged.
    """
    if sor_filter != "on":
        return model_path, {}, "disabled", None, [], None, None
    try:
        output_dir = training_dir / "sor" / attempt_id
        command_filter = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(filter_script),
            "--model-snapshot",
            str(model_path),
            "--output-dir",
            str(output_dir),
            # Conservative band preset: the only render-lossless configuration
            # from the Stage 1 room evidence (codex.md 2026-08-20).
            "--nb-neighbors",
            "30",
            "--std-ratio",
            "2.0",
            "--band-opacity",
            "0.05",
        ]
        _adapter_progress(context, "gaussian_sor_filter", 0.70)
        completed = _run_adapter_command(command_filter, context, project_root, env=env)
        filtered_model_path = output_dir / "filtered-model.pt"
        record_path = output_dir / "filter-record.json"
        mask_path = output_dir / "filter-mask.npz"
        if not all(
            path.is_file() for path in (filtered_model_path, record_path, mask_path)
        ):
            raise ReconstructionError("SOR filter output is incomplete")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        metrics: dict[str, int | float | str | bool] = {
            "gaussian_sor_filter_input_count": int(record["input_count"]),
            "gaussian_sor_filter_kept_count": int(record["kept_count"]),
            "gaussian_sor_filter_removed_count": int(record["removed_count"]),
        }
        log_lines = [
            f"gaussian_sor_filter_command={' '.join(command_filter)}",
            *(
                [f"gaussian_sor_filter_stdout={completed.stdout.strip()}"]
                if completed.stdout.strip()
                else []
            ),
        ]
        return filtered_model_path, metrics, "available", None, log_lines, record_path, mask_path
    except (KeyError, OSError, ValueError, ReconstructionError) as exc:
        if context.cancel_requested is not None and context.cancel_requested():
            raise
        return (
            model_path,
            {},
            "unavailable",
            str(exc),
            [f"gaussian_sor_filter_error={exc}"],
            None,
            None,
        )


def _read_video_registration_recovery(
    selection: dict[str, Any], diagnostics_path: Path
) -> tuple[dict[str, int | float | str | bool], list[str]]:
    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionError(
            "video registration recovery diagnostics are missing or invalid"
        ) from exc
    status = diagnostics.get("status")
    rounds = diagnostics.get("rounds")
    if (
        diagnostics.get("schema_version") != 1
        or diagnostics.get("method") != "incremental_colmap"
        or status not in {"not_needed", "recovered", "partial", "unavailable"}
        or not isinstance(rounds, list)
        or any(not isinstance(round_record, dict) for round_record in rounds)
    ):
        raise ReconstructionError("video registration recovery diagnostics are invalid")
    try:
        final_selected_count = int(selection["selected_count"])
        recovery_selected_count = int(selection.get("recovery_selected_count", 0))
        initial_selected_count = int(
            diagnostics.get(
                "initial_selected_count",
                final_selected_count - recovery_selected_count,
            )
        )
        base_selected_count = int(selection["base_selected_count"])
        adaptive_selected_count = int(selection["adaptive_selected_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("standard_v2 selection counts are invalid") from exc
    if (
        initial_selected_count + recovery_selected_count != final_selected_count
        or base_selected_count + adaptive_selected_count != initial_selected_count
    ):
        raise ReconstructionError("standard_v2 selection counts are inconsistent")

    accepted_rounds = sum(bool(round_record.get("accepted")) for round_record in rounds)
    pair_count = sum(int(round_record.get("pair_count", 0)) for round_record in rounds)
    elapsed_seconds = float(
        diagnostics.get(
            "elapsed_seconds",
            sum(float(round_record.get("elapsed_seconds", 0.0)) for round_record in rounds),
        )
    )
    metrics: dict[str, int | float | str | bool] = {
        "video_initial_selected_count": initial_selected_count,
        "video_base_selected_count": base_selected_count,
        "video_adaptive_selected_count": adaptive_selected_count,
        "video_recovery_selected_count": recovery_selected_count,
        "video_registration_recovery_status": str(status),
        "video_registration_recovery_rounds": len(rounds),
        "video_registration_recovery_accepted_rounds": accepted_rounds,
        "video_registration_recovery_pair_count": pair_count,
        "video_registration_recovery_elapsed_seconds": elapsed_seconds,
    }
    reason = diagnostics.get("reason")
    if reason is not None:
        metrics["video_registration_recovery_reason"] = str(reason)
    final_adjustment = diagnostics.get("final_bundle_adjustment")
    if isinstance(final_adjustment, dict):
        elapsed = final_adjustment.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            metrics["video_registration_recovery_final_ba_seconds"] = float(
                elapsed
            )
        metrics["video_registration_recovery_final_ba_accepted"] = bool(
            final_adjustment.get("accepted")
        )
        metrics["video_registration_recovery_final_ba_cpu_fallback"] = bool(
            final_adjustment.get("fallback_to_cpu")
        )

    timeline_fields = {
        "selected_count": "selected_count",
        "registered_count": "registered_count",
        "registration_rate": "registration_rate",
        "temporal_coverage": "temporal_coverage",
        "maximum_registered_gap_seconds": "max_gap_seconds",
        "gap_violation_count": "gap_violation_count",
        "gap_violation_total_seconds": "gap_violation_total_seconds",
        "gap_violation_excess_seconds": "gap_violation_excess_seconds",
        "sparse_point_count": "sparse_point_count",
    }
    for source_name, metric_prefix in (("initial", "pre"), ("final", "post")):
        timeline = diagnostics.get(source_name)
        if not isinstance(timeline, dict):
            continue
        for source_key, metric_suffix in timeline_fields.items():
            value = timeline.get(source_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metric_key = f"video_registration_recovery_{metric_prefix}_{metric_suffix}"
            metrics[metric_key] = (
                int(value)
                if source_key
                in {
                    "selected_count",
                    "registered_count",
                    "gap_violation_count",
                    "sparse_point_count",
                }
                else float(value)
            )
    initial = diagnostics.get("initial")
    final = diagnostics.get("final")
    if isinstance(initial, dict) and isinstance(final, dict):
        metrics["video_registration_recovery_registered_gain"] = int(
            final["registered_count"]
        ) - int(initial["registered_count"])

    log_lines = [
        "video_registration_recovery_method=incremental_colmap",
        f"video_registration_recovery_status={status}",
        f"video_registration_recovery_rounds={len(rounds)}",
        f"video_registration_recovery_selected={recovery_selected_count}",
    ]
    if reason is not None:
        log_lines.append(f"video_registration_recovery_reason={reason}")
    if isinstance(final_adjustment, dict):
        log_lines.append(
            "video_registration_recovery_final_ba="
            f"accepted={str(bool(final_adjustment.get('accepted'))).lower()} "
            f"cpu_fallback={str(bool(final_adjustment.get('fallback_to_cpu'))).lower()} "
            f"elapsed_seconds={final_adjustment.get('elapsed_seconds', 0)}"
        )
    for round_record in rounds:
        before = round_record.get("before")
        after = round_record.get("after")
        before_gap = (
            before.get("maximum_registered_gap_seconds")
            if isinstance(before, dict)
            else None
        )
        after_gap = (
            after.get("maximum_registered_gap_seconds")
            if isinstance(after, dict)
            else None
        )
        log_lines.append(
            "video_registration_recovery_round="
            f"{round_record.get('round')} "
            f"mode={round_record.get('mode', 'augmentation')} "
            f"candidates={round_record.get('candidate_count', 0)} "
            f"materialized={round_record.get('materialized_count', 0)} "
            f"pairs={round_record.get('pair_count', 0)} "
            f"accepted={str(bool(round_record.get('accepted'))).lower()} "
            f"reason={round_record.get('reason', 'none')} "
            f"before_max_gap={before_gap} after_max_gap={after_gap}"
        )
    return metrics, log_lines


def _write_video_registration_diagnostics(
    selection: dict[str, Any], cameras_path: Path, output_path: Path
) -> tuple[
    dict[str, float],
    dict[str, int | float | str | bool],
    list[dict[str, float]],
]:
    selected_payload = selection.get("selected")
    if not isinstance(selected_payload, list) or not selected_payload:
        raise ReconstructionError("video selection metadata has no selected frames")
    selected_by_name: dict[str, dict[str, Any]] = {}
    for item in selected_payload:
        if not isinstance(item, dict):
            raise ReconstructionError("video selection metadata is invalid")
        name = Path(str(item.get("path", ""))).name
        if not name or name in selected_by_name:
            raise ReconstructionError("video selection frame names are invalid")
        selected_by_name[name] = item
    try:
        cameras_payload = json.loads(cameras_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionError("cannot read COLMAP cameras for video registration") from exc
    cameras = {
        int(camera["camera_id"]): camera
        for camera in cameras_payload.get("cameras", [])
        if isinstance(camera, dict) and "camera_id" in camera
    }
    registered_by_name = {
        Path(str(image.get("name", ""))).name: image
        for image in cameras_payload.get("images", [])
        if isinstance(image, dict)
    }
    try:
        selected_timestamps = {
            name: float(item["time_seconds"])
            for name, item in selected_by_name.items()
        }
        timeline = analyze_registration_timeline(
            selected_timestamps,
            registered_by_name,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("video selection metadata is invalid") from exc
    registered_names = timeline["registered_names"]
    selected_count = int(timeline["selected_count"])
    registered_count = int(timeline["registered_count"])
    registration_rate = float(timeline["registration_rate"])
    temporal_coverage = float(timeline["temporal_coverage"])
    gap_violations = timeline["gap_violations"]
    gate_failures: list[str] = []
    if registered_count < MIN_VIDEO_REGISTERED_COUNT:
        gate_failures.append(
            f"registered_count {registered_count} is below the "
            f"{MIN_VIDEO_REGISTERED_COUNT}-frame minimum"
        )
    if registration_rate < MIN_VIDEO_REGISTRATION_RATE:
        gate_failures.append(
            f"registration_rate {registration_rate:.3f} is below "
            f"{MIN_VIDEO_REGISTRATION_RATE:.3f}"
        )
    if temporal_coverage < MIN_VIDEO_TEMPORAL_COVERAGE:
        gate_failures.append(
            f"temporal_coverage {temporal_coverage:.3f} is below "
            f"{MIN_VIDEO_TEMPORAL_COVERAGE:.3f}"
        )
    records = []
    for name, selected in sorted(
        selected_by_name.items(), key=lambda item: float(item[1]["time_seconds"])
    ):
        registered = registered_by_name.get(name)
        camera = cameras.get(int(registered["camera_id"])) if registered is not None else None
        records.append(
            {
                "filename": name,
                "source_time_seconds": float(selected["time_seconds"]),
                "registered": registered is not None,
                "image_id": int(registered["image_id"]) if registered is not None else None,
                "camera_id": int(registered["camera_id"]) if registered is not None else None,
                "camera_model": camera.get("model") if camera is not None else None,
            }
        )
    payload = {
        "schema_version": 1,
        "profile": str(selection.get("profile", "unknown")),
        "selected_count": selected_count,
        "registered_count": registered_count,
        "registration_rate": registration_rate,
        "temporal_coverage": temporal_coverage,
        "maximum_registered_gap_seconds": timeline[
            "maximum_registered_gap_seconds"
        ],
        "maximum_registered_gap_threshold_seconds": timeline[
            "maximum_registered_gap_threshold_seconds"
        ],
        "gap_violation_count": timeline["gap_violation_count"],
        "gap_violation_total_seconds": timeline["gap_violation_total_seconds"],
        "gap_violation_excess_seconds": timeline["gap_violation_excess_seconds"],
        "gap_violations": gap_violations,
        "gate": {
            "passed": not gate_failures,
            "failures": gate_failures,
            "minimum_registered_count": MIN_VIDEO_REGISTERED_COUNT,
            "minimum_registration_rate": MIN_VIDEO_REGISTRATION_RATE,
            "minimum_temporal_coverage": MIN_VIDEO_TEMPORAL_COVERAGE,
        },
        "frames": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if gate_failures:
        raise ReconstructionError(
            "video_registration_quality_gate_failed: " + "; ".join(gate_failures)
        )
    timestamps = {name: selected_timestamps[name] for name in registered_names}
    metrics: dict[str, int | float | str | bool] = {
        "video_registered_count": registered_count,
        "video_registration_rate": registration_rate,
        "video_registration_temporal_coverage": temporal_coverage,
        "video_registration_max_gap_seconds": float(
            timeline["maximum_registered_gap_seconds"]
        ),
    }
    if gap_violations:
        metrics.update(
            video_registration_gap_violation_count=len(gap_violations),
            video_registration_gap_violation_total_seconds=float(
                timeline["gap_violation_total_seconds"]
            ),
            video_registration_gap_violation_excess_seconds=float(
                timeline["gap_violation_excess_seconds"]
            ),
        )
    return timestamps, metrics, gap_violations


def _automatic_test_evaluation_enabled(_trainer_id: str) -> bool:
    return False


def _adapter_progress(context: ReconstructionContext, stage: str, progress: float) -> None:
    if context.progress_callback is not None:
        context.progress_callback(stage, progress)


def _run_adapter_command(
    command: list[str],
    context: ReconstructionContext,
    cwd: Path,
    *,
    env: dict[str, str] | None,
    poll_callback: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        if context.cancel_requested is None:
            return subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        from image3d_scenegraph.worker import run_cancellable_command

        return run_cancellable_command(
            command,
            cwd=cwd,
            env=env,
            cancel_requested=context.cancel_requested,
            poll_callback=poll_callback,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
        raise ReconstructionError(f"project 3DGS command failed:\n{details}") from exc


class MockPointCloudAdapter:
    backend = "mock"
    output_type = "point_cloud"

    def run(self, context: ReconstructionContext) -> ReconstructionResult:
        point_cloud_path = context.job_dir / "geometry" / "points.ply"
        self._write_mock_geometry(point_cloud_path)
        return ReconstructionResult(
            stage="mock_reconstruction",
            assets={"point_cloud": "geometry/points.ply"},
            metrics={"num_points": 5},
            log_lines=[
                "geometry_backend=mock",
                "output_type=point_cloud",
                "adapter=MockPointCloudAdapter",
            ],
        )

    def _write_mock_geometry(self, path: Path) -> None:
        points = [
            (-0.5, -0.5, 1.0, 255, 80, 80),
            (0.5, -0.5, 1.0, 80, 255, 80),
            (0.5, 0.5, 1.0, 80, 80, 255),
            (-0.5, 0.5, 1.0, 255, 255, 80),
            (0.0, 0.0, 0.5, 255, 255, 255),
        ]
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        body = [f"{x} {y} {z} {r} {g} {b}" for x, y, z, r, g, b in points]
        path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


class VggtPointCloudAdapter:
    backend = "vggt"
    output_type = "point_cloud"

    def run(self, context: ReconstructionContext) -> ReconstructionResult:
        if context.mode == "video":
            raise ReconstructionError("VGGT adapter currently expects image inputs, not video files")

        project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
        script_path = project_root / "scripts" / "run_vggt_pointcloud.py"
        if not script_path.exists():
            raise ReconstructionError(f"VGGT runner missing: {script_path}")

        image_dir = context.job_dir / "input" / "images"
        command = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(script_path),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(context.job_dir),
            "--device",
            os.environ.get("IMAGE3D_VGGT_DEVICE", "cuda"),
            "--max-images",
            str(_positive_int_option(context, "vggt_max_images", "IMAGE3D_VGGT_MAX_IMAGES", 8)),
            "--max-points",
            os.environ.get("IMAGE3D_VGGT_MAX_POINTS", "200000"),
            "--conf-percentile",
            os.environ.get("IMAGE3D_VGGT_CONF_PERCENTILE", "50"),
            "--batch-size",
            str(_positive_int_option(context, "vggt_batch_size", "IMAGE3D_VGGT_BATCH_SIZE", 8)),
            "--overlap-size",
            str(_positive_int_option(context, "vggt_overlap_size", "IMAGE3D_VGGT_OVERLAP_SIZE", 4)),
        ]
        if os.environ.get("IMAGE3D_VGGT_USE_POINT_MAP") == "1":
            command.append("--use-point-map")

        env = os.environ.copy()
        if os.environ.get("IMAGE3D_VGGT_KEEP_LD_LIBRARY_PATH") != "1":
            env.pop("LD_LIBRARY_PATH", None)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            if context.cancel_requested is None:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                from image3d_scenegraph.worker import run_cancellable_command

                completed = run_cancellable_command(
                    command,
                    cwd=project_root,
                    env=env,
                    cancel_requested=context.cancel_requested,
                )
        except subprocess.CalledProcessError as exc:
            details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
            raise ReconstructionError(f"VGGT reconstruction failed:\n{details}") from exc

        point_cloud_path = context.job_dir / "geometry" / "points.ply"
        cameras_path = context.job_dir / "geometry" / "cameras.json"
        if not point_cloud_path.exists() or not cameras_path.exists():
            raise ReconstructionError("VGGT reconstruction did not produce points.ply and cameras.json")

        runner_log = context.job_dir / "logs" / "run.log"
        metrics = _parse_key_value_metrics(runner_log)
        log_lines = [
            "geometry_backend=vggt",
            "output_type=point_cloud",
            "adapter=VggtPointCloudAdapter",
            f"runner={' '.join(command)}",
            *[line for line in runner_log.read_text(encoding="utf-8").splitlines() if line],
        ]
        if completed.stdout.strip():
            log_lines.append(f"stdout={completed.stdout.strip()}")

        return ReconstructionResult(
            stage="vggt_reconstruction",
            assets={
                "point_cloud": "geometry/points.ply",
                "cameras": "geometry/cameras.json",
            },
            metrics=metrics,
            log_lines=log_lines,
        )


class ColmapPointCloudAdapter:
    backend = "colmap"
    output_type = "point_cloud"

    def run(self, context: ReconstructionContext) -> ReconstructionResult:
        if context.mode == "video":
            raise ReconstructionError("COLMAP adapter currently expects image inputs, not video files")

        project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
        script_path = project_root / "scripts" / "run_colmap_sparse.py"
        if not script_path.exists():
            raise ReconstructionError(f"COLMAP runner missing: {script_path}")

        image_dir = context.job_dir / "input" / "images"
        command = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(script_path),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(context.job_dir),
            "--matcher",
            os.environ.get("IMAGE3D_COLMAP_MATCHER", "sequential"),
        ]

        try:
            if context.cancel_requested is None:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                from image3d_scenegraph.worker import run_cancellable_command

                completed = run_cancellable_command(
                    command,
                    cwd=project_root,
                    cancel_requested=context.cancel_requested,
                )
        except subprocess.CalledProcessError as exc:
            details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
            raise ReconstructionError(f"COLMAP reconstruction failed:\n{details}") from exc

        point_cloud_path = context.job_dir / "geometry" / "points.ply"
        cameras_path = context.job_dir / "geometry" / "cameras.json"
        if not point_cloud_path.exists() or not cameras_path.exists():
            raise ReconstructionError("COLMAP reconstruction did not produce points.ply and cameras.json")

        runner_log = context.job_dir / "logs" / "run.log"
        metrics = _parse_key_value_metrics(runner_log)
        log_lines = [
            "geometry_backend=colmap",
            "output_type=point_cloud",
            "adapter=ColmapPointCloudAdapter",
            f"runner={' '.join(command)}",
            *[line for line in runner_log.read_text(encoding="utf-8").splitlines() if line],
        ]
        if completed.stdout.strip():
            log_lines.append(f"stdout={completed.stdout.strip()}")

        return ReconstructionResult(
            stage="colmap_sparse_reconstruction",
            assets={
                "point_cloud": "geometry/points.ply",
                "cameras": "geometry/cameras.json",
            },
            metrics=metrics,
            log_lines=log_lines,
        )


class ColmapVggtPointCloudAdapter:
    backend = "colmap_vggt"
    output_type = "point_cloud"

    def run(self, context: ReconstructionContext) -> ReconstructionResult:
        if context.mode == "video":
            raise ReconstructionError("COLMAP+VGGT adapter currently expects image inputs, not video files")

        project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
        script_path = project_root / "scripts" / "run_colmap_vggt_dense.py"
        if not script_path.exists():
            raise ReconstructionError(f"COLMAP+VGGT runner missing: {script_path}")

        image_dir = context.job_dir / "input" / "images"
        fusion_mode = os.environ.get("IMAGE3D_COLMAP_VGGT_FUSION_MODE", "points")
        if fusion_mode not in {"points", "tsdf"}:
            raise ReconstructionError("IMAGE3D_COLMAP_VGGT_FUSION_MODE must be 'points' or 'tsdf'")
        grouping = _choice_option(
            context,
            "colmap_vggt_grouping",
            "IMAGE3D_COLMAP_VGGT_GROUPING",
            "sequential",
            {"sequential", "covisibility"},
        )
        batch_size = _positive_int_option(
            context, "vggt_batch_size", "IMAGE3D_COLMAP_VGGT_BATCH_SIZE", 4
        )
        overlap_size = _positive_int_option(
            context,
            "colmap_vggt_overlap_size",
            "IMAGE3D_COLMAP_VGGT_OVERLAP_SIZE",
            2,
        )
        if batch_size < 2 or overlap_size >= batch_size:
            raise ReconstructionError(
                "colmap_vggt_overlap_size must be smaller than vggt_batch_size, which must be at least 2"
            )
        command = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(script_path),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(context.job_dir),
            "--device",
            os.environ.get("IMAGE3D_VGGT_DEVICE", "cuda"),
            "--matcher",
            os.environ.get("IMAGE3D_COLMAP_VGGT_MATCHER", "exhaustive"),
            "--vggt-batch-size",
            str(batch_size),
            "--vggt-overlap-size",
            str(overlap_size),
            "--vggt-grouping",
            grouping,
            "--fusion-mode",
            fusion_mode,
            "--max-points",
            str(_positive_int_option(context, "colmap_vggt_max_points", "IMAGE3D_COLMAP_VGGT_MAX_POINTS", 2_000_000)),
            "--conf-percentile",
            str(_percentile_option(context, "colmap_vggt_conf_percentile", "IMAGE3D_COLMAP_VGGT_CONF_PERCENTILE", 50.0)),
            "--confidence-threshold-scope",
            _choice_option(
                context,
                "colmap_vggt_confidence_threshold_scope",
                "IMAGE3D_COLMAP_VGGT_CONFIDENCE_THRESHOLD_SCOPE",
                "global",
                {"global", "per_frame"},
            ),
            "--consistency-support-policy",
            _choice_option(
                context,
                "colmap_vggt_consistency_support_policy",
                "IMAGE3D_COLMAP_VGGT_CONSISTENCY_SUPPORT_POLICY",
                "any_support",
                {"any_support", "adaptive_two"},
            ),
            "--point-budget-policy",
            _choice_option(
                context,
                "colmap_vggt_point_budget_policy",
                "IMAGE3D_COLMAP_VGGT_POINT_BUDGET_POLICY",
                "random",
                {"random", "spatial_balanced"},
            ),
        ]

        env = os.environ.copy()
        if os.environ.get("IMAGE3D_VGGT_KEEP_LD_LIBRARY_PATH") != "1":
            env.pop("LD_LIBRARY_PATH", None)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            if context.cancel_requested is None:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                from image3d_scenegraph.worker import run_cancellable_command

                completed = run_cancellable_command(
                    command,
                    cwd=project_root,
                    env=env,
                    cancel_requested=context.cancel_requested,
                )
        except subprocess.CalledProcessError as exc:
            details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
            raise ReconstructionError(f"COLMAP+VGGT reconstruction failed:\n{details}") from exc

        point_cloud_path = context.job_dir / "geometry" / "points.ply"
        cameras_path = context.job_dir / "geometry" / "cameras.json"
        fusion_diagnostics_path = context.job_dir / "diagnostics" / "fusion.json"
        visibility_graph_path = context.job_dir / "diagnostics" / "visibility_graph.json"
        scale_disagreement_path = context.job_dir / "diagnostics" / "scale_disagreement.json"
        consistency_path = context.job_dir / "diagnostics" / "consistency.json"
        if not all(
            path.exists()
            for path in [
                point_cloud_path,
                cameras_path,
                fusion_diagnostics_path,
                visibility_graph_path,
                scale_disagreement_path,
                consistency_path,
            ]
        ):
            raise ReconstructionError("COLMAP+VGGT reconstruction did not produce required geometry and diagnostics")

        runner_log = context.job_dir / "logs" / "run.log"
        metrics = _parse_key_value_metrics(runner_log)
        log_lines = [
            "geometry_backend=colmap_vggt",
            "output_type=point_cloud",
            "adapter=ColmapVggtPointCloudAdapter",
            f"runner={' '.join(command)}",
            *[line for line in runner_log.read_text(encoding="utf-8").splitlines() if line],
        ]
        if completed.stdout.strip():
            log_lines.append(f"stdout={completed.stdout.strip()}")

        return ReconstructionResult(
            stage="colmap_vggt_dense_reconstruction",
            assets={
                "point_cloud": "geometry/points.ply",
                "cameras": "geometry/cameras.json",
                "fusion_diagnostics": "diagnostics/fusion.json",
                "visibility_graph": "diagnostics/visibility_graph.json",
                "scale_disagreement_diagnostics": "diagnostics/scale_disagreement.json",
                "consistency_diagnostics": "diagnostics/consistency.json",
            },
            metrics=metrics,
            log_lines=log_lines,
        )


def _parse_key_value_metrics(path: Path) -> dict[str, int | float | str | bool]:
    metrics: dict[str, int | float | str | bool] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {
            "num_images",
            "registered_images",
            "scaled_images",
            "colmap_points",
            "num_groups",
            "batch_size",
            "vggt_batch_size",
            "vggt_overlap_size",
            "overlap_size",
            "num_points",
            "max_points",
            "integrated_frames",
            "point_budget_input_points",
            "point_budget_output_points",
            "point_budget_quantization_bits",
            "point_budget_occupied_codes",
            "factorial_output_count",
            "point_budget_sensitivity_output_count",
            "gpu_peak_memory_bytes",
        }:
            metrics[key] = int(value)
        elif key == "point_budget_applied":
            metrics[key] = value.lower() == "true"
        elif key in {
            "conf_percentile",
            "model_load_seconds",
            "inference_seconds",
            "colmap_seconds",
            "vggt_seconds",
            "elapsed_seconds",
            "scale_median",
            "scale_min",
            "scale_max",
            "scale_observations_median",
            "scale_log_mad_median",
            "consistency_confidence_threshold",
            "consistency_confidence_percentile",
            "consistency_confidence_threshold_min",
            "consistency_confidence_threshold_median",
            "consistency_confidence_threshold_max",
            "consistency_relative_threshold",
            "consistency_acceptance_rate",
            "consistency_residual_p50",
            "consistency_residual_p90",
            "tsdf_voxel_length",
            "tsdf_sdf_trunc",
            "tsdf_depth_trunc",
            "tsdf_full_sparse_diagonal",
            "tsdf_robust_sparse_diagonal",
        }:
            metrics[key] = float(value)
        elif key in {
            "consistency_candidates",
            "consistency_accepted",
            "consistency_rejected",
            "consistency_unverified",
            "consistency_supported",
            "consistency_stride",
            "consistency_multi_visible",
            "consistency_policy_rejected_supported",
        }:
            metrics[key] = int(value)
        else:
            metrics[key] = value
    return metrics


def _positive_int_option(context: ReconstructionContext, key: str, env_key: str, default: int) -> int:
    value = context.options.get(key)
    if value is None:
        value = int(os.environ.get(env_key, str(default)))
    value = int(value)
    if value <= 0:
        raise ReconstructionError(f"{key} must be positive")
    return value


def _percentile_option(context: ReconstructionContext, key: str, env_key: str, default: float) -> float:
    value = context.options.get(key)
    if value is None:
        value = float(os.environ.get(env_key, str(default)))
    value = float(value)
    if value < 0 or value >= 100:
        raise ReconstructionError(f"{key} must be between 0 and 99")
    return value


def _choice_option(
    context: ReconstructionContext,
    key: str,
    env_key: str,
    default: str,
    choices: set[str],
) -> str:
    value = str(context.options.get(key, os.environ.get(env_key, default)))
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ReconstructionError(f"{key} must be one of: {allowed}")
    return value


def get_reconstruction_adapter(
    geometry_backend: str, output_type: str
) -> ReconstructionAdapter:
    if geometry_backend not in VALID_GEOMETRY_BACKENDS:
        allowed = ", ".join(sorted(VALID_GEOMETRY_BACKENDS))
        raise ReconstructionError(
            f"unsupported geometry_backend '{geometry_backend}', expected one of: {allowed}"
        )
    if output_type not in VALID_OUTPUT_TYPES:
        allowed = ", ".join(sorted(VALID_OUTPUT_TYPES))
        raise ReconstructionError(
            f"unsupported output_type '{output_type}', expected one of: {allowed}"
        )

    if geometry_backend == "project_3dgs" and output_type == "gaussian_splat":
        return ProjectGaussianAdapter()
    if geometry_backend == "mock" and output_type == "point_cloud":
        return MockPointCloudAdapter()
    if geometry_backend == "vggt" and output_type in {"point_cloud", "mesh"}:
        return VggtPointCloudAdapter()
    if geometry_backend == "colmap" and output_type in {"point_cloud", "mesh"}:
        return ColmapPointCloudAdapter()
    if geometry_backend == "colmap_vggt" and output_type in {"point_cloud", "mesh"}:
        return ColmapVggtPointCloudAdapter()

    raise ReconstructionError(
        f"geometry_backend '{geometry_backend}' with output_type '{output_type}' is not implemented"
    )
