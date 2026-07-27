from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


VALID_GEOMETRY_BACKENDS = {"mock", "vggt", "colmap", "colmap_vggt", "dust3r", "mast3r", "nerfstudio_3dgs"}
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
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
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
            completed = subprocess.run(
                command,
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
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
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
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
