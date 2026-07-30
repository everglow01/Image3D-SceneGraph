from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from image3d_scenegraph.gaussian.config import (
    GaussianConfigError,
    ResolvedGaussianConfig,
    canonical_config_json,
    resolved_config_record,
)
from image3d_scenegraph.geometry.adapters import (
    ReconstructionContext,
    ReconstructionError,
    get_reconstruction_adapter,
)


VALID_MODES = {"image", "multi_image", "video", "panorama"}
MESH_METHODS = {"poisson", "ball_pivoting", "alpha_shape"}
MESH_OPTION_DEFAULTS = {
    "method": "poisson",
    "voxel_size": 0.05,
    "normal_radius": 0.2,
    "normal_max_nn": 30,
    "statistical_neighbors": 24,
    "statistical_std_ratio": 2.0,
    "radius_outlier_neighbors": 0,
    "radius_outlier_radius": 0.0,
    "poisson_depth": 8,
    "density_trim_quantile": 0.1,
    "component_min_ratio": 0.03,
    "edge_trim_quantile": 0.98,
    "edge_trim_factor": 2.5,
    "max_triangles": 120_000,
    "alpha": 0.0,
}


@dataclass(frozen=True)
class UploadedInput:
    filename: str
    content: bytes
    content_type: str | None = None


class JobError(ValueError):
    """Raised when a job request violates the local job contract."""


class JobStore:
    def __init__(self, output_root: Path | str | None = None) -> None:
        default_root = os.environ.get("IMAGE3D_OUTPUT_ROOT", "outputs/jobs")
        self.output_root = Path(output_root or default_root)

    def create_mock_job(self, mode: str, files: list[UploadedInput]) -> dict[str, Any]:
        return self.create_job(mode, files, geometry_backend="mock", output_type="point_cloud")

    def create_job(
        self,
        mode: str,
        files: list[UploadedInput],
        geometry_backend: str = "mock",
        output_type: str = "point_cloud",
        options: dict[str, int | float | str] | None = None,
        *,
        gaussian_config: ResolvedGaussianConfig | None = None,
    ) -> dict[str, Any]:
        self._validate_request(mode, files)
        try:
            gaussian_config_record = (
                resolved_config_record(gaussian_config) if gaussian_config is not None else None
            )
        except GaussianConfigError as exc:
            raise JobError(str(exc)) from exc

        job_id = self._new_job_id()
        job_dir = self.job_dir(job_id)
        self._create_job_dirs(job_dir)

        input_assets = self._write_inputs(job_dir, mode, files)
        try:
            adapter = get_reconstruction_adapter(geometry_backend, output_type)
        except ReconstructionError as exc:
            raise JobError(str(exc)) from exc

        try:
            reconstruction = adapter.run(
                ReconstructionContext(
                    job_id=job_id,
                    job_dir=job_dir,
                    mode=mode,
                    input_assets=input_assets,
                    options=options or {},
                )
            )
        except ReconstructionError as exc:
            raise JobError(str(exc)) from exc

        alignment_assets, alignment_metrics, alignment_log_lines = self._try_align_point_cloud(
            job_dir, reconstruction.assets
        )
        geometry_assets = {
            **reconstruction.assets,
            **alignment_assets,
        }
        mesh_assets: dict[str, str] = {}
        mesh_metrics: dict[str, int | float | str | bool] = {}
        mesh_log_lines: list[str] = []
        stage = reconstruction.stage
        if output_type == "mesh":
            mesh_assets, mesh_metrics, mesh_log_lines = self._build_mesh(job_dir, geometry_assets)
            stage = "mesh_reconstruction"

        scene = self._build_mock_scene(job_id, mode)
        self._write_json(job_dir / "scene_graph" / "scene.json", scene)

        gaussian_log_lines: list[str] = []
        if gaussian_config_record is not None:
            gaussian_log_lines = [
                f"gaussian_config_schema_version={gaussian_config_record['schema_version']}",
                f"gaussian_requested_profile={gaussian_config_record['requested_profile']}",
                f"gaussian_effective_config_hash={gaussian_config_record['effective_config_hash']}",
                "gaussian_effective_config="
                + canonical_config_json(gaussian_config_record["effective_config"]),
            ]
        log_text = "\n".join(
            [
                f"job_id={job_id}",
                f"mode={mode}",
                f"num_inputs={len(files)}",
                f"stage={stage}",
                "status=done",
                *gaussian_log_lines,
                *reconstruction.log_lines,
                *alignment_log_lines,
                *mesh_log_lines,
                "",
            ]
        )
        (job_dir / "logs" / "run.log").write_text(log_text, encoding="utf-8")

        assets = {
            **geometry_assets,
            **mesh_assets,
            "scene_graph": "scene_graph/scene.json",
            "log": "logs/run.log",
        }
        metrics = {
            "num_inputs": len(input_assets),
            "num_objects": len(scene["objects"]),
            **reconstruction.metrics,
            **alignment_metrics,
            **mesh_metrics,
        }
        manifest = self._build_manifest(
            job_id,
            mode,
            input_assets,
            scene,
            geometry_backend=geometry_backend,
            output_type=output_type,
            stage=stage,
            assets=assets,
            metrics=metrics,
            gaussian_config=gaussian_config_record,
        )
        if output_type == "mesh":
            self._with_existing_mesh_variants(job_dir, manifest)
        self._write_json(job_dir / "manifest.json", manifest)
        return manifest

    def get_manifest(self, job_id: str) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        manifest_path = job_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(job_id)
        manifest = self._with_existing_alignment_assets(job_dir, self._read_json(manifest_path))
        return self._with_existing_mesh_variants(job_dir, manifest)

    def get_scene(self, job_id: str) -> dict[str, Any]:
        scene_path = self.job_dir(job_id) / "scene_graph" / "scene.json"
        if not scene_path.exists():
            raise FileNotFoundError(job_id)
        return self._read_json(scene_path)

    def build_mesh_variant(self, job_id: str, requested_options: dict[str, Any]) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        manifest_path = job_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(job_id)

        manifest = self._with_existing_alignment_assets(job_dir, self._read_json(manifest_path))
        self._with_existing_mesh_variants(job_dir, manifest)
        assets = manifest.setdefault("assets", {})
        source_asset = assets.get("point_cloud_aligned") or assets.get("point_cloud")
        if not isinstance(source_asset, str):
            raise JobError("mesh variant requires a point_cloud asset")

        options = self._normalize_mesh_options(requested_options)
        variant_id = f"{options['method']}_{uuid.uuid4().hex[:8]}"
        mesh_asset = f"geometry/mesh_{variant_id}.glb"
        diagnostics_asset = f"diagnostics/mesh_{variant_id}.json"
        _, metrics, log_lines = self._run_mesh(
            job_dir,
            source_asset,
            mesh_asset,
            diagnostics_asset,
            options,
        )
        diagnostics = self._read_json(job_dir / diagnostics_asset)
        variant = self._mesh_variant(
            variant_id=variant_id,
            mesh_asset=mesh_asset,
            diagnostics_asset=diagnostics_asset,
            source_asset=source_asset,
            diagnostics=diagnostics,
            metrics=metrics,
            label=self._mesh_label(str(options["method"])),
        )
        variants = manifest.setdefault("mesh_variants", [])
        if not isinstance(variants, list):
            variants = []
            manifest["mesh_variants"] = variants
        variants.append(variant)
        self._write_json(manifest_path, manifest)
        with (job_dir / "logs" / "run.log").open("a", encoding="utf-8") as log_file:
            log_file.write("\n" + "\n".join([f"mesh_variant={variant_id}", *log_lines]) + "\n")
        return manifest

    def get_asset_path(self, job_id: str, asset_path: str) -> Path:
        job_dir = self.job_dir(job_id).resolve()
        candidate = (job_dir / asset_path).resolve()
        if job_dir != candidate and job_dir not in candidate.parents:
            raise JobError("asset path escapes job directory")
        if not candidate.is_file():
            raise FileNotFoundError(asset_path)
        return candidate

    def build_zip(self, job_id: str) -> Path:
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            raise FileNotFoundError(job_id)
        bundle_path = self.output_root / f"{job_id}.zip"
        if bundle_path.exists():
            bundle_path.unlink()
        shutil.make_archive(str(bundle_path.with_suffix("")), "zip", job_dir)
        return bundle_path

    def job_dir(self, job_id: str) -> Path:
        return self.output_root / job_id

    def _validate_request(self, mode: str, files: list[UploadedInput]) -> None:
        if mode not in VALID_MODES:
            allowed = ", ".join(sorted(VALID_MODES))
            raise JobError(f"unsupported mode '{mode}', expected one of: {allowed}")
        if not files:
            raise JobError("at least one input file is required")
        if mode == "image" and len(files) != 1:
            raise JobError("image mode requires exactly one file")
        if mode == "multi_image" and len(files) < 2:
            raise JobError("multi_image mode requires at least two files")
        if mode == "video" and len(files) != 1:
            raise JobError("video mode requires exactly one file")
        if mode == "panorama" and len(files) != 1:
            raise JobError("panorama mode requires exactly one file")

    def _new_job_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"

    def _create_job_dirs(self, job_dir: Path) -> None:
        for relative in [
            "input/images",
            "frames",
            "geometry/depth",
            "semantic/masks",
            "scene_graph",
            "logs",
        ]:
            (job_dir / relative).mkdir(parents=True, exist_ok=False)

    def _write_inputs(
        self, job_dir: Path, mode: str, files: list[UploadedInput]
    ) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        for index, uploaded in enumerate(files):
            filename = self._safe_relative_path(uploaded.filename, index)
            if mode == "video":
                destination = job_dir / "input" / Path(filename).name
            else:
                destination = job_dir / "input" / "images" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = self._deduplicate_destination(destination, index)
            destination.write_bytes(uploaded.content)
            assets.append(
                {
                    "filename": destination.name,
                    "path": destination.relative_to(job_dir).as_posix(),
                    "content_type": uploaded.content_type,
                    "size_bytes": len(uploaded.content),
                }
            )
        return assets

    def _safe_relative_path(self, filename: str, index: int) -> str:
        normalized = filename.replace("\\", "/")
        parts: list[str] = []
        for raw_part in normalized.split("/"):
            part = raw_part.strip()
            if not part or part in {".", ".."}:
                continue
            parts.append(Path(part).name)
        if not parts:
            return f"input_{index:03d}.bin"
        return "/".join(parts)

    def _deduplicate_destination(self, destination: Path, index: int) -> Path:
        if not destination.exists():
            return destination
        return destination.with_name(f"{destination.stem}_{index:03d}{destination.suffix}")

    def _try_align_point_cloud(
        self, job_dir: Path, assets: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, int | float | str | bool], list[str]]:
        point_cloud_asset = assets.get("point_cloud")
        if not point_cloud_asset:
            return {}, {"alignment_status": "skipped_no_point_cloud"}, ["alignment_status=skipped_no_point_cloud"]

        input_path = job_dir / point_cloud_asset
        if not input_path.is_file():
            return {}, {"alignment_status": "skipped_missing_point_cloud"}, ["alignment_status=skipped_missing_point_cloud"]

        output_asset = "geometry/points_aligned.ply"
        diagnostics_asset = "diagnostics/alignment.json"
        output_path = job_dir / output_asset
        diagnostics_path = job_dir / diagnostics_asset

        try:
            align_module = self._load_alignment_module()
            result = align_module.align_pointcloud(
                input_path=input_path,
                output_path=output_path,
                diagnostics_output=diagnostics_path,
                sample_size=int(os.environ.get("IMAGE3D_ALIGNMENT_SAMPLE_SIZE", "50000")),
                ransac_iterations=int(os.environ.get("IMAGE3D_ALIGNMENT_RANSAC_ITERATIONS", "400")),
                min_plane_inlier_ratio=float(os.environ.get("IMAGE3D_ALIGNMENT_MIN_PLANE_RATIO", "0.08")),
                seed=int(os.environ.get("IMAGE3D_ALIGNMENT_SEED", "42")),
            )
        except (Exception, SystemExit) as exc:
            reason = str(exc) or exc.__class__.__name__
            return (
                {},
                {
                    "alignment_status": "failed",
                    "alignment_reason": reason,
                },
                [
                    "alignment_status=failed",
                    f"alignment_reason={reason}",
                ],
            )

        if not output_path.is_file() or not diagnostics_path.is_file():
            return (
                {},
                {
                    "alignment_status": "failed",
                    "alignment_reason": "alignment outputs missing",
                },
                [
                    "alignment_status=failed",
                    "alignment_reason=alignment outputs missing",
                ],
            )

        source_plane = result.get("source_plane", {})
        inlier_ratio = source_plane.get("inlier_ratio")
        metrics: dict[str, int | float | str | bool] = {"alignment_status": "aligned"}
        if isinstance(inlier_ratio, (int, float)):
            metrics["alignment_plane_inlier_ratio"] = float(inlier_ratio)
        return (
            {
                "point_cloud_aligned": output_asset,
                "alignment_diagnostics": diagnostics_asset,
            },
            metrics,
            [
                "alignment_status=aligned",
                f"alignment_output={output_asset}",
                f"alignment_diagnostics={diagnostics_asset}",
            ],
        )

    def _build_mesh(
        self, job_dir: Path, assets: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, int | float | str | bool], list[str]]:
        source_asset = assets.get("point_cloud_aligned") or assets.get("point_cloud")
        if not source_asset:
            raise JobError("mesh output requires a point_cloud asset")

        return self._run_mesh(
            job_dir,
            source_asset,
            "geometry/mesh.glb",
            "diagnostics/mesh.json",
            self._mesh_options_from_environment(),
        )

    def _run_mesh(
        self,
        job_dir: Path,
        source_asset: str,
        mesh_asset: str,
        diagnostics_asset: str,
        options: dict[str, int | float | str],
    ) -> tuple[dict[str, str], dict[str, int | float | str | bool], list[str]]:

        source_path = job_dir / source_asset
        if not source_path.is_file():
            raise JobError(f"mesh source point cloud is missing: {source_asset}")

        mesh_path = job_dir / mesh_asset
        diagnostics_path = job_dir / diagnostics_asset

        project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
        script_path = project_root / "scripts" / "mesh_from_pointcloud.py"
        if not script_path.exists():
            raise JobError(f"mesh runner missing: {script_path}")

        command = [
            os.environ.get("IMAGE3D_PYTHON", sys.executable),
            str(script_path),
            str(source_path),
            str(mesh_path),
            "--diagnostics-output",
            str(diagnostics_path),
            "--method",
            str(options["method"]),
            "--voxel-size",
            str(options["voxel_size"]),
            "--normal-radius",
            str(options["normal_radius"]),
            "--normal-max-nn",
            str(options["normal_max_nn"]),
            "--statistical-neighbors",
            str(options["statistical_neighbors"]),
            "--statistical-std-ratio",
            str(options["statistical_std_ratio"]),
            "--radius-outlier-neighbors",
            str(options["radius_outlier_neighbors"]),
            "--radius-outlier-radius",
            str(options["radius_outlier_radius"]),
            "--poisson-depth",
            str(options["poisson_depth"]),
            "--density-trim-quantile",
            str(options["density_trim_quantile"]),
            "--component-min-ratio",
            str(options["component_min_ratio"]),
            "--edge-trim-quantile",
            str(options["edge_trim_quantile"]),
            "--edge-trim-factor",
            str(options["edge_trim_factor"]),
            "--max-triangles",
            str(options["max_triangles"]),
            "--alpha",
            str(options["alpha"]),
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
            raise JobError(f"mesh reconstruction failed:\n{details}") from exc

        if not mesh_path.is_file() or not diagnostics_path.is_file():
            raise JobError("mesh reconstruction did not produce mesh.glb and mesh.json")

        diagnostics = self._read_json(diagnostics_path)
        metrics = self._mesh_metrics(source_asset, diagnostics)

        log_lines = [
            "mesh_status=built",
            f"mesh_source={source_asset}",
            f"mesh_output={mesh_asset}",
            f"mesh_diagnostics={diagnostics_asset}",
            f"mesh_runner={' '.join(command)}",
        ]
        if completed.stdout.strip():
            log_lines.append(f"mesh_stdout={completed.stdout.strip()}")

        return (
            {
                "mesh": mesh_asset,
                "mesh_diagnostics": diagnostics_asset,
            },
            metrics,
            log_lines,
        )

    def _mesh_metrics(self, source_asset: str, diagnostics: dict[str, Any]) -> dict[str, int | float | str | bool]:
        metrics: dict[str, int | float | str | bool] = {
            "mesh_status": "built",
            "mesh_source": source_asset,
            "mesh_method": str(diagnostics.get("method", "")),
        }
        for key, metric_key in {
            "vertices": "mesh_vertices",
            "triangles": "mesh_triangles",
            "processed_points": "mesh_processed_points",
        }.items():
            value = diagnostics.get(key)
            if isinstance(value, (int, float)):
                metrics[metric_key] = int(value)
        cleanup = diagnostics.get("cleanup")
        if isinstance(cleanup, dict):
            for key, metric_key in {
                "component_count": "mesh_component_count",
                "long_edge_removed_triangles": "mesh_long_edge_removed_triangles",
                "small_component_removed_triangles": "mesh_small_component_removed_triangles",
            }.items():
                value = cleanup.get(key)
                if isinstance(value, (int, float)):
                    metrics[metric_key] = int(value)
        return metrics

    def _mesh_options_from_environment(self) -> dict[str, int | float | str]:
        requested = {
            "method": os.environ.get("IMAGE3D_MESH_METHOD", "poisson"),
            "voxel_size": os.environ.get("IMAGE3D_MESH_VOXEL_SIZE", "0.05"),
            "normal_radius": os.environ.get("IMAGE3D_MESH_NORMAL_RADIUS", "0.2"),
            "normal_max_nn": os.environ.get("IMAGE3D_MESH_NORMAL_MAX_NN", "30"),
            "statistical_neighbors": os.environ.get("IMAGE3D_MESH_STATISTICAL_NEIGHBORS", "24"),
            "statistical_std_ratio": os.environ.get("IMAGE3D_MESH_STATISTICAL_STD_RATIO", "2.0"),
            "radius_outlier_neighbors": os.environ.get("IMAGE3D_MESH_RADIUS_OUTLIER_NEIGHBORS", "0"),
            "radius_outlier_radius": os.environ.get("IMAGE3D_MESH_RADIUS_OUTLIER_RADIUS", "0.0"),
            "poisson_depth": os.environ.get("IMAGE3D_MESH_POISSON_DEPTH", "8"),
            "density_trim_quantile": os.environ.get("IMAGE3D_MESH_DENSITY_TRIM_QUANTILE", "0.1"),
            "component_min_ratio": os.environ.get("IMAGE3D_MESH_COMPONENT_MIN_RATIO", "0.03"),
            "edge_trim_quantile": os.environ.get("IMAGE3D_MESH_EDGE_TRIM_QUANTILE", "0.98"),
            "edge_trim_factor": os.environ.get("IMAGE3D_MESH_EDGE_TRIM_FACTOR", "2.5"),
            "max_triangles": os.environ.get("IMAGE3D_MESH_MAX_TRIANGLES", "120000"),
            "alpha": os.environ.get("IMAGE3D_MESH_ALPHA", "0.0"),
        }
        return self._normalize_mesh_options(requested)

    def _normalize_mesh_options(self, requested_options: dict[str, Any] | None) -> dict[str, int | float | str]:
        options: dict[str, Any] = dict(MESH_OPTION_DEFAULTS)
        if requested_options:
            unknown_options = sorted(set(requested_options) - set(MESH_OPTION_DEFAULTS))
            if unknown_options:
                raise JobError(f"unsupported mesh options: {', '.join(unknown_options)}")
            options.update(requested_options)

        try:
            options["method"] = str(options["method"])
            for key in {
                "voxel_size",
                "normal_radius",
                "statistical_std_ratio",
                "radius_outlier_radius",
                "density_trim_quantile",
                "component_min_ratio",
                "edge_trim_quantile",
                "edge_trim_factor",
                "alpha",
            }:
                options[key] = float(options[key])
            for key in {
                "normal_max_nn",
                "statistical_neighbors",
                "radius_outlier_neighbors",
                "poisson_depth",
                "max_triangles",
            }:
                options[key] = int(options[key])
        except (TypeError, ValueError) as exc:
            raise JobError("mesh options must use numeric values") from exc

        if options["method"] not in MESH_METHODS:
            raise JobError(f"unsupported mesh method '{options['method']}'")
        for key in {"voxel_size", "normal_radius", "statistical_std_ratio", "edge_trim_factor"}:
            if options[key] <= 0:
                raise JobError(f"mesh {key} must be positive")
        for key in {"normal_max_nn", "statistical_neighbors", "radius_outlier_neighbors", "poisson_depth", "max_triangles"}:
            if options[key] < 0:
                raise JobError(f"mesh {key} must be non-negative")
        for key in {"density_trim_quantile", "component_min_ratio", "edge_trim_quantile"}:
            if not 0.0 <= options[key] < 1.0:
                raise JobError(f"mesh {key} must be between 0 and 1")
        if options["radius_outlier_radius"] < 0 or options["alpha"] < 0:
            raise JobError("mesh radius and alpha values must be non-negative")
        return options

    def _with_existing_mesh_variants(self, job_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        variants = manifest.get("mesh_variants")
        if isinstance(variants, list):
            return manifest

        manifest["mesh_variants"] = []
        assets = manifest.get("assets")
        if not isinstance(assets, dict):
            return manifest
        mesh_asset = assets.get("mesh")
        diagnostics_asset = assets.get("mesh_diagnostics")
        if not isinstance(mesh_asset, str) or not isinstance(diagnostics_asset, str):
            return manifest

        diagnostics_path = job_dir / diagnostics_asset
        if not (job_dir / mesh_asset).is_file() or not diagnostics_path.is_file():
            return manifest
        try:
            diagnostics = self._read_json(diagnostics_path)
        except (json.JSONDecodeError, OSError):
            return manifest

        metrics = manifest.get("metrics")
        source_asset = ""
        if isinstance(metrics, dict):
            source_asset = str(metrics.get("mesh_source", ""))
        if not source_asset:
            source_asset = str(assets.get("point_cloud_aligned") or assets.get("point_cloud") or "")
        manifest["mesh_variants"].append(
            self._mesh_variant(
                variant_id="baseline",
                mesh_asset=mesh_asset,
                diagnostics_asset=diagnostics_asset,
                source_asset=source_asset,
                diagnostics=diagnostics,
                metrics=self._mesh_metrics(source_asset, diagnostics),
                label=f"{self._mesh_label(str(diagnostics.get('method', 'mesh')))} baseline",
            )
        )
        return manifest

    def _mesh_variant(
        self,
        *,
        variant_id: str,
        mesh_asset: str,
        diagnostics_asset: str,
        source_asset: str,
        diagnostics: dict[str, Any],
        metrics: dict[str, int | float | str | bool],
        label: str,
    ) -> dict[str, Any]:
        options = diagnostics.get("options")
        return {
            "id": variant_id,
            "label": label,
            "method": str(diagnostics.get("method", "")),
            "mesh_asset": mesh_asset,
            "diagnostics_asset": diagnostics_asset,
            "source_asset": source_asset,
            "options": options if isinstance(options, dict) else {},
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _mesh_label(self, method: str) -> str:
        return method.replace("_", " ").title()

    def _load_alignment_module(self) -> Any:
        project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
        scripts_dir = project_root / "scripts"
        script_path = scripts_dir / "align_pointcloud.py"
        if not script_path.exists():
            raise FileNotFoundError(f"alignment script missing: {script_path}")

        scripts_dir_text = str(scripts_dir)
        if scripts_dir_text not in sys.path:
            sys.path.insert(0, scripts_dir_text)
        spec = importlib_util.spec_from_file_location("image3d_scenegraph_align_pointcloud", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load alignment script: {script_path}")
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_mock_scene(self, job_id: str, mode: str) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "mode": mode,
            "coordinate_system": "mock_camera",
            "objects": [
                {
                    "id": "obj_000",
                    "label": "scene_proxy",
                    "confidence": 1.0,
                    "center": [0.0, 0.0, 1.0],
                    "extent": [1.0, 1.0, 1.0],
                    "source": "mock",
                }
            ],
            "relations": [],
            "diagnostics": {
                "scale_recovered": False,
                "physical_checks": [],
            },
        }

    def _build_manifest(
        self,
        job_id: str,
        mode: str,
        input_assets: list[dict[str, Any]],
        scene: dict[str, Any],
        geometry_backend: str,
        output_type: str,
        stage: str,
        assets: dict[str, str],
        metrics: dict[str, int | float | str | bool],
        gaussian_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        manifest = {
            "job_id": job_id,
            "status": "done",
            "stage": stage,
            "progress": 1.0,
            "mode": mode,
            "input_type": self._input_type(mode),
            "geometry_backend": geometry_backend,
            "output_type": output_type,
            "created_at": created_at,
            "inputs": input_assets,
            "assets": assets,
            "metrics": metrics,
        }
        if gaussian_config is not None:
            manifest["gaussian_config"] = gaussian_config
        return manifest

    def _input_type(self, mode: str) -> str:
        if mode == "panorama":
            return "equirectangular_panorama"
        return mode

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _with_existing_alignment_assets(self, job_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        assets = manifest.setdefault("assets", {})
        metrics = manifest.setdefault("metrics", {})
        aligned_asset = "geometry/points_aligned.ply"
        diagnostics_asset = "diagnostics/alignment.json"

        if "point_cloud_aligned" not in assets and (job_dir / aligned_asset).is_file():
            assets["point_cloud_aligned"] = aligned_asset
        if "alignment_diagnostics" not in assets and (job_dir / diagnostics_asset).is_file():
            assets["alignment_diagnostics"] = diagnostics_asset
        if assets.get("point_cloud_aligned") and "alignment_status" not in metrics:
            metrics["alignment_status"] = "aligned"
            diagnostics_path = job_dir / diagnostics_asset
            if diagnostics_path.is_file():
                try:
                    alignment = self._read_json(diagnostics_path)
                    inlier_ratio = alignment.get("source_plane", {}).get("inlier_ratio")
                    if isinstance(inlier_ratio, (int, float)):
                        metrics["alignment_plane_inlier_ratio"] = float(inlier_ratio)
                except (json.JSONDecodeError, OSError):
                    pass
        return manifest
