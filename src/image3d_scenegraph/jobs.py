from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable

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
LIFECYCLE_SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"done", "failed", "cancelled"}
MAX_ATTEMPTS = 3
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


class JobCancelled(RuntimeError):
    """Raised when a queued or running local job is cancelled."""


class JobStore:
    def __init__(self, output_root: Path | str | None = None) -> None:
        default_root = os.environ.get("IMAGE3D_OUTPUT_ROOT", "outputs/jobs")
        self.output_root = Path(output_root or default_root)
        self._state_lock = threading.RLock()

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
        """Run one job synchronously for compatibility with direct callers."""
        queued = self.enqueue_job(
            mode,
            files,
            geometry_backend=geometry_backend,
            output_type=output_type,
            options=options,
            gaussian_config=gaussian_config,
        )
        result = self.execute_job(queued["job_id"], cancellable=False)
        if result["status"] != "done":
            error = result.get("error") or {}
            raise JobError(str(error.get("message", "job execution failed")))
        return result

    def enqueue_job(
        self,
        mode: str,
        files: list[UploadedInput],
        geometry_backend: str = "mock",
        output_type: str = "point_cloud",
        options: dict[str, int | float | str] | None = None,
        *,
        gaussian_config: ResolvedGaussianConfig | None = None,
    ) -> dict[str, Any]:
        """Persist a validated job and return before reconstruction starts."""
        self._validate_request(mode, files)
        try:
            gaussian_config_record = (
                resolved_config_record(gaussian_config) if gaussian_config is not None else None
            )
        except GaussianConfigError as exc:
            raise JobError(str(exc)) from exc
        try:
            get_reconstruction_adapter(geometry_backend, output_type)
        except ReconstructionError as exc:
            raise JobError(str(exc)) from exc

        job_id = self._new_job_id()
        job_dir = self.job_dir(job_id)
        self._create_job_dirs(job_dir, queued=True)
        input_assets = self._write_inputs(job_dir, mode, files)
        now = self._timestamp()
        attempt = self._attempt_record("attempt-001", "fresh", None, now)
        manifest: dict[str, Any] = {
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "mode": mode,
            "input_type": self._input_type(mode),
            "geometry_backend": geometry_backend,
            "output_type": output_type,
            "created_at": now,
            "updated_at": now,
            "queued_at": now,
            "started_at": None,
            "completed_at": None,
            "cancel_requested_at": None,
            "active_attempt_id": attempt["attempt_id"],
            "attempts": [attempt],
            "error": None,
            "inputs": input_assets,
            "assets": {},
            "metrics": {"num_inputs": len(input_assets)},
        }
        if gaussian_config_record is not None:
            manifest["gaussian_config"] = gaussian_config_record
        request = {
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "job_id": job_id,
            "mode": mode,
            "geometry_backend": geometry_backend,
            "output_type": output_type,
            "options": options or {},
            "gaussian_config": gaussian_config_record,
        }
        self._write_json(job_dir / "request.json", request)
        self._write_json(job_dir / "manifest.json", manifest)
        return manifest

    def execute_job(self, job_id: str, *, cancellable: bool = True) -> dict[str, Any]:
        """Claim and execute one queued attempt."""
        job_dir = self.job_dir(job_id)
        with self._state_lock:
            manifest = self.get_manifest(job_id)
            if manifest.get("status") != "queued":
                return manifest
            now = self._timestamp()
            manifest.update(
                status="running",
                stage="geometry_reconstruction",
                progress=0.05,
                started_at=now,
                updated_at=now,
                error=None,
            )
            attempt = self._active_attempt(manifest)
            attempt.update(status="running", started_at=now)
            self._write_json(job_dir / "manifest.json", manifest)

        request = self._read_json(job_dir / "request.json")
        attempt_id = str(manifest["active_attempt_id"])
        workspace = self._create_attempt_workspace(job_dir, attempt_id)
        cancel_requested = (lambda: self.is_cancel_requested(job_id)) if cancellable else None
        try:
            result = self._run_prepared_job(
                job_id,
                workspace,
                manifest,
                request,
                cancel_requested=cancel_requested,
            )
            if cancel_requested is not None and cancel_requested():
                raise JobCancelled("job cancellation requested")
            self._set_running_stage(job_id, "publishing", 0.95)
            self._publish_workspace(job_dir, workspace, attempt_id)
            now = self._timestamp()
            result.update(
                lifecycle_schema_version=LIFECYCLE_SCHEMA_VERSION,
                status="done",
                progress=1.0,
                updated_at=now,
                queued_at=manifest.get("queued_at"),
                started_at=manifest.get("started_at"),
                completed_at=now,
                cancel_requested_at=manifest.get("cancel_requested_at"),
                active_attempt_id=attempt_id,
                attempts=manifest["attempts"],
                error=None,
            )
            attempt.update(status="done", completed_at=now, error=None)
            self._write_attempt_log(job_dir, attempt_id, "status=done\n")
            self._write_json(job_dir / "manifest.json", result)
            return result
        except JobCancelled as exc:
            self._preserve_workspace(job_dir, workspace, attempt_id)
            return self._finish_unsuccessful(job_id, "cancelled", "cancelled", str(exc))
        except (JobError, ReconstructionError, OSError, ValueError) as exc:
            self._preserve_workspace(job_dir, workspace, attempt_id)
            return self._finish_unsuccessful(job_id, "failed", "execution_failed", str(exc))
        except Exception as exc:
            self._preserve_workspace(job_dir, workspace, attempt_id)
            return self._finish_unsuccessful(
                job_id, "failed", "unexpected_error", str(exc) or exc.__class__.__name__
            )

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Idempotently cancel queued work or request cancellation of running work."""
        with self._state_lock:
            manifest = self.get_manifest(job_id)
            status = manifest.get("status")
            if status in TERMINAL_STATUSES:
                return manifest
            now = self._timestamp()
            manifest["cancel_requested_at"] = manifest.get("cancel_requested_at") or now
            manifest["updated_at"] = now
            if status == "queued":
                manifest.update(
                    status="cancelled",
                    stage="cancelled",
                    completed_at=now,
                    error={"code": "cancelled", "message": "job cancelled before execution"},
                )
                attempt = self._active_attempt(manifest)
                attempt.update(status="cancelled", completed_at=now, error=manifest["error"])
                self._write_attempt_log(
                    self.job_dir(job_id), str(manifest["active_attempt_id"]), "status=cancelled\n"
                )
            self._write_json(self.job_dir(job_id) / "manifest.json", manifest)
            return manifest

    def retry_job(self, job_id: str) -> dict[str, Any]:
        """Queue a bounded clean retry with a new immutable attempt identity."""
        with self._state_lock:
            manifest = self.get_manifest(job_id)
            if manifest.get("status") not in {"failed", "cancelled"}:
                raise JobError("only failed or cancelled jobs can be retried")
            attempts = manifest.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise JobError("legacy jobs without attempt history cannot be retried")
            if len(attempts) >= MAX_ATTEMPTS:
                raise JobError(f"retry limit reached ({MAX_ATTEMPTS} attempts)")
            parent_id = str(manifest["active_attempt_id"])
            attempt_id = f"attempt-{len(attempts) + 1:03d}"
            now = self._timestamp()
            attempts.append(self._attempt_record(attempt_id, "retry", parent_id, now))
            manifest.update(
                status="queued",
                stage="queued",
                progress=0.0,
                updated_at=now,
                queued_at=now,
                started_at=None,
                completed_at=None,
                cancel_requested_at=None,
                active_attempt_id=attempt_id,
                error=None,
                assets={},
                metrics={"num_inputs": len(manifest.get("inputs", []))},
            )
            self._write_json(self.job_dir(job_id) / "manifest.json", manifest)
            return manifest

    def list_queued_jobs(self) -> list[str]:
        if not self.output_root.is_dir():
            return []
        queued: list[tuple[str, str]] = []
        for directory in self.output_root.iterdir():
            manifest_path = directory / "manifest.json"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = self._read_json(manifest_path)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if manifest.get("status") == "queued":
                queued.append((str(manifest.get("queued_at", "")), str(manifest.get("job_id", directory.name))))
        return [job_id for _, job_id in sorted(queued)]

    def recover_interrupted_jobs(self) -> list[str]:
        """Fail work that cannot still be running after local worker restart."""
        if not self.output_root.is_dir():
            return []
        recovered: list[str] = []
        for directory in self.output_root.iterdir():
            manifest_path = directory / "manifest.json"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = self._read_json(manifest_path)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if manifest.get("status") not in {"running", "exporting"}:
                continue
            job_id = str(manifest.get("job_id", directory.name))
            code = "cancelled" if manifest.get("cancel_requested_at") else "worker_interrupted"
            status = "cancelled" if code == "cancelled" else "failed"
            self._quarantine_unpublished_outputs(directory, str(manifest.get("active_attempt_id", "unknown")))
            self._finish_unsuccessful(
                job_id,
                status,
                code,
                "worker stopped before the attempt reached a terminal state",
            )
            recovered.append(job_id)
        return recovered

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._state_lock:
            manifest = self._read_json(self.job_dir(job_id) / "manifest.json")
            return bool(manifest.get("cancel_requested_at"))

    def _run_prepared_job(
        self,
        job_id: str,
        workspace: Path,
        queued_manifest: dict[str, Any],
        request: dict[str, Any],
        *,
        cancel_requested: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        mode = str(request["mode"])
        geometry_backend = str(request["geometry_backend"])
        output_type = str(request["output_type"])
        options = request.get("options")
        if not isinstance(options, dict):
            raise JobError("persisted job options must be an object")
        input_assets = queued_manifest.get("inputs")
        if not isinstance(input_assets, list):
            raise JobError("persisted job inputs must be an array")
        self._check_cancel(cancel_requested)
        try:
            adapter = get_reconstruction_adapter(geometry_backend, output_type)
            reconstruction = adapter.run(
                ReconstructionContext(
                    job_id=job_id,
                    job_dir=workspace,
                    mode=mode,
                    input_assets=input_assets,
                    options=options,
                    cancel_requested=cancel_requested,
                )
            )
        except ReconstructionError as exc:
            if cancel_requested is not None and cancel_requested():
                raise JobCancelled("job cancellation requested") from exc
            raise JobError(str(exc)) from exc

        self._check_cancel(cancel_requested)
        self._set_running_stage(job_id, "alignment", 0.55)
        alignment_assets, alignment_metrics, alignment_log_lines = self._try_align_point_cloud(
            workspace, reconstruction.assets
        )
        self._check_cancel(cancel_requested)
        geometry_assets = {**reconstruction.assets, **alignment_assets}
        mesh_assets: dict[str, str] = {}
        mesh_metrics: dict[str, int | float | str | bool] = {}
        mesh_log_lines: list[str] = []
        stage = reconstruction.stage
        if output_type == "mesh":
            self._set_running_stage(job_id, "mesh_reconstruction", 0.75)
            mesh_assets, mesh_metrics, mesh_log_lines = self._build_mesh(
                workspace, geometry_assets, cancel_requested=cancel_requested
            )
            stage = "mesh_reconstruction"
        self._check_cancel(cancel_requested)

        scene = self._build_mock_scene(job_id, mode)
        self._write_json(workspace / "scene_graph" / "scene.json", scene)
        gaussian_config_record = request.get("gaussian_config")
        gaussian_log_lines: list[str] = []
        if isinstance(gaussian_config_record, dict):
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
                f"num_inputs={len(input_assets)}",
                f"stage={stage}",
                "status=done",
                *gaussian_log_lines,
                *reconstruction.log_lines,
                *alignment_log_lines,
                *mesh_log_lines,
                "",
            ]
        )
        (workspace / "logs" / "run.log").write_text(log_text, encoding="utf-8")
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
        result = self._build_manifest(
            job_id,
            mode,
            input_assets,
            scene,
            geometry_backend=geometry_backend,
            output_type=output_type,
            stage=stage,
            assets=assets,
            metrics=metrics,
            gaussian_config=gaussian_config_record if isinstance(gaussian_config_record, dict) else None,
        )
        result["created_at"] = queued_manifest["created_at"]
        if output_type == "mesh":
            self._with_existing_mesh_variants(workspace, result)
        return result

    def _attempt_record(
        self, attempt_id: str, kind: str, parent_attempt_id: str | None, created_at: str
    ) -> dict[str, Any]:
        return {
            "attempt_id": attempt_id,
            "kind": kind,
            "parent_attempt_id": parent_attempt_id,
            "status": "queued",
            "created_at": created_at,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

    def _active_attempt(self, manifest: dict[str, Any]) -> dict[str, Any]:
        active_id = manifest.get("active_attempt_id")
        attempts = manifest.get("attempts")
        if not isinstance(attempts, list):
            raise JobError("job attempt history is missing")
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("attempt_id") == active_id:
                return attempt
        raise JobError("active job attempt is missing")

    def _create_attempt_workspace(self, job_dir: Path, attempt_id: str) -> Path:
        attempt_root = job_dir / "lifecycle" / "attempts" / attempt_id
        attempt_root.mkdir(parents=True, exist_ok=True)
        workspace = attempt_root / "workspace"
        if workspace.exists():
            raise JobError(f"attempt workspace already exists: {attempt_id}")
        workspace.mkdir()
        (workspace / "input").symlink_to(job_dir / "input", target_is_directory=True)
        for relative in [
            "frames",
            "geometry/depth",
            "diagnostics",
            "semantic/masks",
            "scene_graph",
            "logs",
        ]:
            (workspace / relative).mkdir(parents=True, exist_ok=True)
        self._write_attempt_log(job_dir, attempt_id, "status=running\n")
        return workspace

    def _publish_workspace(self, job_dir: Path, workspace: Path, attempt_id: str) -> None:
        for name in ["frames", "geometry", "diagnostics", "semantic", "scene_graph"]:
            source = workspace / name
            destination = job_dir / name
            if destination.exists():
                raise JobError(f"cannot publish over existing output: {name}")
            os.rename(source, destination)
        run_log = workspace / "logs" / "run.log"
        if not run_log.is_file():
            raise JobError("completed attempt did not produce logs/run.log")
        os.rename(run_log, job_dir / "logs" / "run.log")
        shutil.rmtree(workspace)
        self._write_attempt_log(job_dir, attempt_id, "status=published\n")

    def _preserve_workspace(self, job_dir: Path, workspace: Path, attempt_id: str) -> None:
        if not workspace.exists():
            return
        preserved = job_dir / "lifecycle" / "attempts" / attempt_id / "partial"
        if preserved.exists():
            shutil.rmtree(preserved)
        os.rename(workspace, preserved)

    def _quarantine_unpublished_outputs(self, job_dir: Path, attempt_id: str) -> None:
        target = job_dir / "lifecycle" / "attempts" / attempt_id / "partial_published"
        for name in ["frames", "geometry", "diagnostics", "semantic", "scene_graph"]:
            source = job_dir / name
            if not source.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            destination = target / name
            if destination.exists():
                shutil.rmtree(destination)
            os.rename(source, destination)
        run_log = job_dir / "logs" / "run.log"
        if run_log.exists():
            target.mkdir(parents=True, exist_ok=True)
            os.rename(run_log, target / "run.log")

    def _finish_unsuccessful(
        self, job_id: str, status: str, code: str, message: str
    ) -> dict[str, Any]:
        with self._state_lock:
            job_dir = self.job_dir(job_id)
            manifest = self._read_json(job_dir / "manifest.json")
            now = self._timestamp()
            error = {"code": code, "message": message}
            manifest.update(
                status=status,
                stage=status,
                updated_at=now,
                completed_at=now,
                error=error,
                assets={},
            )
            attempt = self._active_attempt(manifest)
            attempt.update(status=status, completed_at=now, error=error)
            self._write_attempt_log(job_dir, str(manifest["active_attempt_id"]), f"status={status}\nerror={code}: {message}\n")
            self._write_json(job_dir / "manifest.json", manifest)
            return manifest

    def _set_running_stage(self, job_id: str, stage: str, progress: float) -> None:
        with self._state_lock:
            manifest_path = self.job_dir(job_id) / "manifest.json"
            manifest = self._read_json(manifest_path)
            if manifest.get("status") != "running":
                return
            manifest.update(stage=stage, progress=progress, updated_at=self._timestamp())
            self._write_json(manifest_path, manifest)

    def _write_attempt_log(self, job_dir: Path, attempt_id: str, text: str) -> None:
        path = job_dir / "logs" / f"{attempt_id}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def _check_cancel(self, cancel_requested: Callable[[], bool] | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise JobCancelled("job cancellation requested")

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def get_manifest(self, job_id: str) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        manifest_path = job_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(job_id)
        manifest = self._with_existing_alignment_assets(job_dir, self._read_json(manifest_path))
        if manifest.get("status") != "done":
            return manifest
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

    def _create_job_dirs(self, job_dir: Path, *, queued: bool = False) -> None:
        relative_dirs = ["input/images", "logs", "lifecycle/attempts"]
        if not queued:
            relative_dirs.extend(
                ["frames", "geometry/depth", "diagnostics", "semantic/masks", "scene_graph"]
            )
        for relative in relative_dirs:
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
        self,
        job_dir: Path,
        assets: dict[str, str],
        *,
        cancel_requested: Callable[[], bool] | None = None,
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
            cancel_requested=cancel_requested,
        )

    def _run_mesh(
        self,
        job_dir: Path,
        source_asset: str,
        mesh_asset: str,
        diagnostics_asset: str,
        options: dict[str, int | float | str],
        *,
        cancel_requested: Callable[[], bool] | None = None,
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
            if cancel_requested is None:
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
                    cancel_requested=cancel_requested,
                )
        except JobCancelled:
            raise
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
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

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
