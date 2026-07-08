from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image3d_scenegraph.geometry.adapters import (
    ReconstructionContext,
    ReconstructionError,
    get_reconstruction_adapter,
)


VALID_MODES = {"image", "multi_image", "video", "panorama"}


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
        options: dict[str, int | float] | None = None,
    ) -> dict[str, Any]:
        self._validate_request(mode, files)

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

        scene = self._build_mock_scene(job_id, mode)
        self._write_json(job_dir / "scene_graph" / "scene.json", scene)

        log_text = "\n".join(
            [
                f"job_id={job_id}",
                f"mode={mode}",
                f"num_inputs={len(files)}",
                f"stage={reconstruction.stage}",
                "status=done",
                *reconstruction.log_lines,
                "",
            ]
        )
        (job_dir / "logs" / "run.log").write_text(log_text, encoding="utf-8")

        assets = {
            **reconstruction.assets,
            "scene_graph": "scene_graph/scene.json",
            "log": "logs/run.log",
        }
        metrics = {
            "num_inputs": len(input_assets),
            "num_objects": len(scene["objects"]),
            **reconstruction.metrics,
        }
        manifest = self._build_manifest(
            job_id,
            mode,
            input_assets,
            scene,
            geometry_backend=geometry_backend,
            output_type=output_type,
            stage=reconstruction.stage,
            assets=assets,
            metrics=metrics,
        )
        self._write_json(job_dir / "manifest.json", manifest)
        return manifest

    def get_manifest(self, job_id: str) -> dict[str, Any]:
        manifest_path = self.job_dir(job_id) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(job_id)
        return self._read_json(manifest_path)

    def get_scene(self, job_id: str) -> dict[str, Any]:
        scene_path = self.job_dir(job_id) / "scene_graph" / "scene.json"
        if not scene_path.exists():
            raise FileNotFoundError(job_id)
        return self._read_json(scene_path)

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
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
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
