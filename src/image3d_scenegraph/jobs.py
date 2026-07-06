from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        self._validate_request(mode, files)

        job_id = self._new_job_id()
        job_dir = self.job_dir(job_id)
        self._create_job_dirs(job_dir)

        input_assets = self._write_inputs(job_dir, mode, files)
        self._write_mock_geometry(job_dir / "geometry" / "points.ply")

        scene = self._build_mock_scene(job_id, mode)
        self._write_json(job_dir / "scene_graph" / "scene.json", scene)

        log_text = "\n".join(
            [
                f"job_id={job_id}",
                f"mode={mode}",
                f"num_inputs={len(files)}",
                "stage=mock_reconstruction",
                "status=done",
                "",
            ]
        )
        (job_dir / "logs" / "run.log").write_text(log_text, encoding="utf-8")

        manifest = self._build_manifest(job_id, mode, input_assets, scene)
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
            filename = self._safe_filename(uploaded.filename, index)
            if mode == "video":
                destination = job_dir / "input" / filename
            else:
                destination = job_dir / "input" / "images" / filename
            destination.write_bytes(uploaded.content)
            assets.append(
                {
                    "filename": filename,
                    "path": destination.relative_to(job_dir).as_posix(),
                    "content_type": uploaded.content_type,
                    "size_bytes": len(uploaded.content),
                }
            )
        return assets

    def _safe_filename(self, filename: str, index: int) -> str:
        name = Path(filename).name.strip()
        if not name or name in {".", ".."}:
            return f"input_{index:03d}.bin"
        return name

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
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "job_id": job_id,
            "status": "done",
            "stage": "mock_reconstruction",
            "progress": 1.0,
            "mode": mode,
            "input_type": self._input_type(mode),
            "created_at": created_at,
            "inputs": input_assets,
            "assets": {
                "point_cloud": "geometry/points.ply",
                "scene_graph": "scene_graph/scene.json",
                "log": "logs/run.log",
            },
            "metrics": {
                "num_inputs": len(input_assets),
                "num_points": 5,
                "num_objects": len(scene["objects"]),
            },
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
