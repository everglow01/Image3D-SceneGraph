from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from image3d_scenegraph.gaussian.trainers import get_gaussian_trainer_specs
from image3d_scenegraph.geometry.colmap import resolve_colmap_executable


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    label: str
    supported_outputs: tuple[str, ...]
    available: bool
    reason: str | None
    setup_command: str | None
    options: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.backend_id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
            "supported_outputs": list(self.supported_outputs),
            "setup_command": self.setup_command,
            **(self.options or {}),
        }


def get_backend_specs(project_root: Path | str | None = None) -> list[BackendSpec]:
    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", "."))
    external_root = Path(os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external"))
    checkpoint_root = Path(os.environ.get("IMAGE3D_CHECKPOINT_ROOT", root / "checkpoints"))
    colmap = resolve_colmap_executable(root)

    return [
        BackendSpec(
            backend_id="mock",
            label="Mock",
            supported_outputs=("point_cloud",),
            available=True,
            reason=None,
            setup_command=None,
        ),
        _external_model_spec(
            backend_id="vggt",
            label="VGGT",
            repo_path=external_root / "vggt",
            checkpoint_hint=checkpoint_root / "vggt" / "facebook--VGGT-1B" / "model.safetensors",
            supported_outputs=("point_cloud", "mesh"),
            adapter_implemented=True,
        ),
        BackendSpec(
            backend_id="colmap",
            label="COLMAP",
            supported_outputs=("point_cloud", "mesh"),
            available=colmap is not None,
            reason=None if colmap is not None else "colmap executable not found",
            setup_command="uv run python scripts/setup_colmap_cuda.py --install",
        ),
        _colmap_vggt_spec(
            colmap=colmap,
            repo_path=external_root / "vggt",
            checkpoint_hint=checkpoint_root / "vggt" / "facebook--VGGT-1B" / "model.safetensors",
        ),
        _external_model_spec(
            backend_id="dust3r",
            label="DUSt3R",
            repo_path=external_root / "dust3r",
            checkpoint_hint=checkpoint_root / "dust3r",
            supported_outputs=("point_cloud",),
        ),
        _external_model_spec(
            backend_id="mast3r",
            label="MASt3R",
            repo_path=external_root / "mast3r",
            checkpoint_hint=checkpoint_root / "mast3r",
            supported_outputs=("point_cloud",),
        ),
        _project_gaussian_spec(root, colmap),
    ]


def get_backend_status_payload(project_root: Path | str | None = None) -> dict[str, Any]:
    specs = get_backend_specs(project_root)
    return {"backends": [spec.to_dict() for spec in specs]}


def _project_gaussian_spec(
    project_root: Path, colmap: Path | None
) -> BackendSpec:
    trainers = get_gaussian_trainer_specs(project_root)
    available_trainers = [trainer for trainer in trainers if trainer.available]
    reasons = [
        f"{trainer.label}: {trainer.reason}"
        for trainer in trainers
        if trainer.reason is not None
    ]
    colmap_available = colmap is not None
    if not colmap_available:
        reasons.append("colmap executable not found")
    ffmpeg = shutil.which(os.environ.get("IMAGE3D_FFMPEG_BIN") or "ffmpeg")
    ffprobe = shutil.which(os.environ.get("IMAGE3D_FFPROBE_BIN") or "ffprobe")
    video_available = bool(ffmpeg and ffprobe)
    external_root = Path(
        os.environ.get("IMAGE3D_EXTERNAL_ROOT", project_root / "external")
    )
    checkpoint_root = Path(
        os.environ.get("IMAGE3D_CHECKPOINT_ROOT", project_root / "checkpoints")
    )
    vggt_repo = external_root / "vggt"
    vggt_checkpoint = (
        checkpoint_root / "vggt" / "facebook--VGGT-1B" / "model.safetensors"
    )
    vggt_base_missing = [
        label
        for available, label in (
            (vggt_repo.is_dir(), f"VGGT repo missing: {vggt_repo}"),
            (vggt_checkpoint.is_file(), f"VGGT checkpoint missing: {vggt_checkpoint}"),
        )
        if not available
    ]
    tracker_checkpoint = Path(
        os.environ.get(
            "IMAGE3D_VGGSFM_TRACKER_CHECKPOINT",
            checkpoint_root / "vggt" / "vggsfm_v2_tracker.pt",
        )
    )
    dinov2_repo = external_root / "dinov2"
    lightglue_repo = external_root / "lightglue"
    dinov2_checkpoint = Path(
        os.environ.get(
            "IMAGE3D_DINOV2_CHECKPOINT",
            checkpoint_root / "vggt" / "dinov2_vitb14_reg4_pretrain.pth",
        )
    )
    aliked_checkpoint = Path(
        os.environ.get(
            "IMAGE3D_ALIKED_CHECKPOINT",
            checkpoint_root
            / "vggt"
            / "torch-hub"
            / "checkpoints"
            / "aliked-n16.pth",
        )
    )
    vggt_ba_missing = [
        *vggt_base_missing,
        *(
            []
            if dinov2_repo.is_dir()
            else [f"DINOv2 repo missing: {dinov2_repo}"]
        ),
        *(
            []
            if lightglue_repo.is_dir()
            else [f"LightGlue repo missing: {lightglue_repo}"]
        ),
        *(
            []
            if importlib_util.find_spec("pycolmap") is not None
            else ["pycolmap is not installed"]
        ),
        *(
            []
            if importlib_util.find_spec("lightglue") is not None
            else ["LightGlue/ALIKED is not installed"]
        ),
        *(
            []
            if aliked_checkpoint.is_file()
            else [f"ALIKED checkpoint missing: {aliked_checkpoint}"]
        ),
        *(
            []
            if tracker_checkpoint.is_file()
            else [f"VGGSfM tracker checkpoint missing: {tracker_checkpoint}"]
        ),
        *(
            []
            if dinov2_checkpoint.is_file()
            else [f"DINOv2 checkpoint missing: {dinov2_checkpoint}"]
        ),
    ]
    return BackendSpec(
        backend_id="project_3dgs",
        label="Project 3DGS",
        supported_outputs=("gaussian_splat",),
        available=bool(available_trainers) and colmap_available,
        reason=None if available_trainers and colmap_available else "; ".join(reasons),
        setup_command="uv run python scripts/setup_colmap_cuda.py --install && uv run python scripts/setup_gaussian_trainer.py --trainer <trainer>",
        options={
            "gaussian_trainers": [trainer.to_dict() for trainer in trainers],
            "gaussian_geometry_sources": [
                {
                    "id": "colmap",
                    "label": "COLMAP",
                    "available": colmap_available,
                    "reason": None
                    if colmap_available
                    else "colmap executable not found",
                    "experimental": False,
                    "setup_command": "uv run python scripts/setup_colmap_cuda.py --install",
                },
                {
                    "id": "vggt_ba",
                    "label": "VGGT + BA",
                    "available": colmap_available and not vggt_ba_missing,
                    "reason": None
                    if colmap_available and not vggt_ba_missing
                    else "; ".join(
                        (["colmap executable not found"] if not colmap_available else [])
                        + vggt_ba_missing
                    ),
                    "experimental": True,
                    "supported_modes": ["video"],
                    "setup_command": "uv run python scripts/setup_colmap_cuda.py --install && uv run python scripts/setup_model.py --backend vggt --install",
                },
            ],
            "gaussian_postprocessors": [
                {
                    "id": "none",
                    "label": "Disabled",
                    "available": True,
                    "reason": None,
                    "experimental": False,
                    "setup_command": None,
                },
                {
                    "id": "vggt_visibility_v1",
                    "label": "VGGT Train-depth cleanup",
                    "available": not vggt_base_missing,
                    "reason": None
                    if not vggt_base_missing
                    else "; ".join(vggt_base_missing),
                    "experimental": True,
                    "setup_command": "uv run python scripts/setup_model.py --backend vggt --install",
                },
            ],
            "video_ingestion": {
                "available": video_available,
                "reason": None
                if video_available
                else "ffmpeg and ffprobe executables are required",
                "supported_profiles": ["standard_v1"],
                "max_duration_seconds": 600,
                "max_size_bytes": 2 * 1024**3,
                "max_keyframes": 800,
            },
        },
    )


def _external_model_spec(
    *,
    backend_id: str,
    label: str,
    repo_path: Path,
    checkpoint_hint: Path,
    supported_outputs: tuple[str, ...],
    adapter_implemented: bool = False,
) -> BackendSpec:
    missing: list[str] = []
    if not adapter_implemented:
        missing.append("adapter not implemented")
    if not repo_path.exists():
        missing.append(f"repo missing: {repo_path}")
    if not checkpoint_hint.exists():
        missing.append(f"checkpoint path missing: {checkpoint_hint}")

    return BackendSpec(
        backend_id=backend_id,
        label=label,
        supported_outputs=supported_outputs,
        available=not missing,
        reason="; ".join(missing) if missing else None,
        setup_command=f"uv run python scripts/setup_model.py --backend {backend_id}",
    )


def _colmap_vggt_spec(
    *, colmap: Path | None, repo_path: Path, checkpoint_hint: Path
) -> BackendSpec:
    missing: list[str] = []
    if colmap is None:
        missing.append("colmap executable not found")
    if not repo_path.exists():
        missing.append(f"repo missing: {repo_path}")
    if not checkpoint_hint.exists():
        missing.append(f"checkpoint path missing: {checkpoint_hint}")
    return BackendSpec(
        backend_id="colmap_vggt",
        label="COLMAP + VGGT",
        supported_outputs=("point_cloud", "mesh"),
        available=not missing,
        reason="; ".join(missing) if missing else None,
        setup_command="uv run python scripts/setup_colmap_cuda.py --install && uv run python scripts/setup_model.py --backend vggt",
    )
