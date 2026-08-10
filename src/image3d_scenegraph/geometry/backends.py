from __future__ import annotations

import os
from dataclasses import dataclass
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
    return BackendSpec(
        backend_id="project_3dgs",
        label="Project 3DGS",
        supported_outputs=("gaussian_splat",),
        available=bool(available_trainers) and colmap_available,
        reason=None if available_trainers and colmap_available else "; ".join(reasons),
        setup_command="uv run python scripts/setup_colmap_cuda.py --install && uv run python scripts/setup_gaussian_trainer.py --trainer <trainer>",
        options={"gaussian_trainers": [trainer.to_dict() for trainer in trainers]},
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
