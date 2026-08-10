"""Gaussian trainer selection and local availability probes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAINER_IDS = ("project", "graphdeco")
GRAPHDECO_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"


class GaussianTrainerError(ValueError):
    """Raised when a Gaussian trainer choice is invalid."""


@dataclass(frozen=True)
class GaussianTrainerSpec:
    trainer_id: str
    label: str
    available: bool
    reason: str | None
    setup_command: str | None
    revision: str
    license: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.trainer_id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
            "setup_command": self.setup_command,
            "revision": self.revision,
            "license": self.license,
        }


def validate_trainer_id(value: str) -> str:
    if value not in TRAINER_IDS:
        allowed = ", ".join(TRAINER_IDS)
        raise GaussianTrainerError(f"unsupported Gaussian trainer '{value}', expected one of: {allowed}")
    return value


def get_gaussian_trainer_specs(
    project_root: Path | str | None = None,
) -> list[GaussianTrainerSpec]:
    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    external_root = Path(os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external"))
    return [
        _project_spec(),
        _external_spec(
            trainer_id="graphdeco",
            label="Graphdeco official",
            root=external_root / "gaussian-splatting",
            executable="python",
            revision=GRAPHDECO_COMMIT,
            license_name="Graphdeco research/evaluation only",
        ),
    ]


def trainer_record(trainer_id: str, project_root: Path | str | None = None) -> dict[str, Any]:
    trainer_id = validate_trainer_id(trainer_id)
    spec = next(
        spec for spec in get_gaussian_trainer_specs(project_root) if spec.trainer_id == trainer_id
    )
    return {
        "id": spec.trainer_id,
        "label": spec.label,
        "revision": spec.revision,
        "license": spec.license,
    }


def _project_spec() -> GaussianTrainerSpec:
    try:
        import gsplat
        import torch
    except ImportError as exc:
        return GaussianTrainerSpec(
            trainer_id="project",
            label="Project (gsplat)",
            available=False,
            reason=f"optional GPU dependency missing: {exc.name}",
            setup_command="env -u LD_LIBRARY_PATH uv sync --extra gpu --inexact",
            revision="gsplat-1.5.3",
            license="Apache-2.0",
        )
    cuda_available = bool(torch.cuda.is_available())
    return GaussianTrainerSpec(
        trainer_id="project",
        label="Project (gsplat)",
        available=cuda_available,
        reason=None if cuda_available else "CUDA unavailable",
        setup_command="env -u LD_LIBRARY_PATH uv sync --extra gpu --inexact",
        revision=str(getattr(gsplat, "__version__", "gsplat-1.5.3")),
        license="Apache-2.0",
    )


def _external_spec(
    *,
    trainer_id: str,
    label: str,
    root: Path,
    executable: str,
    revision: str,
    license_name: str,
) -> GaussianTrainerSpec:
    python = root / ".venv" / "bin" / executable
    missing: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            missing.append("CUDA unavailable")
    except ImportError:
        missing.append("project Torch is unavailable")
    if not (root / ".git").is_dir():
        missing.append(f"repo missing: {root}")
    if not python.is_file():
        missing.append(f"environment missing: {python}")
    revision_file = root / ".image3d-revision"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").strip() != revision:
        missing.append("installed revision does not match the pinned revision")
    return GaussianTrainerSpec(
        trainer_id=trainer_id,
        label=label,
        available=not missing,
        reason="; ".join(missing) if missing else None,
        setup_command=(
            f"uv run python scripts/setup_gaussian_trainer.py --trainer {trainer_id} --install"
            + (" --accept-research-license" if trainer_id == "graphdeco" else "")
        ),
        revision=revision,
        license=license_name,
    )
