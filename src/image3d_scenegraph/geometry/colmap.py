from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_colmap_executable(project_root: Path | str | None = None) -> Path | None:
    """Resolve an explicit, project-local, or PATH COLMAP executable."""
    configured = os.environ.get("IMAGE3D_COLMAP_BIN")
    if configured:
        return _executable(Path(configured).expanduser())

    root = Path(project_root or os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    external_root = Path(os.environ.get("IMAGE3D_EXTERNAL_ROOT", root / "external")).expanduser()
    local = external_root / "colmap-4-cuda" / "install" / "bin" / "colmap"
    if resolved := _executable(local):
        return resolved

    found = shutil.which("colmap")
    return _executable(Path(found)) if found else None


def _executable(path: Path) -> Path | None:
    resolved = path.resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    return None
