"""Bounded historical browser experiments, separate from manifest-backed assets."""

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def browser_experiment_router(output_root: Path) -> APIRouter:
    router = APIRouter(prefix="/api/gaussian-browser-experiments/browser-ksplat-20260922-v1")

    @router.get("/{filename}")
    @router.head("/{filename}")
    def get_browser_experiment(filename: Literal["level1.ksplat", "level2.ksplat"]) -> FileResponse:
        experiments = output_root.resolve().parent / "experiments"
        root = experiments / "browser-ksplat-20260922-v1"
        path = root / filename
        if (experiments.is_symlink() or root.is_symlink() or path.is_symlink()
                or not path.is_file()):
            raise HTTPException(status_code=404, detail="experiment asset not found")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "private, max-age=300"},
        )

    return router
