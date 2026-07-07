from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


VALID_GEOMETRY_BACKENDS = {"mock", "vggt", "dust3r", "mast3r", "nerfstudio_3dgs"}
VALID_OUTPUT_TYPES = {"point_cloud", "mesh", "gaussian_splat"}


class ReconstructionError(ValueError):
    """Raised when a reconstruction request cannot be served."""


@dataclass(frozen=True)
class ReconstructionContext:
    job_id: str
    job_dir: Path
    mode: str
    input_assets: list[dict[str, Any]]


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

    raise ReconstructionError(
        f"geometry_backend '{geometry_backend}' with output_type '{output_type}' is not implemented"
    )
