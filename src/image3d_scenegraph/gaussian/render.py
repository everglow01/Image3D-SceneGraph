"""Thin project boundary around the approved gsplat rasterizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .model import GaussianModel


NEAR_PLANE_NORMALIZED = 0.01
FAR_PLANE_NORMALIZED = 1e10


@dataclass(frozen=True)
class RenderCamera:
    """Pinhole camera with rigid extrinsics in normalized camera units."""

    image_id: str
    camera_from_normalized: torch.Tensor
    intrinsic: torch.Tensor
    width: int
    height: int


@dataclass(frozen=True)
class RenderResult:
    image: torch.Tensor
    alpha: torch.Tensor
    metadata: dict
    depth: torch.Tensor | None = None


def render_gaussians(
    model: GaussianModel,
    camera: RenderCamera,
    *,
    sh_degree: int,
    background: torch.Tensor | None = None,
    gradient_statistics: bool = False,
    distributed: bool = False,
    render_mode: Literal["RGB", "RGB+ED"] = "RGB",
) -> RenderResult:
    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional GPU environment with `uv sync --extra gpu --inexact`."
        ) from exc
    if gradient_statistics and distributed:
        require_distributed_absgrad()
    means, quats, scales, opacities, sh_coeffs = model.activated()
    image, alpha, metadata = rasterization(
        means,
        quats,
        scales,
        opacities,
        sh_coeffs,
        camera.camera_from_normalized[None],
        camera.intrinsic[None],
        width=camera.width,
        height=camera.height,
        sh_degree=sh_degree,
        packed=not distributed,
        backgrounds=background,
        render_mode=render_mode,
        absgrad=gradient_statistics,
        distributed=distributed,
        near_plane=NEAR_PLANE_NORMALIZED,
        far_plane=FAR_PLANE_NORMALIZED,
    )
    rendered = image[0]
    if render_mode == "RGB+ED":
        return RenderResult(rendered[..., :3], alpha[0], metadata, rendered[..., 3])
    return RenderResult(rendered, alpha[0], metadata)


def require_distributed_absgrad() -> None:
    from gsplat import rendering

    if getattr(rendering, "PROJECT_DISTRIBUTED_ABSGRAD_VERSION", None) != 1:
        raise RuntimeError(
            "Distributed AbsGrad requires the isolated gsplat overlay prepared by "
            "scripts/prepare_gsplat_absgrad.py; the installed package is not modified."
        )


def collect_distributed_absgrad(metadata: dict) -> None:
    """Return pixel-absolute statistics to their Gaussian owners after backward."""
    projected = metadata["means2d"]
    rasterized = metadata["project_absgrad_means2d"]
    counts = metadata["project_absgrad_counts"]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    absolute = getattr(rasterized, "absgrad", None)
    valid = (
        len(counts) == world_size
        and tuple(projected.shape) == (world_size, counts[rank], 2)
        and absolute is not None
        and tuple(absolute.shape) == (1, sum(counts), 2)
        and bool(torch.isfinite(absolute).all())
        and bool((absolute >= 0).all())
    )
    # Every rank must fail before entering the variable-sized exchange.
    ready = torch.tensor(int(valid), device=projected.device, dtype=torch.int32)
    torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
    if not bool(ready):
        raise RuntimeError("Distributed AbsGrad has missing, invalid or misaligned statistics")
    with torch.no_grad():
        outgoing = [part.contiguous() for part in absolute[0].split(counts, dim=0)]
        incoming = [torch.empty_like(projected[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(incoming, outgoing)
        projected.absgrad = torch.stack(incoming)
