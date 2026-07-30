"""Thin project boundary around the approved gsplat rasterizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import GaussianModel


@dataclass(frozen=True)
class RenderCamera:
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


def render_gaussians(
    model: GaussianModel,
    camera: RenderCamera,
    *,
    sh_degree: int,
    background: torch.Tensor | None = None,
    gradient_statistics: bool = False,
) -> RenderResult:
    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional GPU environment with `uv sync --extra gpu --inexact`."
        ) from exc
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
        packed=True,
        backgrounds=background,
        absgrad=gradient_statistics,
    )
    return RenderResult(image[0], alpha[0], metadata)
