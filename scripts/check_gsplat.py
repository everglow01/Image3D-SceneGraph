#!/usr/bin/env python3
"""Smoke-check the pinned gsplat CUDA rasterizer and one smooth gradient."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    try:
        import gsplat
        import torch
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise SystemExit("Install the optional GPU environment with `uv sync --extra gpu --inexact`.") from exc

    version = str(getattr(gsplat, "__version__", "unknown"))
    if version.split("+", 1)[0] != "1.5.3":
        raise SystemExit(f"Expected gsplat 1.5.3, found {version}")
    if not torch.cuda.is_available():
        raise SystemExit("gsplat Stage 2 requires CUDA; torch.cuda.is_available() is false.")

    device = torch.device("cuda")
    means = torch.tensor([[-0.15, 0.0, 2.0], [0.2, 0.05, 2.4]], device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device, requires_grad=True)
    scales = torch.tensor([[0.18, 0.15, 0.12], [0.14, 0.2, 0.12]], device=device, requires_grad=True)
    opacities = torch.tensor([0.75, 0.65], device=device, requires_grad=True)
    colors = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.4, 0.9]], device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor([[[48.0, 0.0, 32.0], [0.0, 48.0, 32.0], [0.0, 0.0, 1.0]]], device=device)

    def loss_at(candidate_colors):
        image, alpha, _meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            candidate_colors,
            viewmats,
            intrinsics,
            width=64,
            height=64,
            packed=False,
        )
        weights = torch.linspace(0.0, 1.0, 64, device=device)[None, None, :, None]
        return (image * weights).sum() + 0.1 * alpha.sum(), image, alpha

    loss, image, alpha = loss_at(colors)
    loss.backward()
    gradients = [means.grad, quats.grad, scales.grad, opacities.grad, colors.grad]
    if not torch.isfinite(image).all() or not torch.isfinite(alpha).all():
        raise SystemExit("gsplat forward produced non-finite output")
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise SystemExit("gsplat backward produced missing or non-finite gradients")

    epsilon = args.finite_difference_epsilon
    with torch.no_grad():
        plus = colors.detach().clone()
        minus = colors.detach().clone()
        plus[0, 0] += epsilon
        minus[0, 0] -= epsilon
        numeric = float((loss_at(plus)[0] - loss_at(minus)[0]) / (2 * epsilon))
    analytic = float(colors.grad[0, 0])
    relative_error = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-6)
    if relative_error > args.relative_tolerance:
        raise SystemExit(
            f"finite-difference mismatch: analytic={analytic:.6g} numeric={numeric:.6g} "
            f"relative_error={relative_error:.6g}"
        )

    properties = torch.cuda.get_device_properties(device)
    print(f"gsplat={gsplat.__version__}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"forward_shape={tuple(image.shape)} alpha_shape={tuple(alpha.shape)}")
    print(f"analytic_gradient={analytic:.8g}")
    print(f"numeric_gradient={numeric:.8g}")
    print(f"relative_error={relative_error:.8g}")


if __name__ == "__main__":
    main()
