"""Project-owned image losses and training schedules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("SSIM inputs must have matching H x W x 3 shapes")
    x = prediction.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    size = min(window_size, prediction.shape[0], prediction.shape[1])
    if size % 2 == 0:
        size -= 1
    if size < 1:
        raise ValueError("SSIM input cannot be empty")
    sigma = max(size / 6.0, 1e-3)
    coordinate = torch.arange(size, dtype=x.dtype, device=x.device) - (size - 1) / 2
    kernel = torch.exp(-(coordinate.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, size, size)
    padding = size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=3)
    mu_y = F.conv2d(y, window, padding=padding, groups=3)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(x * x, window, padding=padding, groups=3) - mu_x2
    sigma_y = F.conv2d(y * y, window, padding=padding, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=3) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().clamp(-1.0, 1.0)


def l1_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float,
    ssim_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = torch.mean(torch.abs(prediction - target))
    ssim = structural_similarity(prediction, target)
    loss = l1_weight * l1 + ssim_weight * (1.0 - ssim)
    return loss, {"l1": float(l1.detach()), "ssim": float(ssim.detach())}


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target).square())
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def exponential_learning_rate(
    initial: float,
    final: float,
    iteration: int,
    total_iterations: int,
    *,
    delay_multiplier: float,
) -> float:
    fraction = min(max(iteration / max(total_iterations, 1), 0.0), 1.0)
    value = math.exp(math.log(initial) * (1.0 - fraction) + math.log(final) * fraction)
    delay = delay_multiplier + (1.0 - delay_multiplier) * math.sin(0.5 * math.pi * fraction)
    return value * delay


def active_sh_degree(iteration: int, schedule: dict) -> int:
    initial = int(schedule["initial_degree"])
    maximum = int(schedule["max_degree"])
    interval = int(schedule["increase_every_iterations"])
    return min(maximum, initial + iteration // interval)
