"""Project-owned 3D Gaussian parameter model."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


SH_DC = 0.28209479177387814
PARAMETER_GROUPS = {
    "position": "means",
    "feature": "sh_coeffs",
    "opacity": "opacity_logits",
    "scaling": "log_scales",
    "rotation": "quats",
}


class GaussianModelError(ValueError):
    """Raised when Gaussian parameters violate the training contract."""


class GaussianModel(nn.Module):
    """Owned Gaussian parameters; gsplat is used only by the render wrapper."""

    def __init__(
        self,
        *,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quats: torch.Tensor,
        opacity_logits: torch.Tensor,
        sh_coeffs: torch.Tensor,
        max_sh_degree: int = 3,
    ) -> None:
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quats = nn.Parameter(quats)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.sh_coeffs = nn.Parameter(sh_coeffs)
        self.validate()

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        colors: torch.Tensor,
        scales: torch.Tensor,
        *,
        initial_opacity: float = 0.1,
        max_sh_degree: int = 3,
    ) -> GaussianModel:
        if points.ndim != 2 or points.shape[1] != 3:
            raise GaussianModelError("initial points must have shape N x 3")
        count = len(points)
        if colors.shape != (count, 3):
            raise GaussianModelError("initial colors must have shape N x 3")
        if scales.shape not in {(count,), (count, 3)}:
            raise GaussianModelError("initial scales must have shape N or N x 3")
        if not 0.0 < initial_opacity < 1.0:
            raise GaussianModelError("initial opacity must be between zero and one")
        if scales.ndim == 1:
            scales = scales[:, None].expand(-1, 3).clone()
        basis_count = (max_sh_degree + 1) ** 2
        sh_coeffs = torch.zeros(
            (count, basis_count, 3), dtype=points.dtype, device=points.device
        )
        sh_coeffs[:, 0] = (colors - 0.5) / SH_DC
        quats = torch.zeros((count, 4), dtype=points.dtype, device=points.device)
        quats[:, 0] = 1.0
        opacity = math.log(initial_opacity / (1.0 - initial_opacity))
        return cls(
            means=points.clone(),
            log_scales=scales.log(),
            quats=quats,
            opacity_logits=torch.full(
                (count,), opacity, dtype=points.dtype, device=points.device
            ),
            sh_coeffs=sh_coeffs,
            max_sh_degree=max_sh_degree,
        )

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def activated(self) -> tuple[torch.Tensor, ...]:
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (
            self.means,
            quats,
            self.log_scales.exp(),
            self.opacity_logits.sigmoid(),
            self.sh_coeffs,
        )

    def parameter_groups(self, learning_rates: dict[str, dict[str, float]]) -> list[dict]:
        groups = []
        for group_name, attribute in PARAMETER_GROUPS.items():
            groups.append(
                {
                    "name": group_name,
                    "params": [getattr(self, attribute)],
                    "lr": float(learning_rates[group_name]["initial"]),
                }
            )
        return groups

    def validate(self, *, max_count: int | None = None) -> None:
        count = self.means.shape[0]
        expected = {
            "means": (count, 3),
            "log_scales": (count, 3),
            "quats": (count, 4),
            "opacity_logits": (count,),
            "sh_coeffs": (count, (self.max_sh_degree + 1) ** 2, 3),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise GaussianModelError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
            if not torch.isfinite(value).all():
                raise GaussianModelError(f"{name} contains non-finite values")
        if count < 1:
            raise GaussianModelError("Gaussian model cannot be empty")
        if max_count is not None and count > max_count:
            raise GaussianModelError(f"Gaussian count {count} exceeds budget {max_count}")
        means, quats, scales, opacities, _ = self.activated()
        if not torch.isfinite(means).all() or not torch.isfinite(scales).all():
            raise GaussianModelError("activated geometry contains non-finite values")
        if (scales <= 0).any():
            raise GaussianModelError("activated scales must be positive")
        if not torch.allclose(
            quats.norm(dim=-1), torch.ones(count, device=quats.device), atol=1e-5
        ):
            raise GaussianModelError("activated quaternions must be unit length")
        if (opacities <= 0).any() or (opacities >= 1).any():
            raise GaussianModelError("activated opacities must be strictly between zero and one")

    def validate_gradients(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                raise GaussianModelError(f"missing gradient for {name}")
            if not torch.isfinite(parameter.grad).all():
                raise GaussianModelError(f"non-finite gradient for {name}")

    @torch.no_grad()
    def replace_rows(self, indices: torch.Tensor, clones: torch.Tensor | None = None) -> None:
        """Keep selected rows and optionally append cloned source rows."""
        if indices.dtype != torch.long or indices.ndim != 1:
            raise GaussianModelError("topology indices must be a one-dimensional int64 tensor")
        if clones is not None and (clones.dtype != torch.long or clones.ndim != 1):
            raise GaussianModelError("clone indices must be a one-dimensional int64 tensor")
        for name in PARAMETER_GROUPS.values():
            source = getattr(self, name).detach()
            value = source[indices]
            if clones is not None and len(clones):
                appended = source[clones].clone()
                if name == "means":
                    scale = self.log_scales.detach()[clones].exp()
                    direction = torch.zeros_like(appended)
                    direction[:, 0] = 1.0
                    appended.add_(0.25 * scale * direction)
                elif name == "log_scales":
                    appended.sub_(math.log(1.6))
                    value = value.clone()
                value = torch.cat((value, appended), dim=0)
            setattr(self, name, nn.Parameter(value.clone()))
        self.validate()

    @torch.no_grad()
    def reset_opacity(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise GaussianModelError("opacity reset value must be between zero and one")
        maximum = math.log(value / (1.0 - value))
        self.opacity_logits.clamp_(max=maximum)

    def named_group_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for group_name, attribute in PARAMETER_GROUPS.items():
            yield group_name, getattr(self, attribute)
