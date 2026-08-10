"""Project-owned 3D Gaussian parameter model."""

from __future__ import annotations

import math

import torch
from torch import nn


SH_DC = 0.28209479177387814


class GaussianModelError(ValueError):
    """Raised when Gaussian parameters violate the training contract."""


class GaussianModel(nn.Module):
    """Owned Gaussian parameters compatible with gsplat strategies."""

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
        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(log_scales),
                "quats": nn.Parameter(quats),
                "opacities": nn.Parameter(opacity_logits),
                "sh0": nn.Parameter(sh_coeffs[:, :1]),
                "shN": nn.Parameter(sh_coeffs[:, 1:]),
            }
        )
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
    def means(self) -> nn.Parameter:
        return self.params["means"]

    @property
    def log_scales(self) -> nn.Parameter:
        return self.params["scales"]

    @property
    def quats(self) -> nn.Parameter:
        return self.params["quats"]

    @property
    def opacity_logits(self) -> nn.Parameter:
        return self.params["opacities"]

    @property
    def sh_coeffs(self) -> torch.Tensor:
        return torch.cat((self.params["sh0"], self.params["shN"]), dim=1)

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

    def optimizers(
        self,
        learning_rates: dict,
        *,
        position_scale: float = 1.0,
    ) -> dict[str, torch.optim.Adam]:
        rates = {
            "means": float(learning_rates["position"]["initial"]) * position_scale,
            "scales": float(learning_rates["scaling"]),
            "quats": float(learning_rates["rotation"]),
            "opacities": float(learning_rates["opacity"]),
            "sh0": float(learning_rates["feature"]),
            "shN": float(learning_rates["feature"]) / 20.0,
        }
        return {
            name: torch.optim.Adam([self.params[name]], lr=rate, eps=1e-15)
            for name, rate in rates.items()
        }

    def validate(self, *, max_count: int | None = None) -> None:
        count = self.means.shape[0]
        basis_count = (self.max_sh_degree + 1) ** 2
        expected = {
            "means": (count, 3),
            "scales": (count, 3),
            "quats": (count, 4),
            "opacities": (count,),
            "sh0": (count, 1, 3),
            "shN": (count, basis_count - 1, 3),
        }
        for name, shape in expected.items():
            value = self.params[name]
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
        if (opacities < 0).any() or (opacities > 1).any():
            raise GaussianModelError("activated opacities must remain within zero and one")

    def validate_gradients(self) -> None:
        for name, parameter in self.params.items():
            if parameter.grad is None:
                raise GaussianModelError(f"missing gradient for {name}")
            if not torch.isfinite(parameter.grad).all():
                raise GaussianModelError(f"non-finite gradient for {name}")

    def snapshot(self) -> dict[str, torch.Tensor | int]:
        return {
            "max_sh_degree": self.max_sh_degree,
            "means": self.means.detach(),
            "log_scales": self.log_scales.detach(),
            "quats": self.quats.detach(),
            "opacity_logits": self.opacity_logits.detach(),
            "sh_coeffs": self.sh_coeffs.detach(),
        }

    def state_dict(self, *args, **kwargs):
        return {
            "means": self.means.detach(),
            "log_scales": self.log_scales.detach(),
            "quats": self.quats.detach(),
            "opacity_logits": self.opacity_logits.detach(),
            "sh_coeffs": self.sh_coeffs.detach(),
        }
