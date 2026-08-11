from __future__ import annotations

import sys
import types

import pytest


torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.render import RenderCamera, render_gaussians


def _model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0.1]),
    )


def _camera() -> RenderCamera:
    return RenderCamera(
        image_id="train-1",
        camera_from_normalized=torch.eye(4),
        intrinsic=torch.eye(3),
        width=2,
        height=2,
    )


def test_render_mode_preserves_rgb_and_exposes_expected_depth(monkeypatch):
    calls: list[str] = []

    def rasterization(*args, render_mode="RGB", **kwargs):
        calls.append(render_mode)
        channels = 4 if render_mode == "RGB+ED" else 3
        rendered = torch.arange(2 * 2 * channels, dtype=torch.float32).reshape(1, 2, 2, channels)
        return rendered, torch.ones(1, 2, 2, 1), {}

    monkeypatch.setitem(sys.modules, "gsplat.rendering", types.SimpleNamespace(rasterization=rasterization))

    rgb = render_gaussians(_model(), _camera(), sh_degree=0)
    rgb_depth = render_gaussians(_model(), _camera(), sh_degree=0, render_mode="RGB+ED")

    assert calls == ["RGB", "RGB+ED"]
    assert rgb.image.shape == (2, 2, 3)
    assert rgb.depth is None
    assert rgb_depth.image.shape == (2, 2, 3)
    assert rgb_depth.depth is not None
    assert rgb_depth.depth.shape == (2, 2)
