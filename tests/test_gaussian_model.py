from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.model import GaussianModel, GaussianModelError
from image3d_scenegraph.gaussian.training_math import (
    active_sh_degree,
    exponential_learning_rate,
    l1_ssim_loss,
)


def make_model(count: int = 3) -> GaussianModel:
    points = torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.1], [-0.2, 0.1, 2.2]])[:count]
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])[:count]
    scales = torch.full((count,), 0.1)
    return GaussianModel.from_points(points, colors, scales)


def test_model_activations_groups_and_topology_are_valid():
    model = make_model()
    learning_rates = {
        name: {"initial": 0.1, "final": 0.01}
        for name in ("position", "feature", "opacity", "scaling", "rotation")
    }

    groups = model.parameter_groups(learning_rates)
    model.replace_rows(torch.tensor([0, 2]), torch.tensor([0]))
    model.reset_opacity(0.01)

    assert [group["name"] for group in groups] == list(learning_rates)
    assert model.count == 3
    assert torch.all(model.activated()[3] <= 0.010001)
    model.validate(max_count=3)


def test_model_rejects_non_finite_and_budget_overflow():
    model = make_model()
    with pytest.raises(GaussianModelError, match="exceeds budget"):
        model.validate(max_count=2)
    model.means.data[0, 0] = math.nan
    with pytest.raises(GaussianModelError, match="non-finite"):
        model.validate()


def test_loss_and_schedules_have_expected_boundaries():
    target = torch.rand(16, 16, 3)
    identical, terms = l1_ssim_loss(target, target, l1_weight=0.8, ssim_weight=0.2)
    initial = exponential_learning_rate(0.1, 0.01, 0, 100, delay_multiplier=0.1)
    final = exponential_learning_rate(0.1, 0.01, 100, 100, delay_multiplier=0.1)

    assert float(identical) == pytest.approx(0.0, abs=1e-6)
    assert terms["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert initial == pytest.approx(0.01)
    assert final == pytest.approx(0.01)
    assert active_sh_degree(0, {"initial_degree": 0, "max_degree": 3, "increase_every_iterations": 10}) == 0
    assert active_sh_degree(31, {"initial_degree": 0, "max_degree": 3, "increase_every_iterations": 10}) == 3
