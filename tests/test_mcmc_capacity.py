from __future__ import annotations

import math
from importlib import import_module

import pytest


torch = pytest.importorskip("torch")

ops = pytest.importorskip("gsplat.strategy.ops")
capacity = import_module("scripts.mcmc_capacity_ops")


def model(opacities, column=False):
    n = len(opacities)
    logits = torch.logit(torch.tensor(opacities))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.arange(n * 3).float().reshape(n, 3)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(.01))),
        "opacities": torch.nn.Parameter(logits[:, None] if column else logits),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)),
        "sh0": torch.nn.Parameter(torch.zeros((n, 1, 3))),
        "shN": torch.nn.Parameter(torch.zeros((n, 15, 3))),
    })
    optimizers = {key: torch.optim.Adam([value], lr=.01) for key, value in params.items()}
    for key, value in params.items():
        optimizers[key].state[value] = {"step": torch.tensor(9300.),
                                       "exp_avg": torch.ones_like(value),
                                       "exp_avg_sq": torch.ones_like(value)}
    return params, optimizers


@pytest.fixture
def formula(monkeypatch):
    calls = []

    def reference(*, opacities, scales, ratios, binoms):
        calls.append(ratios.clone())
        return -torch.expm1(torch.log1p(-opacities.double()) / ratios).float(), scales.clone()

    monkeypatch.setattr(ops, "compute_relocation", reference)
    return calls


def test_copy_budget_obeys_strict_opacity_and_kernel_limits():
    values = torch.tensor([.006, .01, .5, .9999, 0., 1., float("nan"), float("inf")])
    counts = capacity.copy_budget(values, .005, 51)
    assert counts.tolist() == [0, 1, 50, 50, 0, 0, 0, 0]
    for minimum in (.001, .005, .01):
        values = torch.linspace(0, 1, 1001)
        counts = capacity.copy_budget(values, minimum, 51)
        assert (counts >= 0).all() and (counts <= 50).all()
        for alpha, count in zip(values.tolist(), counts.tolist()):
            if count:
                assert -math.expm1(math.log1p(-alpha) / (count + 1)) > max(minimum, 1 / 255)
    with pytest.raises(ValueError, match="bounds"):
        capacity.copy_budget(values, 0., 51)


@pytest.mark.parametrize("column", [False, True])
def test_relocation_caps_real_copies_and_preserves_deferred_rows(formula, column):
    params, optimizers = model([.5] + [1e-6] * 183, column)
    before = {key: value.detach().clone() for key, value in params.items()}
    state = {"history": torch.arange(184, dtype=torch.float64), "counter": 7}
    count = capacity.relocate(params, optimizers, state, torch.arange(184) > 0, torch.zeros((51, 51)))
    assert count == 50 and formula[0].tolist() == [51]
    opacity = params["opacities"].sigmoid().flatten()
    assert (opacity[:51] > .005).all()
    assert abs(-math.expm1(51 * math.log1p(-float(opacity[0]))) - .5) < 1e-6
    for key, value in params.items():
        assert len(value) == 184 and torch.equal(value[51:], before[key][51:])
        state_row = optimizers[key].state[value]
        assert state_row["step"] == 9300
        assert (state_row["exp_avg"][0] == 0).all()
        assert (state_row["exp_avg"][1:] == 1).all()
        assert len(optimizers[key].state) == 1
    assert state["history"].dtype == torch.float64 and state["counter"] == 7


def test_add_uses_same_budget_and_preserves_source_moments(formula):
    params, optimizers = model([.5])
    state = {"history": torch.ones(1, dtype=torch.float64), "counter": 7}
    count = capacity.sample_add(params, optimizers, state, 183, torch.zeros((51, 51)))
    assert count == 50 and formula[0].tolist() == [51]
    for key, value in params.items():
        assert len(value) == 51
        row = optimizers[key].state[value]
        assert row["step"] == 9300 and (row["exp_avg"][0] == 1).all()
        assert (row["exp_avg"][1:] == 0).all()
    assert state["history"].dtype == torch.float64
    assert state["history"].tolist() == [1.] + [0.] * 50
    assert state["counter"] == 7


def test_near_dead_repeated_operations_are_noops_without_rng_or_scale_drift(formula):
    params, optimizers = model([.006, 1e-6])
    originals = dict(params)
    before = {key: value.detach().clone() for key, value in params.items()}
    rng = torch.get_rng_state().clone()
    for _ in range(40):
        assert capacity.relocate(params, optimizers, {}, torch.tensor([False, True]), torch.zeros((51, 51))) == 0
        assert capacity.sample_add(params, optimizers, {}, 183, torch.zeros((51, 51))) == 0
    assert not formula and torch.equal(rng, torch.get_rng_state())
    assert all(params[key] is value and torch.equal(value, before[key]) for key, value in originals.items())


@pytest.mark.parametrize("bad_scale,bad_opacity", [
    (0., None), (-1., None), (float("nan"), None), (float("inf"), None), (1e-40, None),
    (.01, .005), (.01, 0.), (.01, 1.), (.01, float("nan")),
])
def test_invalid_formula_group_is_rejected_before_any_parameter_write(monkeypatch, bad_scale, bad_opacity):
    params, optimizers = model([.5, 1e-6])
    before = {key: value.detach().clone() for key, value in params.items()}

    def invalid(*, opacities, scales, ratios, binoms):
        opacity = torch.full_like(opacities, .2 if bad_opacity is None else bad_opacity)
        return opacity, torch.full_like(scales, bad_scale)

    monkeypatch.setattr(ops, "compute_relocation", invalid)
    assert capacity.relocate(params, optimizers, {}, torch.tensor([False, True]), torch.zeros((51, 51))) == 0
    for key, value in params.items():
        assert torch.equal(value, before[key])
        assert (optimizers[key].state[value]["exp_avg"] == 1).all()


def test_admission_keeps_proposal_target_order_and_actual_ratios(formula, monkeypatch):
    params, optimizers = model([.02, .5] + [1e-6] * 8)
    means = params["means"].detach().clone()
    proposals = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1])
    monkeypatch.setattr(ops, "_multinomial_sample", lambda *args, **kw: proposals)
    assert capacity.relocate(params, optimizers, {}, torch.arange(10) >= 2, torch.zeros((51, 51))) == 7
    assert formula[0].tolist() == [4, 5]
    for slot, donor in enumerate(proposals):
        expected = means[slot + 2] if slot == 6 else means[donor]
        assert torch.equal(params["means"][slot + 2], expected)
