from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_gsplat_mcmc", Path(__file__).resolve().parents[1] / "scripts/prepare_gsplat_mcmc.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_relocation_reference_uses_floor_and_ratio_cap_without_torch():
    path = Path(__file__).resolve().parents[1] / "scripts/smoke_mcmc_stability.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "relocation_reference")
    namespace = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    reference = namespace["relocation_reference"]
    opacity, scale = reference(.006, 2, .005)
    assert opacity == .005
    assert scale == pytest.approx(.601062538, rel=1e-8)
    assert reference(.05, 75, .005) == reference(.05, 51, .005)
    assert reference(.5, 1, .005) == (.5, 1.)
    for alpha, ratio in ((.006, 2), (.01, 10), (.05, 51), (.0001, 75), (.5, 2), (.99, 1)):
        opacity, coefficient = reference(alpha, ratio, .005)
        assert .005 <= opacity < 1 and math.isfinite(coefficient) and coefficient > 0


def test_destination_moments_patch_is_independent(monkeypatch):
    source = "def optimizer_fn(key, v):\n        v[sampled_idxs] = 0\n        return v\n"
    monkeypatch.setitem(prepare.SOURCE_SHA256, "strategy/ops.py", hashlib.sha256(source.encode()).hexdigest())
    patched = prepare.patch_destination_moments(source)
    ast.parse(patched)
    assert "v[dead_indices] = 0" in patched
    assert "mcmc_destination_moments_overlay_v1" in patched
    assert "compute_relocation" not in patched
    with pytest.raises(ValueError, match="audited wheel"):
        prepare.patch_destination_moments(source + "# changed\n")


def test_patch_rejects_unaudited_sources():
    with pytest.raises(ValueError, match="audited wheel"):
        prepare.patch_sources(dict.fromkeys(prepare.SOURCE_SHA256, "wrong source"))


def test_patch_forwards_floor_in_both_paths_and_clamps_before_scale(monkeypatch):
    wrapper = '''from .cuda._wrapper import _make_lazy_cuda_func

def compute_relocation(
    opacities, scales, ratios,
    binoms: Tensor,  # [n_max, n_max]
):
    return _make_lazy_cuda_func("relocation")(
        opacities, scales, ratios, binoms, n_max
    )
'''
    operation = '''def NAME(min_opacity=0.005):
    new_opacities, new_scales = compute_relocation(
        opacities=opacities,
        scales=scales,
        ratios=ratios,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, min=min_opacity)
'''
    kernel = '''#include "Common.h"
#include "Relocation.h"
    int n_max,
    const int n_max,
                    n_max,
    new_opacities[idx] = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx);
    // compute new scale
'''
    source = {"relocation.py": wrapper,
              "strategy/ops.py": operation.replace("NAME", "relocate") + operation.replace("NAME", "sample_add"),
              "cuda/csrc/RelocationCUDA.cu": kernel}
    monkeypatch.setattr(prepare, "SOURCE_SHA256", {
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()
    })
    patched = prepare.patch_sources(source)
    ast.parse(patched["relocation.py"])
    tree = ast.parse(patched["strategy/ops.py"])
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "compute_relocation"]
    assert len(calls) == 2
    assert all(any(key.arg == "min_opacity" for key in call.keywords) for call in calls)
    assert "n_max, min_opacity" in patched["relocation.py"]
    changed = patched["cuda/csrc/RelocationCUDA.cu"]
    assert changed.index("fminf(fmaxf") < changed.index("// compute new scale")
    assert "                    min_opacity," in changed
    assert source["relocation.py"] == wrapper
    with pytest.raises(ValueError, match="not unique"):
        prepare.replace_once("x x", "x", "y")


def test_capacity_patch_is_isolated_and_reports_actual_counts(monkeypatch):
    ops = "# original ops\n"
    strategy = """class Strategy:
    def refine(self, params, optimizers, binoms, state):
            n_relocated_gs = self._relocate_gs(params, optimizers, binoms)
    def relocate(self):
            relocate(
                params=None,
            )
            return n_gs
    def add(self):
            sample_add(
                params=None,
            )
            return n_gs
"""
    monkeypatch.setitem(prepare.SOURCE_SHA256, "strategy/ops.py", hashlib.sha256(ops.encode()).hexdigest())
    monkeypatch.setattr(prepare, "MCMC_SHA256", hashlib.sha256(strategy.encode()).hexdigest())
    patched = prepare.patch_capacity(ops, strategy)
    assert set(patched) == {"strategy/ops.py", "strategy/mcmc.py", "strategy/_project_mcmc_capacity.py"}
    for source in patched.values():
        ast.parse(source)
    assert 'n_gs = relocate(' in patched["strategy/mcmc.py"]
    assert 'n_gs = sample_add(' in patched["strategy/mcmc.py"]
    assert 'state["project_mcmc_relocated_count"] = n_relocated_gs' in patched["strategy/mcmc.py"]
    assert 'mcmc_capacity_overlay_v1' in patched["strategy/ops.py"]
    assert 'relocation.py' not in patched
    assert 'min_opacity=min_opacity' not in patched["strategy/ops.py"]
    with pytest.raises(ValueError, match="audited wheel"):
        prepare.patch_capacity(ops, strategy + "# changed")
