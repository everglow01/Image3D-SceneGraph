from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("degeneration_gate", ROOT / "scripts/smoke_mcmc_degeneration.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class Ratios(list):
    def numel(self):
        return len(self)

    def max(self):
        return max(self)

    def __gt__(self, value):
        return Ratios(x > value for x in self)

    def sum(self):
        return sum(self)


@pytest.mark.parametrize("fail", [False, True])
def test_observers_preserve_calls_results_and_restore_on_failure(tmp_path, monkeypatch, fail):
    calls = []
    output = (None, SimpleNamespace(numel=lambda: 7), None)
    failure = RuntimeError("CUDA allocation failed")

    def original_tiles(*args, **kwargs):
        calls.append((args, kwargs))
        if fail:
            raise failure
        return output

    def original_relocation(*args, **kwargs):
        calls.append((args, kwargs))
        ratios = kwargs["ratios"]
        ratios[:] = [min(51, x) for x in ratios]
        return output

    rendering = SimpleNamespace(isect_tiles=original_tiles)
    ops = SimpleNamespace(compute_relocation=original_relocation)
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rendering=rendering))
    monkeypatch.setitem(sys.modules, "gsplat.strategy", SimpleNamespace(ops=ops))
    monkeypatch.setattr(gate, "tile_estimate", lambda *args: {"bbox_intersections_estimate": 7})
    path = tmp_path / "trace.jsonl"
    ratios = Ratios([184, 2, 184])
    try:
        with gate.observe_gsplat(path):
            assert ops.compute_relocation(ratios=ratios) is output
            actual = rendering.isect_tiles("means", "radii", "depths", 16, 4, 4, packed=False)
            assert not fail and actual is output
    except RuntimeError as exc:
        assert fail and exc is failure
    assert rendering.isect_tiles is original_tiles
    assert ops.compute_relocation is original_relocation
    assert ratios == [51, 2, 51]
    assert len(calls) == 2 and calls[1][1] == {"packed": False}
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["maximum_actual_ratio"] == 184
    assert events[0]["entries_above_51"] == 2
    if fail:
        assert events[-1]["event"] == "tile_failure"
    else:
        assert events[-1]["actual_intersections"] == 7
        assert events[-1]["sorted_index_arrays_bytes"] == 168


def test_main_rejects_local_model_execution_before_torch_import(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate", "--output-dir", str(tmp_path / "new")])
    monkeypatch.setattr(gate.socket, "gethostname", lambda: "not-the-authorized-host")
    previous_torch = sys.modules.get("torch")
    with pytest.raises(SystemExit, match="禁止本机"):
        gate.main()
    assert not (tmp_path / "new").exists()
    assert sys.modules.get("torch") is previous_torch
