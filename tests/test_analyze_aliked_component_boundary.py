from pathlib import Path
import runpy

import numpy as np
import pytest


def test_boundary_pair_selection_and_spatial_support(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "analyze_aliked_component_boundary.py"))
    pairs = [{"left": 1, "right": 2}, {"left": 2, "right": 3}, {"left": 1, "right": 4}, {"left": 3, "right": 4}]
    assert audit["crossing_pairs"](pairs, {2, 3}) == [pairs[0], pairs[3]]
    assert audit["spatial_support"](np.zeros((0, 2)), 100, 200)["count"] == 0
    support = audit["spatial_support"](np.array([[0, 0], [99, 199], [99, 199]]), 100, 200)
    assert support["count"] == 3
    assert support["occupied_4x4_cells"] == 2
    assert support["bbox_fraction"] == pytest.approx(0.99 * 0.995)
