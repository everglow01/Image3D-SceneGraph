from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

from scripts import run_mcmc_backward_boundary as gate
from scripts import run_mcmc_first_divergence as native


def comparisons():
    equal = {"status": "bitwise_equal"}
    phases = {s: {"parameters/means": dict(equal)} for s in gate.snapshots.STAGES}
    boundaries = {"B0": {"inputs/colors": dict(equal), "image_gradients/rgb": dict(equal), "image_gradients/alpha": dict(equal)},
                  "B1": {"outputs/v_colors": dict(equal), "rng/torch": dict(equal)}}
    return phases, boundaries


def test_boundary_attribution_requires_bitwise_inputs():
    p, b = comparisons()
    assert gate.attribution(p, b) == "backward_difference_not_reproduced"
    p["P2"]["parameters/means"]["status"] = "numerically_equivalent"
    assert gate.attribution(p, b) == "difference_downstream_of_rasterizer_backward"
    b["B1"]["outputs/v_colors"]["status"] = "numerically_equivalent"
    assert gate.attribution(p, b) == "rasterizer_backward_differs_with_bitwise_equal_inputs"
    b["B0"]["image_gradients/rgb"]["status"] = "numerically_equivalent"
    assert gate.attribution(p, b) == "image_loss_backward_differs_SSIM_not_yet_isolated"
    b["B0"]["inputs/colors"]["status"] = "numerically_equivalent"
    assert gate.attribution(p, b) == "saved_forward_inputs_or_boundary_state_differ"
    p["P0"]["parameters/means"]["status"] = "numerically_equivalent"
    assert gate.attribution(p, b) == "P0_P1_prerequisite_not_bitwise_equal"


def test_adapter_restores_old_runner_and_blocks_local_execution(tmp_path, monkeypatch):
    before = (native.PROFILE, native.__file__, native.worker, native.FirstDivergenceProbe, native.DEPENDENCIES)
    monkeypatch.setattr(native.socket, "gethostname", lambda: "local")
    monkeypatch.setattr(sys, "argv", ["boundary", "--output-root", str(tmp_path / "new"), "--run", "--authorize-execution", "--branch", "a"])
    with pytest.raises(SystemExit, match="本机禁止"):
        with gate.adapted_runner():
            assert native.PROFILE == gate.PROFILE and native.worker is gate.worker
            assert native.__file__ == gate.__file__
            native.main()
    assert (native.PROFILE, native.__file__, native.worker, native.FirstDivergenceProbe, native.DEPENDENCIES) == before
    assert not (tmp_path / "new").exists() and "torch" not in sys.modules
    assert native.lifecycle.core_hash() == native.lifecycle.CORE


def test_protocol_and_complete_bind_boundaries(tmp_path):
    gate.write_new(tmp_path / "protocol.json", {"profile": gate.PROFILE})
    assert json.loads((tmp_path / "protocol.json").read_text())["backward_boundary"] == gate.CONTRACT
    for rank in (0, 1):
        (tmp_path / f"boundary-rank-{rank}.json").write_text('{}')
    gate.write_new(tmp_path / "complete.json", {"files": {}})
    files = json.loads((tmp_path / "complete.json").read_text())["files"]
    assert set(files) == {"boundary-rank-0.json", "boundary-rank-1.json"}
    with pytest.raises(FileExistsError):
        gate.write_new(tmp_path / "complete.json", {"files": {}})


def test_incomplete_boundaries_cannot_finish(tmp_path):
    p = gate.BoundaryProbe(tmp_path, 0)
    with pytest.raises(ValueError, match="incomplete backward"):
        p.finish()
    assert not list(tmp_path.iterdir())


def test_cgroup_names_are_independent(monkeypatch):
    seen = []
    monkeypatch.setattr(gate, "_NATIVE_CGROUP", lambda name: seen.append(name))
    gate.cgroup("image3d-first-divergence-t1-a-v1.service")
    assert seen == ["image3d-backward-boundary-a-v1.service"]
    with pytest.raises(ValueError, match="unexpected"):
        gate.cgroup("other.service")


def test_schema_missing_tensor_and_optional_inputs(monkeypatch):
    leaves = {(k,): {"type": "tensor"} for k in ()}
    leaves[("rng", "torch")] = {"type": "tensor"}
    for key in ("python", "numpy", "cuda"):
        leaves[("rng", key, 0)] = {"type": "tensor"}
    leaves.update({("outputs", key): {"type": "tensor"} for key in gate.OUTPUTS})
    leaves[("outputs", "v_backgrounds")] = {"type": "NoneType"}
    monkeypatch.setattr(gate.snapshots, "snapshot_leaves", lambda record: leaves)
    gate.validate_boundary_schema({}, "B1")
    del leaves[("outputs", "v_colors")]
    with pytest.raises(ValueError, match="missing boundary"):
        gate.validate_boundary_schema({}, "B1")


def test_original_raster_backward_called_once_between_snapshots():
    tree = ast.parse(Path(gate.__file__).read_text())
    backward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "backward")
    text = ast.unparse(backward)
    assert text.count("original(ctx, v_render_colors, v_render_alphas)") == 1
    assert text.index("'B0'") < text.index("original(ctx") < text.index("'B1'") < text.index("return result")
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("set_rng_state", "set_rng_state_all", "use_deterministic_algorithms") for n in ast.walk(tree))
