from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import pytest

from scripts import smoke_distributed_camera_sizes as gate


def records():
    return {case: {"completed_updates": 5, "equivalent": True, "packed_control_passed": True,
                   "ranks": [{"rank": r, "completed_updates": 5} for r in (0, 1)]} for case in gate.CASES}


def test_spec_swaps_only_camera_assignment_or_ownership():
    for base in ("historical", "mixed"):
        original = gate.case_spec(base)
        changed = gate.case_spec(base + "_camera_swap")
        assert changed.pop("camera_assignment") == [1, 0]
        original.pop("camera_assignment")
        original.pop("name")
        changed.pop("name")
        assert original == changed
    owner = gate.case_spec("mixed_owner_swap")
    mixed = gate.case_spec("mixed")
    assert owner["ids"] == mixed["ids"][::-1]
    assert owner["points"] == mixed["points"]
    assert owner["cameras"] == mixed["cameras"]
    for case in gate.CASES:
        spec = gate.case_spec(case)
        assert sorted(spec["ids"][0] + spec["ids"][1]) == list(range(16))
    assert [len(x) for x in gate.case_spec("unequal_shards")["ids"]] == [7, 9]
    hidden = gate.case_spec("invisible_owner")
    assert all(hidden["points"][i][2] < 0 for i in hidden["ids"][1])


def test_assessment_does_not_pass_missing_or_failed_controls():
    trial = records()
    assert gate.assess(trial)["passed"]
    del trial["historical"]
    assert gate.assess(trial)["status"] == "incomplete"
    trial = records()
    trial["historical"]["equivalent"] = False
    assert gate.assess(trial)["status"] == "not_equivalent"
    trial["equal_small"]["equivalent"] = False
    assert gate.assess(trial)["status"] == "control_blocked"
    trial = records()
    trial["mixed"]["packed_control_passed"] = False
    assert gate.assess(trial)["status"] == "control_blocked"
    for update in (0, 4, 6):
        trial = records()
        trial["mixed"]["ranks"][1]["completed_updates"] = update
        assert gate.assess(trial)["status"] == "incomplete"


def test_local_entry_rejected_before_torch_import(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate", "--output-root", str(tmp_path / "probe")])
    monkeypatch.setattr(gate.socket, "gethostname", lambda: "local")
    before = sys.modules.get("torch")
    with pytest.raises(SystemExit, match="不运行本机模型"):
        gate.main()
    assert sys.modules.get("torch") is before
    assert not (tmp_path / "probe").exists()


def test_gate_wires_protection_and_retains_differences_before_collectives():
    tree = ast.parse(Path(gate.__file__).read_text())
    worker = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"))
    run = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"))
    assert worker.index("before-update.jsonl") < worker.index("shard.validate_gradients()")
    assert worker.index("comparisons.jsonl") < worker.index("torch.distributed.barrier()")
    for expected in ("TrainingMonitor", "host_group=None", "pass_fds=(lease.fileno(),)", "unlimited_host_cgroup(UNIT)"):
        assert expected in run
    assert "packed=False" in worker and "loss / 2" in worker
    assert "step_post_backward" not in worker


def test_assessment_rejects_duplicate_rank_identity():
    assert gate.ATOL == 2e-6 and gate.RTOL == 2e-4
    bad = copy.deepcopy(records())
    bad["equal_large"]["ranks"][1]["rank"] = 0
    assert gate.assess(bad)["status"] == "incomplete"


@pytest.mark.skipif(gate.socket.gethostname() != "i-94B8D131", reason="NumPy comparison checks run remotely")
@pytest.mark.parametrize("actual,expected,passed", [
    ([0.], [0.], True), ([2e-6], [0.], True), ([2.01e-6], [0.], False),
    ([1.0001], [1.], True), ([1.001], [1.], False),
    ([float("nan")], [0.], False), ([float("inf")], [float("inf")], False),
])
def test_comparison_threshold_and_nonfinite_values(actual, expected, passed):
    import numpy as np

    class Array:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    result = gate.comparison(Array(actual), Array(expected))
    assert result["passed"] == passed
    assert result["elements"] == len(actual)
    import json
    json.dumps(result, allow_nan=False)
