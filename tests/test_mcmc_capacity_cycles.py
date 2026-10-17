from __future__ import annotations

import copy
import json
import sys
import time

import pytest

from scripts import run_mcmc_capacity_cycles as cycles


def records():
    return {f"{arm}/{case}": {"initialization_sha256": ["a", "b"], "reference_sha256": ["c", "d"],
        "terminal": "completed", "completed_updates": 1000, "final_metrics": {"raw_psnr": 30.},
        "final_rank_health": [{"effective_visible_count": 4}, {"effective_visible_count": 4}]}
        for arm in ("baseline", "capacity") for case in cycles.CASES}


def test_pair_requires_matching_inputs_and_complete_visible_and_imbalanced_cases():
    baseline = records()
    assert cycles.assess_pair(baseline)["eligible_for_10k"]
    for case in ("visible", "imbalanced"):
        trial = copy.deepcopy(baseline)
        for arm in ("baseline", "capacity"):
            trial[f"{arm}/{case}"].update(terminal="source_exhausted", completed_updates=200)
        result = cycles.assess_pair(trial)
        assert not result["eligible_for_10k"]
        assert len(result["blocking_reasons"]) == 2
    for field in ("initialization_sha256", "reference_sha256"):
        trial = copy.deepcopy(baseline)
        trial["capacity/visible"][field] = ["changed"]
        assert not cycles.assess_pair(trial)["eligible_for_10k"]
    trial = copy.deepcopy(baseline)
    trial["capacity/imbalanced"]["final_rank_health"][1]["effective_visible_count"] = 0
    assert not cycles.assess_pair(trial)["eligible_for_10k"]
    del baseline["capacity/visible"]
    assert not cycles.assess_pair(baseline)["eligible_for_10k"]


def test_negative_control_exhaustion_is_not_a_quality_regression():
    trial = records()
    for arm in ("baseline", "capacity"):
        trial[f"{arm}/no_image_signal"].update(terminal="source_exhausted", completed_updates=200)
    assert cycles.assess_pair(trial)["eligible_for_10k"]


@pytest.mark.parametrize("score,eligible", [(29.5, True), (29.49, False), (float("nan"), False)])
def test_predeclared_quality_veto(score, eligible):
    trial = records()
    trial["capacity/visible"]["final_metrics"]["raw_psnr"] = score
    assert cycles.assess_pair(trial)["eligible_for_10k"] is eligible


def termination():
    return {"requested_updates": 1000, "completed_updates": 200, "terminal": "source_exhausted",
            "exhausted_ranks": [1], "final_rank_health": [{"alive_count": 10}, {"alive_count": 0}]}


def test_early_termination_preserves_resource_and_actual_progress_checks(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(termination()))
    progress = tmp_path / "progress.jsonl"
    progress.write_text("".join(json.dumps({"iteration": i, "world_size": 2, "gaussian_count": 256,
        "batch_view_ids": ["left", "right"]}) + "\n" for i in range(1, 201)))
    telemetry = tmp_path / "memory"
    telemetry.mkdir()
    for rank in (0, 1):
        (telemetry / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "local_rank": rank,
            "peak_reserved_bytes": 100, "monotonic_seconds": time.monotonic(), "finished": True}))
    limits = {**cycles.LIMITS, "max_observed_global_gaussians": 256, "max_main_stage_wall_seconds": 900}
    monitor = cycles.CycleMonitor(progress, telemetry, updates=1000, main_stage=True, limits=limits)
    assert monitor(5., final=True) is None
    assert monitor.record()["optimizer_updates_observed"] == 200
    assert monitor.record()["camera_samples_observed"] == 400
    path = telemetry / "rank-1.json"
    value = json.loads(path.read_text())
    value["peak_reserved_bytes"] = 19 * 1024**3
    path.write_text(json.dumps(value))
    assert monitor(5., final=True) == "rank_1_reserved_exceeded_hardware_budget"


@pytest.mark.parametrize("changes", [
    {"completed_updates": 199}, {"terminal": "completed"}, {"exhausted_ranks": []},
    {"requested_updates": 900}, {"completed_updates": True},
    {"final_rank_health": [{"alive_count": 10}, {"alive_count": 1}]},
])
def test_monitor_does_not_relabel_incomplete_or_invalid_termination(tmp_path, changes):
    report = {**termination(), **changes}
    (tmp_path / "summary.json").write_text(json.dumps(report))
    monitor = cycles.CycleMonitor(tmp_path / "progress.jsonl", tmp_path / "memory", updates=1000, main_stage=True)
    with pytest.raises(ValueError, match="controlled termination"):
        monitor(5., final=True)


def test_local_entry_rejected_before_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cycles", "--output-root", str(tmp_path / "new")])
    monkeypatch.setattr(cycles.socket, "gethostname", lambda: "local")
    before = sys.modules.get("torch")
    with pytest.raises(SystemExit, match="不运行本机模型"):
        cycles.main()
    assert sys.modules.get("torch") is before
    assert not (tmp_path / "new").exists()
