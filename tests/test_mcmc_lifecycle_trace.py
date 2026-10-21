from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
import sys
import time

import pytest

from scripts import mcmc_lifecycle_observer as observer
from scripts import run_mcmc_lifecycle_trace as trace


def test_core_is_frozen_and_no_tensor_import_on_local_entry(tmp_path, monkeypatch):
    assert trace.core_hash() == trace.CORE
    monkeypatch.setattr(trace, "PROJECT", tmp_path)
    root = tmp_path / "outputs/analysis/new"
    monkeypatch.setattr(sys, "argv", ["trace", "--output-root", str(root), "--run", "--authorize-execution"])
    monkeypatch.setattr(trace.socket, "gethostname", lambda: "local")
    before = sys.modules.get("torch")
    with pytest.raises(SystemExit, match="本机禁止"):
        trace.main()
    assert sys.modules.get("torch") is before and not root.exists()


def test_remote_execution_requires_separate_opt_in_before_files(monkeypatch):
    project = Path("/usr/local/3dgs_new/Image3D-SceneGraph")
    monkeypatch.setattr(trace, "PROJECT", project)
    monkeypatch.setattr(trace.socket, "gethostname", lambda: "i-94B8D131")
    monkeypatch.setattr(sys, "argv", ["trace", "--output-root", str(project / "outputs/analysis/new"), "--run"])
    with pytest.raises(SystemExit, match="未获显式执行授权"):
        trace.main()


def test_source_cohorts_reuse_and_invalid_identities():
    r = observer.reuse_record(8, [0, 2, 4], [1, 1, 3])
    assert r["group_sizes"] == {"target": 3, "source": 2, "uninvolved": 3}
    assert r["unique_source_count"] == 2 and r["max_source_reuse"] == 2
    assert r["reuse_histogram"] == {1: 1, 2: 1}
    assert observer.reuse_record(100, list(range(60)), [99] * 60)["formula_copies_above_51_sources"] == 1
    for targets, sources in (([0], [0]), ([0, 0], [1, 1]), ([0], [8]), ([0], []), ([False], [1])):
        with pytest.raises(ValueError):
            observer.reuse_record(8, targets, sources)


def test_survival_separates_never_revived_redied_and_recovered():
    r = observer.survival_summary([False, True, True, True], [12100, 12101, -1, 12110], [True, False, True, True], [12105, 12120, -1, 12130])
    assert r["not_alive_after_relocation"] == 1
    assert r["initially_alive_ever_died"] == 2
    assert r["redeath_fraction_of_initially_alive"] == 2 / 3
    assert r["ever_dead_now_alive"] == 2 and r["dead_at_window_end"] == 1
    assert r["ever_recovered_after_death"] == 3 and r["recovered_then_dead_at_end"] == 1
    assert observer.survival_summary([False], [12100], [False], [-1])["redeath_fraction_of_initially_alive"] is None
    with pytest.raises(ValueError):
        observer.survival_summary([], [], [True], [])


def test_canonical_hash_preserves_type_order_and_nested_values():
    h = observer.digest_state
    assert h({"b": [1, 2], "a": None}) == h({"a": None, "b": [1, 2]})
    assert len({h([1, 2]), h([2, 1]), h((1, 2)), h([1., 2.])}) == 4
    with pytest.raises(ValueError):
        h(float("nan"))
    with pytest.raises(TypeError):
        h(object())


def test_resume_monitor_counts_new_samples_and_rejects_missing_updates(tmp_path):
    progress, telemetry = tmp_path / "progress.jsonl", tmp_path / "telemetry"
    telemetry.mkdir()
    for rank in (0, 1):
        (telemetry / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "local_rank": rank,
            "peak_reserved_bytes": 100, "monotonic_seconds": time.monotonic(), "finished": True}))
    events = [{"iteration": i, "world_size": 2, "batch_view_ids": ["a", "b"], "gaussian_count": 3000000}
              for i in range(trace.START + 1, trace.END + 1)]
    progress.write_text("".join(json.dumps(e) + "\n" for e in events))
    monitor = trace.ResumeMonitor(progress, telemetry)
    assert monitor(1, final=True) is None
    assert monitor.record()["optimizer_updates_observed"] == 200
    assert monitor.record()["camera_samples_observed"] == 400
    progress.write_text(json.dumps(events[1]) + "\n")
    with pytest.raises(ValueError, match="missing or reordered"):
        trace.ResumeMonitor(progress, telemetry)(1)
    progress.write_text(json.dumps(events[0]) + "\n")
    assert trace.ResumeMonitor(progress, telemetry)(1, final=True) == "training_progress_incomplete"


def audits():
    state = dict.fromkeys(("parameters", "adam", "strategy", "camera", "rng"), "a" * 64)
    rows = [{"event": "first_render", "iteration": trace.START + 1, "state": state}]
    rows += [{"event": "step", "iteration": i, "state": state,
              "sampling": [{"count": 1, "targets": "b"*64, "sources": "c"*64}] if i in (trace.EVENT, trace.END) else []}
             for i in range(trace.START + 1, trace.END + 1)]
    return rows + [{"event": "checkpoint", "iteration": trace.END, "state": state}]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def diagnostics(rank):
    groups = {g: {"count": n} for g, n in zip(observer.GROUPS, (1, 1, 1499998))}
    rows = [{"event": "relocation_before", "iteration": trace.EVENT, "groups": groups,
             **observer.reuse_record(1500000, [0], [1])},
            {"event": "relocation_after", "iteration": trace.EVENT, "groups": groups},
            {"event": "cohort_end_before_relocation", "iteration": trace.END, "groups": groups}]
    rows += [{"event": "opacity_update", "iteration": i} for i in range(trace.EVENT + 1, trace.END + 1)]
    rows += [{"event": "gradient", "iteration": i, "parameter": name}
             for i in range(trace.EVENT + 1, trace.END + 1) for name in ("opacities", "scales")]
    rows += [{"event": "contribution", "iteration": i, "camera_rank": rank, "weight_sum": [1.] * 6}
             for i in range(trace.START + 1, trace.END + 1)]
    return rows


def fixture_pair(root):
    p = {"pair": {"environments": {arm: {"hash": arm} for arm in trace.ARMS}}}
    for arm in trace.ARMS:
        for branch in trace.BRANCHES:
            stage = root / arm / branch
            stage.mkdir(parents=True)
            for rank in (0, 1):
                write_rows(stage / f"audit-rank-{rank}.jsonl", audits())
                trace.write_new(stage / f"worker-rank-{rank}.json", {"rank": rank, "checkpoint_state_verified": True,
                    "core": trace.CORE, "environment_hash": arm, "new_updates": 200})
                if branch == "observed":
                    write_rows(stage / f"diagnostic-rank-{rank}.jsonl", diagnostics(rank))
                    with gzip.open(stage / f"cohort-rank-{rank}.json.gz", "wt") as handle:
                        json.dump({"owner_rank": rank, "event": trace.EVENT, "target_slots": [0], "sampled_source_slots": [1]}, handle)
            trace.write_new(stage / "complete.json", {"status": "complete", "arm": arm, "branch": branch,
                "protocol_sha256": "protocol", "resources": {"optimizer_updates_observed": 200, "camera_samples_observed": 400,
                                                              "absolute_iteration": trace.END},
                "files": {f.name: trace.sha256_file(f) for f in stage.iterdir()}})
    return p


def test_gate1_then_gate2_bind_all_files_and_refuse_tampering(tmp_path):
    p = fixture_pair(tmp_path)
    trace.gate1(tmp_path, p, "protocol")
    trace.gate2(tmp_path, p, "protocol")
    assert trace.read(tmp_path / "gate2.json")["status"] == "observed_not_causal_intervention"
    (tmp_path / "repair/observed/diagnostic-rank-1.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="bound file changed"):
        trace.gate2(tmp_path, p, "protocol")


def test_gate1_identifies_original_nondeterminism_and_observer_change(tmp_path):
    fixture_pair(tmp_path)
    a = tmp_path / "baseline/control_a"
    b = tmp_path / "baseline/control_b"
    trace.compare_audits(a, b)
    rows = audits()
    rows[10] = {**rows[10], "state": {**rows[10]["state"], "rng": "different"}}
    write_rows(b / "audit-rank-0.jsonl", rows)
    with pytest.raises(ValueError, match="iteration=12010"):
        trace.compare_audits(a, b)
    write_rows(b / "audit-rank-0.jsonl", audits()[:-1])
    with pytest.raises(ValueError, match="missing or reordered"):
        trace.compare_audits(a, b)


def test_gate2_rejects_failed_gate1_and_unbound_branches(tmp_path):
    trace.write_new(tmp_path / "gate1.json", {"status": "failed", "protocol_sha256": "p"})
    with pytest.raises(ValueError, match="requires passed"):
        trace.gate2(tmp_path, {}, "p")
    (tmp_path / "gate1.json").write_text(json.dumps({"status": "passed", "protocol_sha256": "p", "files": {}}))
    with pytest.raises(ValueError, match="incomplete branch"):
        trace.gate2(tmp_path, {}, "p")


def test_path_and_protocol_budget_guards(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        trace.checked_path(tmp_path, "../outside")
    with pytest.raises(ValueError, match="unsafe"):
        trace.checked_path(tmp_path, "/outside")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="unsafe"):
        trace.checked_path(tmp_path, "link")
    trace.write_new(tmp_path / "protocol.json", {"profile": "bad"})
    with pytest.raises(ValueError, match="SHA"):
        trace.protocol_at(tmp_path, "wrong")


def test_source_wrappers_preserve_original_calls_and_budget():
    tree = ast.parse(Path(observer.__file__).read_text())
    sample = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample")
    assert len([n for n in ast.walk(sample) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "original_sample"]) == 1
    runner = Path(trace.__file__).read_text()
    assert 'resume_iteration=START' in runner and 'stop_after_iteration=END' in runner
    assert 'host_group=None' in runner and 'parent_attempt_id="train-001"' in runner
    assert 'register_step_pre_hook' in Path(observer.__file__).read_text()
    assert 'set_rng_state' not in Path(observer.__file__).read_text()


def test_terminal_component_verification_keeps_normal_checkpoint_contract(tmp_path):
    from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance, CheckpointState, create_attempt, write_checkpoint
    provenance = CheckpointProvenance("a"*64, "b"*64, "c"*64, "d"*64)
    create_attempt(tmp_path, attempt_id="train-001", kind="fresh", provenance=provenance)
    scheduler = {"iteration": trace.START, "nominal_iterations": 30000, "requested_stop_iteration": trace.START}
    state = CheckpointState(model=b"model", optimizer=b"adam", scheduler=json.dumps(scheduler).encode(),
                            densification=b"strategy", rng=b"rng", metric_history=())
    saved = write_checkpoint(tmp_path, attempt_id="train-001", iteration=trace.START, purpose="final", provenance=provenance, state=state)
    assert trace.verify_checkpoint(tmp_path, "train-001", trace.START, provenance.__dict__)["checkpoint_hash"] == saved.checkpoint_hash
    with pytest.raises(ValueError, match="identity"):
        trace.verify_checkpoint(tmp_path, "train-001", trace.START, {**provenance.__dict__, "code_hash": "e"*64})
    component = tmp_path / f"attempts/train-001/checkpoints/iteration_{trace.START:09d}/rng.bin"
    component.write_bytes(b"bad")
    with pytest.raises(ValueError, match="component"):
        trace.verify_checkpoint(tmp_path, "train-001", trace.START, provenance.__dict__)


def test_missing_diagnostic_window_cannot_pass_gate2_even_with_updated_hash(tmp_path):
    p = fixture_pair(tmp_path)
    stage = tmp_path / "baseline/observed"
    rows = diagnostics(0)
    rows = [r for r in rows if not (r["event"] == "gradient" and r["iteration"] == trace.END)]
    file = stage / "diagnostic-rank-0.jsonl"
    write_rows(file, rows)
    c = trace.read(stage / "complete.json")
    c["files"][file.name] = trace.sha256_file(file)
    (stage / "complete.json").write_text(json.dumps(c))
    trace.gate1(tmp_path, p, "protocol")
    with pytest.raises(ValueError, match="incomplete gradient"):
        trace.gate2(tmp_path, p, "protocol")
