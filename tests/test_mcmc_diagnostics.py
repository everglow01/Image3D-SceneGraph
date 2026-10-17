from __future__ import annotations

import ast
import json
import traceback
from pathlib import Path
from types import SimpleNamespace



ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/image3d_scenegraph/gaussian/trainer.py"
TREE = ast.parse(SOURCE.read_text())
NAMES = {"_publish_event", "_record_mcmc_failure", "_mcmc_render_health"}
NAMESPACE = {"Path": Path, "json": json, "traceback": traceback}
exec(compile(ast.Module(body=[node for node in TREE.body
                             if isinstance(node, ast.FunctionDef) and node.name in NAMES],
                        type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


def test_failure_journal_keeps_original_rank_phase_and_cause(tmp_path):
    path = tmp_path / "mcmc-rank-1.jsonl"
    try:
        try:
            raise ValueError("scales contains non-finite values")
        except ValueError as cause:
            raise RuntimeError("rank failed") from cause
    except RuntimeError as error:
        NAMESPACE["_record_mcmc_failure"](path, 1, 24900, "parameter_validation", error)
    record = json.loads(path.read_text())
    assert record["rank"] == 1 and record["iteration"] == 24900
    assert record["phase"] == "parameter_validation"
    assert record["exception_type"] == "RuntimeError"
    assert "scales contains non-finite values" in record["traceback"]
    assert "rank failed" in record["traceback"]


def test_record_write_failure_does_not_replace_training_exception(tmp_path, capsys):
    NAMESPACE["_record_mcmc_failure"](tmp_path / "missing" / "record.jsonl", 0, 1, "render", RuntimeError("OOM"))
    assert "could not be written" in capsys.readouterr().out


def test_intersection_size_uses_global_render_buffer_not_local_point_count():
    metadata = {"isect_ids": SimpleNamespace(numel=lambda: 500_000_000)}
    health = NAMESPACE["_mcmc_render_health"](metadata)
    assert health["tile_intersection_count"] == 500_000_000
    assert health["sorted_index_arrays_bytes"] == 12_000_000_000
    assert health["scope"] == "rendered_camera_after_all_to_all"
    assert NAMESPACE["_mcmc_render_health"]({}) == {"status": "unavailable"}


def test_rank_health_is_published_before_validation_and_summary_collective():
    source = SOURCE.read_text()
    position = source.index('"phase": "after_strategy_before_validation"')
    assert position < source.index("model.validate(max_count=local_cap)", position)
    assert position < source.index("torch.distributed.all_reduce(summary)", position)
    handler = next(node for node in TREE.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_record_mcmc_failure")
    assert "torch" not in ast.unparse(handler)
    assert "all_gather" not in ast.unparse(handler)


def test_capacity_progress_uses_actual_global_relocations_not_requests():
    block = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
                 and any(isinstance(item, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "actual" for target in item.targets)
                         for item in node.body)
                 and "project_mcmc_relocated_count" in ast.unparse(node.test))
    namespace = {"strategy_state": {"project_mcmc_relocated_count": 2},
                 "event": {"mcmc_health_by_rank": [{"relocated_count": 2}, {"relocated_count": 1}]},
                 "mcmc_dead_global": 10}
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert namespace["event"]["mcmc_relocated_count"] == 3
    assert namespace["event"]["mcmc_deferred_count"] == 7


def test_guarded_gate_reuses_active_gpu_monitor_without_host_limit(tmp_path, monkeypatch):
    import shutil
    import socket
    import sys
    from image3d_scenegraph.gaussian import absgrad_resources
    from image3d_scenegraph import gpu_lease
    from scripts import run_mcmc_stability_ablation as ablation

    path = ROOT / "scripts/smoke_mcmc_stability.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "guarded_gate")
    namespace = {"Path": Path, "socket": socket, "shutil": shutil, "sys": sys, "__file__": str(path)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    project = Path("/usr/local/3dgs_new/Image3D-SceneGraph")
    root = project / "outputs/analysis/gate"
    args = SimpleNamespace(output_dir=root, relocation=False, destination_moments=False,
                           resource_telemetry_dir=None, capacity=True)
    checked = []
    monkeypatch.setattr(socket, "gethostname", lambda: "i-94B8D131")
    monkeypatch.setattr(ablation, "PROJECT", project)
    monkeypatch.setattr(ablation, "unlimited_host_cgroup", lambda unit: checked.append(unit))
    monkeypatch.setitem(sys.modules, "run_mcmc_stability_ablation", ablation)
    monkeypatch.setattr(gpu_lease, "require_idle_gpu", lambda: checked.append("idle"))
    class Lease:
        def __init__(self, path):
            assert path == project / "outputs/.gpu.lock"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def fileno(self):
            return 123
    monkeypatch.setattr(gpu_lease, "FileLease", Lease)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=30 * 1024**3))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: None)
    def stage(output, name, command, **kw):
        assert output == root and name == "probe"
        assert "--capacity" in command and "--resource-telemetry-dir" in command
        assert "--guarded" not in command
        assert kw["host_group"] is None and kw["pass_fds"] == (123,)
        monitor = kw["monitor"]
        assert monitor.updates == 12 and monitor.main_stage
        assert monitor.limits["max_main_stage_wall_seconds"] == 300
        assert monitor.limits["max_per_rank_peak_reserved_bytes"] == [18 * 1024**3] * 2
        checked.append("stage")
    monkeypatch.setattr(absgrad_resources, "run_stage", stage)
    namespace["guarded_gate"](args)
    assert checked == ["image3d-mcmc-capacity-20261017-v1.service", "idle", "stage"]
