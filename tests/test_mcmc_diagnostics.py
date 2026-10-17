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
