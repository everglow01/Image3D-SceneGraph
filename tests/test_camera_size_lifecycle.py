from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src/image3d_scenegraph/gaussian/trainer.py"
TREE = ast.parse(SOURCE.read_text())
NAMESPACE = {"Path": Path, "json": json, "TrainingError": ValueError,
             "sha256_file": lambda path: hashlib.sha256(path.read_bytes()).hexdigest()}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef)
    and n.name in {"_execution_end", "_camera_size_overlay_identity"}], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


def test_execution_budget_keeps_nominal_schedule_and_validates_types():
    end = NAMESPACE["_execution_end"]
    assert end(30000, None) == 30000
    assert end(30000, 12000) == 12000
    assert end(16, 12) == 12
    for value in (0, -1, 30001, 12.0, True, "12000"):
        with pytest.raises(ValueError, match="integer"):
            end(30000, value)


def test_budget_loop_lr_validation_and_checkpoint_use_separate_endpoints():
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "train_gaussians")
    code = ast.unparse(function)
    assert "range(start_iteration, end_iteration + 1)" in code
    assert 'config, iteration, total_iterations, scene_scale' in code
    assert "stop_after_iteration is not None and iteration == end_iteration" in code
    assert "iteration=end_iteration" in code
    assert "nominal_iterations=total_iterations" in code
    assert "requested_stop_iteration=stop_after_iteration" in code
    assert "iteration_{end_iteration:09d}" in code
    writer = ast.unparse(next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_write_streaming_checkpoint"))
    assert "execution_budget or {}" in writer


def fixture_overlay(tmp_path):
    package = tmp_path / "gsplat"
    package.mkdir()
    for name in ("__init__.py", "rendering.py", "csrc.so"):
        (package / name).write_bytes(name.encode())
    files = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
             for name in ("__init__.py", "rendering.py", "csrc.so")}
    original = {**files, "rendering.py": "6b5e4101035031afbb26bed7e496382c7502f9d8bc7cc9a432f290674a4d9133"}
    record = {"profile": "camera_sizes_overlay_v1", "files": files, "source_hashes": original,
              "source_rendering_sha256": original["rendering.py"]}
    (tmp_path / "overlay.json").write_text(json.dumps(record))
    return package, record


def test_overlay_identity_checks_hashes_and_file_set(tmp_path):
    package, record = fixture_overlay(tmp_path)
    identity = NAMESPACE["_camera_size_overlay_identity"]
    assert identity(package).startswith("project_camera_sizes_v1_sha256=")
    (package / "rendering.py").write_text("mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        identity(package)
    (package / "rendering.py").write_text("rendering.py")
    (package / "extra.py").write_text("extra")
    with pytest.raises(ValueError, match="identity"):
        identity(package)
    (package / "extra.py").unlink()
    record["source_hashes"]["csrc.so"] = "a" * 64
    (tmp_path / "overlay.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        identity(package)


def test_size_overlay_cannot_mix_with_other_profiles():
    version = ast.unparse(next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_gsplat_version"))
    assert "camera_sizes != 1 or profile is not None" in version
    assert "camera-size rendering import path differs" in version
