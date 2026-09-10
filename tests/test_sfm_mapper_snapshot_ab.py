import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest


spec = importlib.util.spec_from_file_location("mapper_snapshot_ab", Path(__file__).parents[1] / "scripts/run_sfm_mapper_snapshot_ab.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def source_workspace(root):
    workspace = root / "workspace"
    (workspace / "colmap").mkdir(parents=True)
    (workspace / "frames").mkdir()
    names = [f"frame-{i}.jpg" for i in range(1000)]
    with sqlite3.connect(workspace / "colmap/database.db") as db:
        db.execute("CREATE TABLE images(image_id INTEGER, name TEXT)")
        db.executemany("INSERT INTO images VALUES (?, ?)", enumerate(names, 1))
    (workspace / "colmap/v2-mapper-seed.txt").write_text("\n".join(names) + "\n")
    (workspace / "frames/selection.json").write_text(json.dumps({
        "selected": [{"path": name, "time_seconds": i} for i, name in enumerate(names)]
    }))
    return workspace


def test_frozen_databases_have_same_bytes_but_no_shared_inode(tmp_path):
    workspace = source_workspace(tmp_path)
    root = tmp_path / "experiment"
    root.mkdir()
    result = runner.freeze_databases(workspace / "colmap/database.db", root)
    paths = [root / "snapshot.db", root / "incremental/database.db", root / "global/database.db"]
    assert len({p.stat().st_ino for p in paths}) == 3
    assert {runner.sha256_file(p) for p in paths} == {result["snapshot_sha256"]}
    with sqlite3.connect(paths[1]) as db:
        db.execute("DELETE FROM images")
    assert runner.sha256_file(paths[0]) == runner.sha256_file(paths[2])
    assert runner.sha256_file(workspace / "colmap/database.db") == result["source_sha256"]


def test_commands_preserve_baseline_and_cannot_load_rgb(tmp_path):
    inc = runner.arm_commands(Path("colmap"), tmp_path, "incremental", "0")[0]
    glob = runner.arm_commands(Path("colmap"), tmp_path, "global", "0")
    assert [c[1] for c in glob] == ["view_graph_calibrator", "global_mapper"]
    assert inc[inc.index("--Mapper.extract_colors") + 1] == "0"
    assert inc[inc.index("--Mapper.ba_use_gpu") + 1] == "0"
    assert "--Mapper.ba_global_frames_freq" in inc
    for command in (inc, glob[1]):
        assert command[command.index("--image_path") + 1] == str(tmp_path / "no-rgb")
        assert command[command.index("--default_random_seed") + 1] == "0"
        assert str(tmp_path / "seed.txt") in command


@pytest.mark.parametrize("arm_status,exit_code", [("completed", 2), ("execution_failed", 1)])
def test_publication_is_independent_of_quality_and_out_environment(tmp_path, monkeypatch, arm_status, exit_code):
    workspace = source_workspace(tmp_path)
    colmap = tmp_path / "colmap"
    colmap.write_text("fixed fake executable")
    output = tmp_path / "ab"
    calls = []
    monkeypatch.delenv("OUT", raising=False)

    def fake_arm(colmap, root, arm, gpu, timestamps, names, timeout):
        calls.append(arm)
        assert len(timestamps) == len(names) == 1000
        assert not list((root / "no-rgb").iterdir())
        return {"status": arm_status, "selected": None}

    monkeypatch.setattr(runner, "run_arm", fake_arm)
    assert runner.run_experiment(workspace, output, colmap, "0", 10) == exit_code
    assert calls == ["incremental", "global"]
    report = json.loads((output / "comparison.json").read_text())
    assert report["source_unchanged"]
    assert report["comparison"]["status"] == "not_comparable"
    assert report["publication_status"] == "published_evidence"
    assert not output.with_name(".ab.tmp").exists()
    with pytest.raises(FileExistsError):
        runner.run_experiment(workspace, output, colmap, "0", 10)
    assert calls == ["incremental", "global"]


def test_solver_failure_is_logged_without_fallback(tmp_path, monkeypatch):
    (tmp_path / "global").mkdir()
    (tmp_path / "global/database.db").write_bytes(b"database")
    calls = []

    def fail(command, **kwargs):
        calls.append(command[1])
        kwargs["stdout"].write("solver failed\n")
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    result = runner.run_arm(Path("colmap"), tmp_path, "global", "0", {}, {}, 1)
    assert calls == ["view_graph_calibrator"]
    assert result["status"] == "execution_failed"
    assert result["selected"] is None
    assert (tmp_path / "global/view_graph_calibrator.log").read_text() == "solver failed\n"
    assert result["stages"][0]["exit_code"] == 7


def test_timeout_keeps_stage_evidence(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    stages = []
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_stage(tmp_path, "mapper", ["colmap", "mapper"], stages, 1)
    assert stages[0]["status"] == "timeout"
    assert json.loads((tmp_path / "stages.json").read_text())["stages"] == stages


def test_candidate_uses_seed_identity_pose_gates_and_pixels(tmp_path, monkeypatch):
    (tmp_path / "cameras.txt").write_text("1 PINHOLE 64 64 50 50 32 32\n")
    (tmp_path / "points3D.txt").write_text("1 0 0 2 1 2 3 0.00001 1 0\n")
    (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a.jpg\n35 36 1\n")
    monkeypatch.setattr(runner, "build_sfm_pose_health_from_text", lambda **kwargs: {
        "status": "passed", "reason_codes": [], "temporal": {"registration_timeline": {
            "registration_rate": 1, "temporal_coverage": 1,
        }},
    })
    result = runner.evaluate_candidate(tmp_path, tmp_path / "database.db", {"a.jpg": 0}, {1: "a.jpg"})
    assert not result["accepted"]
    assert result["gate_reason_codes"] == ["registered_count_below_gate"]
    assert result["pixel_reprojection"]["median_reprojection_error_pixels"] == 5
    with pytest.raises(ValueError, match="non-seed"):
        runner.evaluate_candidate(tmp_path, tmp_path / "database.db", {"b.jpg": 0}, {1: "a.jpg"})
    (tmp_path / "points3D.txt").write_text("")
    result = runner.evaluate_candidate(tmp_path, tmp_path / "database.db", {"a.jpg": 0}, {1: "a.jpg"})
    assert "evaluation_error" in result and "health" in result


def test_comparison_requires_both_healthy_arms_and_pixel_gate():
    def arm(error):
        return {"status": "completed", "solver_elapsed_seconds": 10,
                "selected": {"registered_count": 900, "point_count": 2000,
                             "registration_timeline": {"temporal_coverage": 1},
                             "pixel_reprojection": {"median_reprojection_error_pixels": error}}}
    arms = {"incremental": arm(0.3), "global": arm(0.5)}
    assert runner.compare_arms(arms)["status"] == "failed_gates"
    arms["global"] = arm(0.31)
    assert runner.compare_arms(arms)["status"] == "passed"
    arms["incremental"]["selected"] = None
    assert runner.compare_arms(arms)["status"] == "not_comparable"
