from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

from scripts import run_camera_size_scene_pair as pair
from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance, CheckpointState, create_attempt, write_checkpoint


def test_pair_does_not_replace_nominal_schedule_or_start_unbounded_training():
    tree = ast.parse(Path(pair.__file__).read_text())
    run = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"))
    assert "'--stop-after-iteration', str(UPDATES)" in run
    assert "updates=UPDATES" in run and "host_group=None" in run
    assert pair.UPDATES == 12000
    assert "'--initialization', 'frozen'" in run
    assert "filter_gaussian_sor" not in run and "final_fit" not in run
    assert "source_hashes" in run


def test_terminal_checkpoint_is_verified_without_deserializing_models(tmp_path):
    provenance = CheckpointProvenance("a"*64, "b"*64, "c"*64, "d"*64)
    create_attempt(tmp_path, attempt_id="train-001", kind="fresh", provenance=provenance)
    scheduler = {"iteration": 12000, "nominal_iterations": 30000, "requested_stop_iteration": 12000}
    state = CheckpointState(model=b"model", optimizer=b"optimizer", scheduler=json.dumps(scheduler).encode(),
        densification=b"strategy", rng=b"rng", metric_history=({"iteration": 12000, "loss": .1},))
    saved = write_checkpoint(tmp_path, attempt_id="train-001", iteration=12000, purpose="final", provenance=provenance, state=state)
    artifact = tmp_path / "attempts/train-001/artifacts"
    artifact.mkdir()
    (artifact / "model.pt").write_bytes(b"candidate")
    evaluation = {"successful_views": 351, "failed_views": [],
        "per_view": [{"image_id": str(i), "psnr": 20., "ssim": .8, "display_psnr": 20., "display_ssim": .8,
                      "rgb_range": {"nonfinite_count": 0}} for i in range(351)],
        "mcmc_health_by_rank": [dict.fromkeys(("activated_scale_nonfinite_count", "log_scale_nan_count",
            "log_scale_negative_inf_count", "log_scale_positive_inf_count", "opacity_nonfinite_count",
            "origin_distance_nonfinite_count"), 0) for _ in range(2)]}
    steps = [3000, 5000, 7000, 10000, 12000]
    (artifact / "progress.jsonl").write_text("".join(json.dumps({"event": "validation", "iteration": i, **evaluation}) + "\n" for i in steps))
    result = {"iteration": 12000, "nominal_iterations": 30000, "requested_stop_iteration": 12000,
        "world_size": 2, "gaussian_cap": 3000000, "termination_reason": "execution_budget",
        "final_checkpoint_path": "attempts/train-001/checkpoints/iteration_000012000",
        "final_checkpoint_hash": saved.checkpoint_hash, "model_path": "attempts/train-001/artifacts/model.pt",
        "final_validation": evaluation}
    (artifact / "result.json").write_text(json.dumps(result))
    protocol = {"dataset_hash": "a"*64, "config_hash": "b"*64, "core": "c"*64,
                "environments": {"baseline": {"hash": "d"*64}},
                "validation_ids": sorted(str(i) for i in range(351)), "validation_steps": steps}
    assert pair.verify_terminal(tmp_path, result, protocol, "baseline")["checkpoint_sha256"] == saved.checkpoint_hash
    with pytest.raises(ValueError, match="budget"):
        pair.verify_terminal(tmp_path, {**result, "iteration": 30000}, protocol, "baseline")
    with pytest.raises(ValueError, match="identity"):
        pair.verify_validation({**evaluation, "per_view": evaluation["per_view"][:-1]}, protocol)
    with pytest.raises(ValueError, match="nonfinite"):
        pair.verify_validation({**evaluation, "per_view": [{**v, "psnr": float("nan")} for v in evaluation["per_view"]]}, protocol)
    with pytest.raises(ValueError, match="schedule"):
        pair.verify_terminal(tmp_path, result, {**protocol, "validation_steps": [12000]}, "baseline")
    component = tmp_path / result["final_checkpoint_path"] / "optimizer.bin"
    component.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="component"):
        pair.verify_terminal(tmp_path, result, protocol, "baseline")


def test_local_entry_does_not_import_torch_or_create_outputs(tmp_path, monkeypatch):
    before = sys.modules.get("torch")
    monkeypatch.setattr(sys, "argv", ["pair", "--prepare", "--output-root", str(tmp_path / "new")])
    monkeypatch.setattr(pair.socket, "gethostname", lambda: "local")
    with pytest.raises(SystemExit, match="不运行本机模型"):
        pair.main()
    assert sys.modules.get("torch") is before and not (tmp_path / "new").exists()


def test_missing_arm_never_produces_success(tmp_path):
    with pytest.raises(SystemExit):
        pair.summarize(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "incomplete_or_unpaired"
