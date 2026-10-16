from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def suite(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("mcmc_suite", ROOT / "scripts/run_mcmc_stability_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    def prepare(args):
        args.output_root.mkdir()
        (args.output_root / "protocol.json").write_text("{}")
    monkeypatch.setattr(module.ablation, "prepare", prepare)
    return module


@pytest.mark.parametrize("failure", [None, "unit", "verification"])
def test_suite_continues_after_candidate_failure_without_retry(suite, tmp_path, monkeypatch, failure):
    calls = []
    def run(root, arm):
        calls.append(arm)
        return 1 if failure == "unit" and arm == "raw_loss" else 0
    def verify(root, arm, sequence):
        if failure == "verification" and arm == "raw_loss":
            raise ValueError("bad artifact hash")
        assert sequence == (None if arm == "baseline" else "sequence")
        return {"arm": arm, "status": "verified", "resources": {"camera_sequence_sha256": "sequence"},
                "selection_raw_psnr": 20., "selection_display_psnr": 21.}
    monkeypatch.setattr(suite, "run_arm", run)
    monkeypatch.setattr(suite, "verify_arm", verify)
    root = tmp_path / "suite"
    assert suite.run_suite(root) == (0 if failure is None else 1)
    assert calls == list(suite.ablation.ARMS)
    summary = json.loads((root / "suite.summary.json").read_text())
    assert summary["arms"][-1]["status"] == "verified"
    assert summary["arms"][1]["status"] == ("verified" if failure is None else "failed")
    with pytest.raises(ValueError, match="already exists"):
        suite.run_suite(root)
    with pytest.raises(FileExistsError):
        suite.run_suite(tmp_path / "another-suite")
    assert calls == list(suite.ablation.ARMS)


def test_missing_baseline_skips_dependents_and_records_summary(suite, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "run_arm", lambda root, arm: calls.append(arm) or 1)
    root = tmp_path / "suite"
    assert suite.run_suite(root) == 1
    assert calls == ["baseline"]
    records = json.loads((root / "suite.summary.json").read_text())["arms"]
    assert [r["status"] for r in records] == ["failed"] + ["skipped"] * 4
    assert all(r["reason"] == "matched_baseline_unavailable" for r in records[1:])


def test_arm_command_is_blocking_unlimited_and_checks_idle_gpu(suite, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "require_idle_gpu", lambda: calls.append("idle"))
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(suite.subprocess, "run", run)
    assert suite.run_arm(tmp_path, "raw_loss") == 1
    assert calls[:2] == [["nvidia-smi", "-L"], "idle"]
    cmd = calls[2]
    assert "--wait" in cmd
    assert all(f"{key}=infinity" in cmd for key in ("MemoryMax", "MemoryHigh", "MemorySwapMax"))
    assert "RuntimeMaxSec=23400" in cmd
    assert cmd[-2:] == ["--arm", "raw_loss"]
    monkeypatch.setattr(suite, "require_idle_gpu", lambda: (_ for _ in ()).throw(RuntimeError("busy")))
    with pytest.raises(RuntimeError, match="busy"):
        suite.run_arm(tmp_path, "position_lr")
    assert len(calls) == 4


def test_verification_streams_checkpoint_and_rejects_corruption(suite, tmp_path):
    root = tmp_path / "experiment"
    arm = "baseline"
    output = root / arm
    training = output / "training/attempts/train-001"
    artifacts = training / "artifacts"
    checkpoint = training / "checkpoints/iteration_000030000"
    artifacts.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value))
    save(root / "protocol.json", {"training_code_hash": "core", "dataset_hash": "dataset"})
    save(root / "baseline.config.json", {"effective_config_hash": "config"})
    digest = hashlib.sha256()
    with (artifacts / "progress.jsonl").open("w") as f:
        for i in range(1, 30001):
            f.write(json.dumps({"iteration": i, "world_size": 2, "batch_view_ids": ["a", "b"]}) + "\n")
            digest.update(b"a\0b\n")
            if i in (3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000):
                f.write(json.dumps({"event": "validation", "iteration": i,
                                    "status": "complete", "successful_views": 351}) + "\n")
    for stage in ("train", "sor", "selection"):
        save(output / f"{stage}.exit.json", {"returncode": 0, "resource_failure": None, "elapsed_seconds": 1.})
    meta = {"iteration": 30000, "purpose": "final", "provenance": {
        "code_hash": "core", "dataset_hash": "dataset", "effective_config_hash": "config"}, "files": {}}
    for name, filename in suite._COMPONENT_FILES.items():
        (checkpoint / filename).write_bytes(name.encode())
        meta["files"][name] = {"path": filename, "bytes": len(name), "sha256": suite.sha256_file(checkpoint / filename)}
    meta["checkpoint_hash"] = hashlib.sha256(suite._json_bytes(meta)).hexdigest()
    save(checkpoint / "checkpoint.json", meta)
    (output / "sor").mkdir()
    (output / "sor/filtered-model.pt").write_bytes(b"model")
    (output / "selection").mkdir()
    save(output / "selection/evaluation.json", {"status": "complete", "successful_views": 351,
        "psnr": {"mean": 20}, "display_psnr": {"mean": 21}, "ssim": {"mean": .9}, "gaussian_count": 2990000})
    complete = {"status": "complete", "arm": arm, "protocol_sha256": suite.sha256_file(root / "protocol.json"),
        "resources": {"optimizer_updates_observed": 30000, "camera_samples_observed": 60000,
                      "max_observed_global_gaussians": 3000000, "camera_sequence_sha256": digest.hexdigest()},
        "selection_model_sha256": suite.sha256_file(output / "sor/filtered-model.pt"),
        "selection_evaluation_sha256": suite.sha256_file(output / "selection/evaluation.json"),
        "selected_iteration": 30000, "final_raw_psnr": {"mean": 20}, "final_max_rgb": 1.2,
        "final_negative_psnr_views": 0, "final_parameter_health_by_rank": [{"dead_fraction": .1}] * 2}
    save(output / "complete.json", complete)
    assert suite.verify_arm(root, arm, None)["status"] == "verified"
    with pytest.raises(ValueError, match="camera sequence"):
        suite.verify_arm(root, arm, "wrong")
    (checkpoint / "optimizer.bin").write_bytes(b"corrupt!!")
    with pytest.raises(ValueError, match="SHA-256"):
        suite.verify_arm(root, arm, None)
