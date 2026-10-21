from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import pytest

from scripts import mcmc_first_divergence as probe
from scripts import run_mcmc_first_divergence as runner


def test_local_entry_rejects_before_tensor_import_or_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "local")
    for mode in ("prepare", "run", "worker", "compare"):
        monkeypatch.setattr(sys, "argv", ["probe", "--output-root", str(tmp_path / "new"), "--" + mode])
        with pytest.raises(SystemExit, match="本机禁止"):
            runner.main()
        assert not (tmp_path / "new").exists()
    assert "torch" not in sys.modules
    assert runner.lifecycle.core_hash() == runner.lifecycle.CORE


def test_explicit_authorization_precedes_protocol_read(monkeypatch):
    monkeypatch.setattr(runner, "PROJECT", Path("/usr/local/3dgs_new/Image3D-SceneGraph"))
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "i-94B8D131")
    for mode in ("prepare", "run", "worker", "compare"):
        monkeypatch.setattr(sys, "argv", ["probe", "--output-root", str(runner.PROJECT / "outputs/analysis/new"), "--" + mode])
        with pytest.raises(SystemExit, match="未获显式授权"):
            runner.main()


def test_numeric_reference_boundaries_and_nonfinite():
    assert not probe.numeric_pair(2e-6, 0.)["mismatch"]
    assert probe.numeric_pair(math.nextafter(2e-6, math.inf), 0.)["mismatch"]
    assert probe.numeric_pair(1e-8, 0.)["relative_unbounded"]
    assert probe.numeric_pair(0., 0.)["rel_error"] == 0
    assert not probe.numeric_pair(-0., 0.)["strict_unequal"]
    assert probe.numeric_pair(1.00000001, 1., exact=True)["mismatch"]
    assert not probe.numeric_pair(1.00000001, 1.)["mismatch"]
    for a, b in ((math.nan, math.nan), (math.inf, math.inf), (0., -math.inf)):
        assert probe.numeric_pair(a, b)["mismatch"]
    assert probe.numeric_pair(2**63 - 1, 2**63 - 2, exact=True)["abs_error"] == 1


def test_comparison_classification_and_stage_order():
    def rows():
        return {str(r): {s: {"parameters/means": {"status": "bitwise_equal"}} for s in probe.STAGES} for r in (0, 1)}
    ranks = rows()
    assert probe.summarize_comparison(ranks)["status"] == "bitwise_equal"
    ranks["1"]["P2"]["parameters/means"]["status"] = "numerically_equivalent"
    report = probe.summarize_comparison(ranks)
    assert report["first_nonbitwise_stage"] == "P2" and report["status"] == "numerically_equivalent"
    ranks["0"]["P1"]["render/metadata/depths"] = {"status": "numerically_inconsistent"}
    report = probe.summarize_comparison(ranks)
    assert report["first_failed_tolerance_or_structure_stage"] == "P1"
    assert report["ranks"]["0"]["first_primary_nonbitwise_stage"] is None
    ranks["0"]["P0"]["parameters/means"]["status"] = "structure_mismatch"
    assert not probe.summarize_comparison(ranks)["p0_prerequisite_passed"]
    del ranks["0"]["P3"]
    with pytest.raises(ValueError, match="missing or reordered"):
        probe.summarize_comparison(ranks)


def test_probe_completion_requires_all_phases_and_calls(tmp_path):
    instance = probe.FirstDivergenceProbe(tmp_path, 0)
    with pytest.raises(ValueError, match="incomplete"):
        instance.finish()
    instance.stages = list(probe.STAGES)
    instance.steps = list(probe.PARAMETERS)
    instance.calls = dict.fromkeys(instance.calls, 1)
    instance.manifests = dict.fromkeys(probe.STAGES, "a" * 64)
    instance.finish()
    record = json.loads((tmp_path / "worker-rank-0.json").read_text())
    assert record["status"] == "probe_complete"
    assert not record["native_training_complete"] and not record["validation_evaluated"]
    with pytest.raises(FileExistsError):
        instance.finish()


def dummy_snapshot(folder, stage):
    files = {}

    def tensor():
        name = f"tensor-{len(files)}.npy"
        (folder / name).write_bytes(b"opaque-test-bytes")
        info = {"file": name, "dtype": "float32", "shape": [1], "kind": "torch", "device": "cuda:0", "stride": [1],
                "sha256": runner.sha256_file(folder / name), "bytes": len(b"opaque-test-bytes"), "canonical_sha256": "a" * 64}
        files[name] = info
        return {"type": "tensor", **info}

    def mapping(values):
        return {"type": "dict", "items": [[{"type": "str", "value": k}, v] for k, v in sorted(values.items())]}

    scalar = {"type": "int", "value": 0}
    values = {"rng": mapping({k: tensor() for k in ("python", "numpy", "torch", "cuda")})}
    if stage in ("P0", "P3", "P4"):
        values.update(parameters=mapping({n: tensor() for n in probe.PARAMETERS}),
            adam=mapping({n: mapping({"state": mapping({k: tensor() for k in ("exp_avg", "exp_avg_sq", "step")}),
                                     "param_groups": mapping({"lr": scalar})}) for n in probe.PARAMETERS}),
            strategy=mapping({"binoms": tensor()}))
        if stage == "P0":
            values["gradient_presence"] = mapping({n: {"type": "bool", "value": False} for n in probe.PARAMETERS})
    elif stage == "P1":
        render = {k: tensor() for k in ("rgb", "alpha", "backward_loss", "ground_truth", "camera_matrix", "intrinsic")}
        render.update(loss_terms=mapping({k: tensor() for k in ("reconstruction", "opacity", "scale")}),
                      metadata=mapping({"depths": tensor()}))
        values["render"] = mapping(render)
    else:
        values["gradients"] = mapping({n: tensor() for n in probe.PARAMETERS})
        values["parameters_adam_sha256"] = {"type": "str", "value": "b" * 64}
    if stage in ("P0", "P1", "P3", "P4"):
        values["camera"] = mapping({"cursor": scalar, "order": {"type": "list", "items": [scalar]}})
    if stage == "P0":
        runtime = {k: scalar for k in ("torch", "gsplat", "cuda", "gpu", "rank", "deterministic_algorithms", "matmul_tf32")}
        runtime["optimizers"] = mapping({n: mapping({"class": scalar}) for n in probe.PARAMETERS})
        values["runtime"] = mapping(runtime)
    tree = mapping(values)
    record = {"tree": tree, "files": files, "tree_sha256": probe.digest_state(tree)}
    probe.write_new(folder / "manifest.json", record)
    return record


def worker_fixture(stage, rank):
    p = probe.FirstDivergenceProbe(stage, rank)
    p.stages, p.steps, p.calls = list(probe.STAGES), list(probe.PARAMETERS), dict.fromkeys(p.calls, 1)
    for s in probe.STAGES:
        folder = stage / f"rank-{rank}" / s
        folder.mkdir(parents=True)
        dummy_snapshot(folder, s)
        p.manifests[s] = runner.sha256_file(folder / "manifest.json")
    p.finish()
    (stage / f"phases-rank-{rank}.jsonl").write_text("".join(json.dumps({"stage": s, "iteration": probe.ITERATION,
        "manifest_sha256": p.manifests[s]}) + "\n" for s in probe.STAGES))


def test_worker_order_hash_and_complete_bindings(tmp_path):
    stage = tmp_path / "a"
    stage.mkdir()
    for r in (0, 1):
        worker_fixture(stage, r)
        runner.verify_worker(stage, r)
    files = {f.name: runner.sha256_file(f) for f in stage.iterdir() if f.is_file()}
    probe.write_new(stage / "complete.json", {"status": "probe_complete", "branch": "a", "protocol_sha256": "f" * 64,
        "resources": {"optimizer_updates_observed": 1, "camera_samples_observed": 2, "absolute_iteration": 12001}, "files": files})
    runner.completed(tmp_path, "a", "f" * 64)
    (stage / "rank-0/P2/manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        runner.completed(tmp_path, "a", "f" * 64)


def test_paths_and_protocol_reject_tampering(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    for name in ("../outside", str(outside), "link/file"):
        with pytest.raises(ValueError, match="unsafe"):
            probe.safe_file(root, name)
    probe.write_new(root / "protocol.json", {"profile": "wrong"})
    with pytest.raises(ValueError, match="SHA"):
        runner.protocol_at(root, "0" * 64)
    with pytest.raises(ValueError, match="unsupported"):
        runner.protocol_at(root, runner.sha256_file(root / "protocol.json"))


def test_monitor_budget_and_missing_phase(tmp_path):
    monitor = runner.ProbeMonitor(tmp_path)
    assert monitor.record()["optimizer_updates_observed"] == 0
    assert monitor(21601) == "probe_timeout"
    assert monitor(1, final=True) == "missing_memory_telemetry"
    monitor.telemetry.mkdir()
    for rank in (0, 1):
        probe.write_new(monitor.telemetry / f"rank-{rank}.json", {"peak_reserved_bytes": 10})
        worker_fixture(tmp_path, rank)
    assert monitor(1, final=True) is None
    assert monitor.record()["optimizer_updates_observed"] == 1
    assert monitor.record()["camera_samples_observed"] == 2
    (monitor.telemetry / "rank-0.json").write_text(json.dumps({"peak_reserved_bytes": 19 * 1024**3}))
    assert monitor(2) == "gpu_reserved_limit"


def test_wrapper_structure_retains_original_operations_and_exact_exit():
    tree = ast.parse(Path(probe.__file__).read_text())
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    backward = ast.unparse(funcs["backward"])
    assert backward.index("'P1'") < backward.index("original_backward(tensor") < backward.index("'P2'")
    assert backward.count("original_backward(tensor") == 1
    after = ast.unparse(funcs["after"])
    assert after.index("post(*a, **kw)") < after.index("'P4'") < after.index("raise ProbeComplete")
    capture = ast.unparse(funcs["capture"])
    assert capture.index("torch.cuda.synchronize") < capture.index("save_snapshot")
    batch = ast.unparse(funcs["batch"])
    assert batch.index("'P0'") < batch.index("originals['_next_camera_batch'](")
    assert issubclass(probe.ProbeComplete, BaseException) and not issubclass(probe.ProbeComplete, Exception)
    runner_tree = ast.parse(Path(runner.__file__).read_text())
    worker = next(n for n in ast.walk(runner_tree) if isinstance(n, ast.FunctionDef) and n.name == "worker")
    assert [ast.unparse(n.type) for n in ast.walk(worker) if isinstance(n, ast.ExceptHandler)] == ["ProbeComplete"]
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
        "set_rng_state", "set_rng_state_all", "use_deterministic_algorithms") for n in ast.walk(tree))


def test_required_snapshot_schema_and_file_integrity(tmp_path):
    folder = tmp_path / "P1"
    folder.mkdir()
    record = dummy_snapshot(folder, "P1")
    probe.validate_stage_schema(probe.verify_snapshot_files(folder), "P1")
    for key, value in record["tree"]["items"]:
        if key["value"] == "render":
            value["items"] = [item for item in value["items"] if item[0]["value"] != "rgb"]
    with pytest.raises(ValueError, match="unreferenced"):
        probe.snapshot_leaves(record)
    node = next(iter(record["files"].values()))
    (folder / node["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file changed"):
        probe.verify_snapshot_files(folder)
