import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from image3d_scenegraph.gaussian.config import resolve_mcmc_config, resolved_config_record  # noqa: E402
from image3d_scenegraph.gaussian.dataset import sha256_file  # noqa: E402
from image3d_scenegraph.gaussian.evaluation import _authorize_test, _finish_test_consumption  # noqa: E402
from image3d_scenegraph.gaussian.trainer import (  # noqa: E402
    FINAL_FIT_ITERATIONS, _final_fit_profile, _id_list_hash,
)

ROOT = Path(__file__).resolve().parents[1]
script = runpy.run_path(str(ROOT / "scripts/evaluate_gaussian_final_fit.py"))
fixtures = runpy.run_path(str(ROOT / "tests/test_gaussian_export.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def evaluation(contract, config_hash, model_hash, split, count=2, offset=0):
    ids = contract["splits"]["test" if split == "test" else "validation"]
    roles = {"validation": "held_out_model_selection", "test": "held_out_final_evaluation",
             "fit_validation": "in_sample_after_train_validation_fit",
             "control_validation": "held_out_after_train_only_control"}
    return {
        "schema_version": 2, "status": "complete", "split": split,
        "quality_role": roles[split], "selection_eligible": split == "validation",
        "num_views": len(ids), "successful_views": len(ids), "failed_views": [],
        "gaussian_count": count,
        "quality_profiles": {
            "primary": "raw_float_v1",
            "raw_float_v1": {"psnr_field": "psnr", "ssim_field": "ssim",
                             "prediction": "unclamped_float", "reference": "float",
                             "quantization": "none"},
            "display_clamped_uint8_v1": {
                "psnr_field": "display_psnr", "ssim_field": "display_ssim",
                "prediction": "clamp_0_1", "reference": "clamp_0_1",
                "quantization": "floor_uint8_then_divide_255",
            },
        },
        "provenance": {"dataset_hash": contract["dataset_hash"],
                       "effective_config_hash": config_hash, "model_sha256": model_hash},
        "per_view": [{"image_id": image_id, "psnr": 20 + index + offset,
                      "ssim": 0.7 + offset / 100, "display_psnr": 20.5 + index + offset,
                      "display_ssim": 0.71 + offset / 100} for index, image_id in enumerate(ids)],
    }


def setup(tmp_path):
    contract = fixtures["contract"]()
    gaussian = fixtures["model"]()
    config = resolve_mcmc_config()
    dataset = write(tmp_path / "dataset.json", contract)
    config_path = write(tmp_path / "config.json", resolved_config_record(config))
    source = tmp_path / "source.pt"
    torch.save({"state_dict": gaussian.state_dict(), "max_sh_degree": 3}, source)
    selection = write(tmp_path / "selection.json", evaluation(
        contract, config.effective_config_hash, sha256_file(source), "validation"))
    directories = []
    for control in (True, False):
        directory = tmp_path / ("train-only" if control else "train-validation")
        directory.mkdir()
        (directory / "model.pt").write_bytes(source.read_bytes())
        profile = _final_fit_profile(config.effective_config, train_only_control=control)
        train = contract["splits"]["train"]
        validation = contract["splits"]["validation"]
        test = contract["splits"]["test"]
        ids = train if control else train + validation
        ev_path = write(directory / "evaluation.json", evaluation(
            contract, config.effective_config_hash, sha256_file(source),
            "control_validation" if control else "fit_validation"))
        write(directory / "record.json", {
            "schema_version": 1, "profile": profile["profile"], "status": "complete",
            "profile_config": profile, "profile_hash": script["_hash"](profile),
            "dataset_hash": contract["dataset_hash"],
            "effective_config_hash": config.effective_config_hash,
            "source_model_sha256": sha256_file(source), "source_model_unchanged": True,
            "selection_evaluation_sha256": sha256_file(selection),
            "final_model_sha256": sha256_file(source), "evaluation_sha256": sha256_file(ev_path),
            "input_splits": ["train"] if control else ["train", "validation"],
            "train_count": len(train), "validation_count": len(validation),
            "fit_view_count": len(ids), "test_count": len(test),
            "fit_view_ids_sha256": _id_list_hash(ids), "test_ids_sha256": _id_list_hash(test),
            "test_rgb": "not_loaded", "optimizer_updates": FINAL_FIT_ITERATIONS,
            "camera_samples": FINAL_FIT_ITERATIONS * 2, "world_size": 2,
            "gaussian_count_before": gaussian.count, "gaussian_count_after": gaussian.count,
            "topology_changed": False, "code_hash": ("a" if control else "b") * 64,
            "environment_hash": "c" * 64,
        })
        events = [{"event": "final_fit", "profile": profile["profile"], "iteration": i,
                   "optimizer_updates": i, "world_size": 2, "gaussian_count": gaussian.count,
                   "batch_view_ids": [ids[((i - 1) * 2 + j) % len(ids)] for j in range(2)]}
                  for i in range(1, FINAL_FIT_ITERATIONS + 1)]
        (directory / "progress.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        directories.append(directory)
    return SimpleNamespace(dataset_contract=dataset, resolved_config_json=config_path,
                           source_model=source, selection_evaluation=selection,
                           train_only_dir=directories[0], train_validation_dir=directories[1],
                           output_dir=tmp_path / "comparison")


def frozen_args(tmp_path):
    args = setup(tmp_path)
    protocol = script["freeze"](args)
    args.protocol_sha256 = sha256_file(args.output_dir / "protocol.json")
    args.dataset_root = tmp_path / "missing-rgb"
    args.authorize_test = False
    return args, protocol


def test_freeze_without_any_rgb_and_reject_overwrite(tmp_path):
    args, protocol = frozen_args(tmp_path)
    assert not (tmp_path / "images").exists()
    assert set(protocol["arms"]) == set(script["ARMS"])
    assert protocol["test_ids"] == ["10", "11"]
    assert not list(args.output_dir.glob("*.test-consumed.json"))
    assert protocol["arms"]["train-only"]["code_hash"] != protocol["arms"]["train-validation"]["code_hash"]
    with pytest.raises(ValueError, match="already exists"):
        script["freeze"](args)
    with pytest.raises(ValueError, match="authorize-test"):
        script["evaluate"](args)
    assert not (args.output_dir / "test-started.json").exists()


@pytest.mark.parametrize(("field", "value"), [
    ("world_size", 1), ("optimizer_updates", 1999), ("camera_samples", 2000),
    ("source_model_sha256", "d" * 64), ("topology_changed", True),
    ("environment_hash", "d" * 64), ("input_splits", ["train", "validation"]),
])
def test_freeze_rejects_noncomparable_control(tmp_path, field, value):
    args = setup(tmp_path)
    path = args.train_only_dir / "record.json"
    record = json.loads(path.read_text())
    record[field] = value
    write(path, record)
    with pytest.raises(ValueError):
        script["freeze"](args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("forbidden_id", ["8", "10"])
def test_control_progress_cannot_sample_validation_or_test(tmp_path, forbidden_id):
    args = setup(tmp_path)
    path = args.train_only_dir / "progress.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["batch_view_ids"][0] = forbidden_id
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(ValueError, match="sampling/budget"):
        script["freeze"](args)
    assert not args.output_dir.exists()


def test_evaluation_preflights_every_arm_before_test_consumption(tmp_path, monkeypatch):
    args, _ = frozen_args(tmp_path)
    args.authorize_test = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    (args.train_validation_dir / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen source changed"):
        script["evaluate"](args)
    assert not list(args.output_dir.glob("*.test-consumed.json"))
    assert not (args.output_dir / "test-started.json").exists()


def test_mock_three_arm_test_and_paired_report(tmp_path, monkeypatch):
    args, protocol = frozen_args(tmp_path)
    args.authorize_test = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    called = []

    def fake_run(**kwargs):
        name = kwargs["output_dir"].name
        called.append(name)
        assert kwargs["split"] == "test"
        consumed = _authorize_test(kwargs["frozen_candidate_path"],
                                   dataset_hash=protocol["dataset_hash"],
                                   config_hash=protocol["effective_config_hash"],
                                   model_hash=protocol["arms"][name]["model_sha256"])
        value = evaluation(kwargs["contract"], protocol["effective_config_hash"],
                           protocol["arms"][name]["model_sha256"], "test",
                           offset={"selection": 0, "train-only": 0.5, "train-validation": 1.5}[name])
        path = write(kwargs["output_dir"] / "evaluation.json", value)
        _finish_test_consumption(consumed, "complete", sha256_file(path))
        return value

    monkeypatch.setitem(script["evaluate"].__globals__, "run_evaluation", fake_run)
    result = script["evaluate"](args)
    assert called == list(script["ARMS"])
    assert result["num_views"] == 2
    assert result["selection_eligible"] is result["promotion_eligible"] is False
    pair = result["comparisons"]["train-validation_minus_train-only"]["metrics"]["psnr"]
    assert pair["delta"]["mean"] == pytest.approx(1.0)
    assert pair["improved_views"] == 2
    with pytest.raises(ValueError, match="already started"):
        script["evaluate"](args)
    assert len(called) == 3
    with pytest.raises(FileExistsError):
        script["report"](args)


def test_report_rejects_missing_or_duplicate_test_views(tmp_path):
    args, protocol = frozen_args(tmp_path)
    value = evaluation(json.loads(args.dataset_contract.read_text()),
                       protocol["effective_config_hash"],
                       protocol["arms"]["selection"]["model_sha256"], "test")
    value["per_view"][1]["image_id"] = "10"
    write(args.output_dir / "selection/evaluation.json", value)
    with pytest.raises(ValueError, match="view IDs"):
        script["report"](args)
    assert not (args.output_dir / "comparison.json").exists()
