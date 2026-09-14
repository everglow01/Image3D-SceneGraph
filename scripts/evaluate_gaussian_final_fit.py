#!/usr/bin/env python3
"""Freeze and evaluate the MCMC final-fit equal-budget comparison on untouched Test views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import image3d_scenegraph.gaussian.evaluation as gaussian_evaluation
from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig, resolved_config_record
from image3d_scenegraph.gaussian.dataset import sha256_file, validate_contract
from image3d_scenegraph.gaussian.evaluation import (
    _distribution,
    load_model_snapshot,
    run_evaluation,
    write_frozen_candidate,
)
from image3d_scenegraph.gaussian.trainer import (
    FINAL_FIT_ITERATIONS,
    _final_fit_profile,
    _final_fit_split_ids,
    _id_list_hash,
)

ARMS = ("selection", "train-only", "train-validation")
METRICS = ("psnr", "ssim", "display_psnr", "display_ssim")
PROFILE = "mcmc_final_fit_generalization_v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolved(path: Path) -> ResolvedGaussianConfig:
    value = _read(path)
    resolved = ResolvedGaussianConfig(
        requested_profile=value["requested_profile"],
        effective_config=value["effective_config"],
        effective_config_hash=value["effective_config_hash"],
    )
    resolved_config_record(resolved)
    _require(resolved.effective_config["strategy"]["name"] == "mcmc_v1", "MCMC only")
    return resolved


def _check_evaluation(value: dict, ids: list[str], split: str, role: str,
                      dataset_hash: str, config_hash: str, model_hash: str) -> None:
    _require(
        value.get("schema_version") == 2
        and value.get("status") == "complete"
        and value.get("split") == split
        and value.get("quality_role") == role
        and value.get("selection_eligible") is (split == "validation")
        and value.get("num_views") == len(ids)
        and value.get("successful_views") == len(ids)
        and value.get("failed_views") == [],
        "evaluation role or completeness mismatch",
    )
    profiles = value.get("quality_profiles", {})
    _require(profiles.get("primary") == "raw_float_v1"
             and profiles.get("raw_float_v1") == {
                 "psnr_field": "psnr", "ssim_field": "ssim",
                 "prediction": "unclamped_float", "reference": "float", "quantization": "none",
             }
             and profiles.get("display_clamped_uint8_v1") == {
                 "psnr_field": "display_psnr", "ssim_field": "display_ssim",
                 "prediction": "clamp_0_1", "reference": "clamp_0_1",
                 "quantization": "floor_uint8_then_divide_255",
             }, "evaluation metric profiles mismatch")
    rows = value["per_view"]
    _require(len(rows) == len(ids) and {row["image_id"] for row in rows} == set(ids),
             "evaluation view IDs mismatch")
    provenance = value.get("provenance", {})
    _require(all(provenance.get(key) == expected for key, expected in {
        "dataset_hash": dataset_hash,
        "effective_config_hash": config_hash,
        "model_sha256": model_hash,
    }.items()), "evaluation provenance mismatch")
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        _require(all(math.isfinite(v) for v in values), "non-finite metric")


def freeze(args: argparse.Namespace) -> dict:
    _require(not args.output_dir.exists(), "comparison output already exists")
    contract = _read(args.dataset_contract)
    validate_contract(contract)
    train, validation, fit = _final_fit_split_ids(contract)
    test = [str(value) for value in contract["splits"]["test"]]
    _require(bool(test), "Test split is empty")
    resolved = _resolved(args.resolved_config_json)
    files = [args.dataset_contract, args.resolved_config_json, args.source_model,
             args.selection_evaluation, Path(__file__).resolve()]
    module_root = Path(gaussian_evaluation.__file__).parent
    files.extend(module_root / name for name in (
        "evaluation.py", "model.py", "render.py", "runtime.py", "training_math.py",
        "config.py", "dataset.py",
    ))
    for directory in (args.train_only_dir, args.train_validation_dir):
        files.extend(directory / name for name in
                     ("model.pt", "record.json", "evaluation.json", "progress.jsonl"))
    sources = {str(path.resolve()): sha256_file(path) for path in files}
    source_hash = sources[str(args.source_model.resolve())]
    selection_hash = sources[str(args.selection_evaluation.resolve())]
    source_model = load_model_snapshot(args.source_model, torch.device("cpu"))
    count = source_model.count
    sh_degree = source_model.max_sh_degree
    _require(sh_degree == int(resolved.effective_config["sh_schedule"]["max_degree"]),
             "source model SH does not match the resolved configuration")
    del source_model
    _check_evaluation(_read(args.selection_evaluation), validation, "validation",
                      "held_out_model_selection", contract["dataset_hash"],
                      resolved.effective_config_hash, source_hash)
    arms = {"selection": {"model": str(args.source_model.resolve()),
                          "model_sha256": source_hash, "optimization_splits": ["train"]}}
    for name, directory in (("train-only", args.train_only_dir),
                            ("train-validation", args.train_validation_dir)):
        control = name == "train-only"
        ids = train if control else fit
        splits = ["train"] if control else ["train", "validation"]
        profile = _final_fit_profile(resolved.effective_config, train_only_control=control)
        record = _read(directory / "record.json")
        model_path = directory / "model.pt"
        expected = {
            "schema_version": 1, "status": "complete", "profile": profile["profile"],
            "profile_config": profile, "profile_hash": _hash(profile),
            "dataset_hash": contract["dataset_hash"],
            "effective_config_hash": resolved.effective_config_hash,
            "source_model_sha256": source_hash, "source_model_unchanged": True,
            "selection_evaluation_sha256": selection_hash,
            "final_model_sha256": sources[str(model_path.resolve())],
            "evaluation_sha256": sources[str((directory / "evaluation.json").resolve())],
            "input_splits": splits, "train_count": len(train),
            "validation_count": len(validation), "fit_view_count": len(ids),
            "test_count": len(test), "fit_view_ids_sha256": _id_list_hash(ids),
            "test_ids_sha256": _id_list_hash(test), "test_rgb": "not_loaded",
            "optimizer_updates": FINAL_FIT_ITERATIONS, "world_size": 2,
            "camera_samples": FINAL_FIT_ITERATIONS * 2,
            "gaussian_count_before": count, "gaussian_count_after": count,
            "topology_changed": False,
        }
        for key, value in expected.items():
            _require(record.get(key) == value, f"{name} record mismatch: {key}")
        events = [json.loads(line) for line in
                  (directory / "progress.jsonl").read_text().splitlines()]
        _require(len(events) == FINAL_FIT_ITERATIONS, "optimizer event count mismatch")
        allowed = set(ids)
        sampled = set()
        for iteration, event in enumerate(events, 1):
            batch = event.get("batch_view_ids", [])
            _require(event.get("event") == "final_fit"
                     and event.get("profile") == profile["profile"]
                     and event.get("iteration") == iteration
                     and event.get("optimizer_updates") == iteration
                     and event.get("world_size") == 2
                     and event.get("gaussian_count") == count
                     and len(batch) == 2 and set(batch) <= allowed,
                     f"{name} sampling/budget mismatch at {iteration}")
            sampled.update(batch)
        _require(sampled == allowed, f"{name} did not sample every fit view")
        model = load_model_snapshot(model_path, torch.device("cpu"))
        _require(model.count == count and model.max_sh_degree == sh_degree,
                 "model count/SH changed")
        del model
        _check_evaluation(
            _read(directory / "evaluation.json"), validation,
            "control_validation" if control else "fit_validation",
            "held_out_after_train_only_control" if control
            else "in_sample_after_train_validation_fit",
            contract["dataset_hash"], resolved.effective_config_hash,
            expected["final_model_sha256"],
        )
        arms[name] = {
            "model": str(model_path.resolve()),
            "model_sha256": expected["final_model_sha256"],
            "optimization_splits": splits, "code_hash": record["code_hash"],
            "environment_hash": record["environment_hash"],
        }
    _require(arms["train-only"]["environment_hash"] == arms["train-validation"]["environment_hash"],
             "fit environments differ; not an equal-environment comparison")
    _verify_sources(sources)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, arm in arms.items():
        candidate = args.output_dir / f"{name}.frozen.json"
        write_frozen_candidate(candidate, candidate_id=name,
                               dataset_hash=contract["dataset_hash"],
                               effective_config_hash=resolved.effective_config_hash,
                               model_sha256=arm["model_sha256"])
        arm["frozen_candidate_sha256"] = sha256_file(candidate)
    protocol = {
        "schema_version": 1, "profile": PROFILE, "status": "frozen",
        "dataset_contract": str(args.dataset_contract.resolve()),
        "resolved_config_json": str(args.resolved_config_json.resolve()),
        "dataset_hash": contract["dataset_hash"],
        "effective_config_hash": resolved.effective_config_hash,
        "test_ids": test, "gaussian_count": count, "sources": sources, "arms": arms,
        "policy": {
            "evaluation_split": "test", "metrics": list(METRICS),
            "view_policy": "all_test_views_no_posthoc_exclusion",
            "selection_eligible": False, "parameter_tuning": False,
            "promotion_eligible": False,
            "comparisons": [["train-only", "selection"],
                            ["train-validation", "selection"],
                            ["train-validation", "train-only"]],
            "interpretation": "Paired single-scene, single-seed novel-RGB-view evidence; "
                              "cameras/SfM are shared, not unseen geometry or temporal stability. "
                              "Positive means alone are not a promotion gate. "
                              "Reused final-fit code hashes may differ and remain disclosed.",
        },
    }
    _write(args.output_dir / "protocol.json", protocol)
    return protocol


def _verify_sources(sources: dict) -> None:
    for path, expected in sources.items():
        _require(sha256_file(Path(path)) == expected, f"frozen source changed: {path}")


def _load_protocol(output_dir: Path, expected_hash: str) -> dict:
    path = output_dir / "protocol.json"
    _require(sha256_file(path) == expected_hash, "protocol hash mismatch")
    protocol = _read(path)
    _require(protocol["profile"] == PROFILE and protocol["status"] == "frozen",
             "invalid comparison protocol")
    _verify_sources(protocol["sources"])
    for name in ARMS:
        _require(sha256_file(output_dir / f"{name}.frozen.json")
                 == protocol["arms"][name]["frozen_candidate_sha256"],
                 "frozen candidate changed")
    return protocol


def evaluate(args: argparse.Namespace) -> dict:
    _require(args.authorize_test, "explicit --authorize-test is required; Test was not consumed")
    protocol = _load_protocol(args.output_dir, args.protocol_sha256)
    _require(torch.cuda.is_available(), "Test rendering requires the authorized remote CUDA host")
    contract = _read(Path(protocol["dataset_contract"]))
    resolved = _resolved(Path(protocol["resolved_config_json"]))
    for name in ARMS:
        _require(not (args.output_dir / name).exists()
                 and not (args.output_dir / f"{name}.frozen.test-consumed.json").exists(),
                 "comparison evaluation already started; do not retry or refreeze")
    _write(args.output_dir / "test-started.json", {"protocol_sha256": args.protocol_sha256})
    for name in ARMS:
        run_evaluation(
            contract=contract, dataset_root=args.dataset_root,
            model_path=Path(protocol["arms"][name]["model"]), resolved_config=resolved,
            split="test", output_dir=args.output_dir / name,
            frozen_candidate_path=args.output_dir / f"{name}.frozen.json",
        )
    return report(args)


def report(args: argparse.Namespace) -> dict:
    protocol = _load_protocol(args.output_dir, args.protocol_sha256)
    evaluations = {}
    hashes = {}
    for name in ARMS:
        path = args.output_dir / name / "evaluation.json"
        value = _read(path)
        _check_evaluation(value, protocol["test_ids"], "test", "held_out_final_evaluation",
                          protocol["dataset_hash"], protocol["effective_config_hash"],
                          protocol["arms"][name]["model_sha256"])
        _require(value.get("gaussian_count") == protocol["gaussian_count"],
                 "Test model count mismatch")
        consumption = _read(args.output_dir / f"{name}.frozen.test-consumed.json")
        hashes[name] = sha256_file(path)
        _require(consumption.get("status") == "complete"
                 and consumption.get("evaluation_sha256") == hashes[name],
                 "Test consumption incomplete or evaluation changed")
        evaluations[name] = {row["image_id"]: row for row in value["per_view"]}
    comparisons = {}
    for after, before in protocol["policy"]["comparisons"]:
        rows = [{"image_id": image_id, **{
            metric: evaluations[after][image_id][metric] - evaluations[before][image_id][metric]
            for metric in METRICS}} for image_id in protocol["test_ids"]]
        comparisons[f"{after}_minus_{before}"] = {
            "per_view": rows,
            "metrics": {metric: {
                "delta": _distribution([row[metric] for row in rows]),
                "improved_views": sum(row[metric] > 0 for row in rows),
                "regressed_views": sum(row[metric] < 0 for row in rows),
                "worst_view_ids": [row["image_id"] for row in
                                   sorted(rows, key=lambda row: row[metric])[:10]],
            } for metric in METRICS},
        }
    result = {
        "schema_version": 1, "profile": PROFILE, "status": "complete",
        "protocol_sha256": args.protocol_sha256, "evaluation_sha256": hashes,
        "quality_role": "held_out_final_evaluation", "selection_eligible": False,
        "promotion_eligible": False, "num_views": len(protocol["test_ids"]),
        "policy": protocol["policy"], "arms": protocol["arms"],
        "metrics": {name: {metric: _distribution([row[metric] for row in values.values()])
                           for metric in METRICS} for name, values in evaluations.items()},
        "comparisons": comparisons,
    }
    _write(args.output_dir / "comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    freezing = commands.add_parser("freeze", help="CPU-only; no dataset RGB is opened")
    for name in ("dataset-contract", "resolved-config-json", "source-model",
                 "selection-evaluation", "train-only-dir", "train-validation-dir", "output-dir"):
        freezing.add_argument(f"--{name}", type=Path, required=True)
    for stage in ("evaluate", "report"):
        command = commands.add_parser(stage)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--protocol-sha256", required=True)
        if stage == "evaluate":
            command.add_argument("--dataset-root", type=Path, required=True)
            command.add_argument("--authorize-test", action="store_true")
    args = parser.parse_args()
    result = {"freeze": freeze, "evaluate": evaluate, "report": report}[args.stage](args)
    print(json.dumps({
        "stage": args.stage, "status": result["status"],
        "output_dir": str(args.output_dir.resolve()),
        "protocol_sha256": sha256_file(args.output_dir / "protocol.json"),
    }, allow_nan=False))


if __name__ == "__main__":
    main()
