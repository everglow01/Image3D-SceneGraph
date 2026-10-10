#!/usr/bin/env python3
"""Report both frozen endpoints; quality exploration never passes the old resource gate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import socket

import numpy as np
from PIL import Image, ImageDraw

from image3d_scenegraph.gaussian.absgrad_resources import (
    QUALITY_LIMITS, STREAMING_HOST_POLICY, STREAMING_UNIT, quality_cgroup, write_json,
)
from image3d_scenegraph.gaussian import absgrad_experiment as runner

VALIDATION_ROI = "outputs/analysis/gaussian-quality-attribution-v1-20260928/roi.json"
VALIDATION_ROI_SHA256 = "5f64c889b169431980bc539f205a12b6f799c35478a96ee9e6abc662abad6afb"
TRAIN_ROI = "outputs/analysis/absgrad-signed-train-roi-20261008-v1/train-roi.json"


def paired_validation(left: dict, right: dict, expected_ids: list[str]) -> dict:
    paired = []
    indices = []
    for record in (left, right):
        rows = record["per_view"]
        index = {str(row["image_id"]): row for row in rows}
        if len(rows) != 377 or len(index) != 377 or set(index) != set(expected_ids):
            raise ValueError("Validation must contain exactly the same 377 unique camera IDs")
        if any(not math.isfinite(float(row[key])) for row in rows for key in ("psnr", "ssim")):
            raise ValueError("non-finite Validation metric")
        indices.append(index)
    global_delta = {}
    for key in ("psnr", "ssim"):
        values = [[float(index[i][key]) for i in expected_ids] for index in indices]
        global_delta[key] = {
            "mean": float(np.mean(values[1]) - np.mean(values[0])),
            "p10": float(np.percentile(values[1], 10) - np.percentile(values[0], 10)),
            "paired_delta_p10": float(np.percentile(np.subtract(values[1], values[0]), 10)),
        }
    for i in expected_ids:
        paired.append({"image_id": i, "signed": indices[0][i], "absolute": indices[1][i],
            "delta": {key: float(indices[1][i][key]) - float(indices[0][i][key]) for key in ("psnr", "ssim")}})
    return {"metric_profile": "raw_float_v1", "global_delta": global_delta, "per_view": paired}


def numerical_gate(rows: list[dict], validation: dict, gates: dict) -> dict:
    if len(rows) != 30 or len({(r["split"], r["image_id"], r["name"]) for r in rows}) != 30:
        raise ValueError("expected 24 Train and 6 Validation unique ROIs")
    checks = {}
    for region in ("chair_legs_and_wheels", "carpet_texture", "wall_corner"):
        control = region == "wall_corner"
        train = [r for r in rows if r["split"] == "train" and r["name"] == region]
        held_out = [r for r in rows if r["split"] == "validation" and r["name"] == region]
        if len(train) != 4 or len(held_out) != 1:
            raise ValueError("missing primary or normal-control ROI")
        threshold = gates["normal_control_train_group_and_validation_roi"] if control else gates["train_primary"]
        mean_delta = {key: float(np.mean([r["delta"][key] for r in train])) for key in ("psnr", "ssim")}
        minimum = {key: threshold[f"min_delta_{key}" + ("_db" if key == "psnr" else "")] if control
            else threshold[f"per_region_group_mean_min_delta_{key}" + ("_db" if key == "psnr" else "")]
            for key in ("psnr", "ssim")}
        improved = sum(all(r["delta"][key] > 0 for key in ("psnr", "ssim")) for r in train)
        passed = all(mean_delta[key] >= minimum[key] for key in minimum)
        if not control:
            passed = passed and improved >= threshold["required_directional_improvement_views_out_of_four"]
        checks[f"train_{region}"] = {"passed": passed, "mean_delta": mean_delta, "improved_views": improved}
        threshold = gates["normal_control_train_group_and_validation_roi"] if control else gates["validation_primary"]
        minimum = {key: threshold[("min_delta_" if control else "each_primary_roi_min_delta_") + key
            + ("_db" if key == "psnr" else "")] for key in ("psnr", "ssim")}
        checks[f"validation_{region}"] = {"passed": all(held_out[0]["delta"][key] >= minimum[key] for key in minimum),
                                         "delta": held_out[0]["delta"]}
    for stat in ("mean", "p10"):
        checks[f"global_{stat}"] = {"passed": all(validation["global_delta"][key][stat]
            >= gates["global_validation"][f"{stat}_min_delta_{key}" + ("_db" if key == "psnr" else "")]
            for key in ("psnr", "ssim"))}
    return {"numerical_passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "visual_review": "pending_all_frozen_regions", "quality_decision": "pending_visual_review" if all(c["passed"] for c in checks.values()) else "fail_numerical_quality",
            "promotion_eligible": False}


def crop_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    import torch
    from image3d_scenegraph.gaussian.training_math import psnr, structural_similarity

    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("ROI image shape mismatch")
    ref = torch.from_numpy(reference.astype(np.float32) / 255)
    pred = torch.from_numpy(prediction.astype(np.float32) / 255)
    values = {"psnr": float(psnr(pred, ref)), "ssim": float(structural_similarity(pred, ref))}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("non-finite ROI metrics")
    return values


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def render_controls(model_path: Path, views, output: Path) -> None:
    import torch
    from image3d_scenegraph.gaussian.evaluation import load_model_snapshot
    from image3d_scenegraph.gaussian.render import render_gaussians

    output.mkdir()
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    model = load_model_snapshot(model_path, device)
    with torch.no_grad():
        for stored in views:
            view = stored.to(device)
            rendered = render_gaussians(model, view.camera, sh_degree=3, background=None)
            if torch.cuda.max_memory_reserved(device) > QUALITY_LIMITS["max_per_rank_peak_reserved_bytes"][0]:
                raise RuntimeError("paired rendering exceeds 18 GiB reserved")
            display = rendered.image.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
            Image.fromarray(display).save(output / f"{view.camera.image_id}.png")
    del model
    torch.cuda.empty_cache()


def resource_report(candidate_native: dict, signed_native: dict | None, candidate_stage: dict,
                    signed_stage: dict | None, historical_limits: dict) -> dict:
    observed = candidate_stage["resources"]
    historical = {
        "identity": "historical_signed_not_current_matched_control",
        "old_resource_gate": "failed_in_original_candidate; retained",
        "gaussian_ratio": observed["max_observed_global_gaussians"] / (historical_limits["max_observed_global_gaussians"] / 2),
        "native_reserved_ratio": [a / (b / 2) for a, b in zip(candidate_native["per_rank_peak_reserved_bytes"],
            historical_limits["max_per_rank_peak_reserved_bytes"], strict=True)],
        "stage_wall_ratio": candidate_stage["elapsed_seconds"] / (historical_limits["max_main_stage_wall_seconds"] / 2),
        "memory_scope": "historical native peaks may have been reset by nested Validation; not a corrected full-lifecycle baseline",
    }
    matched = None
    if signed_stage is not None:
        reference = signed_stage["resources"]
        matched = {
            "identity": "recovered_signed_same_core_and_absolute" if signed_native is None else "current_same_code_signed_and_absolute",
            "signed_max_gaussians": reference["max_observed_global_gaussians"],
            "absolute_max_gaussians": observed["max_observed_global_gaussians"],
            "gaussian_ratio": observed["max_observed_global_gaussians"] / reference["max_observed_global_gaussians"],
            "signed_native_reserved": None if signed_native is None else signed_native["per_rank_peak_reserved_bytes"],
            "absolute_native_reserved": candidate_native["per_rank_peak_reserved_bytes"],
            "signed_observed_reserved": reference["per_rank_peak_reserved_bytes"],
            "signed_observed_stage_seconds": signed_stage["elapsed_seconds"],
            "absolute_stage_seconds": candidate_stage["elapsed_seconds"],
            "native_reserved_ratio": None, "telemetry_reserved_ratio": None,
            "native_time_ratio": None, "stage_wall_ratio": None,
            "memory_scope": "signed interrupted before publication; separate offline recovery; lifecycle ratios unavailable",
        }
        if signed_native is not None:
            matched.update(
                native_reserved_ratio=[a / b for a, b in zip(candidate_native["per_rank_peak_reserved_bytes"], signed_native["per_rank_peak_reserved_bytes"], strict=True)],
                telemetry_reserved_ratio=[a / b for a, b in zip(observed["per_rank_peak_reserved_bytes"], reference["per_rank_peak_reserved_bytes"], strict=True)],
                native_time_ratio=candidate_native["elapsed_seconds"] / signed_native["elapsed_seconds"],
                stage_wall_ratio=candidate_stage["elapsed_seconds"] / signed_stage["elapsed_seconds"],
                memory_scope="native training lifecycle vs native; polled lifecycle including merge vs same polled lifecycle")
    return {"historical_reference": historical, "matched_pair": matched,
        "new_safety_limits": QUALITY_LIMITS, "promotion_eligible": False}


def evaluate(signed: Path, candidate: Path, output: Path) -> None:
    import torch
    from image3d_scenegraph.gaussian.runtime import load_training_views
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig, resolved_config_record

    if output.exists() or output.is_symlink() or output.resolve().parent != candidate.resolve():
        raise ValueError("paired output must be a new direct child of the candidate")
    candidate_record = runner.read_json(candidate / "complete.json")
    candidate_protocol = runner.read_json(candidate / "protocol.json")
    recovered = candidate_protocol["profile"] == runner.RECOVERED_PROFILE
    if candidate_protocol["profile"] in (runner.PROFILE, runner.RECOVERED_PROFILE):
        if recovered:
            runner.continuation_cgroup()
        else:
            quality_cgroup(policy=STREAMING_HOST_POLICY, unit=STREAMING_UNIT)
        gate = runner.checked_json(candidate.parent / "gate.json", candidate_protocol["gate_sha256"])
        runner.validate_matched_gate(gate, recovered=recovered)
        if candidate_protocol["matched_signed"] != runner.receipt(signed):
            raise ValueError("matched signed receipt changed")
        protocol = runner.read_json(signed / "protocol.json")
        baseline = runner.read_json(signed / "signed.config.json")
        signed_record = runner.read_json(signed / "signed/train-only/record.json")
        for key in ("code", "code_hash", "environment_hash", "gate_sha256"):
            if protocol[key] != candidate_protocol[key]:
                raise ValueError(f"matched signed/absolute {key} mismatch")
        if recovered:
            from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance
            verified = runner.recovered_signed_main(CheckpointProvenance(**{
                key: protocol[key] for key in ("dataset_hash", "effective_config_hash", "code_hash", "environment_hash")}))
            if (protocol["signed_main_recovery"] != verified or candidate_protocol["signed_main_recovery"] != verified
                    or runner.read_json(signed / "signed/train-recovery.json") != verified):
                raise ValueError("signed recovery binding changed")
    elif candidate_protocol["profile"] == "absgrad_quality_exploration_v1":
        quality_cgroup()
        protocol = runner.checked_json(signed / "protocol.json", runner.SIGNED_PROTOCOL_SHA256)
        baseline = runner.checked_json(signed / "signed.config.json", runner.SIGNED_CONFIG_SHA256)
        signed_record = runner.checked_json(signed / "signed/train-only/record.json", runner.SIGNED_FINAL_RECORD_SHA256)
    else:
        raise ValueError("paired quality requires the separately authorized exploration protocol")
    if candidate_record["status"] != "absolute_training_complete_quality_pending":
        raise ValueError("candidate training is incomplete")
    config = runner.read_json(candidate / "absolute.config.json")
    resolved_config_record(ResolvedGaussianConfig(requested_profile=config["requested_profile"],
        effective_config=config["effective_config"], effective_config_hash=config["effective_config_hash"]))
    restored = {**config["effective_config"], "densification": {**config["effective_config"]["densification"], "absgrad": False}}
    if config["effective_config_hash"] != runner.ABSOLUTE_CONFIG_HASH or restored != baseline["effective_config"]:
        raise ValueError("paired comparison configuration drift")
    replay = Path(protocol["replay"])
    dataset = runner.read_json(replay / "dataset.json")
    if dataset["dataset_hash"] != protocol["dataset_hash"] or candidate_protocol["dataset_hash"] != protocol["dataset_hash"]:
        raise ValueError("dataset identity drift")
    train_roi = runner.checked_json(runner.PROJECT_ROOT / TRAIN_ROI, runner.ROI_SHA256)
    val_roi = runner.checked_json(runner.PROJECT_ROOT / VALIDATION_ROI, VALIDATION_ROI_SHA256)
    proposal = runner.checked_json(runner.PROJECT_ROOT / "outputs/analysis/absgrad-stage2b-gate-proposal-20261008-v1/gate-proposal.json", runner.QUALITY_SHA256)
    targets = train_roi["targets"]
    if len(targets) != 16 or sum(len(t["regions"]) for t in targets) != 24:
        raise ValueError("frozen Train ROI count drift")
    from gsplat import rendering
    if runner.sha256_file(Path(rendering.__file__)) != protocol["overlay_sha256"]:
        raise ValueError("rendering overlay identity drift")
    protected = {path: runner.sha256_file(path) for path in (signed / "protocol.json", signed / "signed.config.json",
        candidate / "protocol.json", candidate / "absolute.config.json", candidate / "complete.json",
        signed / "signed/train-only/record.json", replay / "dataset.json",
        runner.PROJECT_ROOT / TRAIN_ROI, runner.PROJECT_ROOT / VALIDATION_ROI, Path(rendering.__file__))}
    output.mkdir()
    torch.set_num_threads(2)
    entries = {str(e["image_id"]): e for e in dataset["images"]}
    all_views = load_training_views(dataset, replay, split="train", longest_edge=1920, device=torch.device("cpu"))
    by_id = {camera.image_id: i for i, camera in enumerate(all_views.cameras)}
    ids = [t["image_id"] for t in targets]
    if len(set(ids)) != 16 or not set(ids).issubset(by_id):
        raise ValueError("frozen Train cameras missing or duplicated")
    views = [all_views[by_id[i]] for i in ids]
    endpoints = {}
    contacts = output / "contacts"
    contacts.mkdir()
    for endpoint, final in (("selection", False), ("train-only", True)):
        left_root, right_root = signed / "signed", candidate / "absolute"
        eval_paths = [left_root / endpoint / "evaluation.json", right_root / endpoint / "evaluation.json"]
        expected = [signed_record["evaluation_sha256" if final else "selection_evaluation_sha256"],
                    candidate_record["final_evaluation_sha256" if final else "selection_evaluation_sha256"]]
        evals = []
        models = [left_root / ("train-only/model.pt" if final else "sor/filtered-model.pt"),
                  right_root / ("train-only/model.pt" if final else "sor/filtered-model.pt")]
        for index, path in enumerate(eval_paths):
            value = runner.checked_json(path, expected[index])
            runner.validate_evaluation(path, final=final)
            model_hash = runner.sha256_file(models[index])
            if (value["provenance"]["dataset_hash"] != protocol["dataset_hash"]
                    or value["provenance"]["effective_config_hash"] != (baseline["effective_config_hash"] if index == 0 else runner.ABSOLUTE_CONFIG_HASH)):
                raise ValueError("evaluation dataset/config identity drift")
            if value["provenance"]["model_sha256"] != model_hash:
                raise ValueError("evaluation/model identity drift")
            if final and model_hash != (signed_record["final_model_sha256"] if index == 0 else candidate_record["final_model_sha256"]):
                raise ValueError("final model identity drift")
            evals.append(value)
            protected[path] = expected[index]
            protected[models[index]] = model_hash
        validation = paired_validation(*evals, list(map(str, dataset["splits"]["validation"])))
        endpoint_root = output / endpoint
        endpoint_root.mkdir()
        for arm, model in zip(("signed", "absolute"), models, strict=True):
            render_controls(model, views, endpoint_root / arm)
        rows = []
        for split, definition in (("train", train_roi), ("validation", val_roi)):
            for target in definition["targets"]:
                image_id = str(target["image_id"])
                reference_path = replay / entries[image_id]["path"]
                if runner.sha256_file(reference_path) != target["reference_sha256"]:
                    raise ValueError("ROI reference image changed")
                protected[reference_path] = target["reference_sha256"]
                reference = rgb(reference_path)
                preview_paths = [endpoint_root / arm / f"{image_id}.png" for arm in ("signed", "absolute")] if split == "train" else [p.parent / "previews" / f"{image_id}.png" for p in eval_paths]
                protected.update({p: runner.sha256_file(p) for p in preview_paths})
                predictions = [rgb(p) for p in preview_paths]
                if any(pred.shape != reference.shape for pred in predictions):
                    raise ValueError("reference/preview shape drift")
                for region in target["regions"]:
                    box = region["pixel_xyxy"]
                    h, w = reference.shape[:2]
                    if box != [round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(region["normalized_xyxy"])]:
                        raise ValueError("ROI coordinate drift")
                    x0, y0, x1, y1 = box
                    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                        raise ValueError("ROI outside image")
                    crops = [a[y0:y1, x0:x1] for a in (reference, *predictions)]
                    metrics = [crop_metrics(crops[0], pred) for pred in crops[1:]]
                    row = {"split": split, "image_id": image_id, "name": region["name"],
                        "validation_group": target.get("validation_group", image_id), "pixel_xyxy": box,
                        "signed": metrics[0], "absolute": metrics[1],
                        "delta": {k: metrics[1][k] - metrics[0][k] for k in ("psnr", "ssim")}}
                    rows.append(row)
                    sheet = Image.new("RGB", (960, 280), "white")
                    draw = ImageDraw.Draw(sheet)
                    for column, (label, crop) in enumerate(zip(("reference", "signed", "absolute"), crops, strict=True)):
                        image = Image.fromarray(crop)
                        image.thumbnail((320, 250))
                        sheet.paste(image, (column * 320, 26))
                        draw.text((column * 320 + 4, 5), label, fill="black")
                    sheet.save(contacts / f"{endpoint}-{split}-{image_id}-{region['name']}.png")
        endpoints[endpoint] = {"roi_metric_profile": "display_clamped_uint8_v1_crop_metrics", "rois": rows,
            "validation": validation, "quality_gate": numerical_gate(rows, validation, proposal["proposed_quality_gates"])}
        write_json(endpoint_root / "metrics.json", endpoints[endpoint])
    if any(runner.sha256_file(path) != expected for path, expected in protected.items()):
        raise ValueError("protected comparison input changed during evaluation")
    candidate_stage = runner.read_json(candidate / "absolute/train.exit.json")
    if recovered:
        source = protocol["signed_main_recovery"]
        signed_stage = runner.checked_json(Path(source["source_train_exit"]), source["source_train_exit_sha256"])
        signed_native = None
    else:
        signed_stage = (runner.read_json(signed / "signed/train.exit.json")
            if candidate_protocol["profile"] == runner.PROFILE else None)
        signed_native = runner.read_json(signed / "signed/training/attempts/train-001/artifacts/result.json")
    resource = resource_report(
        runner.read_json(candidate / "absolute/training/attempts/train-001/artifacts/result.json"),
        signed_native, candidate_stage, signed_stage, proposal["proposed_resource_gates"],
    )
    write_json(output / "report.json", {"status": "paired_numerical_report_complete_visual_pending", "endpoints": endpoints,
        "resources": resource, "visual_review": "pending_all_60_endpoint_ROIs", "test_rgb": "not_loaded", "promotion_eligible": False,
        "protected": {str(path): digest for path, digest in protected.items()}})
    with (output / "REPORT.md").open("x") as f:
        f.write("# absolute 完整配对报告\n\n两个端点均报告24个Train ROI、6个Validation ROI和377个逐视角raw指标。\n\n")
        for endpoint, value in endpoints.items():
            f.write(f"## {endpoint}\n\n数值门禁：{'通过' if value['quality_gate']['numerical_passed'] else '不通过'}。视觉审查尚未完成。\n\n")
            f.write("| split | 相机 | ROI | ΔPSNR | ΔSSIM |\n|---|---|---|---:|---:|\n")
            for row in value["rois"]:
                f.write(f"| {row['split']} | {row['image_id']} | {row['name']} | {row['delta']['psnr']:+.6f} | {row['delta']['ssim']:+.6f} |\n")
            f.write("\n")
            for key, stats in value["validation"]["global_delta"].items():
                f.write(f"- raw {key}：均值差 {stats['mean']:+.6f}，P10差 {stats['p10']:+.6f}。\n")
        f.write("\n原资源门禁失败结论保留；新预算只是硬件停止保护。不得产生PASS_FOR_REPLICATION或默认推广。\n")
    write_json(output / "integrity.json", {str(p.relative_to(output)): runner.sha256_file(p) for p in output.rglob("*") if p.is_file()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-experiment", type=Path, required=True)
    parser.add_argument("--candidate-experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "i-94B8D131" or runner.PROJECT_ROOT != Path("/usr/local/3dgs_new/Image3D-SceneGraph"):
        raise ValueError("paired model evaluation is restricted to the authorized remote workspace")
    evaluate(args.signed_experiment, args.candidate_experiment, args.output_dir)


if __name__ == "__main__":
    main()
