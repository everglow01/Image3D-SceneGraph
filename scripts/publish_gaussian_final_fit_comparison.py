#!/usr/bin/env python3
"""Publish an immutable four-model final-fit comparison; never trains or loads RGB."""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.dataset import validate_contract
from image3d_scenegraph.gaussian.evaluation import load_model_snapshot
from image3d_scenegraph.gaussian.export import (
    BROWSER_RENDERER_IMPLEMENTATION, BROWSER_RENDERER_VERSION,
    BROWSER_RENDERER_MAX_SH_DEGREE, _camera_path, _model_rows, _scene_frame,
    write_binary_ply,
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def publish(args) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_id):
        raise ValueError("invalid comparison job ID")
    destination = args.output_root / args.job_id
    if destination.exists():
        raise FileExistsError(destination)
    experiment = args.experiment
    sources = [args.dataset, args.camera_path, experiment / "protocol.json",
               experiment / "comparison.json"]
    contract = read_json(args.dataset)
    validate_contract(contract)  # No dataset root: Test RGB is not hashed or decoded.
    protocol = read_json(experiment / "protocol.json")
    comparison = read_json(experiment / "comparison.json")
    camera_path = read_json(args.camera_path)
    if camera_path != _camera_path(contract):
        raise ValueError("shared camera path does not match the frozen dataset")
    if (protocol.get("dataset_hash") != contract["dataset_hash"]
            or protocol.get("dataset_file_sha256") != sha256_file(args.dataset)
            or protocol.get("splits") != {k: len(v) for k, v in contract["splits"].items()}
            or comparison.get("protocol_sha256") != sha256_file(sources[2])
            or comparison.get("status") != "completed"
            or comparison.get("gates", {}).get("integrity") is not True):
        raise ValueError("comparison protocol identity mismatch")
    descriptors = []
    validation_ids = set(map(str, contract["splits"]["validation"]))
    for arm, source_model in (("mcmc", args.mcmc_selection_model),
                              ("project", args.project_selection_model)):
        arm_dir = experiment / arm
        record_path = arm_dir / "final-fit/record.json"
        record = read_json(record_path)
        model_paths = [source_model, arm_dir / "final-fit/model.pt"]
        eval_paths = [arm_dir / "selection/evaluation.json", arm_dir / "final-fit/evaluation.json"]
        sources.extend([record_path, *model_paths, *eval_paths])
        hashes = [sha256_file(p) for p in model_paths]
        if (record.get("schema_version") != 1 or record.get("status") != "complete"
                or record.get("profile") != "train_validation_v1"
                or record.get("dataset_hash") != contract["dataset_hash"]
                or record.get("effective_config_hash") != protocol["arms"][arm]["config_hash"]
                or hashes[0] != protocol["arms"][arm]["source_model_sha256"]
                or record.get("source_model_sha256") != hashes[0]
                or record.get("final_model_sha256") != hashes[1]
                or record.get("selection_evaluation_sha256") != sha256_file(eval_paths[0])
                or record.get("evaluation_sha256") != sha256_file(eval_paths[1])
                or record.get("source_model_unchanged") is not True
                or record.get("topology_changed") is not False
                or record.get("test_rgb") != "not_loaded"
                or record.get("gaussian_count_before") != record.get("gaussian_count_after")
                or record.get("optimizer_updates") != 2000):
            raise ValueError(f"{arm} final-fit lineage mismatch")
        for index, stage in enumerate(("selection", "final_fit")):
            evaluation = read_json(eval_paths[index])
            role = "held_out_model_selection" if index == 0 else "in_sample_after_train_validation_fit"
            rows = evaluation.get("per_view", [])
            ids = [str(row["image_id"]) for row in rows]
            if (evaluation.get("split") != ("validation" if index == 0 else "fit_validation")
                    or evaluation.get("quality_role") != role
                    or evaluation.get("selection_eligible") is not (index == 0)
                    or evaluation.get("status") != "complete"
                    or evaluation.get("successful_views") != len(validation_ids)
                    or len(ids) != len(validation_ids) or set(ids) != validation_ids
                    or evaluation.get("provenance", {}).get("model_sha256") != hashes[index]
                    or evaluation["provenance"].get("dataset_hash") != contract["dataset_hash"]
                    or evaluation["provenance"].get("effective_config_hash") != record["effective_config_hash"]):
                raise ValueError(f"{arm}/{stage} evaluation identity mismatch")
            summary = {key: evaluation[key]["mean"] for key in
                       ("psnr", "ssim", "display_psnr", "display_ssim")}
            expected = comparison["arms"][arm]["baseline" if index == 0 else "fit"]
            if not all(isinstance(v, (float, int)) and math.isfinite(v)
                       and v == expected[k]["mean"] for k, v in summary.items()):
                raise ValueError("comparison metrics disagree with evaluation")
            descriptors.append((arm, stage, model_paths[index], eval_paths[index],
                                record_path, hashes[index], role, summary, record))
    reuse = None
    if args.reuse_export is not None:
        reuse = read_json(args.reuse_export / "export.json")
        sources += [args.reuse_export / "export.json", args.reuse_export / "scene.ply"]
        if (reuse.get("model_sha256") != sha256_file(args.mcmc_selection_model)
                or reuse.get("browser_sha256") != sha256_file(sources[-1])
                or reuse.get("dataset_hash") != contract["dataset_hash"]
                or reuse.get("sh_degree") != 3):
            raise ValueError("reused MCMC export hash mismatch")
    before = {str(p.resolve()): sha256_file(p) for p in sources}
    staging = args.output_root / ".publication" / (args.job_id + "-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    variants = []
    assets = {"gaussian_camera_path": "camera_path.json", "scene_graph": "scene_graph/scene.json",
              "gaussian_comparison": "comparison.json", "publication_record": "publication.json"}
    link_or_copy(args.camera_path, staging / "camera_path.json")
    link_or_copy(experiment / "comparison.json", staging / "comparison.json")
    for arm, stage, model_path, evaluation_path, record_path, model_hash, role, summary, record in descriptors:
        variant_id = f"{arm}-{stage.replace('_', '-')}"
        folder = staging / "variants" / variant_id
        folder.mkdir(parents=True)
        model = load_model_snapshot(model_path, torch.device("cpu"))
        if model.max_sh_degree != 3 or model.count != record["gaussian_count_after"]:
            raise ValueError("model SH/count mismatch")
        center, radius = _scene_frame(model)
        if reuse is not None and arm == "mcmc" and stage == "selection":
            if reuse.get("gaussian_count") != model.count:
                raise ValueError("reused PLY count mismatch")
            link_or_copy(args.reuse_export / "scene.ply", folder / "scene.ply")
        else:
            write_binary_ply(folder / "scene.ply", _model_rows(model))
        metadata = {
            "schema_version": 2, "format": "project_gaussian_ply_v1",
            "coordinate_frame": "normalized", "world_units": "arbitrary",
            "world_from_normalized": contract["normalization"]["world_from_normalized"],
            "camera_axes": contract["coordinate_system"]["camera_axes"],
            "scene_center": center, "scene_radius_p95": radius,
            "gaussian_count": model.count, "sh_degree": 3, "model_sh_degree": 3,
            "viewer_minimum_opacity": 0.005,
            "dataset_hash": contract["dataset_hash"], "model_sha256": model_hash,
            "effective_config_hash": record["effective_config_hash"],
            "browser_sha256": sha256_file(folder / "scene.ply"),
            "evaluation_sha256": sha256_file(evaluation_path),
            "camera_path_sha256": sha256_file(args.camera_path),
            "metric_role": role, "variant_id": variant_id,
            "browser_renderer": {
                "implementation": BROWSER_RENDERER_IMPLEMENTATION,
                "version": BROWSER_RENDERER_VERSION, "requested_sh_degree": 3,
                "effective_sh_degree": BROWSER_RENDERER_MAX_SH_DEGREE,
            },
        }
        if stage == "final_fit":
            metadata["final_fit"] = {"profile": record["profile"],
                                     "source_model_sha256": record["source_model_sha256"],
                                     "record_sha256": sha256_file(record_path)}
            link_or_copy(record_path, folder / "final-fit.json")
        link_or_copy(evaluation_path, folder / "evaluation.json")
        (folder / "export.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        variants.append({
            "id": variant_id, "label": f"{'MCMC' if arm == 'mcmc' else 'Project v7'} · {'Final-fit' if stage == 'final_fit' else 'Selection'}",
            "trainer": arm, "model_stage": stage, "metric_role": role,
            "scene_splat": f"variants/{variant_id}/scene.ply",
            "export_metadata": f"variants/{variant_id}/export.json",
            "gaussian_count": model.count, "model_sha256": model_hash, **summary,
        })
        del model
    default = next(v for v in variants if v["id"] == "mcmc-final-fit")
    assets.update(scene_splat=default["scene_splat"], gaussian_export_metadata=default["export_metadata"])
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "job_id": args.job_id, "result_kind": "gaussian_comparison",
        "status": "done", "stage": "gaussian_export", "progress": 1.0,
        "mode": "video", "geometry_backend": "project_3dgs", "output_type": "gaussian_splat",
        "created_at": now, "updated_at": now, "inputs": [], "assets": assets,
        "dataset_hash": contract["dataset_hash"], "default_gaussian_variant": default["id"],
        "gaussian_variants": variants, "navigation_status": "not_generated",
        "navigation_reason": "read_only_comparison", "test_rgb": "not_loaded",
        "metrics": {"num_inputs": 0, "num_objects": 0, "gaussian_test_status": "not_run"},
    }
    (staging / "scene_graph").mkdir()
    (staging / "scene_graph/scene.json").write_text(json.dumps({
        "job_id": args.job_id, "mode": "video", "coordinate_system": "normalized_arbitrary",
        "objects": [], "relations": [], "diagnostics": {"scale_recovered": False, "physical_checks": []},
    }) + "\n")
    if any(sha256_file(Path(p)) != h for p, h in before.items()):
        raise ValueError("publication source changed")
    (staging / "publication.json").write_text(json.dumps({
        "source_sha256": before, "source_unchanged": True, "training": "not_run", "test_rgb": "not_loaded",
        "protocol_code_sha": protocol["code_sha"],
    }, indent=2) + "\n")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("experiment", "dataset", "camera-path", "mcmc-selection-model", "project-selection-model", "output-root"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--reuse-export", type=Path)
    parser.add_argument("--job-id", default="train_validation_final_fit_v1_comparison")
    manifest = publish(parser.parse_args())
    print(json.dumps({"job_id": manifest["job_id"], "variants": [v["id"] for v in manifest["gaussian_variants"]]}))


if __name__ == "__main__":
    main()
