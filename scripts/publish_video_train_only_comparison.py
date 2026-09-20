#!/usr/bin/env python3
"""Publish the completed video runner's two Train-only exports without loading RGB."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.dataset import validate_contract
from image3d_scenegraph.gaussian.export import _camera_path


def publish(args) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_id):
        raise ValueError("invalid comparison job ID")
    destination = args.output_root / args.job_id
    if destination.exists():
        raise FileExistsError(destination)
    root = args.experiment.resolve()
    sources = {}

    def track(path):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("publication source escapes experiment")
        digest = sha256_file(path)
        if path in sources and sources[path] != digest:
            raise ValueError("publication source changed")
        sources[path] = digest
        return digest

    def read(path):
        track(path)
        return json.loads(path.read_text())

    protocol = read(root / "protocol.json")
    if (sources[root / "protocol.json"] != args.protocol_sha256
            or protocol.get("status") != "frozen"
            or protocol.get("profile") not in ("num4_new_4k_train_only_v1", "num4_retake_1080_train_only_v1")
            or (protocol.get("world_size"), protocol.get("main_updates"),
                protocol.get("final_fit_updates"), protocol.get("final_fit_camera_samples")) != (2, 30000, 2000, 4000)
            or protocol.get("test") != "not_authorized"):
        raise ValueError("frozen video protocol mismatch")
    for name, expected in protocol["files"].items():
        if track(Path(name)) != expected:
            raise ValueError("frozen protocol file changed")
    dataset_path = root / "shared/gaussian/replay/dataset.json"
    if protocol["files"].get(str(dataset_path)) != track(dataset_path):
        raise ValueError("dataset is not bound to protocol")
    dataset = read(dataset_path)
    validate_contract(dataset)  # No dataset root: publication never reads image bytes.
    if (dataset["dataset_hash"] != protocol["dataset_hash"]
            or protocol["split_counts"] != {k: len(v) for k, v in dataset["splits"].items()}):
        raise ValueError("dataset identity mismatch")
    validation_ids = set(map(str, dataset["splits"]["validation"]))
    variants = []
    copies = []
    for arm in ("project", "mcmc"):
        folder = root / arm
        complete = read(folder / "complete.json")
        if (complete.get("arm") != arm or complete.get("protocol_sha256") != args.protocol_sha256
                or complete.get("test") != "not_run"):
            raise ValueError(f"{arm} completion mismatch")
        for stage in ("train", "sor", "selection", "train-only", "export"):
            if read(folder / f"{stage}.exit.json").get("returncode") != 0:
                raise ValueError(f"{arm}/{stage} did not complete")
        for key, relative in (("selection_evaluation_sha256", "selection/evaluation.json"),
                              ("final_fit_record_sha256", "train-only/record.json"),
                              ("final_evaluation_sha256", "train-only/evaluation.json")):
            if track(folder / relative) != complete[key]:
                raise ValueError(f"{arm} completion hash mismatch")
        result = read(folder / "training/attempts/train-001/artifacts/result.json")
        record = read(folder / "train-only/record.json")
        evaluation = read(folder / "train-only/evaluation.json")
        metadata = read(folder / "export/export.json")
        model_hash = track(folder / "train-only/model.pt")
        config_path = root / f"{arm}.config.json"
        config = read(config_path)
        if protocol["files"].get(str(config_path)) != sources[config_path]:
            raise ValueError(f"{arm} config is not bound to protocol")
        config_hash = config["effective_config_hash"]
        if ((result.get("iteration"), result.get("world_size")) != (30000, 2)
                or record.get("status") != "complete" or record.get("profile") != "train_only_control_v1"
                or record.get("dataset_hash") != dataset["dataset_hash"]
                or record.get("effective_config_hash") != config_hash
                or record.get("input_splits") != ["train"]
                or (record.get("optimizer_updates"), record.get("camera_samples"), record.get("world_size")) != (2000, 4000, 2)
                or record.get("topology_changed") is not False or record.get("source_model_unchanged") is not True
                or record.get("test_rgb") != "not_loaded"
                or record.get("gaussian_count_before") != record.get("gaussian_count_after")
                or record.get("final_model_sha256") != model_hash
                or record.get("source_model_sha256") != track(folder / "sor/filtered-model.pt")
                or record.get("selection_evaluation_sha256") != complete["selection_evaluation_sha256"]
                or record.get("evaluation_sha256") != complete["final_evaluation_sha256"]):
            raise ValueError(f"{arm} Train-only lineage mismatch")
        ids = [str(row["image_id"]) for row in evaluation.get("per_view", [])]
        provenance = evaluation.get("provenance", {})
        if (evaluation.get("status") != "complete" or evaluation.get("split") != "control_validation"
                or evaluation.get("quality_role") != "held_out_after_train_only_control"
                or evaluation.get("selection_eligible") is not False
                or evaluation.get("test_rgb") != "not_loaded" or evaluation.get("failed_views") != []
                or evaluation.get("successful_views") != len(validation_ids)
                or len(ids) != len(validation_ids) or set(ids) != validation_ids
                or provenance.get("model_sha256") != model_hash
                or provenance.get("dataset_hash") != dataset["dataset_hash"]
                or provenance.get("effective_config_hash") != config_hash):
            raise ValueError(f"{arm} Validation identity mismatch")
        summary = {key: evaluation[key]["mean"] for key in ("psnr", "ssim", "display_psnr", "display_ssim")}
        if not all(type(v) in (int, float) and math.isfinite(v) for v in summary.values()):
            raise ValueError("nonfinite comparison metric")
        camera_path = folder / "export/camera_path.json"
        camera = read(camera_path)
        final_fit = metadata.get("final_fit", {})
        count = metadata.get("gaussian_count")
        if (metadata.get("model_sha256") != model_hash
                or metadata.get("dataset_hash") != dataset["dataset_hash"]
                or metadata.get("effective_config_hash") != config_hash
                or metadata.get("evaluation_sha256") != complete["final_evaluation_sha256"]
                or metadata.get("browser_sha256") != track(folder / "export/scene.ply")
                or metadata.get("camera_path_sha256") != sources[camera_path]
                or camera != _camera_path(dataset)
                or metadata.get("world_from_normalized") != dataset["normalization"]["world_from_normalized"]
                or metadata.get("world_units") != "arbitrary" or metadata.get("coordinate_frame") != "normalized"
                or metadata.get("sh_degree") != 3 or type(count) is not int or count <= 0
                or count != record["gaussian_count_after"] or count != evaluation.get("gaussian_count")
                or metadata.get("checkpoint_hash") != result.get("final_checkpoint_hash")
                or not re.fullmatch(r"[a-f0-9]{64}", str(metadata.get("checkpoint_hash")))
                or final_fit.get("profile") != record["profile"]
                or final_fit.get("record_sha256") != complete["final_fit_record_sha256"]
                or final_fit.get("final_model_sha256") != model_hash
                or final_fit.get("evaluation_role") != evaluation["quality_role"]):
            raise ValueError(f"{arm} export identity mismatch")
        variant_id = f"{arm}-train-only"
        relative = f"variants/{variant_id}"
        variants.append({
            "id": variant_id, "label": f"{'Project v7' if arm == 'project' else 'MCMC'} · 30k + 2k Train-only",
            "trainer": arm, "model_stage": "train_only", "metric_role": evaluation["quality_role"],
            "scene_splat": f"{relative}/scene.ply", "export_metadata": f"{relative}/export.json",
            "gaussian_count": count, "model_sha256": model_hash, **summary,
        })
        copies.extend((folder / source, f"{relative}/{target}") for source, target in (
            ("export/scene.ply", "scene.ply"), ("export/export.json", "export.json"),
            ("train-only/evaluation.json", "evaluation.json"), ("train-only/record.json", "train-only.json"),
            ("complete.json", "complete.json")))
    default = variants[0]
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "job_id": args.job_id, "result_kind": "gaussian_comparison", "status": "done",
        "stage": "gaussian_export", "progress": 1.0, "mode": "video",
        "geometry_backend": "project_3dgs", "output_type": "gaussian_splat",
        "created_at": now, "updated_at": now, "inputs": [],
        "dataset_hash": dataset["dataset_hash"], "default_gaussian_variant": default["id"],
        "gaussian_variants": variants, "navigation_status": "not_generated",
        "navigation_reason": "read_only_comparison", "test_rgb": "not_loaded",
        "metrics": {"num_inputs": 0, "num_objects": 0, "gaussian_test_status": "not_run"},
        "assets": {"scene_splat": default["scene_splat"], "gaussian_export_metadata": default["export_metadata"],
                   "gaussian_camera_path": "camera_path.json", "scene_graph": "scene_graph/scene.json",
                   "publication_record": "publication.json"},
    }
    staging = args.output_root / ".publication" / (args.job_id + "-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    copies.append((root / "project/export/camera_path.json", "camera_path.json"))
    for source, relative in copies:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".ply":
            os.link(source, target)
        else:
            shutil.copyfile(source, target)
        if sha256_file(target) != sources[source]:
            raise ValueError("published asset hash mismatch")
    if any(sha256_file(path) != digest for path, digest in sources.items()):
        raise ValueError("publication source changed")
    publication = {"source_sha256": {str(p): h for p, h in sources.items()}, "source_unchanged": True,
                   "protocol_sha256": args.protocol_sha256, "protocol_code_sha": protocol["code"],
                   "training": "not_run", "test_rgb": "not_loaded"}
    scene = {"job_id": args.job_id, "mode": "video", "coordinate_system": "normalized_arbitrary",
             "objects": [], "relations": [], "diagnostics": {"scale_recovered": False, "physical_checks": []}}
    for relative, value in (("publication.json", publication), ("scene_graph/scene.json", scene), ("manifest.json", manifest)):
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    manifest = publish(parser.parse_args())
    print(json.dumps({"job_id": manifest["job_id"], "variants": [v["id"] for v in manifest["gaussian_variants"]]}))


if __name__ == "__main__":
    main()
