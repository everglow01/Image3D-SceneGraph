import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from image3d_scenegraph.file_integrity import sha256_file  # noqa: E402
from image3d_scenegraph.gaussian.export import _camera_path, _model_rows, write_binary_ply  # noqa: E402
from image3d_scenegraph.jobs import JobError, JobStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
publish = runpy.run_path(str(ROOT / "scripts/publish_video_train_only_comparison.py"))["publish"]
fixtures = runpy.run_path(str(ROOT / "tests/test_gaussian_export.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def setup(tmp_path):
    root = tmp_path / "experiment"
    dataset = fixtures["contract"]()
    dataset_path = write(root / "shared/gaussian/replay/dataset.json", dataset)
    protocol = {"profile": "num4_retake_1080_train_only_v1", "status": "frozen", "code": "a" * 40,
                "world_size": 2, "main_updates": 30000, "final_fit_updates": 2000, "final_fit_camera_samples": 4000,
                "test": "not_authorized", "dataset_hash": dataset["dataset_hash"],
                "split_counts": {k: len(v) for k, v in dataset["splits"].items()},
                "files": {str(dataset_path): sha256_file(dataset_path)}}
    for arm in ("project", "mcmc"):
        folder = root / arm
        config_path = write(root / f"{arm}.config.json", {"effective_config_hash": "c" * 64})
        protocol["files"][str(config_path)] = sha256_file(config_path)
        model = write(folder / "train-only/model.pt", {"fixture_model": arm})
        source = write(folder / "sor/filtered-model.pt", {"fixture_source": arm})
        selection = write(folder / "selection/evaluation.json", {})
        evaluation = write(folder / "train-only/evaluation.json", {
            "status": "complete", "split": "control_validation", "quality_role": "held_out_after_train_only_control",
            "selection_eligible": False, "test_rgb": "not_loaded", "failed_views": [], "successful_views": 2,
            "per_view": [{"image_id": i} for i in dataset["splits"]["validation"]], "gaussian_count": 2,
            "provenance": {"model_sha256": sha256_file(model), "dataset_hash": dataset["dataset_hash"],
                           "effective_config_hash": "c" * 64},
            **{k: {"mean": v} for k, v in {"psnr": 25, "ssim": .8, "display_psnr": 25.1, "display_ssim": .81}.items()},
        })
        record = write(folder / "train-only/record.json", {
            "status": "complete", "profile": "train_only_control_v1", "dataset_hash": dataset["dataset_hash"],
            "effective_config_hash": "c" * 64, "input_splits": ["train"], "optimizer_updates": 2000,
            "camera_samples": 4000, "world_size": 2, "topology_changed": False, "source_model_unchanged": True,
            "test_rgb": "not_loaded", "gaussian_count_before": 2, "gaussian_count_after": 2,
            "final_model_sha256": sha256_file(model), "source_model_sha256": sha256_file(source),
            "selection_evaluation_sha256": sha256_file(selection), "evaluation_sha256": sha256_file(evaluation),
        })
        camera = write(folder / "export/camera_path.json", _camera_path(dataset))
        ply = folder / "export/scene.ply"
        write_binary_ply(ply, _model_rows(fixtures["model"]()))
        write(folder / "export/export.json", {
            "model_sha256": sha256_file(model), "dataset_hash": dataset["dataset_hash"], "effective_config_hash": "c" * 64,
            "evaluation_sha256": sha256_file(evaluation), "browser_sha256": sha256_file(ply),
            "camera_path_sha256": sha256_file(camera), "world_from_normalized": dataset["normalization"]["world_from_normalized"],
            "world_units": "arbitrary", "coordinate_frame": "normalized", "sh_degree": 3, "gaussian_count": 2,
            "checkpoint_hash": "d" * 64, "final_fit": {"profile": "train_only_control_v1", "record_sha256": sha256_file(record),
            "final_model_sha256": sha256_file(model), "evaluation_role": "held_out_after_train_only_control"},
        })
        write(folder / "training/attempts/train-001/artifacts/result.json", {
            "iteration": 30000, "world_size": 2, "final_checkpoint_hash": "d" * 64})
        for stage in ("train", "sor", "selection", "train-only", "export"):
            write(folder / f"{stage}.exit.json", {"returncode": 0})
    protocol_path = write(root / "protocol.json", protocol)
    for arm in ("project", "mcmc"):
        folder = root / arm
        write(folder / "complete.json", {"arm": arm, "protocol_sha256": sha256_file(protocol_path), "test": "not_run",
            "selection_evaluation_sha256": sha256_file(folder / "selection/evaluation.json"),
            "final_fit_record_sha256": sha256_file(folder / "train-only/record.json"),
            "final_evaluation_sha256": sha256_file(folder / "train-only/evaluation.json")})
    return SimpleNamespace(experiment=root, output_root=tmp_path / "jobs", job_id="retake-comparison",
                           protocol_sha256=sha256_file(protocol_path))


def test_publish_completed_exports_without_images_or_model_loading(tmp_path):
    args = setup(tmp_path)
    before = {p: sha256_file(p) for p in args.experiment.rglob("*") if p.is_file()}
    manifest = publish(args)
    destination = args.output_root / args.job_id
    assert manifest["default_gaussian_variant"] == "project-train-only"
    assert [v["trainer"] for v in manifest["gaussian_variants"]] == ["project", "mcmc"]
    for v in manifest["gaussian_variants"]:
        assert v["metric_role"] == "held_out_after_train_only_control"
        assert (destination / v["scene_splat"]).stat().st_ino == (args.experiment / v["trainer"] / "export/scene.ply").stat().st_ino
        assert (destination / v["export_metadata"]).read_bytes() == (args.experiment / v["trainer"] / "export/export.json").read_bytes()
    assert all(sha256_file(p) == digest for p, digest in before.items())
    store = JobStore(args.output_root)
    assert store.list_jobs()[0]["result_kind"] == "gaussian_comparison"
    assert store.get_manifest(args.job_id) == manifest
    for action in (store.cancel_job, store.retry_job, store.request_navigation_assets, store.build_zip):
        with pytest.raises(JobError):
            action(args.job_id)
    with pytest.raises(FileExistsError):
        publish(args)


@pytest.mark.parametrize("relative,field,value", [
    ("complete.json", "protocol_sha256", "0" * 64),
    ("export.exit.json", "returncode", 1),
    ("train-only/record.json", "input_splits", ["train", "validation"]),
    ("train-only/record.json", "optimizer_updates", 1999),
    ("train-only/evaluation.json", "quality_role", "in_sample_after_train_validation_fit"),
    ("train-only/evaluation.json", "per_view", [{"image_id": "test-view"}]),
    ("export/export.json", "browser_sha256", "0" * 64),
    ("export/export.json", "checkpoint_hash", None),
    ("export/camera_path.json", "schema_version", -1),
])
def test_invalid_evidence_never_publishes(tmp_path, relative, field, value):
    args = setup(tmp_path)
    path = args.experiment / "mcmc" / relative
    data = json.loads(path.read_text())
    data[field] = value
    write(path, data)
    if relative in ("train-only/record.json", "train-only/evaluation.json"):
        complete_path = args.experiment / "mcmc/complete.json"
        complete = json.loads(complete_path.read_text())
        key = "final_fit_record_sha256" if "record" in relative else "final_evaluation_sha256"
        complete[key] = sha256_file(path)
        if relative == "train-only/evaluation.json":
            record_path = args.experiment / "mcmc/train-only/record.json"
            record = json.loads(record_path.read_text())
            record["evaluation_sha256"] = sha256_file(path)
            write(record_path, record)
            complete["final_fit_record_sha256"] = sha256_file(record_path)
        write(complete_path, complete)
    with pytest.raises(ValueError):
        publish(args)
    assert not (args.output_root / args.job_id).exists()
    assert not (args.output_root / ".publication").exists()


def test_wrong_protocol_and_escaping_paths_are_rejected(tmp_path):
    args = setup(tmp_path)
    args.protocol_sha256 = "0" * 64
    with pytest.raises(ValueError, match="protocol"):
        publish(args)
    args.job_id = "../escaped"
    with pytest.raises(ValueError, match="job ID"):
        publish(args)
    args.job_id = "retake-comparison"
    protocol_path = args.experiment / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    outside = write(tmp_path / "outside.json", {})
    protocol["files"][str(outside)] = sha256_file(outside)
    write(protocol_path, protocol)
    args.protocol_sha256 = sha256_file(protocol_path)
    with pytest.raises(ValueError, match="escapes"):
        publish(args)
