import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from image3d_scenegraph.file_integrity import sha256_file  # noqa: E402
from image3d_scenegraph.gaussian.export import _camera_path, _model_rows, write_binary_ply, read_gaussian_ply  # noqa: E402
from image3d_scenegraph.jobs import JobStore, JobError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
publish = runpy.run_path(str(ROOT / "scripts/publish_gaussian_final_fit_comparison.py"))["publish"]
fixtures = runpy.run_path(str(ROOT / "tests/test_gaussian_export.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def setup(tmp_path):
    model = fixtures["model"]()
    contract = fixtures["contract"]()
    dataset = write(tmp_path / "dataset.json", contract)
    camera = write(tmp_path / "camera.json", _camera_path(contract))
    experiment = tmp_path / "experiment"
    protocol = {"dataset_hash": contract["dataset_hash"], "dataset_file_sha256": sha256_file(dataset),
                "code_sha": "a" * 40, "splits": {k: len(v) for k, v in contract["splits"].items()}, "arms": {}}
    comparison = {"status": "completed", "gates": {"integrity": True}, "arms": {}}
    models = {}
    for arm in ("project", "mcmc"):
        source = tmp_path / f"{arm}.pt"
        torch.save({"state_dict": model.state_dict(), "max_sh_degree": 3}, source)
        final = experiment / arm / "final-fit/model.pt"
        final.parent.mkdir(parents=True)
        final.write_bytes(source.read_bytes())
        models[arm] = source
        evaluations = []
        for stage in ("selection", "final-fit"):
            ev = {"status": "complete", "successful_views": 2,
                  "split": "validation" if stage == "selection" else "fit_validation",
                  "quality_role": "held_out_model_selection" if stage == "selection" else "in_sample_after_train_validation_fit",
                  "selection_eligible": stage == "selection",
                  "provenance": {"dataset_hash": contract["dataset_hash"], "effective_config_hash": "c" * 64, "model_sha256": sha256_file(source)},
                  "per_view": [{"image_id": i} for i in contract["splits"]["validation"]],
                  **{k: {"mean": v} for k, v in {"psnr": 25, "ssim": 0.8, "display_psnr": 25.1, "display_ssim": 0.81}.items()}}
            evaluations.append(write(experiment / arm / stage / "evaluation.json", ev))
            comparison["arms"].setdefault(arm, {})["baseline" if stage == "selection" else "fit"] = ev
        write(experiment / arm / "final-fit/record.json", {
            "schema_version": 1, "profile": "train_validation_v1", "status": "complete",
            "dataset_hash": contract["dataset_hash"], "effective_config_hash": "c" * 64,
            "source_model_sha256": sha256_file(source), "final_model_sha256": sha256_file(final),
            "selection_evaluation_sha256": sha256_file(evaluations[0]), "evaluation_sha256": sha256_file(evaluations[1]),
            "source_model_unchanged": True, "topology_changed": False, "test_rgb": "not_loaded",
            "gaussian_count_before": model.count, "gaussian_count_after": model.count, "optimizer_updates": 2000,
        })
        protocol["arms"][arm] = {"source_model_sha256": sha256_file(source), "config_hash": "c" * 64}
    write(experiment / "protocol.json", protocol)
    comparison["protocol_sha256"] = sha256_file(experiment / "protocol.json")
    write(experiment / "comparison.json", comparison)
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    write_binary_ply(reuse / "scene.ply", _model_rows(model))
    write(reuse / "export.json", {"model_sha256": sha256_file(models["mcmc"]),
                                 "browser_sha256": sha256_file(reuse / "scene.ply"), "dataset_hash": contract["dataset_hash"],
                                 "sh_degree": 3, "gaussian_count": model.count})
    return SimpleNamespace(experiment=experiment, dataset=dataset, camera_path=camera,
                           project_selection_model=models["project"], mcmc_selection_model=models["mcmc"],
                           reuse_export=reuse, output_root=tmp_path / "jobs", job_id="comparison")


def test_publish_four_variants_and_read_only_actions(tmp_path):
    args = setup(tmp_path)
    before = sha256_file(args.mcmc_selection_model)
    manifest = publish(args)
    job = args.output_root / args.job_id
    assert len(manifest["gaussian_variants"]) == 4
    for v in manifest["gaussian_variants"]:
        assert len(read_gaussian_ply(job / v["scene_splat"])["x"]) == 2
    assert (job / "variants/mcmc-selection/scene.ply").stat().st_ino == (args.reuse_export / "scene.ply").stat().st_ino
    assert sha256_file(args.mcmc_selection_model) == before
    store = JobStore(args.output_root)
    assert store.list_jobs()[0]["result_kind"] == "gaussian_comparison"
    assert store.get_manifest(args.job_id) == manifest
    for action in (store.cancel_job, store.retry_job, store.request_navigation_assets, store.build_zip):
        with pytest.raises(JobError):
            action(args.job_id)
    with pytest.raises(JobError):
        store.build_mesh_variant(args.job_id, {})
    with pytest.raises(FileExistsError):
        publish(args)


def test_bad_source_hash_does_not_publish(tmp_path):
    args = setup(tmp_path)
    args.mcmc_selection_model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="lineage"):
        publish(args)
    assert not (args.output_root / args.job_id).exists()
