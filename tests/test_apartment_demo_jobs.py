from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def test_demo_submission_uses_copies_and_keeps_b_independent(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "apartment_demo", scripts / "run_apartment_demo_jobs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "dataset"
    control = tmp_path / "control"
    control.mkdir()
    name = "10/10_DSC0001.jpg"
    original = root / "images-jpeg-2k" / name
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original RGB fixture")
    metadata = {
        "schema_version": 1,
        "images": {name: {"camera_id": "10", "capture_index": 0}},
    }
    calls = []

    class Store:
        def enqueue_job(self, mode, files, **kwargs):
            assert mode == "multi_image"
            assert kwargs["geometry_backend"] == "project_3dgs"
            assert kwargs["output_type"] == "gaussian_splat"
            staged = files[0].staged_path
            assert staged.read_bytes() == original.read_bytes()
            assert staged.stat().st_ino != original.stat().st_ino
            staged.unlink()
            calls.append(kwargs["options"])
            return {"job_id": f"job-{len(calls)}"}

    a = module.submit(Store(), root, metadata, control, "project")
    b = module.submit(Store(), root, metadata, control, "mcmc", a["job_id"])
    assert a["job_id"] != b["job_id"]
    assert original.read_bytes() == b"original RGB fixture"
    assert calls[0]["sfm_pairing"] == "rig_neighbors_vocab_v1"
    assert json.loads(calls[0]["sfm_capture_metadata"]) == metadata
    assert "sfm_capture_metadata" not in calls[1]
    assert calls[1]["gaussian_geometry_source_job_id"] == a["job_id"]
    for options in calls:
        assert options["gaussian_longest_edge"] == 2048
        assert options["gaussian_final_fit"] == "train_validation_v1"
        assert options["gaussian_sor_filter"] == "on"
    assert calls[0]["gaussian_recovery_prune"] == "on"
    assert calls[1]["gaussian_recovery_prune"] == "off"
    module.write_new(control / ".once", {"approved": True})
    with pytest.raises(FileExistsError):
        module.write_new(control / ".once", {"approved": True})
