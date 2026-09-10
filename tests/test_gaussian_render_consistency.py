from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from image3d_scenegraph.gaussian.dataset import DatasetContractError, contract_hash, sha256_file
from image3d_scenegraph.gaussian.export import _model_rows, write_binary_ply
from image3d_scenegraph.gaussian.importer import import_inria_ply
from image3d_scenegraph.gaussian.runtime import load_evaluation_views
from test_gaussian_export import contract, model

spec = importlib.util.spec_from_file_location("render_audit", Path(__file__).parents[1] / "scripts/audit_gaussian_render_consistency.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_validation_subset_only_opens_requested_reference(tmp_path):
    value = contract()
    (tmp_path / "images").mkdir()
    image = tmp_path / "images/8.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image)
    value["images"][8]["sha256"] = sha256_file(image)
    value["dataset_hash"] = contract_hash(value)
    # All other files, including Train/Test RGB, are absent deliberately.
    views = load_evaluation_views(value, tmp_path, split="validation", longest_edge=1280,
                                  device=torch.device("cpu"), image_ids=["8"])
    assert [v.camera.image_id for v in views] == ["8"]
    assert views[0].image.shape == (64, 64, 3)


@pytest.mark.parametrize("split,ids", [("test", ["10"]), ("validation", ["10"]),
                                       ("validation", ["8", "8"]), ("validation", []),
                                       ("validation", ["unknown"])])
def test_invalid_subset_rejected_before_any_rgb(tmp_path, split, ids):
    with pytest.raises(DatasetContractError):
        load_evaluation_views(contract(), tmp_path, split=split, longest_edge=1280,
                              device=torch.device("cpu"), image_ids=ids)


def test_roundtrip_preserves_all_rows_and_source(tmp_path):
    original = model()
    ply = tmp_path / "source.ply"
    write_binary_ply(ply, _model_rows(original))
    before = sha256_file(ply)
    snapshot = tmp_path / "roundtrip.pt"
    import_inria_ply(ply, snapshot)
    restored = audit.load_model_snapshot(snapshot, torch.device("cpu"))
    difference = audit.attribute_difference(original, restored)
    assert all(d["max_absolute"] == 0 for d in difference.values())
    assert sha256_file(ply) == before
    with torch.no_grad():
        restored.means[0, 0] += 0.1
    assert audit.attribute_difference(original, restored)["means"]["max_absolute"] > 0.09


def test_image_metrics_and_existing_output(tmp_path):
    image = torch.zeros((32, 32, 3))
    same = audit.image_difference(image, image)
    assert same["mae"] == 0 and same["psnr"] == 120
    assert audit.image_difference(image, image + 0.1)["psnr"] == pytest.approx(20)
    with pytest.raises(ValueError, match="same-shape"):
        audit.image_difference(image, image[:1])
    with pytest.raises(FileExistsError):
        audit.run(SimpleNamespace(output_dir=tmp_path))
