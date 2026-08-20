from __future__ import annotations

import numpy as np
import torch

from image3d_scenegraph.gaussian.evaluation import load_model_snapshot
from image3d_scenegraph.gaussian.export import PLY_FIELDS, write_binary_ply
from image3d_scenegraph.gaussian.importer import import_inria_ply


def test_import_inria_ply_applies_similarity_to_means_and_scales(tmp_path):
    rows = np.zeros((1, len(PLY_FIELDS)), dtype=np.float32)
    rows[0, :3] = [1.0, 2.0, 3.0]
    rows[0, 6:9] = [0.1, 0.2, 0.3]
    rows[0, 54] = 0.0
    rows[0, 55:58] = np.log([0.1, 0.2, 0.3])
    rows[0, 58] = 1.0
    source = tmp_path / "native.ply"
    write_binary_ply(source, rows)
    transform = np.eye(4)
    transform[:3, :3] *= 2.0
    transform[:3, 3] = [1.0, -1.0, 0.5]
    destination = tmp_path / "model.pt"

    record = import_inria_ply(
        source,
        destination,
        normalized_from_source=transform,
        trainer={"id": "graphdeco"},
    )
    model = load_model_snapshot(destination, torch.device("cpu"))

    assert record["gaussian_count"] == 1
    assert torch.allclose(model.means, torch.tensor([[3.0, 3.0, 6.5]]))
    assert torch.allclose(model.log_scales.exp(), torch.tensor([[0.2, 0.4, 0.6]]))
    assert torch.allclose(model.quats, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
