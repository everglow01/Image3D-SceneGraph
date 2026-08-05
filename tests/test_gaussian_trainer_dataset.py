from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from image3d_scenegraph.gaussian.dataset import build_colmap_contract
from image3d_scenegraph.gaussian.initialization import InitializationResult
from image3d_scenegraph.gaussian.trainer_dataset import prepare_external_dataset


def _dataset(root: Path) -> tuple[dict, InitializationResult]:
    images = root / "images"
    images.mkdir(parents=True)
    camera = {
        "camera_id": 1,
        "model": "PINHOLE",
        "width": 64,
        "height": 64,
        "params": [50.0, 50.0, 32.0, 32.0],
    }
    records = []
    for index in range(12):
        Image.new("RGB", (64, 64), (index, 10, 20)).save(images / f"{index:02d}.png")
        angle = 2 * np.pi * index / 12
        center = np.array([np.cos(angle), np.sin(angle), 0.1])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        translation = -rotation @ center
        # Valid rotation here is simple enough to reuse the smoke helper convention.
        from scripts.smoke_gaussian_trainer import rotmat_to_qvec

        records.append(
            {
                "image_id": index + 1,
                "qvec": rotmat_to_qvec(rotation),
                "tvec": translation.tolist(),
                "camera_id": 1,
                "name": f"{index:02d}.png",
            }
        )
    (root / "cameras.json").write_text(
        json.dumps(
            {"coordinate_system": "colmap_world", "cameras": [camera], "images": records}
        )
    )
    contract = build_colmap_contract(
        dataset_id="trainer-dataset-test",
        dataset_root=root,
        image_root="images",
        cameras_path="cameras.json",
    )
    initialization = InitializationResult(
        points=np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32),
        colors=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        scales=np.array([0.1, 0.1], dtype=np.float32),
        diagnostics={},
    )
    return contract, initialization


def test_external_datasets_preserve_frozen_splits_and_points(tmp_path):
    root = tmp_path / "source"
    contract, initialization = _dataset(root)

    graph = tmp_path / "graph"
    graph_record = prepare_external_dataset(
        trainer="graphdeco",
        contract=contract,
        dataset_root=root,
        initialization=initialization,
        output_dir=graph,
    )
    nerfstudio = tmp_path / "nerfstudio"
    ns_record = prepare_external_dataset(
        trainer="nerfstudio",
        contract=contract,
        dataset_root=root,
        initialization=initialization,
        output_dir=nerfstudio,
    )

    assert graph_record["splits"] == ns_record["splits"] == contract["splits"]
    assert graph_record["initialization_points_sha256"] == ns_record[
        "initialization_points_sha256"
    ]
    expected_validation = {
        Path(image["path"]).name
        for image in contract["images"]
        if image["image_id"] in contract["splits"]["validation"]
    }
    assert set((graph / "sparse/0/test.txt").read_text().splitlines()) == expected_validation
    transforms = json.loads((nerfstudio / "transforms.json").read_text())
    assert set(transforms["val_filenames"]) == {
        f"images/{name}" for name in expected_validation
    }
    assert len(transforms["test_filenames"]) == len(contract["splits"]["test"])
