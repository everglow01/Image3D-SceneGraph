from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from image3d_scenegraph.gaussian.dataset import contract_hash, sha256_file
from image3d_scenegraph.gaussian.export import (
    GaussianExportError,
    PLY_FIELDS,
    _camera_path,
    _model_rows,
    _scene_frame,
    write_binary_ply,
    export_gaussians,
    read_gaussian_ply,
    write_deterministic_zip,
)
from image3d_scenegraph.gaussian.model import GaussianModel


def model() -> GaussianModel:
    return GaussianModel.from_points(
        torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.full((2,), 0.1),
    )


def contract() -> dict:
    images = []
    for index in range(12):
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = [np.cos(index), np.sin(index), 0.0]
        camera_from_world = np.linalg.inv(world_from_camera)
        images.append(
            {
                "image_id": str(index),
                "path": f"images/{index}.png",
                "width": 64,
                "height": 64,
                "sha256": "a" * 64,
                "intrinsic": [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
                "distortion": {"state": "none", "model": "PINHOLE", "params": []},
                "camera_from_world": camera_from_world.tolist(),
                "world_from_camera": world_from_camera.tolist(),
            }
        )
    value = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "coordinate_system": {
            "camera_convention": "opencv",
            "camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "world_frame": "raw",
            "world_units": "arbitrary",
            "raw_from_world": np.eye(4).tolist(),
            "world_from_raw": np.eye(4).tolist(),
        },
        "normalization": {
            "method": "fixture",
            "center_world": [0.0, 0.0, 0.0],
            "radius_world": 1.0,
            "normalized_from_world": np.eye(4).tolist(),
            "world_from_normalized": np.eye(4).tolist(),
        },
        "source": {"camera_format": "fixture", "camera_path": "cameras.json", "camera_sha256": "b" * 64, "image_root": "images"},
        "images": images,
        "splits": {
            "train": [str(index) for index in range(8)],
            "validation": ["8", "9"],
            "test": ["10", "11"],
        },
        "initialization": {"coordinate_frame": "world", "asset": None, "sha256": None},
    }
    value["dataset_hash"] = contract_hash(value)
    return value


def test_canonical_ply_round_trips_all_owned_attributes(tmp_path):
    gaussian = model()
    rows = _model_rows(gaussian)
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"

    write_binary_ply(first, rows)
    write_binary_ply(second, rows)
    decoded = read_gaussian_ply(first)

    assert first.read_bytes() == second.read_bytes()
    assert tuple(decoded) == PLY_FIELDS
    assert np.array_equal(decoded["x"], gaussian.means[:, 0].detach().numpy())
    assert np.array_equal(decoded["f_dc_0"], gaussian.sh_coeffs[:, 0, 0].detach().numpy())
    assert np.array_equal(decoded["opacity"], gaussian.opacity_logits.detach().numpy())
    assert np.array_equal(decoded["scale_2"], gaussian.log_scales[:, 2].detach().numpy())
    assert np.allclose(
        decoded["rot_0"],
        gaussian.activated()[1][:, 0].detach().numpy(),
    )


def test_scene_frame_uses_robust_center_and_radius():
    inliers = [[index / 100.0, 0.0, 0.0] for index in range(20)]
    gaussian = GaussianModel.from_points(
        torch.tensor([*inliers, [100.0, 0.0, 0.0]]),
        torch.ones((21, 3)),
        torch.full((21,), 0.1),
    )

    center, radius = _scene_frame(gaussian)

    assert center == pytest.approx([0.1, 0.0, 0.0])
    assert radius == pytest.approx(0.1)


def test_deterministic_bundle_has_safe_sorted_entries(tmp_path):
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    write_deterministic_zip(first, {"z/b.txt": source_b, "a.txt": source_a})
    write_deterministic_zip(second, {"a.txt": source_a, "z/b.txt": source_b})

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt", "z/b.txt"]
    with pytest.raises(GaussianExportError, match="unsafe bundle"):
        write_deterministic_zip(tmp_path / "bad.zip", {"../escape": source_a})


def test_filtered_export_verifies_and_bundles_postprocess_provenance(tmp_path):
    gaussian = model()
    model_path = tmp_path / "filtered-model.pt"
    torch.save(
        {
            "state_dict": gaussian.state_dict(),
            "max_sh_degree": gaussian.max_sh_degree,
        },
        model_path,
    )
    value = contract()
    config_hash = "c" * 64
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "provenance": {
                    "dataset_hash": value["dataset_hash"],
                    "effective_config_hash": config_hash,
                    "model_sha256": sha256_file(model_path),
                }
            }
        ),
        encoding="utf-8",
    )
    mask_path = tmp_path / "filter-mask.npz"
    np.savez_compressed(mask_path, keep=np.array([True, True]))
    record_path = tmp_path / "diagnostics.json"
    record_path.write_text(
        json.dumps(
            {
                "profile": "vggt_visibility_v1",
                "source_model_sha256": "d" * 64,
                "filtered_model_sha256": sha256_file(model_path),
                "mask_sha256": sha256_file(mask_path),
                "counts": {"input": 2, "kept": 2, "removed": 0},
            }
        ),
        encoding="utf-8",
    )

    original = export_gaussians(
        model_path=model_path,
        contract=value,
        config_record={"effective_config_hash": config_hash},
        evaluation_path=evaluation_path,
        output_dir=tmp_path / "original-export",
    )
    assert "postprocess" not in original

    result = export_gaussians(
        model_path=model_path,
        contract=value,
        config_record={"effective_config_hash": config_hash},
        evaluation_path=evaluation_path,
        output_dir=tmp_path / "export",
        postprocess_record_path=record_path,
        postprocess_mask_path=mask_path,
    )

    assert result["postprocess"]["profile"] == "vggt_visibility_v1"
    with zipfile.ZipFile(tmp_path / "export" / "result.zip") as archive:
        assert "postprocess/diagnostics.json" in archive.namelist()
        assert "postprocess/filter-mask.npz" in archive.namelist()


def test_filtered_export_rejects_mismatched_postprocess_hash(tmp_path):
    gaussian = model()
    model_path = tmp_path / "filtered-model.pt"
    torch.save(
        {
            "state_dict": gaussian.state_dict(),
            "max_sh_degree": gaussian.max_sh_degree,
        },
        model_path,
    )
    value = contract()
    config_hash = "c" * 64
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "provenance": {
                    "dataset_hash": value["dataset_hash"],
                    "effective_config_hash": config_hash,
                    "model_sha256": sha256_file(model_path),
                }
            }
        ),
        encoding="utf-8",
    )
    mask_path = tmp_path / "filter-mask.npz"
    np.savez_compressed(mask_path, keep=np.array([True, True]))
    record_path = tmp_path / "diagnostics.json"
    record_path.write_text(
        json.dumps(
            {
                "filtered_model_sha256": sha256_file(model_path),
                "mask_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GaussianExportError, match="mask hash mismatch"):
        export_gaussians(
            model_path=model_path,
            contract=value,
            config_record={"effective_config_hash": config_hash},
            evaluation_path=evaluation_path,
            output_dir=tmp_path / "export",
            postprocess_record_path=record_path,
            postprocess_mask_path=mask_path,
        )


def test_camera_path_stays_in_normalized_trusted_bound():
    value = contract()
    path = _camera_path(value)
    assert len(path["keyframes"]) == 2

    value["images"][8]["world_from_camera"][0][3] = 3.0
    with pytest.raises(GaussianExportError, match="trusted"):
        _camera_path(value)
