from __future__ import annotations

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


@pytest.mark.parametrize("layout", ["contiguous", "strided", "fortran", "big_endian", "empty"])
def test_streamed_ply_matches_previous_writer_byte_for_byte(tmp_path, layout):
    rows = np.arange(8 * len(PLY_FIELDS), dtype=np.float32).reshape(8, len(PLY_FIELDS)) / 7
    if layout == "strided":
        rows = rows[::-2, ::-1]
    elif layout == "fortran":
        rows = np.asfortranarray(rows)
    elif layout == "big_endian":
        rows = rows.astype(">f8")
    elif layout == "empty":
        rows = rows[:0]
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        "comment Image3D-SceneGraph canonical Gaussian schema v1",
        f"element vertex {len(rows)}",
        *(f"property float {name}" for name in PLY_FIELDS), "end_header", "",
    ]).encode("ascii")
    expected = header + rows.astype("<f4", copy=False).tobytes(order="C")
    path = tmp_path / "scene.ply"
    write_binary_ply(path, rows)
    assert path.read_bytes() == expected
    with pytest.raises(FileExistsError):
        write_binary_ply(path, rows)


def test_streamed_zip_matches_previous_writer_without_whole_file_reads(tmp_path, monkeypatch):
    source = tmp_path / "scene.ply"
    source.write_bytes(bytes(range(256)) * (32 * 1024 + 1))
    empty = tmp_path / "empty"
    empty.touch()
    entries = {"gaussian/scene.ply": source, "empty": empty}
    previous = tmp_path / "previous.zip"
    with zipfile.ZipFile(previous, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name].read_bytes())

    def reject_whole_file_read(_path):
        raise AssertionError("bundle sources must be read in bounded chunks")

    current = tmp_path / "current.zip"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", reject_whole_file_read)
        write_deterministic_zip(current, entries)
    assert current.read_bytes() == previous.read_bytes()
    with zipfile.ZipFile(current) as archive:
        assert archive.read("gaussian/scene.ply") == source.read_bytes()
        assert archive.read("empty") == b""
    with pytest.raises(GaussianExportError, match="already exists"):
        write_deterministic_zip(current, entries)
    write_deterministic_zip(current, entries, overwrite=True)
    assert current.read_bytes() == previous.read_bytes()


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
    metadata = json.loads(
        (tmp_path / "original-export" / "export.json").read_text()
    )
    assert metadata["schema_version"] == 2
    assert metadata["sh_degree"] == metadata["model_sh_degree"] == 3
    assert metadata["browser_renderer"] == {
        "implementation": "@mkkellogg/gaussian-splats-3d",
        "version": "0.4.7",
        "requested_sh_degree": 3,
        "effective_sh_degree": 2,
        "verified_max_sh_degree": 2,
        "verification_profile": "fixed_camera_browser_v1",
    }
    assert json.loads(
        (tmp_path / "original-export" / "camera_path.json").read_text()
    )["schema_version"] == 1
    assert json.loads(
        (tmp_path / "original-export" / "bundle.json").read_text()
    )["schema_version"] == 1

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


@pytest.mark.parametrize("train_only", [False, True])
def test_final_fit_export_preserves_selection_sor_lineage(tmp_path, train_only):
    profile = "train_only_control_v1" if train_only else "train_validation_v1"
    gaussian = model()
    source_model = tmp_path / "selection-sor.pt"
    final_model = tmp_path / "final-fit.pt"
    payload = {
        "state_dict": gaussian.state_dict(),
        "max_sh_degree": gaussian.max_sh_degree,
    }
    torch.save(payload, source_model)
    torch.save(payload, final_model)
    value = contract()
    config_hash = "c" * 64
    evaluation_path = tmp_path / "fit-evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "split": "control_validation" if train_only else "fit_validation",
                "quality_role": "held_out_after_train_only_control" if train_only else "in_sample_after_train_validation_fit",
                "selection_eligible": False,
                "provenance": {
                    "dataset_hash": value["dataset_hash"],
                    "effective_config_hash": config_hash,
                    "model_sha256": sha256_file(final_model),
                },
            }
        ),
        encoding="utf-8",
    )
    mask_path = tmp_path / "filter-mask.npz"
    np.savez_compressed(mask_path, keep=np.array([True, True]))
    sor_record = tmp_path / "sor-record.json"
    sor_record.write_text(
        json.dumps(
            {
                "profile": "sor_v1",
                "source_model_sha256": "d" * 64,
                "filtered_model_sha256": sha256_file(source_model),
                "mask_sha256": sha256_file(mask_path),
                "counts": {"input": 2, "kept": 2, "removed": 0},
            }
        ),
        encoding="utf-8",
    )
    final_fit_record = tmp_path / "final-fit-record.json"
    final_fit_record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": profile,
                "input_splits": ["train"] if train_only else ["train", "validation"],
                "profile_hash": "e" * 64,
                "status": "complete",
                "dataset_hash": value["dataset_hash"],
                "effective_config_hash": config_hash,
                "source_model_sha256": sha256_file(source_model),
                "final_model_sha256": sha256_file(final_model),
                "evaluation_sha256": sha256_file(evaluation_path),
                "test_rgb": "not_loaded",
                "topology_changed": False,
            }
        ),
        encoding="utf-8",
    )

    result = export_gaussians(
        model_path=final_model,
        contract=value,
        config_record={"effective_config_hash": config_hash},
        evaluation_path=evaluation_path,
        output_dir=tmp_path / "export",
        postprocess_record_path=sor_record,
        postprocess_mask_path=mask_path,
        final_fit_record_path=final_fit_record,
    )

    assert result["final_fit"]["profile"] == profile
    assert result["final_fit"]["source_model_sha256"] == sha256_file(source_model)
    with zipfile.ZipFile(tmp_path / "export" / "result.zip") as archive:
        assert "postprocess/diagnostics.json" in archive.namelist()
        assert "final-fit/record.json" in archive.namelist()

    broken = json.loads(final_fit_record.read_text())
    broken["source_model_sha256"] = "0" * 64
    final_fit_record.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(GaussianExportError, match="filtered model hash mismatch"):
        export_gaussians(
            model_path=final_model,
            contract=value,
            config_record={"effective_config_hash": config_hash},
            evaluation_path=evaluation_path,
            output_dir=tmp_path / "broken-export",
            postprocess_record_path=sor_record,
            postprocess_mask_path=mask_path,
            final_fit_record_path=final_fit_record,
        )


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
