from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from image3d_scenegraph.gaussian.export import PLY_FIELDS, read_gaussian_ply, write_binary_ply


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import filter_gaussian_sor as sor  # noqa: E402


def logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def gaussian_row(x: float, y: float, z: float, opacity: float) -> np.ndarray:
    row = np.zeros(len(PLY_FIELDS), dtype=np.float32)
    row[0:3] = (x, y, z)
    row[54] = logit(opacity)
    row[55:58] = np.log(0.01)
    row[58] = 1.0
    return row


def scene_rows() -> np.ndarray:
    # Dense 4x4x4 grid at the origin plus two isolated outliers: one faint
    # (free-space haze) and one high-opacity (legitimate isolated content).
    rows = [
        gaussian_row(x * 0.1, y * 0.1, z * 0.1, 0.9)
        for x in range(4)
        for y in range(4)
        for z in range(4)
    ]
    rows.append(gaussian_row(5.0, 0.0, 0.0, 0.01))
    rows.append(gaussian_row(0.0, 5.0, 0.0, 0.9))
    return np.stack(rows)


def test_full_variant_removes_all_isolated_outliers():
    rows = scene_rows()

    kept, keep = sor.apply_sor_filter(rows, nb_neighbors=6, std_ratio=1.0)

    assert keep.shape == (len(rows),)
    assert int(keep.sum()) == 64
    assert kept.shape == (64, len(PLY_FIELDS))
    assert not keep[64]
    assert not keep[65]


def test_band_variant_protects_high_opacity_outliers():
    rows = scene_rows()

    kept, keep = sor.apply_sor_filter(
        rows, nb_neighbors=6, std_ratio=1.0, band_opacity=0.05
    )

    assert int(keep.sum()) == 65
    assert not keep[64]  # faint outlier removed
    assert keep[65]  # high-opacity outlier protected


def test_removal_limit_rejects_overaggressive_filter():
    keep = np.zeros(100, dtype=bool)
    keep[:40] = True

    with pytest.raises(ValueError, match="safety limit"):
        sor.check_removal_limit(keep, 0.5)

    sor.check_removal_limit(keep, 0.61)  # 0.60 removed is within a 0.61 limit


@pytest.fixture()
def snapshot_path(tmp_path):
    torch = pytest.importorskip("torch")
    rows = scene_rows()
    count = len(rows)
    payload = {
        "max_sh_degree": 0,
        "state_dict": {
            "means": torch.from_numpy(rows[:, :3].astype(np.float32)),
            "log_scales": torch.from_numpy(rows[:, 55:58].astype(np.float32)),
            "quats": torch.from_numpy(
                np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
            ),
            "opacity_logits": torch.from_numpy(rows[:, 54].astype(np.float32)),
            "sh_coeffs": torch.zeros(count, 1, 3),
        },
    }
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path


def test_model_snapshot_mode_filters_rows_and_writes_records(
    tmp_path, snapshot_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    out_dir = tmp_path / "filtered"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_gaussian_sor.py",
            "--model-snapshot",
            str(snapshot_path),
            "--output-dir",
            str(out_dir),
            "--nb-neighbors",
            "6",
            "--std-ratio",
            "1.0",
            "--band-opacity",
            "0.05",
        ],
    )

    sor.main()

    record = json.loads((out_dir / "filter-record.json").read_text(encoding="utf-8"))
    assert record["variant"] == "band"
    assert record["input_count"] == 66
    assert record["kept_count"] == 65
    assert record["removed_count"] == 1
    for key in (
        "source_model_sha256",
        "filtered_model",
        "filtered_model_sha256",
        "mask_sha256",
    ):
        assert record[key]
    assert (out_dir / "filter-mask.npz").is_file()

    from image3d_scenegraph.gaussian.evaluation import load_model_snapshot

    filtered = load_model_snapshot(out_dir / "filtered-model.pt", torch.device("cpu"))
    assert filtered.means.shape[0] == 65
    assert filtered.max_sh_degree == 0


def test_write_binary_ply_round_trips_through_reader(tmp_path):
    rows = np.arange(5 * len(PLY_FIELDS), dtype=np.float32).reshape(5, len(PLY_FIELDS))

    path = tmp_path / "round_trip.ply"
    write_binary_ply(path, rows)
    fields = read_gaussian_ply(path)
    recovered = np.column_stack([fields[name] for name in PLY_FIELDS])

    assert np.array_equal(recovered, rows)
