from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_pointcloud import read_ply_points_and_colors  # noqa: E402
from export_support_diagnostic_plys import (  # noqa: E402
    CATEGORY_RULES,
    DiagnosticPlyError,
    export_support_diagnostic_plys,
)
from run_vggt_pointcloud import write_ply  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(tmp_path: Path) -> dict[str, Path]:
    points_path = tmp_path / "points.ply"
    diagnostics_path = tmp_path / "support_points.npz"
    diagnostics_index_path = tmp_path / "support_points.json"
    consistency_path = tmp_path / "consistency.json"
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    write_ply(points_path, points, np.zeros((5, 3), dtype=np.uint8))
    np.savez_compressed(
        diagnostics_path,
        support_counts=np.asarray([1, 0, 2, 0, 1], dtype=np.uint16),
        visible_counts=np.asarray([1, 0, 2, 0, 1], dtype=np.uint16),
        overlap_disagreement=np.asarray(
            [np.nan, 0.02, 0.019, 0.5, np.inf], dtype=np.float32
        ),
        confidence=np.arange(5, dtype=np.float32),
    )
    diagnostics_index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "point_order": "exactly matches geometry/points.ply vertex order",
                "point_count": 5,
                "sidecar": diagnostics_path.name,
                "sidecar_sha256": sha256(diagnostics_path),
            }
        ),
        encoding="utf-8",
    )
    consistency_path.write_text(
        json.dumps(
            {
                "accepted_points": 5,
                "supported_points": 3,
                "unverified_points": 2,
                "relative_threshold": 0.02,
            }
        ),
        encoding="utf-8",
    )
    return {
        "points_path": points_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics_index_path": diagnostics_index_path,
        "consistency_path": consistency_path,
        "output_dir": tmp_path / "exports",
    }


def test_exports_category_plys_in_final_point_order(tmp_path):
    paths = write_fixture(tmp_path)

    payload = export_support_diagnostic_plys(**paths)

    expected_indices = {
        "supported": [0, 2, 4],
        "unverified": [1, 3],
        "high_disagreement": [1, 3],
    }
    source_points, _ = read_ply_points_and_colors(paths["points_path"])
    for name, indices in expected_indices.items():
        points, colors = read_ply_points_and_colors(paths["output_dir"] / f"{name}.ply")
        assert np.array_equal(points, source_points[indices])
        assert np.all(colors == np.asarray(CATEGORY_RULES[name]["color"], dtype=np.uint8))
        assert payload["exports"][name]["count"] == len(indices)

    assert payload["finite_overlap_disagreement_count"] == 3
    assert payload["consistency_count_check"]["supported"]["matches"] is True
    assert payload["protocol"]["categories_are_independent"] is True
    assert payload["exports"]["unverified"]["count"] == payload["exports"]["high_disagreement"]["count"]
    assert export_support_diagnostic_plys(**paths, check=True) == payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "SHA-256 mismatch"),
        ("order", "exact final PLY row order"),
        ("point_count", "point count differs"),
        ("consistency", "supported count"),
        ("threshold", "relative_threshold differs"),
        ("array_length", "has shape"),
    ],
)
def test_rejects_inconsistent_sources(tmp_path, mutation, message):
    paths = write_fixture(tmp_path)
    index = json.loads(paths["diagnostics_index_path"].read_text(encoding="utf-8"))
    consistency = json.loads(paths["consistency_path"].read_text(encoding="utf-8"))
    if mutation == "hash":
        index["sidecar_sha256"] = "0" * 64
    elif mutation == "order":
        index["point_order"] = "unknown"
    elif mutation == "point_count":
        index["point_count"] = 4
    elif mutation == "consistency":
        consistency["supported_points"] = 2
    elif mutation == "threshold":
        consistency["relative_threshold"] = 0.03
    else:
        np.savez_compressed(
            paths["diagnostics_path"],
            support_counts=np.ones(4, dtype=np.uint16),
            visible_counts=np.ones(5, dtype=np.uint16),
            overlap_disagreement=np.ones(5, dtype=np.float32),
        )
        index["sidecar_sha256"] = sha256(paths["diagnostics_path"])
    paths["diagnostics_index_path"].write_text(json.dumps(index), encoding="utf-8")
    paths["consistency_path"].write_text(json.dumps(consistency), encoding="utf-8")

    with pytest.raises(DiagnosticPlyError, match=message):
        export_support_diagnostic_plys(**paths)
