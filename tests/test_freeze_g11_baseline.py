from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "freeze_g11_baseline.py"
SPEC = importlib.util.spec_from_file_location("freeze_g11_baseline", SCRIPT_PATH)
assert SPEC and SPEC.loader
FREEZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZER)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "job"
    image_paths = ["input/images/a.jpg", "input/images/b.jpg"]
    for relative_path, content in zip(image_paths, (b"image-a", b"image-b"), strict=True):
        path = job_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manifest = {
        "job_id": "test-job",
        "inputs": [
            {
                "filename": Path(relative_path).name,
                "path": relative_path,
                "size_bytes": (job_dir / relative_path).stat().st_size,
            }
            for relative_path in image_paths
        ],
        "metrics": {
            "num_inputs": 2,
            "registered_images": 2,
            "scaled_images": 2,
            "num_points": 15,
            "colmap_points": 4,
            "matcher": "exhaustive",
            "vggt_batch_size": 4,
            "vggt_overlap_size": 2,
            "overlap_size": 0,
            "vggt_grouping": "sequential",
            "fusion_mode": "points",
            "fusion_intrinsics": "colmap",
            "conf_percentile": 50.0,
            "confidence_threshold_scope": "per_frame",
            "consistency_support_policy": "adaptive_two",
            "max_points": 30,
            "point_budget_policy": "spatial_balanced",
            "point_budget_input_points": 15,
            "point_budget_output_points": 15,
            "point_budget_applied": False,
            "num_groups": 1,
            "colmap_seconds": 2.0,
            "vggt_seconds": 1.0,
            "elapsed_seconds": 4.0,
        },
    }
    write_json(job_dir / "manifest.json", manifest)

    run_values = {
        "job_id": "test-job",
        "num_inputs": "2",
        "registered_images": "2",
        "scaled_images": "2",
        "num_points": "15",
        "matcher": "exhaustive",
        "vggt_batch_size": "4",
        "vggt_overlap_size": "2",
        "overlap_size": "0",
        "vggt_grouping": "sequential",
        "fusion_mode": "points",
        "fusion_intrinsics": "colmap",
        "conf_percentile": "50.0",
        "confidence_threshold_scope": "per_frame",
        "consistency_support_policy": "adaptive_two",
        "max_points": "30",
        "point_budget_policy": "spatial_balanced",
        "point_budget_input_points": "15",
        "point_budget_output_points": "15",
        "point_budget_applied": "false",
        "num_groups": "1",
    }
    run_log = [
        "runner=python scripts/run_colmap_vggt_dense.py --image-dir input/images",
        *(f"{key}={value}" for key, value in run_values.items()),
    ]
    path = job_dir / "logs/run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(run_log) + "\n", encoding="utf-8")

    fusion = {
        "intrinsics_source": "colmap",
        "registered_images": 2,
        "point_budget": {
            "policy": "spatial_balanced",
            "input_points": 15,
            "output_points": 15,
            "applied": False,
        },
        "cross_view_filter": {
            "support_policy": "adaptive_two",
            "confidence_threshold_scope": "per_frame",
        },
        "images": [
            {"depth_scale": 2.0, "scale_observations": 10, "scale_log_mad": 0.1},
            {"depth_scale": 4.0, "scale_observations": 20, "scale_log_mad": 0.3},
        ],
    }
    write_json(job_dir / "diagnostics/fusion.json", fusion)

    images = [
        {
            "candidate_points": 10,
            "accepted_points": 8,
            "rejected_points": 2,
            "unverified_points": 3,
            "supported_points": 5,
            "multi_visible_points": 4,
            "policy_rejected_supported_points": 1,
        },
        {
            "candidate_points": 9,
            "accepted_points": 7,
            "rejected_points": 2,
            "unverified_points": 2,
            "supported_points": 5,
            "multi_visible_points": 3,
            "policy_rejected_supported_points": 1,
        },
    ]
    consistency = {
        "support_policy": "adaptive_two",
        "confidence_threshold_scope": "per_frame",
        "candidate_points": 19,
        "accepted_points": 15,
        "rejected_points": 4,
        "unverified_points": 5,
        "supported_points": 10,
        "multi_visible_points": 7,
        "policy_rejected_supported_points": 2,
        "residual_p50": 0.01,
        "residual_p90": 0.02,
        "images": images,
    }
    write_json(job_dir / "diagnostics/consistency.json", consistency)
    write_json(
        job_dir / "diagnostics/visibility_graph.json",
        {"image_count": 2, "directed_edge_count": 2},
    )
    write_json(
        job_dir / "diagnostics/alignment.json",
        {
            "status": "aligned",
            "num_points": 15,
            "target_axis": "z",
            "translate_plane_to_zero": True,
            "source_plane": {"inlier_ratio": 0.25, "rms_distance": 0.04},
            "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
    )
    for relative_path in ("geometry/points.ply", "geometry/points_aligned.ply"):
        path = job_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode())
    return job_dir


def test_build_baseline_hashes_inputs_and_summarizes_diagnostics(tmp_path: Path) -> None:
    job_dir = make_job(tmp_path)

    baseline = FREEZER.build_baseline(job_dir)

    assert baseline["inputs"]["count"] == 2
    assert baseline["inputs"]["files"][0]["sha256"] == FREEZER.sha256_file(
        job_dir / "input/images/a.jpg"
    )
    assert len(baseline["inputs"]["inventory_sha256"]) == 64
    assert baseline["scale_distribution"]["depth_scale"]["p50"] == 3.0
    assert baseline["support_distribution"]["totals"]["accepted_points"] == 15
    assert baseline["support_distribution"]["accepted_unverified_fraction"] == pytest.approx(
        1 / 3
    )
    assert baseline["phase3_point_budget"]["inactive"] is True
    assert "not activated" in baseline["phase3_point_budget"]["conclusion"]
    assert baseline["configuration"]["effective_overlap_size"] == 0
    assert baseline["views"]["complete"] is False


def test_build_baseline_rejects_manifest_size_mismatch(tmp_path: Path) -> None:
    job_dir = make_job(tmp_path)
    manifest_path = job_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0]["size_bytes"] += 1
    write_json(manifest_path, manifest)

    with pytest.raises(FREEZER.BaselineError, match="inconsistent size"):
        FREEZER.build_baseline(job_dir)


def test_build_baseline_rejects_metric_disagreement(tmp_path: Path) -> None:
    job_dir = make_job(tmp_path)
    consistency_path = job_dir / "diagnostics/consistency.json"
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    consistency["accepted_points"] = 14
    write_json(consistency_path, consistency)

    with pytest.raises(FREEZER.BaselineError, match="inconsistent output points"):
        FREEZER.build_baseline(job_dir)


def test_serialized_baseline_is_byte_stable(tmp_path: Path) -> None:
    job_dir = make_job(tmp_path)

    first = FREEZER.serialized(FREEZER.build_baseline(job_dir))
    second = FREEZER.serialized(FREEZER.build_baseline(job_dir))

    assert first == second
