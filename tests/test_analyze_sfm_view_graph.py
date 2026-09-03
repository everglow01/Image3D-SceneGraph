from __future__ import annotations

import gzip
import hashlib
import json

from scripts.analyze_sfm_view_graph import analyze_job


def test_analyze_job_reads_legacy_diagnostics_without_writing(tmp_path) -> None:
    job = tmp_path / "job"
    diagnostics = job / "diagnostics" / "sfm"
    diagnostics.mkdir(parents=True)
    pair_index = diagnostics / "pairs.json.gz"
    pair_index.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "schema_version": 1,
                    "pairs": [
                        {
                            "pair_key": "1-2",
                            "image_ids": [1, 2],
                            "candidate_match_count": 4,
                            "inlier_count": 3,
                            "geometric_config": 3,
                            "detail_shard": "diagnostics/sfm/pair.json.gz",
                        }
                    ],
                }
            ).encode(),
            mtime=0,
        )
    )
    (diagnostics / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_run_id": "run-1",
                "runs": [
                    {
                        "run_id": "run-1",
                        "pair_index_path": "diagnostics/sfm/pairs.json.gz",
                    }
                ],
                "images": [
                    {"colmap_image_id": 1, "registered": True},
                    {"colmap_image_id": 2, "registered": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(job): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in job.rglob("*")
        if path.is_file()
    }

    summary = analyze_job(job)

    after = {
        path.relative_to(job): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in job.rglob("*")
        if path.is_file()
    }
    assert summary["verified_edge_count"] == 1
    assert summary["match_totals"]["guided_inliers"] == 0
    assert before == after
