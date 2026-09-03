from __future__ import annotations

import pytest

from image3d_scenegraph.geometry.view_graph import ViewGraphError, summarize_view_graph


def test_summarize_view_graph_reports_components_guided_matches_and_video_gaps() -> None:
    images = [
        {
            "colmap_image_id": 1,
            "registered": True,
            "source_time_seconds": 0.0,
        },
        {
            "colmap_image_id": 2,
            "registered": False,
            "source_time_seconds": 1.0,
        },
        {
            "colmap_image_id": 3,
            "registered": True,
            "source_time_seconds": 4.0,
        },
        {
            "colmap_image_id": 4,
            "registered": False,
            "source_time_seconds": 5.0,
        },
    ]
    pairs = [
        {
            "image_ids": [1, 3],
            "candidate_match_count": 10,
            "candidate_inlier_count": 6,
            "guided_inlier_count": 2,
            "inlier_count": 8,
            "outlier_count": 4,
            "geometric_config": 3,
        },
        {
            "image_ids": [1, 2],
            "candidate_match_count": 5,
            "candidate_inlier_count": 2,
            "guided_inlier_count": 0,
            "inlier_count": 2,
            "outlier_count": 3,
            "geometric_config": 2,
        },
        {
            "image_ids": [3, 4],
            "candidate_match_count": 0,
            "candidate_inlier_count": 0,
            "guided_inlier_count": 0,
            "inlier_count": 0,
            "outlier_count": 0,
            "geometric_config": 0,
        },
    ]

    summary = summarize_view_graph(images, pairs)

    assert summary["verified_edge_count"] == 2
    assert summary["connected_component_count"] == 2
    assert summary["largest_component_node_count"] == 3
    assert summary["isolated_node_count"] == 1
    assert summary["degree_one_node_count"] == 2
    assert summary["match_totals"] == {
        "candidate": 15,
        "candidate_inliers": 8,
        "guided_inliers": 2,
        "verified": 10,
        "outliers": 7,
    }
    assert summary["geometric_config_counts"] == {"0": 1, "2": 1, "3": 1}
    assert summary["degree_distribution"]["p90"] == pytest.approx(1.7)
    assert summary["video"]["registered_gap_count"] == 1
    assert summary["video"]["directly_bridged_registered_gap_count"] == 1
    assert summary["video"]["verified_edge_time_span_seconds"]["max"] == 4.0


def test_summarize_view_graph_reads_legacy_pair_counts() -> None:
    summary = summarize_view_graph(
        [{"colmap_image_id": 1}, {"colmap_image_id": 2}],
        [
            {
                "image_ids": [1, 2],
                "candidate_match_count": 4,
                "inlier_count": 3,
                "geometric_config": 3,
            }
        ],
    )

    assert summary["match_totals"] == {
        "candidate": 4,
        "candidate_inliers": 3,
        "guided_inliers": 0,
        "verified": 3,
        "outliers": 1,
    }


def test_summarize_view_graph_rejects_inconsistent_counts() -> None:
    with pytest.raises(ViewGraphError, match="inconsistent"):
        summarize_view_graph(
            [{"colmap_image_id": 1}, {"colmap_image_id": 2}],
            [
                {
                    "image_ids": [1, 2],
                    "candidate_match_count": 1,
                    "candidate_inlier_count": 1,
                    "guided_inlier_count": 1,
                    "inlier_count": 1,
                    "outlier_count": 0,
                    "geometric_config": 3,
                }
            ],
        )
