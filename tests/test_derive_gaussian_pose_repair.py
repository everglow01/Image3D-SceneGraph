from __future__ import annotations

import json
from pathlib import Path

from image3d_scenegraph.gaussian.dataset import (
    build_colmap_contract,
    validate_contract,
)
from image3d_scenegraph.gaussian.replay import validate_replay_bundle
from scripts.derive_gaussian_pose_repair import derive_gaussian_pose_repair
from test_gaussian_dataset import write_colmap_fixture


def test_pose_repair_derives_new_replay_without_mutating_sources(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    write_colmap_fixture(source, count=20)
    contract = build_colmap_contract(
        dataset_id="repair-parent",
        dataset_root=source,
        image_root="images",
        cameras_path="cameras.json",
    )
    dataset_path = source / "dataset.json"
    dataset_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    excluded = int(contract["splits"]["train"][0])
    retained = [
        int(image_id)
        for image_id in contract["splits"]["train"]
        if int(image_id) != excluded
    ][:4]
    points = source / "points3D.txt"
    points.write_text(
        "# points\n"
        + _point_row(1, [excluded, *retained[:2]])
        + _point_row(2, [excluded, *retained[:3]])
        + _point_row(3, retained[:3]),
        encoding="utf-8",
    )
    source_bytes = {
        path: path.read_bytes()
        for path in (dataset_path, source / "cameras.json", points)
    }
    output = tmp_path / "derived"

    record = derive_gaussian_pose_repair(
        dataset_contract=dataset_path,
        dataset_root=source,
        points_path=points,
        excluded_image_ids={excluded},
        output_dir=output,
        max_initial_points=10,
    )

    assert record["status"] == "repaired_derivative"
    assert record["training_started"] is False
    assert record["policy"]["point_coordinates"] == (
        "preserved_parent_sparse_no_bundle_adjustment"
    )
    assert record["excluded_images"][0]["image_id"] == str(excluded)
    assert record["counts"]["points_removed_track_support"] == 1
    assert record["counts"]["observations_removed"] == 2
    assert all(path.read_bytes() == content for path, content in source_bytes.items())

    replay_record = validate_replay_bundle(output / "replay")
    derived = json.loads(
        (output / "replay" / "dataset.json").read_text(encoding="utf-8")
    )
    validate_contract(derived, output / "replay")
    assert replay_record["dataset_hash"] == derived["dataset_hash"]
    assert derived["dataset_hash"] != contract["dataset_hash"]
    assert derived["normalization"] != contract["normalization"]
    assert str(excluded) not in {
        str(image["image_id"]) for image in derived["images"]
    }
    for split in ("train", "validation", "test"):
        assert derived["splits"][split] == [
            image_id
            for image_id in contract["splits"][split]
            if str(image_id) != str(excluded)
        ]

    filtered_camera = json.loads(
        (output / "replay" / "cameras.json").read_text(encoding="utf-8")
    )
    assert str(excluded) not in {
        str(image["image_id"]) for image in filtered_camera["images"]
    }
    filtered_points = (
        output / "preparation" / "repair" / "points3D.txt"
    ).read_text(encoding="utf-8")
    filtered_rows = [
        line.split()
        for line in filtered_points.splitlines()
        if line and not line.startswith("#")
    ]
    assert all(
        excluded not in {int(value) for value in row[8::2]}
        for row in filtered_rows
    )
    assert json.loads((output / "repair.json").read_text(encoding="utf-8")) == record


def _point_row(point_id: int, image_ids: list[int]) -> str:
    tracks = " ".join(f"{image_id} {point_id}" for image_id in image_ids)
    return f"{point_id} {point_id} 0 2 255 0 0 0.2 {tracks}\n"
