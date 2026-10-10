import json

import pytest

from image3d_scenegraph.geometry.rig_pairing import (
    adjacency_pairs,
    capture_metadata,
    write_adjacency,
)
from scripts.prepare_eyeful_capture_metadata import prepare


def data():
    return {
        "schema_version": 1,
        "images": {
            f"{camera}/{frame}.jpg": {"camera_id": camera, "capture_index": frame}
            for camera in ("10", "11")
            for frame in (0, 1, 2, 30)
        },
    }


def test_capture_pairs_cover_within_camera_cross_camera_and_adjacent_captures(tmp_path):
    value = data()
    capture_metadata(value, set(value["images"]))
    pairs = adjacency_pairs(value)
    assert ("10/0.jpg", "11/0.jpg") in pairs
    assert ("10/0.jpg", "11/1.jpg") in pairs
    assert ("10/0.jpg", "10/30.jpg") in pairs
    assert ("10/0.jpg", "11/30.jpg") not in pairs
    assert len(pairs) == len(set(pairs)) and all(a < b for a, b in pairs)
    shuffled = {
        "schema_version": 1,
        "images": dict(reversed(list(value["images"].items()))),
    }
    assert adjacency_pairs(shuffled) == pairs
    record = write_adjacency(tmp_path / "pairs.txt", value)
    assert record["adjacency_pair_count"] == len(pairs)
    with pytest.raises(FileExistsError):
        write_adjacency(tmp_path / "pairs.txt", value)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "folder", "negative", "pose"])
def test_capture_metadata_rejects_ambiguous_or_extra_information(kind):
    value = data()
    names = set(value["images"])
    row = value["images"]["10/1.jpg"]
    if kind == "missing":
        value["images"].pop("10/1.jpg")
    elif kind == "duplicate":
        row["capture_index"] = 0
    elif kind == "folder":
        row["camera_id"] = "wrong"
    elif kind == "negative":
        row["capture_index"] = -1
    else:
        row["pose"] = [1, 0, 0]
    with pytest.raises(ValueError):
        capture_metadata(value, names)


def test_eyeful_export_uses_train_capture_identity_not_poses(tmp_path):
    names = ["10/10_DSC0001", "10/10_DSC0010", "11/11_DSC0001", "11/11_DSC0010"]
    for name in names:
        p = tmp_path / "images-jpeg-2k" / (name + ".jpg")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture")
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": names, "test": ["17/17_DSC0001"]})
    )
    records = [
        {
            "cameraId": name,
            "frameId": 0 if name.endswith("0001") else 1,
            "K": "must not use",
            "T": "must not use",
        }
        for name in names[:-1]
    ]
    (tmp_path / "cameras.json").write_text(json.dumps({"KRT": records}))
    result = prepare(tmp_path)
    assert result["images"]["11/11_DSC0010.jpg"]["capture_index"] == 1
    assert set(result["images"]) == {n + ".jpg" for n in names}
    assert "must not use" not in json.dumps(result)
