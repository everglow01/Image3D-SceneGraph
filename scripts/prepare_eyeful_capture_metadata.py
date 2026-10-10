"""Export Train-only capture identities; never import Eyeful camera poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image3d_scenegraph.geometry.rig_pairing import capture_metadata


def prepare(root: Path) -> dict:
    splits = json.loads((root / "splits.json").read_text())
    train, test = set(splits["train"]), set(splits["test"])
    if train & test:
        raise ValueError("official Train and Test overlap")
    records = json.loads((root / "cameras.json").read_text())["KRT"]
    captures = {}
    for row in records:
        if row["cameraId"] not in train:
            continue
        camera, filename = row["cameraId"].split("/")
        if not filename.startswith(camera + "_"):
            raise ValueError("unrecognized Eyeful capture name")
        token = filename[len(camera) + 1 :]
        index = row["frameId"]
        if token in captures and captures[token] != index:
            raise ValueError("Eyeful synchronized capture identifiers disagree")
        captures[token] = index
    images = {}
    for name in sorted(train):
        camera, filename = name.split("/")
        if not filename.startswith(camera + "_"):
            raise ValueError("unrecognized Eyeful capture name")
        token = filename[len(camera) + 1 :]
        if (
            token not in captures
            or not (root / "images-jpeg-2k" / (name + ".jpg")).is_file()
        ):
            raise ValueError("missing Train image or unverified capture identity")
        images[name + ".jpg"] = {"camera_id": camera, "capture_index": captures[token]}
    return capture_metadata({"schema_version": 1, "images": images}, set(images))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.dataset_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True)


if __name__ == "__main__":
    main()
