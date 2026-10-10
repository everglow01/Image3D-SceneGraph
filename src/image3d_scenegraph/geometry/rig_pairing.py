"""Explicit capture metadata and bounded adjacency for multi-camera stills."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path, PurePosixPath

PROFILE = "rig_neighbors_vocab_v1"
NEIGHBORS = 10
MAX_PAIRS = 1_000_000


def capture_metadata(value: str | dict, names: set[str]) -> dict:
    data = json.loads(value) if isinstance(value, str) else value
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "images"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("invalid capture metadata schema")
    images = data["images"]
    if (
        not isinstance(images, dict)
        or set(images) != names
        or not 2 <= len(images) <= 10_000
    ):
        raise ValueError("capture metadata must cover exactly the Job images")
    seen = set()
    for name, item in images.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or any(c.isspace() for c in name)
        ):
            raise ValueError(
                "capture image names must be safe relative paths without whitespace"
            )
        if not isinstance(item, dict) or set(item) != {"camera_id", "capture_index"}:
            raise ValueError(
                "capture metadata requires camera_id and capture_index only"
            )
        camera, index = item["camera_id"], item["capture_index"]
        if (
            not isinstance(camera, str)
            or not camera
            or len(camera) > 128
            or camera != path.parent.as_posix()
            or camera == "."
        ):
            raise ValueError(
                "capture camera_id must identify the image parent directory"
            )
        if (
            type(index) is not int
            or not 0 <= index <= 1_000_000_000
            or (camera, index) in seen
        ):
            raise ValueError("duplicate or invalid camera/capture index")
        seen.add((camera, index))
    if not 2 <= len({camera for camera, _ in seen}) <= 64:
        raise ValueError("rig pairing requires between 2 and 64 camera groups")
    return data


def adjacency_pairs(metadata: dict) -> list[tuple[str, str]]:
    cameras, captures = defaultdict(list), defaultdict(list)
    for name, item in metadata["images"].items():
        cameras[item["camera_id"]].append((item["capture_index"], name))
        captures[item["capture_index"]].append(name)
    pairs = set()

    def add(first, second):
        if first != second:
            pairs.add(tuple(sorted((first, second))))
        if len(pairs) > MAX_PAIRS:
            raise ValueError("rig adjacency exceeds the fixed pair budget")

    for entries in cameras.values():
        entries.sort()
        for i, (_, name) in enumerate(entries):
            for _, other in entries[i + 1 : i + 1 + NEIGHBORS]:
                add(name, other)
    capture_ids = sorted(captures)
    for i, capture in enumerate(capture_ids):
        for first, second in combinations(sorted(captures[capture]), 2):
            add(first, second)
        if i:
            for first, second in product(
                captures[capture_ids[i - 1]], captures[capture]
            ):
                add(first, second)
    return sorted(pairs)


def write_adjacency(path: Path, metadata: dict) -> dict:
    pairs = adjacency_pairs(metadata)
    with path.open("x", encoding="utf-8") as stream:
        for first, second in pairs:
            stream.write(f"{first} {second}\n")
    return {
        "profile": PROFILE,
        "same_camera_neighbors": NEIGHBORS,
        "adjacent_capture_window": 1,
        "retrieval_images": 100,
        "adjacency_pair_count": len(pairs),
        "input_image_count": len(metadata["images"]),
    }
