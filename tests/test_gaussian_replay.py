from __future__ import annotations

import json
import os

import numpy as np
import pytest

from image3d_scenegraph.gaussian.dataset import (
    build_colmap_contract,
    sha256_file,
    with_initialization,
    write_contract,
)
from image3d_scenegraph.gaussian.initialization import (
    InitializationError,
    load_frozen_initialization,
    sparse_initialization,
    write_initialization,
)
from image3d_scenegraph.gaussian.replay import (
    ReplayError,
    build_replay_bundle,
    validate_replay_bundle,
)
from test_gaussian_dataset import write_colmap_fixture


def _replay_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_colmap_fixture(source)
    points = source / "points3D.txt"
    points.write_text(
        "1 0 0 2 255 0 0 0.2 1 1 2 1 3 1\n",
        encoding="utf-8",
    )
    initialized = sparse_initialization(points, np.eye(4), max_points=10)
    initialization = source / "initialization" / "sparse.npz"
    diagnostics = initialization.with_suffix(".json")
    write_initialization(initialization, diagnostics, initialized)
    contract = build_colmap_contract(
        dataset_id="fixture",
        dataset_root=source,
        image_root="images",
        cameras_path="cameras.json",
    )
    contract = with_initialization(
        contract,
        asset="initialization/sparse.npz",
        asset_sha256=sha256_file(initialization),
    )
    dataset = source / "dataset.json"
    write_contract(dataset, contract)
    return source, dataset, initialization, diagnostics, contract


def test_replay_bundle_hardlinks_and_validates_frozen_inputs(tmp_path):
    source, dataset, initialization, diagnostics, contract = _replay_fixture(tmp_path)
    replay = tmp_path / "replay"

    record = build_replay_bundle(
        contract=contract,
        dataset_path=dataset,
        dataset_root=source,
        initialization_path=initialization,
        diagnostics_path=diagnostics,
        replay_root=replay,
    )
    frozen = load_frozen_initialization(
        replay / "initialization" / "sparse.npz",
        replay / "initialization" / "sparse.json",
        expected_sha256=contract["initialization"]["sha256"],
    )

    assert record == validate_replay_bundle(replay)
    assert record == build_replay_bundle(
        contract=contract,
        dataset_path=dataset,
        dataset_root=source,
        initialization_path=initialization,
        diagnostics_path=diagnostics,
        replay_root=replay,
    )
    assert record["image_count"] == 12
    assert len(frozen.points) == 1
    assert os.stat(source / "images" / "frame_000.jpg").st_ino == os.stat(
        replay / "images" / "frame_000.jpg"
    ).st_ino


def test_replay_bundle_rejects_image_tampering(tmp_path):
    source, dataset, initialization, diagnostics, contract = _replay_fixture(tmp_path)
    replay = tmp_path / "replay"
    build_replay_bundle(
        contract=contract,
        dataset_path=dataset,
        dataset_root=source,
        initialization_path=initialization,
        diagnostics_path=diagnostics,
        replay_root=replay,
    )
    image = replay / "images" / "frame_000.jpg"
    image.unlink()
    image.write_bytes(b"tampered")

    with pytest.raises(ReplayError, match="image hash mismatch"):
        validate_replay_bundle(replay)


def test_frozen_initialization_rejects_nonfinite_scale(tmp_path):
    _source, _dataset, initialization, diagnostics, _contract = _replay_fixture(tmp_path)

    with np.load(initialization, allow_pickle=False) as payload:
        points = payload["points"]
        colors = payload["colors"]
        scales = payload["scales"]
    scales[0] = np.nan
    np.savez(initialization, points=points, colors=colors, scales=scales)
    asset_hash = sha256_file(initialization)
    diagnostics_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    diagnostics_payload["asset_sha256"] = asset_hash
    diagnostics.write_text(json.dumps(diagnostics_payload), encoding="utf-8")

    with pytest.raises(InitializationError, match="finite and positive"):
        load_frozen_initialization(
            initialization,
            diagnostics,
            expected_sha256=asset_hash,
        )
