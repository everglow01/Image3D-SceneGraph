from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image

from image3d_scenegraph.gaussian import runtime
from image3d_scenegraph.gaussian.runtime import TrainingViews, load_training_views, view_cameras
from image3d_scenegraph.gaussian.trainer import _release_training_views


def make_views(tmp_path, monkeypatch, *, distorted=False):
    monkeypatch.setattr(runtime, "validate_contract", lambda *_args: None)
    images = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        pixels = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3) + index
        Image.fromarray(pixels).save(path)
        images.append({
            "image_id": str(index), "path": path.name,
            "distortion": {"state": "distorted", "model": "OPENCV", "params": [0, 0, 0, 0]}
            if distorted else {"state": "none"},
            "intrinsic": [[10, 0, 6], [0, 10, 4], [0, 0, 1]],
            "camera_from_world": torch.eye(4).tolist(),
        })
    contract = {
        "splits": {"train": ["0", "1"], "validation": ["2"], "test": ["3"]},
        "normalization": {"normalized_from_world": torch.eye(4).tolist()},
        "images": images,
    }
    return contract


@pytest.mark.parametrize("distorted", [False, True])
def test_lazy_images_preserve_pixels_cameras_and_byte_bound(tmp_path, monkeypatch, distorted):
    contract = make_views(tmp_path, monkeypatch, distorted=distorted)
    calls = []
    original = runtime._ViewSource.load

    def load(source):
        calls.append(source.camera.image_id)
        return original(source)

    monkeypatch.setattr(runtime._ViewSource, "load", load)
    views = load_training_views(contract, tmp_path, split="train", longest_edge=6, device=torch.device("cpu"))
    assert isinstance(views, TrainingViews)
    assert calls == []
    assert [camera.image_id for camera in view_cameras(views)] == ["0", "1"]
    assert calls == []
    per_view = 6 * 4 * 3 * (4 if distorted else 1)
    views.cache_max_bytes = per_view
    first = views[0]
    with Image.open(tmp_path / "0.png") as image:
        expected = torch.from_numpy(np.asarray(image.convert("RGB").resize((6, 4), Image.Resampling.LANCZOS)).copy())
    intrinsic = np.array([[5, 0, 3], [0, 5, 2], [0, 0, 1]], dtype=np.float64)
    if distorted:
        expected = runtime._undistort_image(expected.float() / 255, intrinsic, contract["images"][0]["distortion"])
    assert torch.equal(first.image, expected)
    assert torch.equal(first.camera.intrinsic, torch.tensor(intrinsic, dtype=torch.float32))
    assert views[0] is first
    assert calls == ["0"]
    assert views.cached_bytes == per_view
    assert views[1].camera.image_id == "1"
    assert views.cached_bytes == per_view
    assert list(views._cache) == [1]
    assert torch.equal(views[0].image, first.image)
    assert calls == ["0", "1", "0"]
    views.cache_max_bytes = per_view - 1
    views._cache.clear()
    views.cached_bytes = 0
    assert views[0].image.numel() == 72
    assert views.cached_bytes == 0
    assert len(views._cache) == 0


def test_final_fit_concat_stays_lazy_and_preserves_order(tmp_path, monkeypatch):
    contract = make_views(tmp_path, monkeypatch)
    train = load_training_views(contract, tmp_path, split="train", longest_edge=3840, device=torch.device("cpu"))
    validation = load_training_views(contract, tmp_path, split="validation", longest_edge=3840, device=torch.device("cpu"))
    fit = train + validation
    assert train.cached_bytes == validation.cached_bytes == fit.cached_bytes == 0
    assert [c.image_id for c in view_cameras(fit)] == ["0", "1", "2"]
    assert [c.image_id for c in view_cameras(fit[1:])] == ["1", "2"]
    assert fit[-1].camera.image_id == "2"
    with pytest.raises(IndexError):
        fit[-4]
    with pytest.raises(IndexError):
        fit[3]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    for index in [1, 0, 1, 2]:
        fit[index].to(torch.device("cpu"))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert train.cached_bytes == validation.cached_bytes == 0
    _release_training_views(train, validation, fit)
    assert len(train) == len(validation) == len(fit) == 0
    assert fit.cached_bytes == 0


def test_native_uhd_view_is_not_silently_downsampled(tmp_path, monkeypatch):
    contract = make_views(tmp_path, monkeypatch)
    Image.new("RGB", (3840, 2160), (127, 64, 255)).save(tmp_path / "0.png")
    views = load_training_views(contract, tmp_path, split="train", longest_edge=3840, device=torch.device("cpu"))
    assert (views.cameras[0].width, views.cameras[0].height) == (3840, 2160)
    assert views.cached_bytes == 0
    assert views[0].image.shape == (2160, 3840, 3)
    assert views.cached_bytes == 3840 * 2160 * 3


def test_lazy_loader_does_not_decode_test_rgb(tmp_path, monkeypatch):
    contract = make_views(tmp_path, monkeypatch)
    decoded = []
    original = runtime._ViewSource.load

    def load(source):
        decoded.append(source.camera.image_id)
        return original(source)

    monkeypatch.setattr(runtime._ViewSource, "load", load)
    views = load_training_views(contract, tmp_path, split="train", longest_edge=3840, device=torch.device("cpu"))
    assert [view.camera.image_id for view in views] == ["0", "1"]
    assert decoded == ["0", "1"]
    with pytest.raises(runtime.DatasetContractError, match="only train or validation"):
        load_training_views(contract, tmp_path, split="test", longest_edge=3840, device=torch.device("cpu"))
