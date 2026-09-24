import json
from pathlib import Path

import numpy as np
import torch

from image3d_scenegraph.gaussian.cloud_render import CloudCamera
from image3d_scenegraph.gaussian.visible_selection import (
    FrontLayer,
    VisiblePicker,
    polygon_pixels,
)


def test_picker_adapter_uses_front_contributors_and_maps_active_to_original_ids(
    monkeypatch,
):
    import gsplat.cuda._wrapper as wrapper

    class Model:
        def activated(self):
            return (
                torch.zeros((2, 3)),
                torch.zeros((2, 4)),
                torch.ones((2, 3)),
                torch.tensor([0.9, 0.9]),
                torch.zeros((2, 16, 3)),
            )

    def project(*args, **kwargs):
        radii = torch.ones((1, 2, 2), dtype=torch.int32)
        means = torch.tensor([[[32.0, 32.0], [32.0, 32.0]]])
        depths = torch.tensor([[1.0, 3.0]])
        conics = torch.zeros((1, 2, 3))
        return radii, means, depths, conics, None

    def indices(
        start,
        stop,
        trans,
        means,
        conics,
        opacity,
        width,
        height,
        tile_size,
        offsets,
        flat,
    ):
        pixel = 32 * width + 32
        if trans[0, 32, 32].item() == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        return torch.tensor([0, 1]), torch.tensor([pixel, pixel]), torch.tensor([0, 0])

    monkeypatch.setattr(wrapper, "fully_fused_projection", project)
    monkeypatch.setattr(
        wrapper,
        "isect_tiles",
        lambda *args, **kwargs: (None, torch.arange(2), torch.arange(2)),
    )
    monkeypatch.setattr(
        wrapper,
        "isect_offset_encode",
        lambda *args, **kwargs: torch.zeros((1, 4, 4), dtype=torch.int64),
    )
    monkeypatch.setattr(wrapper, "rasterize_to_indices_in_range", indices)
    camera = CloudCamera(
        np.eye(4), np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]]), 64, 64
    )
    picker = VisiblePicker(Model(), camera, torch.tensor([5, 13]))
    result = picker.pick([[30, 30], [34, 30], [34, 34], [30, 34]], [0.01, 100])
    assert result["ids"].tolist() == [5]
    assert result["confident_pixels"] == 1
    assert result["uncertain_pixels"] == 15


def test_local_p0_uses_the_same_front_layer_reference_cases():
    cases = json.loads(
        (Path(__file__).parent / "fixtures/gaussian_front_layers.json").read_text()
    )
    for case in cases:
        layer = FrontLayer(1, case["tolerance"])
        for source_id, depth, alpha in case["contributions"]:
            layer.add([source_id], [0], [depth], [alpha])
        ids, _ = layer.result()
        assert ids.tolist() == case["selected"], case["name"]


def test_front_layer_occlusion_and_transparency_abstention():
    layer = FrontLayer(3, 0.02)
    layer.add(
        [10, 20, 11, 21, 12, 13, 22],
        [0, 0, 1, 1, 2, 2, 2],
        [1, 3, 1, 3, 1, 1.01, 3],
        [0.9, 0.9, 0.2, 0.9, 0.3, 0.4, 0.9],
    )
    ids, confident = layer.result()
    assert set(ids) == {10, 12, 13}
    assert confident.tolist() == [True, False, True]


def test_front_layer_batches_do_not_replace_transparent_foreground_with_background():
    layer = FrontLayer(1, 0.02)
    layer.add([5], [0], [1], [0.2])
    layer.add([8], [0], [3], [0.99])
    ids, confident = layer.result()
    assert not ids.size
    assert not confident.any()
    assert layer.done[0]


def test_sparse_early_floater_does_not_hide_a_dominant_surface():
    layer = FrontLayer(1, 0.02)
    layer.add([3], [0], [1], [0.02])
    layer.add([8], [0], [3], [0.9])
    ids, confident = layer.result()
    assert ids.tolist() == [8]
    assert confident[0]
    assert layer.candidates[0] == [8]


def test_front_layer_same_layer_can_span_batches():
    layer = FrontLayer(1, 0.02)
    layer.add([1], [0], [1], [0.3])
    layer.add([2], [0], [1.01], [0.4])
    ids, confident = layer.result()
    assert ids.tolist() == [1, 2]
    assert confident[0]
    assert np.isclose(layer.coverage[0], 0.58)


def test_roi_uses_pixel_centers_and_polygon_not_only_bounding_box():
    polygon = [[0, 0], [3, 0], [0, 3]]
    assert polygon_pixels(
        polygon, np.array([0.5, 2.5]), np.array([0.5, 2.5])
    ).tolist() == [True, False]
