from __future__ import annotations

import ast
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from scripts import prepare_gsplat_camera_sizes as patch


def helper():
    namespace = {"torch": SimpleNamespace(cat=lambda values, dim: sum(values, []))}
    exec(compile(patch.HELPER, "camera-size-helper", "exec"), namespace)
    return namespace["_project_target_camera_sizes"]


def test_helper_preserves_single_and_uniform_projection_call():
    for sizes in [None, [(64, 64), (64, 64)]]:
        calls = []
        def projection(*args, **kwargs):
            calls.append(args)
            return "unchanged"
        result = helper()(projection, sizes, None, None, None, None, [0, 1], [2, 3], 64, 64)
        assert result == "unchanged" and len(calls) == 1
        assert calls[0][-2:] == (64, 64)


@pytest.mark.parametrize("optional", [False, True])
def test_helper_uses_target_sizes_and_preserves_camera_order(optional):
    calls = []
    def projection(*args, **kwargs):
        calls.append(args)
        value = [(args[4][0], args[-2], args[-1])]
        return value, value, value, value, value if optional else None
    result = helper()(projection, [(64, 64), (96, 80)], None, None, None, None,
                      [0, 1], [2, 3], 64, 64, packed=False, sparse_grad=False, camera_model="pinhole")
    assert len(calls) == 2 and result[0] == [(0, 64, 64), (1, 96, 80)]
    assert result[4] == (result[0] if optional else None)


@pytest.mark.parametrize("extra", [{"packed": True}, {"sparse_grad": True}, {"camera_model": "fisheye"}])
def test_helper_rejects_unsupported_projection(extra):
    with pytest.raises(ValueError, match="dense pinhole"):
        helper()(None, [(64, 64), (96, 80)], None, None, None, None, [0, 1], [2, 3], 64, 64,
                 **{**{"camera_model": "pinhole"}, **extra})


def test_patch_rejects_unpinned_source_and_is_syntactically_valid():
    with pytest.raises(ValueError, match="audited"):
        patch.patch_rendering("untrusted source")
    ast.parse(patch.HELPER)


@pytest.mark.skipif(socket.gethostname() != "i-94B8D131", reason="audited source only on remote")
def test_patch_matches_real_installed_source():
    import importlib.util
    source = Path(importlib.util.find_spec("gsplat").origin).parent / "rendering.py"
    result = patch.patch_rendering(source.read_text())
    ast.parse(result)
    assert result.count("camera_sizes = list(zip(widths, heights))") == 1
    assert "with_ut or with_eval3d" in result


def test_overlay_rejects_content_mutation_and_wrong_package_path(tmp_path):
    import json
    import hashlib
    package = tmp_path / "gsplat"
    package.mkdir()
    file = package / "rendering.py"
    file.write_text("patched")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    record = {"profile": patch.PROFILE, "source_rendering_sha256": patch.RENDERING_SHA256,
              "files": {"rendering.py": digest}, "source_hashes": {"rendering.py": patch.RENDERING_SHA256}}
    (tmp_path / "overlay.json").write_text(json.dumps(record))
    assert patch.verify_overlay(tmp_path, package)["profile"] == patch.PROFILE
    with pytest.raises(ValueError, match="path mismatch"):
        patch.verify_overlay(tmp_path, tmp_path)
    file.write_text("mutated")
    with pytest.raises(ValueError, match="files changed"):
        patch.verify_overlay(tmp_path, package)
