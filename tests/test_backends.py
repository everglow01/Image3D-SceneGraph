from __future__ import annotations

from backend.main import create_app
from image3d_scenegraph.geometry.backends import get_backend_specs


def test_backend_specs_report_mock_available(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))
    monkeypatch.setenv("IMAGE3D_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PATH", "")

    specs = {spec.backend_id: spec for spec in get_backend_specs(tmp_path)}

    assert specs["mock"].available is True
    assert specs["mock"].supported_outputs == ("point_cloud",)
    assert specs["vggt"].available is False
    assert "repo missing" in (specs["vggt"].reason or "")
    assert specs["colmap"].available is False
    assert "colmap executable not found" in (specs["colmap"].reason or "")
    assert specs["colmap_vggt"].available is False
    assert "colmap executable not found" in (specs["colmap_vggt"].reason or "")


def test_backend_specs_prefer_project_local_colmap(tmp_path, monkeypatch):
    external_root = tmp_path / "external"
    colmap = external_root / "colmap-4-cuda" / "install" / "bin" / "colmap"
    colmap.parent.mkdir(parents=True)
    colmap.write_text("#!/bin/sh\n", encoding="utf-8")
    colmap.chmod(0o755)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("IMAGE3D_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PATH", "")

    specs = {spec.backend_id: spec for spec in get_backend_specs(tmp_path)}

    assert specs["colmap"].available is True
    assert "setup_colmap_cuda.py" in (specs["colmap"].setup_command or "")


def test_backend_specs_keep_vggt_disabled_until_checkpoint_exists(tmp_path, monkeypatch):
    external_root = tmp_path / "external"
    checkpoint_root = tmp_path / "checkpoints"
    (external_root / "vggt").mkdir(parents=True)
    (checkpoint_root / "vggt").mkdir(parents=True)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("IMAGE3D_CHECKPOINT_ROOT", str(checkpoint_root))

    specs = {spec.backend_id: spec for spec in get_backend_specs(tmp_path)}

    assert specs["vggt"].available is False
    assert "checkpoint path missing" in (specs["vggt"].reason or "")


def test_backend_specs_report_vggt_available_with_local_assets(tmp_path, monkeypatch):
    external_root = tmp_path / "external"
    checkpoint_root = tmp_path / "checkpoints"
    (external_root / "vggt").mkdir(parents=True)
    checkpoint_path = checkpoint_root / "vggt" / "facebook--VGGT-1B" / "model.safetensors"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fake-checkpoint")
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("IMAGE3D_CHECKPOINT_ROOT", str(checkpoint_root))

    specs = {spec.backend_id: spec for spec in get_backend_specs(tmp_path)}

    assert specs["vggt"].available is True
    assert specs["vggt"].reason is None
    assert specs["vggt"].supported_outputs == ("point_cloud", "mesh")


def test_project_gaussian_reports_vggt_postprocess_separately_from_ba(
    tmp_path, monkeypatch
):
    external_root = tmp_path / "external"
    checkpoint_root = tmp_path / "checkpoints"
    (external_root / "vggt").mkdir(parents=True)
    vggt_checkpoint = (
        checkpoint_root / "vggt" / "facebook--VGGT-1B" / "model.safetensors"
    )
    vggt_checkpoint.parent.mkdir(parents=True)
    vggt_checkpoint.write_bytes(b"fake-vggt")
    colmap = external_root / "colmap-4-cuda" / "install" / "bin" / "colmap"
    colmap.parent.mkdir(parents=True)
    colmap.write_text("#!/bin/sh\n", encoding="utf-8")
    colmap.chmod(0o755)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("IMAGE3D_CHECKPOINT_ROOT", str(checkpoint_root))
    monkeypatch.setenv("PATH", "")

    project = {
        spec.backend_id: spec for spec in get_backend_specs(tmp_path)
    }["project_3dgs"]
    geometry = {
        option["id"]: option
        for option in project.options["gaussian_geometry_sources"]
    }
    postprocess = {
        option["id"]: option
        for option in project.options["gaussian_postprocessors"]
    }

    assert geometry["colmap"]["available"] is True
    assert geometry["vggt_ba"]["available"] is False
    assert "DINOv2 repo missing" in geometry["vggt_ba"]["reason"]
    assert "setup_model.py --backend vggt --install" in geometry["vggt_ba"][
        "setup_command"
    ]
    assert postprocess["none"]["available"] is True
    assert postprocess["vggt_visibility_v1"]["available"] is True


def test_project_gaussian_reports_complete_vggt_ba_dependencies(
    tmp_path, monkeypatch
):
    external_root = tmp_path / "external"
    checkpoint_root = tmp_path / "checkpoints"
    (external_root / "vggt").mkdir(parents=True)
    (external_root / "dinov2").mkdir(parents=True)
    (external_root / "lightglue").mkdir(parents=True)
    for path in (
        checkpoint_root / "vggt" / "facebook--VGGT-1B" / "model.safetensors",
        checkpoint_root / "vggt" / "dinov2_vitb14_reg4_pretrain.pth",
        checkpoint_root / "vggt" / "vggsfm_v2_tracker.pt",
        checkpoint_root / "vggt" / "torch-hub" / "checkpoints" / "aliked-n16.pth",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weight")
    colmap = external_root / "colmap-4-cuda" / "install" / "bin" / "colmap"
    colmap.parent.mkdir(parents=True)
    colmap.write_text("#!/bin/sh\n", encoding="utf-8")
    colmap.chmod(0o755)
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("IMAGE3D_CHECKPOINT_ROOT", str(checkpoint_root))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        "image3d_scenegraph.geometry.backends.importlib_util.find_spec",
        lambda name: object() if name in {"pycolmap", "lightglue"} else None,
    )

    project = {
        spec.backend_id: spec for spec in get_backend_specs(tmp_path)
    }["project_3dgs"]
    geometry = {
        option["id"]: option
        for option in project.options["gaussian_geometry_sources"]
    }

    assert geometry["vggt_ba"]["available"] is True
    assert geometry["vggt_ba"]["supported_modes"] == ["video"]


def test_backends_api_route_is_registered(tmp_path):
    app = create_app(output_root=tmp_path / "jobs")

    paths = {route.path for route in app.routes}

    assert "/api/backends" in paths
