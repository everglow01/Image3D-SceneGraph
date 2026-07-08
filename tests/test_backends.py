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


def test_backends_api_route_is_registered(tmp_path):
    app = create_app(output_root=tmp_path / "jobs")

    paths = {route.path for route in app.routes}

    assert "/api/backends" in paths
