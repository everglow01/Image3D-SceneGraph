from __future__ import annotations

from image3d_scenegraph.gaussian.trainers import (
    GRAPHDECO_COMMIT,
    NERFSTUDIO_COMMIT,
    get_gaussian_trainer_specs,
    validate_trainer_id,
)


def test_trainer_registry_reports_pinned_external_environments(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE3D_EXTERNAL_ROOT", str(tmp_path / "external"))

    specs = {spec.trainer_id: spec for spec in get_gaussian_trainer_specs(tmp_path)}

    assert tuple(specs) == ("project", "graphdeco", "nerfstudio")
    assert specs["graphdeco"].revision == GRAPHDECO_COMMIT
    assert specs["nerfstudio"].revision == NERFSTUDIO_COMMIT
    assert specs["graphdeco"].available is False
    assert "repo missing" in (specs["graphdeco"].reason or "")
    assert "--accept-research-license" in (specs["graphdeco"].setup_command or "")


def test_validate_trainer_id_rejects_unknown():
    assert validate_trainer_id("project") == "project"
    try:
        validate_trainer_id("unknown")
    except ValueError as exc:
        assert "unsupported Gaussian trainer" in str(exc)
    else:
        raise AssertionError("unknown trainer was accepted")
