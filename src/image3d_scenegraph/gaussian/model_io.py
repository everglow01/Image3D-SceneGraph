"""Native model files; streaming writes and atomic snapshot replacement."""
from __future__ import annotations

import io
import os
from pathlib import Path
import uuid

import torch

from .model import GaussianModel


def save_torch_file(payload: dict, path: Path, *, replace: bool = False) -> None:
    if path.is_symlink() or (path.exists() and (not replace or not path.is_file())):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def model_payload(model: GaussianModel) -> dict:
    snapshot = model.snapshot()
    return {"max_sh_degree": int(snapshot["max_sh_degree"]), "state_dict": {
        name: value.cpu() for name, value in snapshot.items() if name != "max_sh_degree"}}


def save_model_snapshot(model: GaussianModel, path: Path, *, replace: bool = False) -> None:
    save_torch_file(model_payload(model), path, replace=replace)


def load_model_snapshot(content: bytes | Path, device: torch.device) -> GaussianModel:
    if isinstance(content, Path):
        payload = torch.load(content, map_location=device, weights_only=True, mmap=device.type == "cpu")
    else:
        payload = torch.load(io.BytesIO(content), map_location=device, weights_only=True)
    state = payload["state_dict"]
    if "log_scales" not in state:
        state = {
            "means": state["params.means"], "log_scales": state["params.scales"],
            "quats": state["params.quats"], "opacity_logits": state["params.opacities"],
            "sh_coeffs": torch.cat((state["params.sh0"], state["params.shN"]), dim=1),
        }
    model = GaussianModel(
        means=state["means"], log_scales=state["log_scales"], quats=state["quats"],
        opacity_logits=state["opacity_logits"], sh_coeffs=state["sh_coeffs"],
        max_sh_degree=int(payload["max_sh_degree"]),
    ).to(device)
    model.validate()
    return model
