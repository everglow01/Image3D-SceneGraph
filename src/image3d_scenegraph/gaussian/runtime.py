"""Runtime image and camera loading for the Gaussian dataset contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import DatasetContractError, validate_contract
from .render import RenderCamera


@dataclass(frozen=True)
class TrainingView:
    camera: RenderCamera
    image: torch.Tensor


def load_training_views(
    contract: dict[str, Any],
    dataset_root: Path,
    *,
    split: str,
    longest_edge: int,
    device: torch.device,
) -> list[TrainingView]:
    if split not in {"train", "validation"}:
        raise DatasetContractError("the trainer can load only train or validation views")
    return load_views(
        contract,
        dataset_root,
        split=split,
        longest_edge=longest_edge,
        device=device,
    )


def load_evaluation_views(
    contract: dict[str, Any],
    dataset_root: Path,
    *,
    split: str,
    longest_edge: int,
    device: torch.device,
) -> list[TrainingView]:
    if split not in {"validation", "test"}:
        raise DatasetContractError("the evaluator can load only validation or test views")
    return load_views(
        contract,
        dataset_root,
        split=split,
        longest_edge=longest_edge,
        device=device,
    )


def load_views(
    contract: dict[str, Any],
    dataset_root: Path,
    *,
    split: str,
    longest_edge: int,
    device: torch.device,
) -> list[TrainingView]:
    validate_contract(contract, dataset_root)
    if split not in contract["splits"]:
        raise DatasetContractError(f"unknown dataset split: {split}")
    selected_ids = set(contract["splits"][split])
    normalized_from_world = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    world_from_normalized = np.linalg.inv(normalized_from_world)
    views = []
    for entry in contract["images"]:
        image_id = str(entry["image_id"])
        if image_id not in selected_ids:
            continue
        distortion = entry["distortion"]
        path = (dataset_root / entry["path"]).resolve()
        with Image.open(path) as source:
            source = source.convert("RGB")
            width, height = source.size
            scale = min(1.0, longest_edge / max(width, height))
            target_width = max(1, int(round(width * scale)))
            target_height = max(1, int(round(height * scale)))
            if (target_width, target_height) != (width, height):
                source = source.resize((target_width, target_height), Image.Resampling.LANCZOS)
            image = torch.from_numpy(np.asarray(source, dtype=np.float32).copy() / 255.0).to(
                device
            )
        intrinsic = np.asarray(entry["intrinsic"], dtype=np.float64).copy()
        intrinsic[0] *= target_width / width
        intrinsic[1] *= target_height / height
        image = _undistort_image(image, intrinsic, distortion)
        camera_from_world = np.asarray(entry["camera_from_world"], dtype=np.float64)
        camera_from_normalized = camera_from_world @ world_from_normalized
        camera = RenderCamera(
            image_id=image_id,
            camera_from_normalized=torch.tensor(
                camera_from_normalized, dtype=torch.float32, device=device
            ),
            intrinsic=torch.tensor(intrinsic, dtype=torch.float32, device=device),
            width=target_width,
            height=target_height,
        )
        views.append(TrainingView(camera, image))
    if len(views) != len(selected_ids):
        raise DatasetContractError(f"split {split} does not resolve to every contracted image")
    return views


def _undistort_image(
    image: torch.Tensor,
    intrinsic: np.ndarray,
    distortion: dict[str, Any],
) -> torch.Tensor:
    if distortion.get("state") == "none":
        return image
    model = distortion.get("model")
    params = [float(value) for value in distortion.get("params", [])]
    if model == "SIMPLE_RADIAL" and len(params) == 1:
        k1, k2, p1, p2 = params[0], 0.0, 0.0, 0.0
    elif model == "RADIAL" and len(params) == 2:
        k1, k2, p1, p2 = params[0], params[1], 0.0, 0.0
    elif model == "OPENCV" and len(params) == 4:
        k1, k2, p1, p2 = params
    else:
        raise DatasetContractError(f"trainer cannot undistort camera model: {model} {params}")
    height, width = image.shape[:2]
    y, x = torch.meshgrid(
        torch.arange(height, dtype=image.dtype, device=image.device),
        torch.arange(width, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    normalized_x = (x - cx) / fx
    normalized_y = (y - cy) / fy
    radius2 = normalized_x.square() + normalized_y.square()
    radial = 1.0 + k1 * radius2 + k2 * radius2.square()
    distorted_x = normalized_x * radial + 2 * p1 * normalized_x * normalized_y + p2 * (
        radius2 + 2 * normalized_x.square()
    )
    distorted_y = normalized_y * radial + p1 * (
        radius2 + 2 * normalized_y.square()
    ) + 2 * p2 * normalized_x * normalized_y
    pixel_x = distorted_x * fx + cx
    pixel_y = distorted_y * fy + cy
    grid = torch.stack(
        (
            2.0 * pixel_x / max(width - 1, 1) - 1.0,
            2.0 * pixel_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )[None]
    sampled = F.grid_sample(
        image.permute(2, 0, 1)[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0].permute(1, 2, 0)
