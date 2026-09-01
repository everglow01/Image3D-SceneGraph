#!/usr/bin/env python3
"""Generate a tiny known-camera scene and smoke the project 3DGS trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image3d_scenegraph.gaussian.config import resolve_internal_config
from image3d_scenegraph.gaussian.dataset import build_colmap_contract, write_contract
from image3d_scenegraph.gaussian.initialization import InitializationResult
from image3d_scenegraph.gaussian.model import GaussianModel
from image3d_scenegraph.gaussian.render import RenderCamera, render_gaussians
from image3d_scenegraph.gaussian.trainer import TrainingCancelled, train_gaussians


def camera_pose(angle: float, radius: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.3], dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    camera_from_world = np.eye(4)
    camera_from_world[:3, :3] = rotation
    camera_from_world[:3, 3] = -rotation @ center
    return center, camera_from_world


def rotmat_to_qvec(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    qw = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    qx = np.copysign(np.sqrt(max(0.0, 1.0 + 2 * rotation[0, 0] - trace)) / 2.0, rotation[2, 1] - rotation[1, 2])
    qy = np.copysign(np.sqrt(max(0.0, 1.0 + 2 * rotation[1, 1] - trace)) / 2.0, rotation[0, 2] - rotation[2, 0])
    qz = np.copysign(np.sqrt(max(0.0, 1.0 + 2 * rotation[2, 2] - trace)) / 2.0, rotation[1, 0] - rotation[0, 1])
    return [float(qw), float(qx), float(qy), float(qz)]


def generate_scene(root: Path) -> dict:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    width = height = 64
    intrinsic = np.array([[52.0, 0.0, 32.0], [0.0, 52.0, 32.0], [0.0, 0.0, 1.0]])
    cameras = [{"camera_id": 1, "model": "PINHOLE", "width": width, "height": height, "params": [52.0, 52.0, 32.0, 32.0]}]
    image_records = []
    poses = []
    for index in range(12):
        _, camera_from_world = camera_pose(2 * np.pi * index / 12)
        poses.append(camera_from_world)
        image_records.append(
            {
                "image_id": index + 1,
                "qvec": rotmat_to_qvec(camera_from_world[:3, :3]),
                "tvec": camera_from_world[:3, 3].tolist(),
                "camera_id": 1,
                "name": f"view_{index:02d}.png",
            }
        )

    device = torch.device("cuda")
    target = GaussianModel.from_points(
        torch.tensor([[-0.2, 0.0, 0.0], [0.18, 0.05, 0.03], [0.0, -0.18, -0.04]], device=device),
        torch.tensor([[0.9, 0.15, 0.1], [0.1, 0.8, 0.2], [0.1, 0.25, 0.9]], device=device),
        torch.tensor([0.16, 0.14, 0.13], device=device),
        initial_opacity=0.8,
        max_sh_degree=0,
    )
    world_from_normalized = np.eye(4)
    world_from_normalized[:3, :3] *= np.linalg.norm(np.array([2.0, 0.0, 0.3]))
    with torch.no_grad():
        for index, camera_from_world in enumerate(poses):
            camera = RenderCamera(
                image_id=str(index + 1),
                camera_from_normalized=torch.tensor(camera_from_world @ world_from_normalized, dtype=torch.float32, device=device),
                intrinsic=torch.tensor(intrinsic, dtype=torch.float32, device=device),
                width=width,
                height=height,
            )
            image = render_gaussians(
                target,
                camera,
                sh_degree=0,
                background=torch.ones(3, device=device),
            ).image
            Image.fromarray(image.clamp(0, 1).mul(255).byte().cpu().numpy()).save(
                images_dir / image_records[index]["name"]
            )
    (root / "cameras.json").write_text(
        json.dumps({"coordinate_system": "colmap_world", "cameras": cameras, "images": image_records}),
        encoding="utf-8",
    )
    contract = build_colmap_contract(
        dataset_id="generated-gaussian-smoke-v1",
        dataset_root=root,
        image_root="images",
        cameras_path="cameras.json",
    )
    write_contract(root / "dataset.json", contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/r2_7/synthetic-smoke"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--trainer", choices=["project", "mcmc"], default="project")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing smoke output: {args.output_dir}")
    dataset_root = args.output_dir / "dataset"
    contract = generate_scene(dataset_root)
    radius = float(contract["normalization"]["radius_world"])
    initial_normalized = np.array(
        [[-0.16, 0.02, 0.01], [0.15, 0.03, 0.02], [0.02, -0.15, -0.02]],
        dtype=np.float32,
    )
    initial_colors = np.array(
        [[150, 80, 60], [70, 150, 80], [70, 80, 150]], dtype=np.uint8
    )
    if args.trainer == "mcmc":
        generator = np.random.RandomState(20260729)
        initial_normalized = np.repeat(initial_normalized, 10, axis=0)
        initial_normalized += generator.normal(0.0, 0.02, initial_normalized.shape).astype(
            np.float32
        )
        initial_colors = np.repeat(initial_colors, 10, axis=0)
    initialization = InitializationResult(
        points=initial_normalized,
        colors=initial_colors,
        scales=np.full(len(initial_normalized), 0.13, dtype=np.float32),
        diagnostics={"kind": "generated_smoke", "radius_world": radius},
    )
    if args.trainer == "mcmc":
        if args.iterations < 300:
            raise SystemExit("MCMC smoke requires at least 300 iterations")
        overrides = {
            "iterations": args.iterations,
            "resolution": {"longest_edge": 64},
            "sh_schedule": {
                "initial_degree": 0,
                "max_degree": 0,
                "increase_every_iterations": 1,
            },
            "densification": {
                "enabled": True,
                "start_iteration": 100,
                "end_iteration": args.iterations - 100,
                "every_iterations": 100,
            },
            "opacity_reset": {
                "every_iterations": 100,
                "recovery_prune": {"window_iterations": 100},
            },
            "evaluation": {
                "validation_iterations": [args.iterations // 2, args.iterations],
            },
        }
        resolved = resolve_internal_config("mcmc_v1", overrides=overrides)
    else:
        overrides = {
            "iterations": args.iterations,
            "resolution": {"longest_edge": 64},
            "learning_rate": {
                "position": {"initial": 0.01, "final": 0.001},
                "feature": 0.05,
                "opacity": 0.02,
                "scaling": 0.01,
                "rotation": 0.005,
            },
            "sh_schedule": {
                "initial_degree": 0,
                "max_degree": 0,
                "increase_every_iterations": 1,
            },
            "densification": {
                "enabled": True,
                "start_iteration": 1,
                "end_iteration": max(2, args.iterations - 1),
                "every_iterations": max(1, args.iterations // 4),
                "gradient_threshold": 0.00000001,
                "scale_threshold": 0.01,
            },
            "opacity_reset": {
                "enabled": True,
                "every_iterations": max(1, args.iterations // 4),
            },
            "evaluation": {
                "validation_iterations": [args.iterations // 2, args.iterations],
            },
        }
        resolved = resolve_internal_config(overrides=overrides)
    interrupted_dir = args.output_dir / "resumed"
    cancel = False

    def progress(event: dict) -> None:
        nonlocal cancel
        if event.get("iteration") == args.iterations // 2 and event.get("loss") is not None:
            cancel = True

    try:
        train_gaussians(
            contract=contract,
            dataset_root=dataset_root,
            initialization=initialization,
            resolved_config=resolved,
            run_dir=interrupted_dir,
            progress_callback=progress,
            cancel_requested=lambda: cancel,
        )
    except TrainingCancelled:
        pass
    else:
        raise SystemExit("smoke interruption did not cancel")
    result = train_gaussians(
        contract=contract,
        dataset_root=dataset_root,
        initialization=initialization,
        resolved_config=resolved,
        run_dir=interrupted_dir,
        attempt_id="train-002",
        attempt_kind="resume",
        parent_attempt_id="train-001",
        resume_iteration=args.iterations // 2,
    )
    if not result.final_loss < result.initial_loss:
        raise SystemExit(f"loss did not decrease: {result.initial_loss} -> {result.final_loss}")
    if args.trainer == "mcmc":
        progress_path = interrupted_dir / result.progress_path
        events = [
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
        ]
        refinements = [event for event in events if event.get("mcmc_refinement")]
        if not refinements or not any(event.get("mcmc_added_count", 0) > 0 for event in refinements):
            raise SystemExit("MCMC smoke did not exercise refinement growth")
        if result.strategy_name != "mcmc_v1" or result.gaussian_count > 3_000_000:
            raise SystemExit("MCMC smoke violated strategy or global-cap provenance")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
