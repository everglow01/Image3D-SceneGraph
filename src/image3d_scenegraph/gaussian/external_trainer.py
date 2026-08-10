"""Subprocess lifecycle for pinned external Gaussian trainers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ResolvedGaussianConfig, resolved_config_record
from .dataset import sha256_file
from .importer import import_inria_ply
from .initialization import InitializationResult
from .trainer import TrainingError, TrainingResult
from .trainer_dataset import prepare_external_dataset
from .trainers import get_gaussian_trainer_specs, trainer_record, validate_trainer_id


_LOSS_PATTERN = re.compile(r"(?:loss|Loss)[^0-9+\-.]*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)")


def train_external_gaussians(
    *,
    trainer_id: str,
    contract: dict[str, Any],
    dataset_root: Path,
    initialization: InitializationResult,
    resolved_config: ResolvedGaussianConfig,
    run_dir: Path,
    attempt_id: str,
    cancel_requested: Callable[[], bool] | None = None,
) -> TrainingResult:
    trainer_id = validate_trainer_id(trainer_id)
    if trainer_id == "project":
        raise TrainingError("external trainer runner cannot dispatch the project trainer")
    project_root = Path(os.environ.get("IMAGE3D_PROJECT_ROOT", ".")).resolve()
    spec = next(
        spec
        for spec in get_gaussian_trainer_specs(project_root)
        if spec.trainer_id == trainer_id
    )
    if not spec.available:
        raise TrainingError(f"{spec.label} is unavailable: {spec.reason}")

    config = resolved_config.effective_config
    artifact_dir = run_dir / "attempts" / attempt_id / "artifacts"
    preparation_dir = run_dir / "preparation" / attempt_id
    native_dataset = preparation_dir / f"{trainer_id}-dataset"
    native_output = run_dir / "native" / attempt_id / trainer_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    native_output.mkdir(parents=True, exist_ok=False)
    integrity = prepare_external_dataset(
        trainer=trainer_id,
        contract=contract,
        dataset_root=dataset_root,
        initialization=initialization,
        output_dir=native_dataset,
    )
    config_record = resolved_config_record(resolved_config)
    (preparation_dir / "dataset.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    (preparation_dir / "effective_config.json").write_text(
        json.dumps(config_record, indent=2) + "\n", encoding="utf-8"
    )

    iterations = int(config["iterations"])
    seed = int(config["seed"])
    command = _graphdeco_command(
        project_root, native_dataset, native_output, iterations, seed
    )
    command_record = {
        "trainer": trainer_record(trainer_id, project_root),
        "command": command,
        "command_hash": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "integrity": integrity,
    }
    (native_output / "command.json").write_text(
        json.dumps(command_record, indent=2) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    completed = _run(command, cwd=project_root, cancel_requested=cancel_requested)
    elapsed = time.perf_counter() - started
    (native_output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (native_output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    native_ply = _find_native_ply(native_output)
    losses = _native_losses(native_output, completed, project_root)
    trainer_info = command_record["trainer"] | {
        "command_hash": command_record["command_hash"]
    }
    model_path = artifact_dir / "model.pt"
    transform = np.asarray(
        contract["normalization"]["normalized_from_world"], dtype=np.float64
    )
    imported = import_inria_ply(
        native_ply,
        model_path,
        normalized_from_source=transform,
        trainer=trainer_info,
    )
    progress_path = artifact_dir / "progress.jsonl"
    progress_path.write_text(
        "".join(
            json.dumps({"iteration": index, "loss": value}) + "\n"
            for index, value in enumerate(losses, start=1)
        ),
        encoding="utf-8",
    )
    if not losses:
        raise TrainingError(f"{trainer_id} trainer did not emit machine-readable loss")
    initial_loss = losses[0]
    final_loss = losses[-1]
    result = TrainingResult(
        iteration=iterations,
        candidate_iteration=iterations,
        gaussian_count=int(imported["gaussian_count"]),
        initial_loss=initial_loss,
        final_loss=final_loss,
        validation={"status": "external_common_validation_pending"},
        final_validation={"status": "external_common_validation_pending"},
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
        elapsed_seconds=elapsed,
        final_checkpoint_hash=sha256_file(native_ply),
        final_checkpoint_path=native_ply.relative_to(run_dir).as_posix(),
        model_path=model_path.relative_to(run_dir).as_posix(),
        result_path=(artifact_dir / "result.json").relative_to(run_dir).as_posix(),
        progress_path=progress_path.relative_to(run_dir).as_posix(),
    )
    payload = {
        **result.__dict__,
        "trainer": trainer_info,
        "native_ply": native_ply.relative_to(run_dir).as_posix(),
        "native_ply_sha256": sha256_file(native_ply),
        "loss_status": "complete",
    }
    (artifact_dir / "result.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return result


def _graphdeco_command(
    project_root: Path,
    dataset: Path,
    output: Path,
    iterations: int,
    seed: int = 0,
) -> list[str]:
    repo = project_root / "external" / "gaussian-splatting"
    return [
        str(repo / ".venv" / "bin" / "python"),
        str(project_root / "scripts" / "run_graphdeco_training.py"),
        "--graphdeco-root", str(repo),
        "--seed", str(seed),
        "-s", str(dataset),
        "-m", str(output),
        "--eval",
        "--disable_viewer",
        "--iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--resolution", "1",
    ]


def _run(
    command: list[str],
    *,
    cwd: Path,
    cancel_requested: Callable[[], bool] | None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["CC"] = "/usr/bin/gcc-11"
    env["CXX"] = "/usr/bin/g++-11"
    env["MAX_JOBS"] = "1"
    try:
        if cancel_requested is None:
            return subprocess.run(
                command, cwd=cwd, env=env, check=True, capture_output=True, text=True
            )
        from image3d_scenegraph.worker import run_cancellable_command

        return run_cancellable_command(
            command, cwd=cwd, env=env, cancel_requested=cancel_requested
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        raise TrainingError(f"external Gaussian trainer failed:\n{details}") from exc


def _find_native_ply(output: Path) -> Path:
    candidates = sorted(output.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates or not candidates[-1].is_file():
        raise TrainingError("graphdeco trainer did not produce a Gaussian PLY")
    return candidates[-1]


def _native_losses(
    output: Path,
    completed: subprocess.CompletedProcess[str],
    project_root: Path,
) -> list[float]:
    event_python = project_root / "external" / "gaussian-splatting" / ".venv" / "bin" / "python"
    events = sorted(output.glob("events.out.tfevents.*"))
    if events:
        tags = ["train_loss_patches/total_loss"]
        script = (
            "import json,sys;from tensorboard.backend.event_processing.event_accumulator "
            "import EventAccumulator;a=EventAccumulator(sys.argv[1],size_guidance={'scalars':0});"
            "a.Reload();tags=json.loads(sys.argv[2]);"
            "print(json.dumps([e.value for t in tags if t in a.Tags().get('scalars',[]) "
            "for e in a.Scalars(t)]))"
        )
        parsed = subprocess.run(
            [str(event_python), "-c", script, str(events[-1]), json.dumps(tags)],
            capture_output=True,
            text=True,
        )
        if parsed.returncode == 0:
            try:
                values = [float(value) for value in json.loads(parsed.stdout)]
            except (ValueError, TypeError, json.JSONDecodeError):
                values = []
            if values:
                return values
    return _parse_losses(completed.stdout + "\n" + completed.stderr)


def _parse_losses(text: str) -> list[float]:
    values = []
    for match in _LOSS_PATTERN.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    return values
