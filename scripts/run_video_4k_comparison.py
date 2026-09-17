"""Frozen two-arm native-resolution video experiment; never evaluates Test RGB."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.config import resolve_internal_config, resolved_config_record
from image3d_scenegraph.gaussian.replay import validate_replay_bundle
from image3d_scenegraph.gaussian.runtime import VIEW_CACHE_MAX_BYTES
from image3d_scenegraph.geometry.adapters import ProjectGaussianAdapter, ReconstructionContext
from image3d_scenegraph.video.keyframes import probe_video

PROFILE = "num4_new_4k_train_only_v1"
NATIVE_1080_PROFILE = "num4_retake_1080_train_only_v1"
ARMS = ("project", "mcmc")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configs(longest_edge: int = 3840) -> dict:
    return {
        arm: resolved_config_record(resolve_internal_config(
            "standard_v1" if arm == "project" else "mcmc_v1",
            overrides={
                "resolution": {"longest_edge": longest_edge},
                "opacity_reset": {"recovery_prune": {"enabled": arm == "project"}},
            },
        ))
        for arm in ARMS
    }


def revision() -> str:
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def require_resources(root: Path, *, minimum_free_gib: int) -> None:
    import torch

    if torch.cuda.device_count() != 2:
        raise ValueError("this experiment requires exactly two visible CUDA devices")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).strip()
    if active:
        raise ValueError("GPU compute processes are already running; do not overlap experiments")
    if shutil.disk_usage(root).free < minimum_free_gib * 1024**3:
        raise ValueError(f"at least {minimum_free_gib} GiB free disk is required before this stage")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) * 1024 < 12 * 1024**3:
        raise ValueError("at least 12 GiB available host memory is required")


def run_command(output: Path, name: str, arguments: list[str]) -> None:
    require_resources(output, minimum_free_gib=8)
    command = [sys.executable, *arguments]
    print(f"stage={name} started", flush=True)
    started = time.time()
    with (output / f"{name}.log").open("x") as log:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=24 * 3600)
    write_json(output / f"{name}.exit.json", {"returncode": completed.returncode, "elapsed_seconds": time.time() - started})
    if completed.returncode:
        raise RuntimeError(f"{name} failed; retained log: {output / (name + '.log')}")
    print(f"stage={name} complete", flush=True)


def reuse_video_preparation(parent: Path, shared: Path, source_sha256: str) -> dict:
    original = parent / "shared"
    selection_path = original / "frames/selection.json"
    selection = read_json(selection_path)
    previous_probe = read_json(original / "diagnostics/video_probe.json")
    if (previous_probe.get("source", {}).get("sha256") != source_sha256
            or previous_probe.get("duration_seconds") != selection.get("duration_seconds")):
        raise ValueError("retained video probe identity mismatch")
    frontend_path = original / "diagnostics/sfm_frontend_contract.json"
    frontend = read_json(frontend_path)
    recovery_path = original / "diagnostics/sfm_pose_recovery.json"
    recovery = read_json(recovery_path)
    database = original / "colmap/database.db"
    if (selection.get("profile") != "video_keyframes_standard_v2"
            or selection.get("source_sha256") != source_sha256
            or frontend.get("initial_video_selection_sha256") != sha256_file(selection_path)
            or frontend.get("v2_mapper_seed_count") != 1000):
        raise ValueError("retained 1000-seed preparation identity mismatch")
    if sha256_file(database) != recovery.get("source_database_sha256"):
        raise ValueError("retained feature database changed after the failed run")
    selected = selection["selected"]
    if len(selected) != selection["selected_count"] or len({item["path"] for item in selected}) != len(selected):
        raise ValueError("retained selected-frame list is incomplete or duplicated")
    for item in selected:
        source = (original / item["path"]).resolve()
        destination = (shared / item["path"]).resolve()
        if not source.is_relative_to(original.resolve()) or not destination.is_relative_to(shared.resolve()):
            raise ValueError("retained selected frame escapes its workspace")
        if sha256_file(source) != item["sha256"]:
            raise ValueError("retained selected frame hash mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    metadata = ["frames/selection.json", "diagnostics/video_probe.json", "diagnostics/video_keyframe_timing.json", "diagnostics/video_keyframes.jpg"]
    for relative in metadata:
        destination = shared / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, destination)
    sources = [database, frontend_path, recovery_path, *(original / name for name in metadata)]
    return {
        "profile": "retained_frontend_seed2500_v1",
        "source_experiment": str(parent), "source_database": str(database),
        "source_database_sha256": recovery["source_database_sha256"],
        "source_frontend_contract": str(frontend_path),
        "source_files": {str(path): sha256_file(path) for path in sources},
        "selected_count": len(selected), "mapper_seed_limit": 2500,
        "initial_feature_extraction": "reused", "initial_feature_matching": "reused",
    }


def prepare(
    source: Path, root: Path, *, reuse_experiment: Path | None = None,
    longest_edge: int = 3840, mapper_seed_limit: int = 1000,
) -> None:
    if longest_edge not in (1920, 3840) or mapper_seed_limit not in (1000, 2500, 3500):
        raise ValueError("unsupported resolution or Mapper seed budget")
    if reuse_experiment is not None:
        if longest_edge != 3840:
            raise ValueError("retained preparation requires the original UHD input")
        mapper_seed_limit = 2500
    profile = PROFILE if longest_edge == 3840 else NATIVE_1080_PROFILE
    if root.exists():
        raise ValueError("experiment root already exists; inspect instead of retrying")
    require_resources(root.parent, minimum_free_gib=30 if reuse_experiment is not None else 40)
    code = revision()
    probe = probe_video(source)
    expected_size = (longest_edge, longest_edge * 9 // 16)
    if (probe["source_width"], probe["source_height"]) != expected_size:
        raise ValueError(f"this frozen experiment expects native {expected_size[0]}x{expected_size[1]} input")
    root.mkdir()
    write_json(root / "prepare-started.json", {
        "code": code, "source": str(source), "probe": probe, "profile": profile,
        "longest_edge": longest_edge, "mapper_seed_limit": mapper_seed_limit,
    })
    shared = root / "shared"
    for directory in ("input", "geometry", "diagnostics", "logs"):
        (shared / directory).mkdir(parents=True, exist_ok=False)
    os.link(source, shared / "input" / source.name)
    records = configs(longest_edge)
    for arm, record in records.items():
        write_json(root / f"{arm}.config.json", record)
    options = {
        "gaussian_trainer": "project", "gaussian_geometry_source": "colmap",
        "gaussian_longest_edge": longest_edge, "gaussian_prepare_only": True,
        "v2_mapper_seed_limit": mapper_seed_limit,
        "gaussian_config_record": json.dumps(records["project"]),
        "gaussian_final_fit": "off", "gaussian_sor_filter": "on", "gaussian_postprocess": "none",
        "video_keyframe_profile": "standard_v2", "video_rotation": "auto",
        "sfm_feature_profile": "sift_v1", "sfm_local_matcher": "bruteforce",
        "sfm_pairing": "sequential_loop", "sfm_geometric_verification": "default_v1",
        "sfm_camera_calibration": "shared_opencv_v1", "sfm_mapper": "incremental",
    }
    reused = None
    if reuse_experiment is not None:
        reused = reuse_video_preparation(reuse_experiment, shared, probe["source"]["sha256"])
        write_json(root / "reuse.json", reused)
        options.update(
            video_preparation_reused=True,
            sfm_reuse_feature_database=reused["source_database"],
            sfm_reuse_frontend_contract=reused["source_frontend_contract"],
            sfm_reuse_database_sha256=reused["source_database_sha256"],
        )
    result = ProjectGaussianAdapter().run(ReconstructionContext(
        job_id=profile, job_dir=shared, mode="video",
        input_assets=[{"path": f"input/{source.name}"}], options=options,
        progress_callback=lambda stage, progress: print(f"stage={stage} progress={progress}", flush=True),
    ))
    if result.stage != "gaussian_prepared":
        raise ValueError("preparation unexpectedly entered a training lifecycle")
    write_json(root / "preparation.json", asdict(result))
    replay = shared / "gaussian" / "replay"
    validate_replay_bundle(replay)
    dataset = read_json(replay / "dataset.json")
    dimensions = [(entry["width"], entry["height"]) for entry in dataset["images"]]
    if max(max(size) for size in dimensions) <= longest_edge * 0.8:
        raise ValueError("native-resolution preparation was unexpectedly downsampled")
    if sha256_file(source) != probe["source"]["sha256"]:
        raise ValueError("source video changed during preparation")
    files = [replay / "dataset.json", replay / "replay.json", *(root / f"{arm}.config.json" for arm in ARMS)]
    if reused is not None:
        for path, expected in reused["source_files"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"retained source changed during geometry: {path}")
        original = reuse_experiment / "shared"
        for item in read_json(original / "frames/selection.json")["selected"]:
            if sha256_file(original / item["path"]) != item["sha256"]:
                raise ValueError("retained image changed during geometry")
        files.extend([root / "reuse.json", shared / "diagnostics/sfm_frontend_contract.json"])
    for name in ("sfm_mapper_failure.json", "sfm_mapper_failure.log", "sfm_pose_recovery.json"):
        path = shared / "diagnostics" / name
        if path.is_file():
            files.append(path)
    write_json(root / "protocol.json", {
        "profile": profile, "status": "frozen", "code": code,
        "source": str(source), "source_sha256": probe["source"]["sha256"],
        "replay": str(replay), "dataset_hash": dataset["dataset_hash"],
        "files": {str(path): sha256_file(path) for path in files},
        "split_counts": {key: len(value) for key, value in dataset["splits"].items()},
        "image_dimensions": sorted(set(dimensions)), "longest_edge": longest_edge,
        "geometry_variant": "retained_frontend_seed2500_v1" if reused is not None else f"fresh_seed{mapper_seed_limit}_v1",
        "mapper_seed_limit": mapper_seed_limit,
        "world_size": 2, "main_updates": 30000, "main_camera_samples": 60000,
        "final_fit_profile": "train_only_control_v1", "final_fit_updates": 2000,
        "final_fit_camera_samples": 4000, "view_cache_max_bytes_per_collection": VIEW_CACHE_MAX_BYTES,
        "intermediate_validation_pngs": False, "validation_metrics_at_all_scheduled_steps": True,
        "test": "not_authorized", "promotion_eligible": False,
    })


def load_protocol(root: Path, expected_sha256: str) -> dict:
    if sha256_file(root / "protocol.json") != expected_sha256:
        raise ValueError("protocol hash mismatch")
    protocol = read_json(root / "protocol.json")
    if protocol["profile"] not in (PROFILE, NATIVE_1080_PROFILE) or protocol["status"] != "frozen" or protocol["code"] != revision():
        raise ValueError("protocol/code identity mismatch")
    for path, expected in protocol["files"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"frozen file changed: {path}")
    validate_replay_bundle(Path(protocol["replay"]))
    return protocol


def run_arm(root: Path, arm: str, expected_sha256: str) -> None:
    protocol = load_protocol(root, expected_sha256)
    require_resources(root, minimum_free_gib=20)
    output = root / arm
    output.mkdir(exist_ok=False)
    write_json(output / "started.json", {"arm": arm, "protocol_sha256": expected_sha256})
    replay = Path(protocol["replay"])
    dataset = replay / "dataset.json"
    config = root / f"{arm}.config.json"
    training = output / "training"
    common = ["--dataset-contract", str(dataset), "--dataset-root", str(replay), "--resolved-config-json", str(config)]
    run_command(output, "train", [
        "scripts/run_gaussian_training.py", *common, "--run-dir", str(training),
        "--trainer", arm, "--initialization", "frozen", "--distributed", "--no-intermediate-previews",
    ])
    result = read_json(training / "attempts" / "train-001" / "artifacts" / "result.json")
    if result["iteration"] != 30000 or result["world_size"] != 2:
        raise ValueError("main training budget mismatch")
    source_model = training / result["model_path"]
    sor = output / "sor"
    run_command(output, "sor", [
        "scripts/filter_gaussian_sor.py", "--model-snapshot", str(source_model),
        "--output-dir", str(sor), "--nb-neighbors", "30", "--std-ratio", "2.0", "--band-opacity", "0.05",
    ])
    selection = output / "selection"
    model = sor / "filtered-model.pt"
    run_command(output, "selection", [
        "scripts/evaluate_gaussian.py", *common, "--model", str(model),
        "--split", "validation", "--output-dir", str(selection),
        "--progress", str(training / result["progress_path"]),
    ])
    final_fit = output / "train-only"
    run_command(output, "train-only", [
        "scripts/run_gaussian_final_fit.py", *common, "--source-model", str(model),
        "--selection-evaluation", str(selection / "evaluation.json"), "--output-dir", str(final_fit),
        "--train-only-control", "--distributed",
    ])
    record = read_json(final_fit / "record.json")
    if (record["optimizer_updates"], record["camera_samples"], record["input_splits"], record["topology_changed"]) != (2000, 4000, ["train"], False):
        raise ValueError("Train-only record violates the frozen budget")
    run_command(output, "export", [
        "scripts/export_gaussian.py", "--model", str(final_fit / "model.pt"),
        "--dataset-contract", str(dataset), "--resolved-config-json", str(config),
        "--evaluation", str(final_fit / "evaluation.json"), "--output-dir", str(output / "export"),
        "--checkpoint-hash", result["final_checkpoint_hash"],
        "--postprocess-record", str(sor / "filter-record.json"), "--postprocess-mask", str(sor / "filter-mask.npz"),
        "--final-fit-record", str(final_fit / "record.json"),
    ])
    write_json(output / "complete.json", {
        "arm": arm, "protocol_sha256": expected_sha256,
        "selection_evaluation_sha256": sha256_file(selection / "evaluation.json"),
        "final_fit_record_sha256": sha256_file(final_fit / "record.json"),
        "final_evaluation_sha256": sha256_file(final_fit / "evaluation.json"),
        "test": "not_run", "promotion_eligible": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source", type=Path, required=True)
    preparation.add_argument("--output-dir", type=Path, required=True)
    preparation.add_argument("--longest-edge", type=int, choices=(1920, 3840), default=3840)
    preparation.add_argument("--mapper-seed-limit", type=int, choices=(1000, 2500, 3500), default=1000)
    reuse = commands.add_parser("prepare-reuse")
    reuse.add_argument("--source-experiment", type=Path, required=True)
    reuse.add_argument("--output-dir", type=Path, required=True)
    arm = commands.add_parser("run-arm")
    arm.add_argument("--output-dir", type=Path, required=True)
    arm.add_argument("--arm", choices=ARMS, required=True)
    arm.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.source.resolve(), args.output_dir.resolve(),
            longest_edge=args.longest_edge, mapper_seed_limit=args.mapper_seed_limit,
        )
    elif args.command == "prepare-reuse":
        parent = args.source_experiment.resolve()
        source = Path(read_json(parent / "prepare-started.json")["source"])
        prepare(source.resolve(), args.output_dir.resolve(), reuse_experiment=parent)
    else:
        run_arm(args.output_dir.resolve(), args.arm, args.protocol_sha256)


if __name__ == "__main__":
    main()
