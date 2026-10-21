#!/usr/bin/env python3
"""Implement-only research entry: explicitly authorized, bounded 12k MCMC observation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import gzip
import os
import json
from pathlib import Path
import shutil
import socket
import sys

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.absgrad_resources import TrainingMonitor, run_stage
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
from scripts.mcmc_lifecycle_observer import START, EVENT, END, ATOL, RTOL, GROUPS, LifecycleObserver, digest_state, reuse_record
from scripts.run_camera_size_lifecycle import OVERLAY
from scripts.run_camera_size_scene_pair import verify_terminal, verify_validation
from scripts.run_mcmc_stability_ablation import PROJECT, LIMITS, unlimited_host_cgroup
from scripts.prepare_gsplat_camera_sizes import file_hashes, verify_overlay

PROFILE = "mcmc_lifecycle_observer_v1"
SOURCE = PROJECT / "outputs/analysis/apartment-camera-size-pair-20261021-v1"
SOURCE_PROTOCOL = "51b81adcdfbd0cfdae8f68946c6d55dc5d45fcca39e6c26040df97d0828439bb"
CORE = "5941370c16bb04301db2f59477468eebe62326201d12c487980a1dfb51dcde8a"
ARMS = ("baseline", "repair")
BRANCHES = ("control_a", "control_b", "observed")
CORE_FILES = ("dataset.py", "model.py", "model_io.py", "evaluation.py", "checkpoint.py", "absgrad_resources.py",
              "readiness.py", "render.py", "runtime.py", "trainer.py", "training_math.py", "initialization.py")


def core_hash():
    digest = hashlib.sha256()
    for name in CORE_FILES:
        digest.update(name.encode())
        digest.update((PROJECT / "src/image3d_scenegraph/gaussian" / name).read_bytes())
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checked_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe report path")
    return path


def check_files(root, files):
    for name, digest in files.items():
        if sha256_file(checked_path(root, name)) != digest:
            raise ValueError(f"bound file changed: {name}")


class ResumeMonitor(TrainingMonitor):
    def __init__(self, progress, telemetry):
        super().__init__(progress, telemetry, updates=END, main_stage=True, limits=LIMITS)
        self.iteration = START

    def record(self):
        result = super().record()
        result.update(optimizer_updates_observed=self.iteration - START,
                      camera_samples_observed=2 * (self.iteration - START), absolute_iteration=self.iteration)
        return result


def prepare(root):
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    if core_hash() != CORE or sha256_file(SOURCE / "protocol.json") != SOURCE_PROTOCOL:
        raise ValueError("frozen source core or protocol differs")
    pair = read(SOURCE / "protocol.json")
    check_files(Path("/"), {k.lstrip("/"): v for k, v in pair["files"].items()})
    if file_hashes(Path(pair["source_package"])) != pair["source_hashes"]:
        raise ValueError("original package changed")
    verify_overlay(OVERLAY, OVERLAY / "gsplat")
    replay = validate_replay_bundle(Path(pair["replay"]))
    bound = {str(SOURCE / "protocol.json"): SOURCE_PROTOCOL}
    parents = {}
    for arm in ARMS:
        training = SOURCE / arm / "training"
        result_path = training / "attempts/train-001/artifacts/result.json"
        result = read(result_path)
        hashes = verify_terminal(training, result, pair, arm)
        complete = read(SOURCE / arm / "complete.json")
        if complete["status"] != "complete" or any(complete[k] != v for k, v in hashes.items()):
            raise ValueError("source pair is not complete")
        for path in (result_path, SOURCE / arm / "complete.json", training / "attempts/train-001/attempt.json"):
            bound[str(path)] = sha256_file(path)
        parents[arm] = {"training": str(training), "checkpoint_hash": hashes["checkpoint_sha256"]}
    for path in (Path(__file__).resolve(), Path(__file__).with_name("mcmc_lifecycle_observer.py").resolve(),
                 SOURCE / "config.json"):
        bound[str(path)] = sha256_file(path)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "protocol.json", {"profile": PROFILE, "core": CORE, "source_protocol": SOURCE_PROTOCOL,
        "source": str(SOURCE), "pair": pair, "parents": parents, "bound_files": bound, "replay_record": replay,
        "start": START, "event": EVENT, "end": END, "updates_per_branch": END - START,
        "nominal_iterations": 30000, "arms": list(ARMS), "branches": list(BRANCHES),
        "observer_identity": {"script_sha256": sha256_file(Path(__file__).with_name("mcmc_lifecycle_observer.py")),
            "comparison": "exact_canonical_state_and_sampling", "contribution_atol": ATOL, "contribution_rtol": RTOL,
            "contribution_steps": [START + 1, END], "contribution_cadence": "every_training_step",
            "hooks": ["camera_batch", "training_render", "strategy_pre_post", "relocation", "original_sampler", "adam_pre_post"]},
        "limits": LIMITS, "host_limits": None, "gate3": "not_authorized_or_implemented",
        "test_rgb": "not_decoded_or_evaluated", "status": "prepared_not_executed"})


def protocol_at(root, expected, *, runtime=False):
    if not expected or sha256_file(root / "protocol.json") != expected:
        raise ValueError("explicit frozen protocol SHA required")
    p = read(root / "protocol.json")
    if (p["profile"], p["core"], p["start"], p["event"], p["end"], p["nominal_iterations"], p["branches"], p["arms"]) != (
            PROFILE, CORE, START, EVENT, END, 30000, list(BRANCHES), list(ARMS)):
        raise ValueError("unsupported protocol or execution budget")
    if p["limits"] != LIMITS or p["host_limits"] is not None:
        raise ValueError("resource contract changed")
    if runtime:
        if core_hash() != CORE:
            raise ValueError("source core changed")
        for name, digest in {**p["pair"]["files"], **p["bound_files"]}.items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"frozen input changed: {name}")
        if file_hashes(Path(p["pair"]["source_package"])) != p["pair"]["source_hashes"]:
            raise ValueError("original package changed")
        verify_overlay(OVERLAY, OVERLAY / "gsplat")
    return p


def verify_checkpoint(training, attempt_id, iteration, expected):
    from image3d_scenegraph.gaussian.checkpoint import _COMPONENT_FILES, _json_bytes, load_attempt
    path = training / "attempts" / attempt_id / "checkpoints" / f"iteration_{iteration:09d}"
    metadata = read(path / "checkpoint.json")
    if metadata["provenance"] != expected or metadata["iteration"] != iteration or metadata["purpose"] != "final":
        raise ValueError("checkpoint identity differs")
    if asdict(load_attempt(training, attempt_id).provenance) != expected:
        raise ValueError("attempt provenance differs")
    payload = {k: v for k, v in metadata.items() if k != "checkpoint_hash"}
    if hashlib.sha256(_json_bytes(payload)).hexdigest() != metadata["checkpoint_hash"]:
        raise ValueError("checkpoint metadata changed")
    if set(metadata["files"]) != set(_COMPONENT_FILES):
        raise ValueError("checkpoint components differ")
    for key, name in _COMPONENT_FILES.items():
        entry = metadata["files"][key]
        file = checked_path(path, name)
        if entry["path"] != name or file.stat().st_size != entry["bytes"] or sha256_file(file) != entry["sha256"]:
            raise ValueError("checkpoint component changed")
    if read(path / "scheduler.bin") != {"iteration": iteration, "nominal_iterations": 30000, "requested_stop_iteration": iteration}:
        raise ValueError("scheduler differs")
    return metadata


def parent_provenance(p, arm):
    pair = p["pair"]
    return {"dataset_hash": pair["dataset_hash"], "effective_config_hash": pair["config_hash"],
            "code_hash": CORE, "environment_hash": pair["environments"][arm]["hash"]}


def run_branch(args, p):
    arm, branch, root = args.arm, args.branch, args.output_root
    unit = f"image3d-mcmc-lifecycle-{arm}-{branch}-v1.service"
    unlimited_host_cgroup(unit)
    if branch != "control_a":
        completed(root, p, arm, "control_a", args.protocol_sha256)
    if branch == "observed":
        completed(root, p, arm, "control_b", args.protocol_sha256)
        compare_audits(root / arm / "control_a", root / arm / "control_b")
    with FileLease(PROJECT / "outputs/.gpu.lock") as lease:
        require_idle_gpu()
        from image3d_scenegraph.gaussian.replay import validate_replay_bundle
        validate_replay_bundle(Path(p["pair"]["replay"]))
        parent = Path(p["parents"][arm]["training"])
        metadata = verify_checkpoint(parent, "train-001", START, parent_provenance(p, arm))
        if metadata["checkpoint_hash"] != p["parents"][arm]["checkpoint_hash"]:
            raise ValueError("parent checkpoint changed")
        size = sum(e["bytes"] for e in metadata["files"].values())
        if shutil.disk_usage(PROJECT).free < max(20 * 1024**3, 4 * size + 4 * 1024**3):
            raise ValueError("insufficient space for parent copy, final checkpoint and staging")
        stage = root / arm / branch
        stage.mkdir(parents=True, exist_ok=False)
        destination = stage / "training/attempts/train-001"
        destination.mkdir(parents=True)
        shutil.copyfile(parent / "attempts/train-001/attempt.json", destination / "attempt.json")
        source = parent / f"attempts/train-001/checkpoints/iteration_{START:09d}"
        shutil.copytree(source, destination / f"checkpoints/iteration_{START:09d}")
        verify_checkpoint(stage / "training", "train-001", START, parent_provenance(p, arm))
        prefix = ["/usr/bin/env", "-u", "PYTHONPATH"] if arm == "baseline" else ["/usr/bin/env", "PYTHONPATH=" + str(OVERLAY)]
        command = prefix + [sys.executable, str(Path(__file__).resolve()), "--output-root", str(root), "--worker",
            "--arm", arm, "--branch", branch, "--protocol-sha256", args.protocol_sha256, "--authorize-execution",
            "--lease-fd", str(lease.fileno())]
        monitor = ResumeMonitor(stage / "training/attempts/trace-001/artifacts/progress.jsonl", stage / "memory")
        run_stage(stage, "resume", command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
        protocol_at(root, args.protocol_sha256, runtime=True)
        verify_checkpoint(parent, "train-001", START, parent_provenance(p, arm))
        final = verify_checkpoint(stage / "training", "trace-001", END, parent_provenance(p, arm))
        result_path = stage / "training/attempts/trace-001/artifacts/result.json"
        result = read(result_path)
        if (result["iteration"], result["nominal_iterations"], result["requested_stop_iteration"], result["termination_reason"],
                result["candidate_iteration"], result["world_size"], result["gaussian_cap"], result["final_checkpoint_hash"]) != (
                END, 30000, END, "execution_budget", END, 2, 3000000, final["checkpoint_hash"]):
            raise ValueError("resume result differs from fixed window")
        verify_validation(result["final_validation"], p["pair"])
        files = list(stage.glob("audit-rank-*.jsonl")) + list(stage.glob("worker-rank-*.json"))
        files += list(stage.glob("diagnostic-rank-*.jsonl")) + list(stage.glob("cohort-rank-*.json.gz"))
        files += [result_path, stage / "resume.exit.json"]
        write_new(stage / "complete.json", {"status": "complete", "arm": arm, "branch": branch,
            "protocol_sha256": args.protocol_sha256, "resources": monitor.record(), "checkpoint_sha256": final["checkpoint_hash"],
            "model_sha256": sha256_file(checked_path(stage / "training", result["model_path"])),
            "files": {str(f.relative_to(stage)): sha256_file(f) for f in files},
            "selection_scope": "resumed_window_only_not_historical_best"})


def worker(local_rank, rank, world_size, args):
    import torch
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig
    from image3d_scenegraph.gaussian.initialization import load_frozen_initialization
    from image3d_scenegraph.gaussian.checkpoint import load_checkpoint, CheckpointProvenance
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    from unittest.mock import patch

    if world_size != 2 or rank != local_rank:
        raise ValueError("exactly two local ranks required")
    torch.cuda.set_device(local_rank)
    p = protocol_at(args.output_root, args.protocol_sha256, runtime=True)
    import gsplat
    expected_package = Path(p["pair"]["source_package"]) if args.arm == "baseline" else OVERLAY / "gsplat"
    if Path(gsplat.__file__).resolve().parent != expected_package.resolve():
        raise ValueError("actual gsplat import path differs from frozen environment")
    stage = args.output_root / args.arm / args.branch
    cfg = read(Path(p["source"]) / "config.json")
    resolved = ResolvedGaussianConfig(cfg["requested_profile"], cfg["effective_config"], cfg["effective_config_hash"])
    replay = Path(p["pair"]["replay"])
    contract, bundle = read(replay / "dataset.json"), p["replay_record"]
    initialization = load_frozen_initialization(replay / bundle["initialization_path"], replay / bundle["initialization_diagnostics_path"],
                                               expected_sha256=bundle["initialization_sha256"])
    provenance = trainer.training_provenance(dataset_hash=contract["dataset_hash"], effective_config_hash=resolved.effective_config_hash, world_size=2)
    if asdict(provenance) != parent_provenance(p, args.arm):
        raise ValueError("runtime core/environment differs from parent")
    observer = LifecycleObserver(stage, rank, resolved.effective_config, args.branch == "observed")
    original_writer = trainer._write_streaming_checkpoint
    verified = False

    def writer(*positional, **kwargs):
        nonlocal verified
        before = observer.state()
        record = original_writer(*positional, **kwargs)
        if observer.state() != before:
            raise ValueError("checkpoint save changed training state")
        loaded = load_checkpoint(stage / "training", "trace-001", END, expected_provenance=CheckpointProvenance(**asdict(provenance)))
        state = loaded.state
        device = torch.device("cpu")
        model = trainer._load_model(trainer._checkpoint_rank_bytes(state.model, rank, 2), device)
        adam = trainer._torch_load(trainer._checkpoint_rank_bytes(state.optimizer, rank, 2), device)
        dense = trainer._torch_load(trainer._checkpoint_rank_bytes(state.densification, rank, 2), device)
        rng = trainer._torch_load(trainer._checkpoint_rank_bytes(state.rng, rank, 2), device)
        if (digest_state(dict(model.params.items())) != before["parameters"] or digest_state(adam) != before["adam"]
                or digest_state(dense["strategy_state"]) != before["strategy"] or digest_state(rng) != before["rng"]
                or dense["camera_order"] != observer.camera["order"] or dense["camera_cursor"] != observer.camera["cursor"]):
            raise ValueError("saved tensor state differs from live terminal state")
        observer.emit("audit", {"event": "checkpoint", "iteration": END, "state": before})
        verified = True
        return record

    failure_root = stage / f"failure-rank-{rank}"
    failure_root.mkdir(exist_ok=False)
    with memory_telemetry(stage / "memory", local_rank, rank), failure_record(failure_root, args.arm, args.branch):
        with observer.installed(), patch.object(trainer, "_write_streaming_checkpoint", writer):
            trainer.train_gaussians(contract=contract, dataset_root=replay, initialization=initialization,
                resolved_config=resolved, run_dir=stage / "training", attempt_id="trace-001", attempt_kind="resume",
                parent_attempt_id="train-001", resume_iteration=START, local_rank=local_rank, world_rank=rank, world_size=2,
                stop_after_iteration=END, save_intermediate_previews=False,
                cancel_requested=lambda: (stage / "cancel").exists())
        if not verified:
            raise ValueError("terminal checkpoint was not verified")
        write_new(stage / f"worker-rank-{rank}.json", {"rank": rank, "checkpoint_state_verified": True,
            "core": provenance.code_hash, "environment_hash": provenance.environment_hash, "new_updates": END - START})


def audit_rows(stage, rank):
    rows = [json.loads(line) for line in (stage / f"audit-rank-{rank}.jsonl").read_text().splitlines()]
    expected = [("first_render", START + 1)] + [("step", i) for i in range(START + 1, END + 1)] + [("checkpoint", END)]
    if [(r["event"], r["iteration"]) for r in rows] != expected:
        raise ValueError("missing or reordered audit states")
    for row in rows:
        if set(row["state"]) != {"parameters", "adam", "strategy", "camera", "rng"}:
            raise ValueError("missing canonical state")
    for row in rows[1:-1]:
        if len(row["sampling"]) != (1 if row["iteration"] in (EVENT, END) else 0):
            raise ValueError("missing or unexpected source sampling")
    return rows


def compare_audits(left, right):
    for rank in (0, 1):
        a, b = audit_rows(left, rank), audit_rows(right, rank)
        for x, y in zip(a, b):
            if x != y:
                raise ValueError(f"state/sampling mismatch rank={rank} event={x['event']} iteration={x['iteration']}")


def completed(root, p, arm, branch, digest):
    stage = root / arm / branch
    c = read(stage / "complete.json")
    if (c["status"], c["arm"], c["branch"], c["protocol_sha256"]) != ("complete", arm, branch, digest):
        raise ValueError("branch completion identity differs")
    if (c["resources"]["optimizer_updates_observed"], c["resources"]["camera_samples_observed"], c["resources"]["absolute_iteration"]) != (200, 400, END):
        raise ValueError("branch execution budget differs")
    required = {f"{name}-rank-{rank}.{suffix}" for rank in (0, 1) for name, suffix in (("audit", "jsonl"), ("worker", "json"))}
    if branch == "observed":
        required |= {f"{name}-rank-{rank}.{suffix}" for rank in (0, 1) for name, suffix in (("diagnostic", "jsonl"), ("cohort", "json.gz"))}
    if not required <= c["files"].keys():
        raise ValueError("missing bound rank files")
    check_files(stage, c["files"])
    for rank in (0, 1):
        w = read(stage / f"worker-rank-{rank}.json")
        if w != {"rank": rank, "checkpoint_state_verified": True, "core": CORE,
                 "environment_hash": p["pair"]["environments"][arm]["hash"], "new_updates": 200}:
            raise ValueError("worker identity or checkpoint validation failed")
    return c


def gate1(root, p, digest):
    files = {}
    for arm in ARMS:
        for branch in BRANCHES:
            completed(root, p, arm, branch, digest)
            name = f"{arm}/{branch}/complete.json"
            files[name] = sha256_file(root / name)
        try:
            compare_audits(root / arm / "control_a", root / arm / "control_b")
        except ValueError as exc:
            raise ValueError(f"control_reproducibility_failed {arm}: {exc}") from exc
        try:
            compare_audits(root / arm / "control_a", root / arm / "observed")
        except ValueError as exc:
            raise ValueError(f"observer_equivalence_failed {arm}: {exc}") from exc
    write_new(root / "gate1.json", {"status": "passed", "protocol_sha256": digest, "files": files,
        "scope": "exact_state_and_sampling_full_observer_enabled", "gate2": "not_analyzed"})


def gate2(root, p, digest):
    gate = read(root / "gate1.json")
    if gate["status"] != "passed" or gate["protocol_sha256"] != digest:
        raise ValueError("Gate 2 requires passed matching Gate 1")
    if set(gate["files"]) != {f"{a}/{b}/complete.json" for a in ARMS for b in BRANCHES}:
        raise ValueError("Gate 1 has incomplete branch coverage")
    check_files(root, gate["files"])
    reports = {}
    for arm in ARMS:
        for branch in BRANCHES:
            completed(root, p, arm, branch, digest)
        compare_audits(root / arm / "control_a", root / arm / "control_b")
        compare_audits(root / arm / "control_a", root / arm / "observed")
        reports[arm] = {}
        for rank in (0, 1):
            stage = root / arm / "observed"
            rows = [json.loads(line) for line in (stage / f"diagnostic-rank-{rank}.jsonl").read_text().splitlines()]
            selected = {event: [r for r in rows if r["event"] == event] for event in (
                "relocation_before", "relocation_after", "cohort_end_before_relocation", "opacity_update", "gradient", "contribution")}
            if any(len(selected[e]) != 1 for e in ("relocation_before", "relocation_after", "cohort_end_before_relocation")):
                raise ValueError("incomplete cohort endpoints")
            for event, expected in (("relocation_before", EVENT), ("relocation_after", EVENT), ("cohort_end_before_relocation", END)):
                row = selected[event][0]
                if row["iteration"] != expected or set(row["groups"]) != set(GROUPS):
                    raise ValueError("wrong cohort identity/window")
                if sum(g["count"] for g in row["groups"].values()) != 1500000:
                    raise ValueError("cohort partition is not complete")
            if [r["iteration"] for r in selected["opacity_update"]] != list(range(EVENT + 1, END + 1)):
                raise ValueError("incomplete cohort optimizer window")
            for parameter in ("opacities", "scales"):
                if [r["iteration"] for r in selected["gradient"] if r["parameter"] == parameter] != list(range(EVENT + 1, END + 1)):
                    raise ValueError("incomplete gradient window")
            if [r["iteration"] for r in selected["contribution"]] != list(range(START + 1, END + 1)):
                raise ValueError("contribution observation was not enabled for Gate 1")
            for r in selected["contribution"]:
                if len(r["weight_sum"]) != 6 or r["camera_rank"] != rank:
                    raise ValueError("invalid owner contribution mapping")
            with gzip.open(stage / f"cohort-rank-{rank}.json.gz", "rt") as handle:
                identities = json.load(handle)
            if (identities["owner_rank"], identities["event"]) != (rank, EVENT):
                raise ValueError("cohort owner/event differs")
            reuse = reuse_record(1500000, identities["target_slots"], identities["sampled_source_slots"])
            before = selected["relocation_before"][0]
            for key, value in reuse.items():
                if json.dumps(before[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                    raise ValueError("source reuse identity differs")
            if reuse["group_sizes"] != {g: before["groups"][g]["count"] for g in GROUPS}:
                raise ValueError("identity cohort sizes differ")
            reports[arm][str(rank)] = selected
    write_new(root / "gate2.json", {"status": "observed_not_causal_intervention", "protocol_sha256": digest,
        "gate1_sha256": sha256_file(root / "gate1.json"), "arms": reports, "groups": list(GROUPS),
        "window": [EVENT, END], "test_rgb": "not_loaded", "gate3": "not_authorized",
        "limits": ["one_late_relocation_event", "image_gradient_is_float_residual", "contribution_is_T_alpha_not_quality",
                   "associations_do_not_identify_causal_effect_of_resetting_adam"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "worker", "gate1", "gate2"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--authorize-prepare", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(PROJECT / "outputs/analysis"):
        raise SystemExit("输出必须是独立研究目录")
    if args.prepare or args.run or args.worker:
        if socket.gethostname() != "i-94B8D131" or str(PROJECT) != "/usr/local/3dgs_new/Image3D-SceneGraph":
            raise SystemExit("本机禁止Torch/模型执行；仅允许指定远端")
        if args.prepare and not args.authorize_prepare:
            raise SystemExit("准备需要单独明确授权参数")
        if (args.run or args.worker) and (not args.authorize_execution or args.arm is None or args.branch is None):
            raise SystemExit("实验未获显式执行授权或缺少分支")
    if args.prepare:
        prepare(args.output_root)
        return
    p = protocol_at(args.output_root, args.protocol_sha256, runtime=args.run or args.worker)
    if args.run:
        run_branch(args, p)
    elif args.worker:
        unlimited_host_cgroup(f"image3d-mcmc-lifecycle-{args.arm}-{args.branch}-v1.service")
        import fcntl
        if args.lease_fd is None or os.readlink(f"/proc/self/fd/{args.lease_fd}") != str(PROJECT / "outputs/.gpu.lock"):
            raise ValueError("worker requires the inherited GPU lease")
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from gsplat.distributed import cli
        if torch.cuda.device_count() != 2:
            raise ValueError("requires exactly two GPUs")
        cli(worker, args, verbose=True)
    elif args.gate1:
        gate1(args.output_root, p, args.protocol_sha256)
    else:
        gate2(args.output_root, p, args.protocol_sha256)


if __name__ == "__main__":
    main()
