"""Independent C/U/O admission and conditional Q1 analysis; phase A stays frozen."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import fcntl
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from unittest.mock import patch

from . import admission, cohort_analysis
from .observer import Endpoint, CohortObserver
from . import run as phase_a

connected, legacy, snapshots = phase_a.connected, phase_a.legacy, phase_a.snapshots
read, write_new, sha256 = phase_a.read, phase_a.write_new, phase_a.sha256
PROJECT, REL = phase_a.PROJECT, phase_a.REL
MODULE = REL.replace('/', '.') + '.run_b'
RUNS = PROJECT / REL / 'runs'
UNIT = 'image3d-observer-admission-v1.service'
A_ROOT = RUNS / 'masks-20261026-v1'
A_SHA = 'ab82dba86d0f3adc3167b703f1ff13c74ca9d1f4e98e21e891c1546df3956426'
A_COMPARISON_SHA = '9b4f950de71ba06a0e9b2df653b5e98cddf36e6ca1a1cf041682b5d39ebce2ae'
FILES = tuple(REL + '/' + n for n in ('run_b.py', 'admission.py', 'observer.py', 'cohort_analysis.py',
    'tests/test_admission.py', 'tests/test_admission_remote.py'))
DEPENDENCIES = tuple(dict.fromkeys((*phase_a.DEPENDENCIES, *FILES)))
CONTRACT = {'profile': 'mcmc_observer_admission_v1', 'phase': 'B', 'cases': list(admission.CASES),
    'start': admission.START, 'event': admission.EVENT, 'end': admission.END, 'owner_size': admission.OWNER_SIZE,
    'nominal_iterations': 30000, 'phase_A_root': str(A_ROOT), 'phase_A_protocol_sha256': A_SHA,
    'phase_A_comparison_sha256': A_COMPARISON_SHA, 'parent_checkpoint_sha256': connected.PARENT_SHA,
    'core': legacy.CORE, 'new_updates_max': 1800, 'training_camera_forwards_max': 3600,
    'loss_backwards_max': 3600, 'contribution_forwards_max': 1200, 'optimizer_group_steps_max': 21600,
    'global_relocation_events_max': 18, 'per_owner_relocation_calls_max': 36,
    'suite_timeout_seconds': 21600, 'collective_timeout_seconds': 120,
    'max_output_bytes': 30 * 1024**3, 'disk_admission_bytes': 38 * 1024**3,
    'max_per_rank_reserved_bytes': 18 * 1024**3, 'host_limits': None,
    'ratio_noise_ceiling': .001, 'minimum_exploratory_owner_effect': .01,
    'loss_floor': [2e-6, 2e-4], 'loss_ceiling': [2e-6, 1e-3],
    'stop': 'after_12200_strategy_and_noise_before_Validation',
    'cohort_stop': 'before_12200_second_relocation',
    'old_gate1': 'not_reclassified', 'cross_run_slot_identity': False,
    'preflight_budget': {'synthetic_updates': 8, 'training_camera_forwards': 16,
        'contribution_forwards': 8, 'reference_contribution_forwards': 2,
        'loss_backwards': 16, 'analytic_autograd_calls': 2, 'optimizer_group_steps': 96,
        'coordinated_rejection_checks': 1, 'rng_rejection_checks': 1, 'real_image_decodes': 0}}


def output_root(path, preparing=False):
    result = phase_a.output_root(path, preparing)
    if not result.name.startswith('admission-'):
        raise ValueError('phase B requires an independent admission-* directory')
    return result


def execution_commit():
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *DEPENDENCIES], cwd=PROJECT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *DEPENDENCIES], cwd=PROJECT, check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()


def configuration(source):
    from scripts import run_mcmc_lifecycle_trace as lifecycle
    value = read(lifecycle.SOURCE / 'config.json')
    cfg = value['effective_config']
    if (value['effective_config_hash'] != source['pair']['config_hash'] or cfg['iterations'] != 30000
            or cfg['strategy']['name'] != 'mcmc_v1' or cfg['strategy']['mcmc_min_opacity'] != .005
            or cfg['strategy']['gaussian_cap'] != 3000000
            or cfg['densification']['every_iterations'] != 100
            or not cfg['densification']['start_iteration'] < admission.EVENT < admission.END < cfg['densification']['end_iteration']
            or any(admission.START < i < admission.END for i in cfg['evaluation']['validation_iterations'])):
        raise ValueError('frozen lifecycle configuration not suitable for this window')
    return value


def source_identity():
    if sha256(A_ROOT / 'protocol.json') != A_SHA or sha256(A_ROOT / 'comparison.json') != A_COMPARISON_SHA:
        raise ValueError('phase-A evidence changed')
    terminal = read(A_ROOT / 'execution.json')
    if (terminal['status'], terminal['exit_code'], terminal['protocol_sha256'], terminal['comparison_sha256']) != (
            'completed', 0, A_SHA, A_COMPARISON_SHA):
        raise ValueError('phase A incomplete')
    phase_a.verify_result(A_ROOT, A_SHA)
    source = connected.native_source()
    meta = connected.parent_metadata(source)
    config = configuration(source)
    from image3d_scenegraph.gaussian.replay import validate_replay_bundle
    replay = validate_replay_bundle(Path(source['pair']['replay']))
    if replay != source['replay_record']:
        raise ValueError('replay identity differs')
    return {'native': source, 'config': config, 'parent_metadata': meta,
            'phase_A_execution_sha256': sha256(A_ROOT / 'execution.json')}


def output_bytes(root):
    total = 0
    for path in root.rglob('*'):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # Telemetry atomically renames its temporary files while we scan.
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError('nonregular entry in phase-B output')
    return total


def reserve(root, extra):
    if extra < 0 or output_bytes(root) + 2 * extra + 2 * 1024**2 > CONTRACT['max_output_bytes']:
        raise ValueError('phase-B output budget exceeded')
    if shutil.disk_usage(root).free < 2 * extra + 4 * 1024**3:
        raise ValueError('phase-B disk reserve')


def prepare(root, preflight):
    commit = execution_commit()
    source = source_identity()
    if shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
        raise ValueError('insufficient disk for frozen nine-case suite')
    # Model endpoints are smaller than the source model+optimizer checkpoint; use the latter as an upper bound.
    checkpoint_bytes = sum(v['bytes'] for v in source['parent_metadata']['files'].values())
    estimated = 10 * checkpoint_bytes + 4 * 1024**3
    if estimated > CONTRACT['max_output_bytes']:
        raise ValueError('source checkpoint and nine endpoints exceed output budget')
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'protocol.json', {**CONTRACT, 'implementation_commit': commit, 'source_identity': source,
        'estimated_output_bytes': estimated, 'code_sha256': {n: sha256(PROJECT / n) for n in DEPENDENCIES},
        'preflight': preflight, 'status': 'prepared_not_executed'})


def protocol_at(root, digest):
    if not digest or sha256(snapshots.safe_file(root, 'protocol.json')) != digest:
        raise ValueError('explicit phase-B protocol SHA required')
    p = read(root / 'protocol.json')
    if any(p.get(k) != v for k, v in CONTRACT.items()) or set(p['code_sha256']) != set(DEPENDENCIES):
        raise ValueError('phase-B contract or dependency coverage differs')
    for name, expected in p['code_sha256'].items():
        if sha256(snapshots.safe_file(PROJECT, name)) != expected:
            raise ValueError('phase-B code changed')
    return p


def verify_inputs(p):
    if verify_preflight(p['preflight']['path']) != p['preflight']:
        raise ValueError('preflight proof changed')
    if execution_commit() != p['implementation_commit'] or source_identity() != p['source_identity']:
        raise ValueError('frozen phase-B source or implementation changed')


def native_artifacts(root, name):
    return root / 'training/attempts' / name / 'artifacts'


def metrics(root, name, p):
    folder = native_artifacts(root, name)
    histories = [read(root / name / f'native-tail-rank-{r}.json') for r in (0, 1)]
    progress = admission.jsonl(folder / 'progress.jsonl')
    if progress != histories[0]:
        raise ValueError('native progress and terminal history differ')
    health = {str(r): admission.jsonl(folder / f'mcmc-rank-{r}.jsonl') for r in (0, 1)}
    train_ids = read(Path(p['source_identity']['native']['pair']['replay']) / 'dataset.json')['splits']['train']
    result = admission.extract(histories[0], health, train_ids)
    other = admission.extract(histories[1], health, train_ids)
    if result != other:
        raise ValueError('rank terminal metrics or camera sequence differ')
    return result


def verify_worker(stage, name, rank):
    import numpy as np
    record = read(stage / f'worker-rank-{rank}.json')
    expected = {'status': 'terminal_complete', 'case': name, 'rank': rank, 'iteration': admission.END,
                'new_updates': 200, 'validation_evaluated': False, 'native_training_complete': False,
                'intermediate_observer': name.startswith('O')}
    if any(record.get(k) != v for k, v in expected.items()) or not record['loaded_train_ids']:
        raise ValueError('worker identity or terminal boundary differs')
    folder = stage / f'endpoint-rank-{rank}'
    if sha256(folder / 'manifest.json') != record['endpoint_manifest_sha256']:
        raise ValueError('endpoint manifest changed')
    snap = snapshots.load_snapshot(folder)
    leaves = snapshots.snapshot_leaves(snap)
    if not all(('parameters', n) in leaves for n in snapshots.PARAMETERS):
        raise ValueError('missing terminal model parameters')
    for name_, node in leaves.items():
        if node['type'] == 'nonfinite':
            raise ValueError('nonfinite endpoint')
        if node['type'] == 'tensor':
            array = np.load(snapshots.safe_file(folder, node['file']), mmap_mode='r', allow_pickle=False)
            if not np.isfinite(array).all():
                raise ValueError('nonfinite endpoint tensor')
            if name_[:1] == ('parameters',) and array.shape[0] != admission.OWNER_SIZE:
                raise ValueError('endpoint owner topology differs')
    if name.startswith('O'):
        observation = record['observation']
        if set(observation['manifests']) != {'cohort-before', 'cohort-initial', 'cohort-final'}:
            raise ValueError('cohort snapshot coverage differs')
        if observation['strategy_posts'] != 200 or observation['contribution_forwards'] != 200:
            raise ValueError('observation budget differs')
        for key, value in observation['manifests'].items():
            folder = stage / f'{key}-rank-{rank}'
            if sha256(folder / 'manifest.json') != value:
                raise ValueError('cohort snapshot changed')
            snapshots.load_snapshot(folder)
        index = stage / f'trajectory-rank-{rank}/index.json'
        if sha256(index) != observation['trajectory_index_sha256']:
            raise ValueError('cohort trajectory index changed')
        value = read(index)
        if [r['iteration'] for r in value['rows']] != list(range(admission.EVENT + 1, admission.END + 1)):
            raise ValueError('trajectory coverage differs')
        for row in value['rows']:
            phase_a.masks.verify_bitset(index.parent, {**row, 'xor_count': row['alive_count']})
    elif record['observation'] is not None or list(stage.glob('*cohort*')) or list(stage.glob('audit-*')):
        raise ValueError('unobserved control contains intermediate observer data')
    return record


def completed(root, name, digest):
    stage = root / name
    value = read(stage / 'complete.json')
    if (value['status'], value['case'], value['protocol_sha256']) != ('terminal_complete', name, digest):
        raise ValueError('incomplete phase-B case')
    required = {str((stage / name_).relative_to(root)) for name_ in
                ('case.exit.json', 'worker-rank-0.json', 'worker-rank-1.json',
                 'native-tail-rank-0.json', 'native-tail-rank-1.json')}
    attempt = root / 'training/attempts' / name / 'attempt.json'
    required.add(str(attempt.relative_to(root)))
    required |= {str((native_artifacts(root, name) / file).relative_to(root))
                 for file in ('progress.jsonl', 'mcmc-rank-0.jsonl', 'mcmc-rank-1.jsonl')}
    if not required <= value['files'].keys():
        raise ValueError('missing case completion bindings')
    for relative, expected in value['files'].items():
        path = snapshots.safe_file(root, relative)
        if not (path.is_relative_to(stage) or path.is_relative_to(native_artifacts(root, name)) or path == attempt) or sha256(path) != expected:
            raise ValueError('case evidence changed or escaped its case')
    exit_record = read(stage / 'case.exit.json')
    if exit_record['returncode'] != 0 or exit_record['resource_failure'] is not None:
        raise ValueError('case execution failed')
    for rank in (0, 1):
        verify_worker(stage, name, rank)
    return value


class CaseMonitor:
    def __init__(self, root, name, deadline):
        self.root, self.name, self.deadline = root, name, deadline
        self.telemetry = root / name / 'memory'
        self.peaks = [0, 0]

    def __call__(self, elapsed, *, final=False):
        if time.monotonic() >= self.deadline:
            return 'suite_timeout'
        if output_bytes(self.root) > CONTRACT['max_output_bytes']:
            return 'suite_output_limit'
        for rank in (0, 1):
            path = self.telemetry / f'rank-{rank}.json'
            if path.exists():
                r = read(path)
                self.peaks[rank] = max(self.peaks[rank], r['peak_reserved_bytes'])
                if self.peaks[rank] > CONTRACT['max_per_rank_reserved_bytes']:
                    return 'gpu_reserved_limit'
                if final and not r['finished']:
                    return 'unfinished_memory_telemetry'
            elif final:
                return 'missing_memory_telemetry'

    def record(self):
        return {'per_rank_peak_reserved_bytes': self.peaks}


def worker(local_rank, rank, world_size, args):
    import torch
    import gsplat
    from scripts import run_mcmc_first_divergence as native
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig
    from image3d_scenegraph.gaussian.initialization import load_frozen_initialization
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    if world_size != 2 or local_rank != rank:
        raise ValueError('exactly two local ranks required')
    torch.cuda.set_device(rank)
    p = protocol_at(args.output_root, args.protocol_sha256)
    verify_inputs(p)
    source = p['source_identity']['native']
    connected.parent_metadata(source, args.output_root / 'training')
    package = Path(source['pair']['source_package'])
    if Path(gsplat.__file__).resolve().parent != package.resolve():
        raise ValueError('original package required')
    spec = find_spec('gsplat.csrc')
    if spec is None or not isinstance(spec.loader, ExtensionFileLoader) or not spec.origin:
        raise ValueError('precompiled extension required; no JIT warmup')
    binary = snapshots.safe_file(package, Path(spec.origin).relative_to(package))
    if sha256(binary) != source['pair']['source_hashes'].get(str(binary.relative_to(package))):
        raise ValueError('extension identity differs')
    if Path(import_module('gsplat.csrc').__file__).resolve() != binary.resolve():
        raise ValueError('loaded extension differs')
    import_module('gsplat.rendering')
    import_module('gsplat.strategy.ops')
    config = p['source_identity']['config']
    resolved = ResolvedGaussianConfig(config['requested_profile'], config['effective_config'], config['effective_config_hash'])
    replay = Path(source['pair']['replay'])
    bundle = source['replay_record']
    contract = read(replay / 'dataset.json')
    initialization = load_frozen_initialization(replay / bundle['initialization_path'], replay / bundle['initialization_diagnostics_path'],
                                               expected_sha256=bundle['initialization_sha256'])
    provenance = trainer.training_provenance(dataset_hash=contract['dataset_hash'], effective_config_hash=resolved.effective_config_hash, world_size=2)
    if asdict(provenance) != native.lifecycle.parent_provenance(source, 'baseline'):
        raise ValueError('native runtime provenance differs')
    stage = args.output_root / args.case
    endpoint = Endpoint(stage, rank, contract['splits']['train'], lambda n: reserve(args.output_root, n))
    observed = (CohortObserver(stage, rank, resolved.effective_config, lambda n: reserve(args.output_root, n))
                if args.case.startswith('O') else None)
    failure = stage / f'failure-rank-{rank}'
    failure.mkdir(exist_ok=False)
    with memory_telemetry(stage / 'memory', rank, rank), failure_record(failure, 'observer_admission', args.case):
        try:
            with endpoint.installed(), observed.installed() if observed else nullcontext():
                trainer.train_gaussians(contract=contract, dataset_root=replay, initialization=initialization,
                    resolved_config=resolved, run_dir=args.output_root / 'training', attempt_id=args.case,
                    attempt_kind='resume', parent_attempt_id='train-001', resume_iteration=admission.START,
                    local_rank=rank, world_rank=rank, world_size=2, stop_after_iteration=admission.END,
                    save_intermediate_previews=False, cancel_requested=lambda: (stage / 'cancel').exists())
        except snapshots.ProbeComplete:
            def finish():
                if not endpoint.completed or endpoint.calls != 1:
                    raise ValueError('terminal boundary incomplete')
                observation = observed.finish() if observed else None
                write_new(stage / f'worker-rank-{rank}.json', {'status': 'terminal_complete', 'case': args.case,
                    'rank': rank, 'iteration': admission.END, 'new_updates': 200,
                    'budget_basis': 'frozen_native_trainer_and_contiguous_200_step_progress_not_control_optimizer_hooks',
                    'endpoint_manifest_sha256': endpoint.manifest, 'intermediate_observer': observed is not None,
                    'loaded_train_ids': sorted(endpoint.loaded), 'observation': observation,
                    'native_training_complete': False, 'validation_evaluated': False})
            endpoint.consensus('terminal-finish', finish)
        else:
            raise ValueError('native trainer passed the evaluation boundary')


def verify_preflight(path):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    value = read(path)
    if (value['status'] != 'passed' or value['profile'] != CONTRACT['profile']
            or value['implementation_commit'] != execution_commit()
            or value['budget'] != CONTRACT['preflight_budget']
            or value['code_sha256'] != {n: sha256(PROJECT / n) for n in DEPENDENCIES}):
        raise ValueError('independent structural preflight missing or stale')
    if set(value['workers']) != {'0', '1'}:
        raise ValueError('preflight did not cover both ranks')
    if (sha256(A_ROOT / 'protocol.json') != A_SHA
            or value['runtime'] != read(A_ROOT / 'protocol.json')['source_inventory']['runtime']):
        raise ValueError('preflight runtime does not match frozen source')
    expected_counts = {'training_camera_forwards': 8, 'contribution_forwards': 4, 'loss_backwards': 8,
                       'optimizer_group_steps': 48, 'synthetic_updates': 8, 'reference_contribution_forwards': 1}
    for rank, row in value['workers'].items():
        file = snapshots.safe_file(RUNS, Path(row['path']).relative_to(RUNS))
        if sha256(file) != row['sha256']:
            raise ValueError('preflight evidence changed')
        evidence = read(file)
        if (evidence['status'] != 'passed' or evidence['rank'] != int(rank) or evidence['counts'] != expected_counts
                or evidence['coordinated_rejection_checks'] != 1 or evidence['real_image_decodes'] != 0
                or not evidence['active_state_unchanged_by_observation'] or not evidence['terminal_before_evaluation']):
            raise ValueError('structural preflight did not satisfy the frozen budget and invariants')
    return {'path': str(path), 'sha256': sha256(path)}


def calibration_at(root, p, digest):
    values = {}
    for name in admission.CALIBRATION:
        completed(root, name, digest)
        values[name] = metrics(root, name, p)
    expected = {**admission.calibrate(values), 'protocol_sha256': digest,
                'complete_sha256': {n: sha256(root / n / 'complete.json') for n in admission.CALIBRATION}}
    if read(root / 'calibration.json') != expected:
        raise ValueError('calibration changed or contains held-out data')
    return expected


def compare(root, p, digest):
    calibration = calibration_at(root, p, digest)
    accepted = {}
    for name in admission.CASES[3:]:
        completed(root, name, digest)
        value = admission.check_case(name, metrics(root, name, p), calibration)
        if not value['passed'] or read(root / name / 'admission.json') != {**value, 'calibration_sha256': sha256(root / 'calibration.json')}:
            raise ValueError('all independent holdouts must pass before Q1 analysis')
        accepted[name] = value
    write_new(root / 'admission.json', {'status': 'finite_scope_admitted', 'protocol_sha256': digest,
        'calibration_sha256': sha256(root / 'calibration.json'), 'holdouts': accepted,
        'preflight': p['preflight'], 'statistical_equivalence_proven': False, 'old_gate1': 'not_reclassified'})
    comparisons = cohort_analysis.compare_endpoints(root)
    reserve(root, len(json.dumps(comparisons, allow_nan=False).encode()) + 4096)
    write_new(root / 'parameter-comparison.json', comparisons)
    q1 = {name: {str(rank): cohort_analysis.cohort(root / name, rank) for rank in (0, 1)} for name in ('O0', 'O1', 'O2')}
    verify_inputs(p)
    reserve(root, len(json.dumps(q1, allow_nan=False).encode()) + 65536)
    write_new(root / 'comparison.json', {'status': 'analysis_complete', 'protocol_sha256': digest,
        'implementation_commit': p['implementation_commit'], 'admission_sha256': sha256(root / 'admission.json'),
        'parameter_comparison_sha256': sha256(root / 'parameter-comparison.json'),
        'complete_sha256': {n: sha256(root / n / 'complete.json') for n in admission.CASES},
        'Q1': q1, 'old_gate1': 'not_reclassified', 'long_term_causality_proven': False,
        'interpretation': 'accepted_within_window_observer; Q1_associations_not_causal_or_cross_run_identity_proof'})


class AdmissionBlocked(Exception):
    pass


def run(args, p):
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    legacy.unlimited_host_cgroup(UNIT)
    root = args.output_root
    with (root / 'execution.once').open('x'):
        pass
    connected.publish_status(root / 'started.json', {'pid': os.getpid(),
        'proc_start': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        'protocol_sha256': args.protocol_sha256, 'started_unix': time.time()})
    deadline = time.monotonic() + CONTRACT['suite_timeout_seconds']
    terminal = {'status': 'failed', 'exit_code': 1, 'protocol_sha256': args.protocol_sha256}
    accepted, durations = [], []
    def expired(signum, frame):
        raise TimeoutError('entire phase B exceeded six hours')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(CONTRACT['suite_timeout_seconds'])
    try:
        with FileLease(PROJECT / 'outputs/.gpu.lock') as lease:
            require_idle_gpu()
            verify_inputs(p)
            if shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
                raise ValueError('insufficient disk admission')
            source = p['source_identity']['native']
            parent = Path(source['parent_training']) / 'attempts/train-001'
            destination = root / 'training/attempts/train-001'
            destination.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(parent / 'attempt.json', destination / 'attempt.json')
            shutil.copytree(parent / 'checkpoints/iteration_000012000', destination / 'checkpoints/iteration_000012000')
            for path in destination.rglob('*'):
                path.chmod(0o555 if path.is_dir() else 0o444)
            destination.chmod(0o555)
            connected.parent_metadata(source, root / 'training')
            calibration = None
            for position, name in enumerate(admission.CASES):
                if name not in admission.CALIBRATION:
                    calibration = calibration_at(root, p, args.protocol_sha256)
                    if calibration['status'] != 'calibrated':
                        raise ValueError('blocked calibration cannot launch holdouts')
                if durations and time.monotonic() + max(durations) * (len(admission.CASES) - position) + 300 >= deadline:
                    raise TimeoutError('remaining frozen suite cannot fit estimated time budget')
                require_idle_gpu()
                subprocess.run(['nvidia-smi', '-L'], check=True)
                stage = root / name
                stage.mkdir(exist_ok=False)
                started = time.monotonic()
                command = ['/usr/bin/env', '-u', 'PYTHONPATH', sys.executable, '-m', MODULE,
                    '--worker', '--case', name, '--output-root', str(root), '--protocol-sha256', args.protocol_sha256,
                    '--authorize-execution', '--lease-fd', str(lease.fileno())]
                monitor = CaseMonitor(root, name, deadline)
                legacy.run_stage(stage, 'case', command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
                connected.parent_metadata(source, root / 'training')
                for rank in (0, 1):
                    verify_worker(stage, name, rank)
                current = metrics(root, name, p)
                files = [f for folder in (stage, native_artifacts(root, name)) for f in folder.rglob('*') if f.is_file()]
                files.append(root / 'training/attempts' / name / 'attempt.json')
                write_new(stage / 'complete.json', {'status': 'terminal_complete', 'case': name,
                    'protocol_sha256': args.protocol_sha256, 'resources': monitor.record(),
                    'files': {str(f.relative_to(root)): sha256(f) for f in files}})
                durations.append(time.monotonic() - started)
                if name == 'C2':
                    values = {n: metrics(root, n, p) for n in admission.CALIBRATION}
                    calibration = {**admission.calibrate(values), 'protocol_sha256': args.protocol_sha256,
                        'complete_sha256': {n: sha256(root / n / 'complete.json') for n in admission.CALIBRATION}}
                    write_new(root / 'calibration.json', calibration)
                    if calibration['status'] != 'calibrated':
                        terminal['reason'] = 'baseline_unstable_or_insufficient'
                        raise AdmissionBlocked('calibration exceeded frozen noise ceiling')
                if name not in admission.CALIBRATION:
                    check = admission.check_case(name, current, calibration)
                    write_new(stage / 'admission.json', {**check, 'calibration_sha256': sha256(root / 'calibration.json')})
                    if not check['passed']:
                        terminal['reason'] = admission.classify_stop(name, accepted)
                        raise AdmissionBlocked(f'{name} outside independent frozen envelope')
                    accepted.append(name)
            compare(root, p, args.protocol_sha256)
            reserve(root, 4096)
        terminal.update(status='completed', exit_code=0, comparison_sha256=sha256(root / 'comparison.json'))
    except AdmissionBlocked as exc:
        terminal.update(status='admission_blocked', exit_code=2, error=str(exc), accepted_holdouts=accepted)
        raise SystemExit(2) from exc
    except BaseException as exc:
        terminal['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        connected.publish_status(root / 'execution.json', {**terminal, 'ended_unix': time.time()})


def monitor(root, digest):
    start = time.monotonic()
    while time.monotonic() - start < CONTRACT['suite_timeout_seconds'] + 300:
        if (root / 'execution.json').exists():
            end = read(root / 'execution.json')
            if end['protocol_sha256'] != digest or {'completed': 0, 'failed': 1, 'admission_blocked': 2}.get(end['status']) != end['exit_code']:
                raise ValueError('phase-B terminal identity differs')
            if end['status'] == 'completed' and sha256(root / 'comparison.json') != end['comparison_sha256']:
                raise ValueError('phase-B comparison changed')
            print(json.dumps(end), flush=True)
            return end['exit_code']
        if (root / 'started.json').exists():
            s = read(root / 'started.json')
            if s['protocol_sha256'] != digest:
                raise ValueError('phase-B start identity differs')
            try:
                alive = Path(f'/proc/{s["pid"]}/stat').read_text().split()[21] == s['proc_start']
            except FileNotFoundError:
                alive = False
            if not alive and not (root / 'execution.json').exists():
                raise RuntimeError('executor disappeared without terminal state')
        elif time.monotonic() - start > 300:
            raise RuntimeError('executor did not start')
        print(json.dumps({'completed_cases': [n for n in admission.CASES if (root / n / 'complete.json').exists()],
                          'status': 'waiting_not_acceptance'}), flush=True)
        time.sleep(30)
    raise TimeoutError('monitor expired; did not stop executor')


def distributed_rank(rank, function, args):
    import torch
    from gsplat.distributed import _distributed_worker
    init = torch.distributed.init_process_group
    def bounded_group(*a, **kw):
        kw['timeout'] = timedelta(seconds=CONTRACT['collective_timeout_seconds'])
        return init(*a, **kw)
    # Apply in the spawned process, not only in its parent. Keep gsplat's communicator warmup.
    with patch.object(torch.distributed, 'init_process_group', bounded_group):
        return _distributed_worker(rank, 2, function, args, None, True)


def launch(function, args):
    import torch
    from gsplat.distributed import _find_free_port
    if torch.cuda.device_count() != 2 or 'OMPI_COMM_WORLD_SIZE' in os.environ:
        raise ValueError('exactly two local GPUs, no multi-node launch')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(_find_free_port())
    torch.multiprocessing.spawn(distributed_rank, args=(function, args), nprocs=2, join=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'worker', 'monitor'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--authorize-prepare', action='store_true')
    parser.add_argument('--authorize-execution', action='store_true')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--case', choices=admission.CASES)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != 'i-94B8D131' or str(PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        raise SystemExit('仅指定远端可执行；本机仅help/NoTorch检查')
    if not (args.authorize_prepare if args.prepare else True if args.monitor else args.authorize_execution):
        raise SystemExit('未获阶段B显式授权')
    if args.case is not None and not args.worker:
        parser.error('--case is worker-only; no arbitrary branch execution')
    args.output_root = output_root(args.output_root, args.prepare)
    if args.prepare:
        if args.preflight_report is None:
            parser.error('independent --preflight-report required before prepare')
        preflight = verify_preflight(args.preflight_report)
        prepare(args.output_root, preflight)
        return
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.monitor:
        raise SystemExit(monitor(args.output_root, args.protocol_sha256))
    if args.run:
        run(args, p)
    else:
        if args.case is None or args.lease_fd is None or os.readlink(f'/proc/self/fd/{args.lease_fd}') != str(PROJECT / 'outputs/.gpu.lock'):
            raise ValueError('worker needs a frozen case and inherited GPU lease')
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        legacy.unlimited_host_cgroup(UNIT)
        launch(worker, args)


if __name__ == '__main__':
    main()
