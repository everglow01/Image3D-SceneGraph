"""Separate authorization for a new common prefix and two source-Adam arms."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from ..e003_minimal_survival import run as base
from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0.run import initialize_backend
from . import analyze
from .probe import InterventionProbe

PROJECT, snapshots, read, sha256, write_new = base.PROJECT, base.snapshots, base.read, base.sha256, base.write_new
REL = 'experiment/mcmc_shard_imbalance/questions/q1_post_relocation_survival/experiments/e004_source_adam_intervention'
MODULE, RUNS = REL.replace('/', '.') + '.run', PROJECT / REL / 'runs'
PROFILE = 'source_adam_intervention_v1'
PREFIX_UNIT = 'image3d-source-adam-prefix-v1.service'
ARMS_UNIT = 'image3d-source-adam-arms-v1.service'
PREFLIGHT_UNIT = 'image3d-source-adam-preflight-v1.service'
FILES = tuple(REL + '/' + n for n in ('run.py', 'probe.py', 'analyze.py', 'tests/test_intervention.py', 'tests/test_intervention_remote.py'))
DEPENDENCIES = tuple(dict.fromkeys((*base.DEPENDENCIES, *FILES)))
CONTRACT = {'profile': PROFILE, 'start': 12000, 'event': 12100, 'end': 12120, 'owner_size': 1500000,
    'nominal_iterations': 30000, 'parent_checkpoint_sha256': base.connected.PARENT_SHA, 'core': base.legacy.CORE,
    'arms': ['A', 'B'], 'intervention': 'restore_only_source_exp_avg_and_exp_avg_sq_all_six_groups',
    'prefix_updates': 100, 'arm_updates': 20, 'total_updates': 140, 'camera_forwards': 280,
    'loss_backwards': 280, 'optimizer_group_steps': 1680, 'global_relocations': 1, 'extra_render_forwards': 0,
    'phase_timeout_seconds': 1800, 'preflight_timeout_seconds': 900, 'collective_timeout_seconds': 120,
    'max_per_rank_reserved_bytes': 18 * 1024**3, 'max_output_bytes': 16 * 1024**3,
    'disk_admission_bytes': 20 * 1024**3, 'disk_reserve_bytes': 4 * 1024**3, 'host_limits': None,
    'atol': snapshots.ATOL, 'rtol': snapshots.RTOL, 'requires_separate_gate0_review': True,
    'long_term_root_cause_proven': False, 'full_numerical_noninterference_proven': False}
PREFLIGHT_COUNTS = {'global_updates': 8, 'camera_forwards': 16, 'loss_backwards': 16, 'optimizer_group_steps': 96,
                    'real_image_decodes': 0, 'extra_render_forwards': 0}
FLAGS = ('native_checkpoint_restore_exact', 'source_reset_target_preserved', 'intervention_whitelist_exact',
         'noise_independent_of_moments', 'same_state_observation_unchanged', 'first_adam_algebra_verified',
         'mapping_and_survival_verified', 'empty_source_owner_supported', 'forbidden_decode_rejected',
         'coordinated_rejection_verified', 'rng_rejection_verified', 'no_validation_or_export')


def execution_commit():
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *DEPENDENCIES], cwd=PROJECT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *DEPENDENCIES], cwd=PROJECT, check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()


def output_root(path, preparing=False):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    if path.parent != RUNS or not path.name.startswith('adam-intervention-') or (preparing and path.exists()):
        raise ValueError('new direct adam-intervention-* output required')
    return path


def reserve(root, extra=0):
    if extra < 0 or base.previous.output_bytes(root) + extra + 2 * 1024**2 > CONTRACT['max_output_bytes']:
        raise ValueError('intervention output budget')
    if shutil.disk_usage(root).free < extra + CONTRACT['disk_reserve_bytes']:
        raise ValueError('intervention disk reserve')


def preflight_required(rank):
    required = {f'initialization-rank-{rank}.json', f'prefix/gate0-rank-{rank}.json', f'prefix/prefix-state-rank-{rank}.json'}
    required.update(f'prefix/{name}-rank-{rank}/manifest.json' for name in ('pre-relocation', 'post-relocation', 'cohort'))
    for case in ('prefix', 'A', 'B'):
        required.add(f'{case}/worker-rank-{rank}.json')
    for case in ('A', 'B'):
        required.add(f'{case}/intervention-rank-{rank}.json')
        required.add(f'{case}/trajectory-rank-{rank}/index.json')
        required.update(f'{case}/{name}-rank-{rank}/manifest.json' for name in ('opacity-before-1', 'opacity-after-1'))
    required.update('training/attempts/prefix/checkpoints/iteration_000000004/' + name
                    for name in ('checkpoint.json', 'model.bin', 'optimizer.bin', 'scheduler.bin', 'densification.bin', 'rng.bin', 'metrics.json'))
    return required


def verify_preflight(path):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    p = read(path)
    if (p['status'] != 'passed' or p['profile'] != PROFILE or p['implementation_commit'] != execution_commit()
            or p['code_sha256'] != {n: sha256(PROJECT / n) for n in DEPENDENCIES}
            or p['runtime'] != base.expected_runtime() or p['counts'] != PREFLIGHT_COUNTS or set(p['workers']) != {'0', '1'}):
        raise ValueError('new intervention preflight required, old gates cannot substitute')
    for rank, row in p['workers'].items():
        file = snapshots.safe_file(RUNS, Path(row['path']).relative_to(RUNS))
        if sha256(file) != row['sha256']:
            raise ValueError('preflight worker changed')
        worker = read(file)
        if worker['rank'] != int(rank) or worker['flags'] != dict.fromkeys(FLAGS, True):
            raise ValueError('incomplete intervention preflight')
        for name, digest in worker['evidence_sha256'].items():
            if sha256(snapshots.safe_file(file.parent, name)) != digest:
                raise ValueError('preflight bound evidence changed')
        required = preflight_required(int(rank))
        if (not required <= worker['evidence_sha256'].keys()
                or worker['counts'] != {'updates': 8, 'forward': 8, 'backward': 8, 'adam': 48}
                or worker['noise_equivalence_fixture_calls'] != 2):
            raise ValueError('preflight evidence or per-rank budget missing')
        for name in required:
            if name.endswith('/manifest.json'):
                manifest = read(file.parent / name)
                if not manifest['files']:
                    raise ValueError('empty snapshot evidence')
                for leaf, item in manifest['files'].items():
                    bound = str(Path(name).parent / leaf)
                    if worker['evidence_sha256'].get(bound) != item['sha256']:
                        raise ValueError('preflight snapshot arrays not bound')
        for case, n in (('prefix', 4), ('A', 2), ('B', 2)):
            w = read(file.parent / case / f'worker-rank-{rank}.json')
            if (w['status'] != 'measured' or w['rank'] != int(rank) or w['case'] != case or w['new_updates'] != n
                    or any(w['calls'][key] != n for key in ('batch', 'render', 'backward', 'raster_forward', 'raster_backward', 'noise', 'strategy'))
                    or w['calls']['optimizer'] != n * 6 or w['calls']['checkpoint'] != int(case == 'prefix')
                    or w['validation_evaluated'] or w['native_training_complete']):
                raise ValueError('synthetic worker counts or phase boundary differ')
        init = read(file.parent / f'initialization-rank-{rank}.json')
        if init['warmup_forward_calls'] or init['warmup_backward_calls']:
            raise ValueError('uncounted preflight warmup')
    return {'path': str(path), 'sha256': sha256(path)}


def prepare(root, preflight):
    source = base.previous.source_identity()
    parent_bytes = sum(x['bytes'] for x in source['parent_metadata']['files'].values())
    full_state_bytes = 3 * 2 * CONTRACT['owner_size'] * 59 * 4
    estimated_peak = parent_bytes + 5 * full_state_bytes + 2 * 1024**3
    if estimated_peak > CONTRACT['max_output_bytes'] or shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
        raise ValueError('prefix/checkpoint transient storage exceeds admission')
    root.mkdir(exist_ok=False)
    write_new(root / 'protocol.json', {**CONTRACT, 'implementation_commit': execution_commit(),
        'code_sha256': {n: sha256(PROJECT / n) for n in DEPENDENCIES}, 'source_identity': source,
        'preflight': preflight, 'estimated_peak_output_bytes': estimated_peak, 'status': 'prefix_prepared_arms_not_authorized'})


def protocol_at(root, digest):
    if not digest or sha256(snapshots.safe_file(root, 'protocol.json')) != digest:
        raise ValueError('explicit prefix protocol SHA required')
    p = read(root / 'protocol.json')
    if (any(p.get(k) != v for k, v in CONTRACT.items()) or set(p['code_sha256']) != set(DEPENDENCIES)
            or any(sha256(snapshots.safe_file(PROJECT, n)) != h for n, h in p['code_sha256'].items())):
        raise ValueError('intervention contract or execution code changed')
    return p


def verify_inputs(p):
    if (execution_commit() != p['implementation_commit'] or base.previous.source_identity() != p['source_identity']
            or verify_preflight(p['preflight']['path']) != p['preflight']):
        raise ValueError('intervention source or preflight changed')


def checkpoint(root, p):
    from scripts import run_mcmc_first_divergence as native
    provenance = native.lifecycle.parent_provenance(p['source_identity']['native'], 'baseline')
    return native.lifecycle.verify_checkpoint(root / 'training', 'prefix', CONTRACT['event'], provenance)


def verify_worker(root, case, rank, p):
    stage = root / case
    w = read(stage / f'worker-rank-{rank}.json')
    n = CONTRACT['prefix_updates'] if case == 'prefix' else CONTRACT['arm_updates']
    counts = {'restore': 1, 'batch': n, 'render': n, 'backward': n, 'raster_forward': n, 'raster_backward': n,
              'optimizer': 6 * n, 'strategy': n, 'relocate': int(case == 'prefix'), 'sample': 0,
              'add': int(case == 'prefix'), 'noise': n, 'terminal': 1, 'checkpoint': int(case == 'prefix')}
    if case == 'prefix':
        counts['sample'] = int(analyze.gate0(stage, rank)['target_count'] > 0)
    if (w['status'] != 'measured' or w['case'] != case or w['rank'] != rank or w['calls'] != counts
            or w['new_updates'] != n or w['validation_evaluated'] or w['native_training_complete']
            or w['protocol_sha256'] != sha256(root / 'protocol.json')):
        raise ValueError('worker lifecycle or counts differ')
    if len(w['schedule']) != n or len(w['noise_schedule']) != n or not w['manifests']:
        raise ValueError('missing worker evidence')
    start = CONTRACT['start'] if case == 'prefix' else CONTRACT['event']
    rows = base.previous.admission.window_rows(read(stage / f'native-tail-rank-{rank}.json'),
        start=start, end=start + n, owner_size=CONTRACT['owner_size'])
    dataset = read(Path(p['source_identity']['native']['pair']['replay']) / 'dataset.json')
    train_ids = set(map(str, dataset['splits']['train']))
    if not w['loaded_train_ids'] or not set(w['loaded_train_ids']) <= train_ids:
        raise ValueError('worker decoded non-Train image')
    for index, (row, schedule, noise) in enumerate(zip(rows, w['schedule'], w['noise_schedule']), start + 1):
        if (schedule['iteration'] != index or noise['iteration'] != index
                or schedule['image_id'] != str(row['batch_view_ids'][rank]) or schedule['sh_degree'] != row['sh_degree']
                or schedule['learning_rates']['means'] != row['position_lr']
                or set(schedule['learning_rates']) != set(snapshots.PARAMETERS)
                or not set(map(str, row['batch_view_ids'])) <= train_ids):
            raise ValueError('native schedule and probe disagree')
    initialization = read(stage / f'initialization-rank-{rank}.json')
    if (initialization['warmup_forward_calls'] or initialization['warmup_backward_calls']
            or initialization['rng_restored'] or not initialization['initialization_outside_measurement']):
        raise ValueError('unverified initialization boundary')
    for name, digest in w['manifests'].items():
        folder = snapshots.safe_file(stage, f'{name}-rank-{rank}')
        if sha256(folder / 'manifest.json') != digest:
            raise ValueError('snapshot manifest binding changed')
        snapshots.load_snapshot(folder)
    if case != 'prefix':
        native_digest = read(root / 'prefix' / f'prefix-state-rank-{rank}.json')['full_state_sha256']
        intervention = read(stage / f'intervention-rank-{rank}.json')
        if (w['restored_digest'] != native_digest or not intervention['only_source_moments_changed']
                or intervention['protected_before'] != intervention['protected_after']
                or intervention['case'] != case or intervention['rank'] != rank):
            raise ValueError('branch origin or intervention whitelist differs')
        for row in intervention['moments'].values():
            for item in row.values():
                if item['after'] != item['expected_retained' if case == 'B' else 'before']:
                    raise ValueError('wrong source moments')
        analyze.trajectory(root / 'prefix', stage, rank, CONTRACT['event'], CONTRACT['end'], CONTRACT['owner_size'])
        analyze.first_step(stage, rank)
    return w


def bind_complete(root, case, p, resources):
    stage = root / case
    workers = {str(r): verify_worker(root, case, r, p) for r in (0, 1)}
    native = root / 'training/attempts' / case
    files = [f for folder in (stage, native) for f in folder.rglob('*') if f.is_file()]
    exit_record = read(stage / 'case.exit.json')
    if exit_record['returncode'] != 0 or exit_record['resource_failure'] is not None:
        raise ValueError('failed worker cannot be completed')
    write_new(stage / 'complete.json', {'case': case, 'protocol_sha256': sha256(root / 'protocol.json'),
        'resources': resources, 'files': {str(f.relative_to(root)): sha256(f) for f in files}})
    return workers


def verify_complete(root, case):
    stage = snapshots.safe_file(root, case)
    c = read(stage / 'complete.json')
    if c['case'] != case or c['protocol_sha256'] != sha256(root / 'protocol.json'):
        raise ValueError('completion identity differs')
    required = {str((stage / n).relative_to(root)) for n in ('worker-rank-0.json', 'worker-rank-1.json', 'case.exit.json')}
    if not required <= c['files'].keys():
        raise ValueError('incomplete completion bindings')
    for rel, digest in c['files'].items():
        file = snapshots.safe_file(root, rel)
        if (not (file.is_relative_to(stage) or file.is_relative_to(root / 'training/attempts' / case)) or sha256(file) != digest):
            raise ValueError('completed evidence changed or escaped stage')
    e = read(stage / 'case.exit.json')
    if e['returncode'] != 0 or e['resource_failure'] is not None:
        raise ValueError('failed completed stage')
    return c


def verified_gate0(root, p):
    verify_complete(root, 'prefix')
    gate = read(root / 'gate0.json')
    owners = {str(r): analyze.gate0(root / 'prefix', r) for r in (0, 1)}
    meta = checkpoint(root, p)
    if (gate['status'] != 'structural_verified_review_required' or gate['owners'] != owners
            or gate['source_protocol_sha256'] != sha256(root / 'protocol.json')
            or gate['checkpoint_hash'] != meta['checkpoint_hash'] or gate['arms_started']
            or gate['moment_magnitude_automatically_admitted']
            or gate['effective_intervention_available'] != any(x['effective_intervention_available'] for x in owners.values())):
        raise ValueError('Gate 0 does not match the completed prefix evidence')
    return gate


def prepare_arms(root, p, digest, reviewed):
    if not reviewed or not digest or sha256(root / 'gate0.json') != digest:
        raise ValueError('explicit reviewed Gate 0 SHA required')
    verify_inputs(p)
    gate = verified_gate0(root, p)
    if not gate['effective_intervention_available']:
        raise ValueError('Gate 0 lacks an effective validated intervention')
    write_new(root / 'arms-protocol.json', {'profile': PROFILE, 'prefix_protocol_sha256': sha256(root / 'protocol.json'),
        'gate0_sha256': digest, 'prefix_complete_sha256': sha256(root / 'prefix/complete.json'),
        'checkpoint_hash': gate['checkpoint_hash'], 'review_acknowledged': True, 'arms': ['A', 'B'],
        'implementation_commit': p['implementation_commit'], 'code_sha256': p['code_sha256']})


def verify_arms(root, p, digest):
    if not digest or sha256(root / 'arms-protocol.json') != digest:
        raise ValueError('explicit arms protocol SHA required')
    a = read(root / 'arms-protocol.json')
    if (a['profile'] != PROFILE or a['arms'] != ['A', 'B'] or not a['review_acknowledged']
            or a['prefix_protocol_sha256'] != sha256(root / 'protocol.json')
            or a['gate0_sha256'] != sha256(root / 'gate0.json')
            or a['prefix_complete_sha256'] != sha256(root / 'prefix/complete.json')
            or a['implementation_commit'] != p['implementation_commit'] or a['code_sha256'] != p['code_sha256']):
        raise ValueError('arms review/source bindings changed')
    gate = verified_gate0(root, p)
    if gate['checkpoint_hash'] != a['checkpoint_hash'] or not gate['effective_intervention_available']:
        raise ValueError('common checkpoint or effective intervention changed')
    return a


def worker(local_rank, rank, world_size, args):
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig
    from image3d_scenegraph.gaussian.initialization import load_frozen_initialization
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    if (world_size, local_rank) != (2, rank):
        raise ValueError('two local ranks required')
    p = protocol_at(args.output_root, args.protocol_sha256)
    verify_inputs(p)
    if args.case != 'prefix':
        verify_arms(args.output_root, p, args.arms_protocol_sha256)
        if args.case == 'B':
            verify_complete(args.output_root, 'A')
    stage = args.output_root / args.case
    base.prime_package(p, rank)
    write_new(stage / f'initialization-rank-{rank}.json',
        {**initialize_backend(), 'rng_restored': False, 'initialization_outside_measurement': True})
    source, config = p['source_identity']['native'], p['source_identity']['config']
    replay = Path(source['pair']['replay'])
    dataset = read(replay / 'dataset.json')
    resolved = ResolvedGaussianConfig(config['requested_profile'], config['effective_config'], config['effective_config_hash'])
    provenance = trainer.training_provenance(dataset_hash=dataset['dataset_hash'], effective_config_hash=resolved.effective_config_hash, world_size=2)
    from scripts import run_mcmc_first_divergence as native
    if asdict(provenance) != native.lifecycle.parent_provenance(source, 'baseline'):
        raise ValueError('native provenance differs')
    bundle = source['replay_record']
    initialization = load_frozen_initialization(replay / bundle['initialization_path'], replay / bundle['initialization_diagnostics_path'], expected_sha256=bundle['initialization_sha256'])
    probe = InterventionProbe(stage, rank, args.case, dataset['splits']['train'], lambda n: reserve(args.output_root, n),
        prefix=args.output_root / 'prefix', training=args.output_root / 'training', provenance=provenance)
    failure = stage / f'failure-rank-{rank}'
    failure.mkdir(exist_ok=False)
    with memory_telemetry(stage / 'memory', rank, rank), failure_record(failure, PROFILE, args.case):
        try:
            with probe.installed():
                trainer.train_gaussians(contract=dataset, dataset_root=replay, initialization=initialization,
                    resolved_config=resolved, run_dir=args.output_root / 'training', attempt_id=args.case, attempt_kind='resume',
                    parent_attempt_id='train-001' if args.case == 'prefix' else 'prefix', resume_iteration=probe.start,
                    local_rank=rank, world_rank=rank, world_size=2, stop_after_iteration=probe.end,
                    save_intermediate_previews=False, cancel_requested=lambda: (stage / 'cancel').exists())
        except snapshots.ProbeComplete:
            probe.consensus('worker-completion', lambda: write_new(stage / f'worker-rank-{rank}.json',
                {**probe.finish(), 'protocol_sha256': args.protocol_sha256, 'arms_protocol_sha256': args.arms_protocol_sha256}))
        else:
            raise ValueError('native training crossed research endpoint')


class Monitor:
    def __init__(self, root, stage, deadline):
        self.root, self.stage, self.deadline = root, stage, deadline
        self.peaks = [0, 0]

    def __call__(self, elapsed, *, final=False):
        if time.monotonic() >= self.deadline:
            return 'phase_timeout'
        try:
            reserve(self.root)
        except ValueError as exc:
            return str(exc)
        for rank in (0, 1):
            file = self.stage / 'memory' / f'rank-{rank}.json'
            if file.exists():
                value = read(file)
                self.peaks[rank] = max(self.peaks[rank], value['peak_reserved_bytes'])
                if self.peaks[rank] > CONTRACT['max_per_rank_reserved_bytes']:
                    return 'gpu_reserved_limit'
                if final and not value['finished']:
                    return 'unfinished_telemetry'
            elif final:
                return 'missing_telemetry'


def run_phase(args, p, phase):
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    root = args.output_root
    base.legacy.unlimited_host_cgroup(PREFIX_UNIT if phase == 'prefix' else ARMS_UNIT)
    with (root / f'{phase}.once').open('x'):
        pass
    terminal = {'status': 'failed', 'exit_code': 1, 'phase': phase, 'protocol_sha256': args.protocol_sha256}
    base.connected.publish_status(root / f'{phase}-started.json', {'pid': os.getpid(),
        'proc_start': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], 'protocol_sha256': args.protocol_sha256})
    deadline = time.monotonic() + CONTRACT['phase_timeout_seconds']
    def expired(signum, frame):
        raise TimeoutError('intervention phase exceeded thirty minutes')
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(CONTRACT['phase_timeout_seconds'])
    try:
        with FileLease(PROJECT / 'outputs/.gpu.lock') as lease:
            require_idle_gpu()
            verify_inputs(p)
            if shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
                raise ValueError('intervention disk admission')
            if phase == 'prefix':
                parent = Path(p['source_identity']['native']['parent_training']) / 'attempts/train-001'
                target = root / 'training/attempts/train-001'
                target.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(parent / 'attempt.json', target / 'attempt.json')
                shutil.copytree(parent / 'checkpoints/iteration_000012000', target / 'checkpoints/iteration_000012000')
                for f in target.rglob('*'):
                    f.chmod(0o555 if f.is_dir() else 0o444)
                target.chmod(0o555)
                base.connected.parent_metadata(p['source_identity']['native'], root / 'training')
            else:
                verify_arms(root, p, args.arms_protocol_sha256)
            for case in (('prefix',) if phase == 'prefix' else ('A', 'B')):
                require_idle_gpu()
                subprocess.run(['nvidia-smi', '-L'], check=True)
                stage = root / case
                stage.mkdir(exist_ok=False)
                monitor = Monitor(root, stage, deadline)
                command = ['/usr/bin/env', '-u', 'PYTHONPATH', sys.executable, '-m', MODULE, '--worker', '--case', case,
                    '--output-root', str(root), '--protocol-sha256', args.protocol_sha256,
                    '--authorize-prefix' if phase == 'prefix' else '--authorize-arms', '--lease-fd', str(lease.fileno())]
                if phase == 'arms':
                    command += ['--arms-protocol-sha256', args.arms_protocol_sha256]
                base.legacy.run_stage(stage, 'case', command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
                bind_complete(root, case, p, {'per_rank_peak_reserved_bytes': monitor.peaks})
                verify_complete(root, case)
            if phase == 'prefix':
                owners = {str(r): analyze.gate0(root / 'prefix', r) for r in (0, 1)}
                report = {'status': 'structural_verified_review_required', 'owners': owners,
                    'effective_intervention_available': any(x['effective_intervention_available'] for x in owners.values()),
                    'checkpoint_hash': checkpoint(root, p)['checkpoint_hash'], 'arms_started': False,
                    'source_protocol_sha256': args.protocol_sha256, 'moment_magnitude_automatically_admitted': False}
                report_name = 'gate0.json'
            else:
                report, report_name = analyze.compare(root), 'comparison.json'
                for rank in (0, 1):
                    if read(root / 'A' / f'worker-rank-{rank}.json')['noise_schedule'] != read(root / 'B' / f'worker-rank-{rank}.json')['noise_schedule']:
                        raise ValueError('arm native RNG/noise draw schedule differs')
                report['arms_protocol_sha256'] = args.arms_protocol_sha256
                verify_arms(root, p, args.arms_protocol_sha256)
            verify_inputs(p)
            reserve(root, len(json.dumps(report).encode()) + 4096)
            write_new(root / report_name, report)
            terminal.update(status='completed', exit_code=0, report=report_name, report_sha256=sha256(root / report_name))
    except BaseException as exc:
        terminal['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        base.connected.publish_status(root / f'{phase}-execution.json', {**terminal, 'ended_unix': time.time()})


def monitor(root, digest, phase):
    started = time.monotonic()
    while time.monotonic() - started < CONTRACT['phase_timeout_seconds'] + 300:
        terminal = root / f'{phase}-execution.json'
        if terminal.exists():
            value = read(terminal)
            if value['protocol_sha256'] != digest or value['phase'] != phase or {'completed': 0, 'failed': 1}.get(value['status']) != value['exit_code']:
                raise ValueError('phase terminal identity differs')
            if value['exit_code'] == 0 and sha256(snapshots.safe_file(root, value['report'])) != value['report_sha256']:
                raise ValueError('phase result changed')
            print(json.dumps(value), flush=True)
            return value['exit_code']
        file = root / f'{phase}-started.json'
        if file.exists():
            s = read(file)
            if s['protocol_sha256'] != digest:
                raise ValueError('phase executor changed')
            try:
                alive = Path(f'/proc/{s["pid"]}/stat').read_text().split()[21] == s['proc_start']
            except FileNotFoundError:
                alive = False
            if not alive and not terminal.exists():
                raise RuntimeError('phase executor disappeared without terminal')
        elif time.monotonic() - started > 300:
            raise TimeoutError('phase did not start')
        print(json.dumps({'phase': phase, 'status': 'waiting_not_acceptance'}), flush=True)
        time.sleep(30)
    raise TimeoutError('read-only monitor expired, executor not stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'prefix', 'prepare-arms', 'arms', 'worker', 'monitor'):
        modes.add_argument('--' + name, action='store_true')
    for name in ('prepare', 'prefix', 'prepare-arms', 'arms'):
        parser.add_argument('--authorize-' + name, action='store_true')
    parser.add_argument('--acknowledge-gate0-review', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--gate0-sha256')
    parser.add_argument('--arms-protocol-sha256')
    parser.add_argument('--case', choices=('prefix', 'A', 'B'))
    parser.add_argument('--monitor-phase', choices=('prefix', 'arms'))
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != 'i-94B8D131' or str(PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        raise SystemExit('仅指定远端可以执行此独立实验')
    if args.case is not None and not args.worker:
        parser.error('不允许跳过固定执行次序')
    phase = 'prefix' if args.prefix or (args.worker and args.case == 'prefix') else 'arms'
    authorized = (args.authorize_prepare if args.prepare else args.authorize_prepare_arms if args.prepare_arms
                  else True if args.monitor else args.authorize_prefix if phase == 'prefix' else args.authorize_arms)
    if not authorized:
        raise SystemExit('需要该阶段独立授权')
    root = args.output_root = output_root(args.output_root, args.prepare)
    if args.prepare:
        if args.preflight_report is None:
            parser.error('需要独立新版前置报告')
        prepare(root, verify_preflight(args.preflight_report))
        return
    p = protocol_at(root, args.protocol_sha256)
    if args.prepare_arms:
        prepare_arms(root, p, args.gate0_sha256, args.acknowledge_gate0_review)
    elif args.monitor:
        if args.monitor_phase is None:
            parser.error('监控必须指定阶段')
        raise SystemExit(monitor(root, args.protocol_sha256, args.monitor_phase))
    elif args.worker:
        if args.case is None or args.lease_fd is None or os.readlink(f'/proc/self/fd/{args.lease_fd}') != str(PROJECT / 'outputs/.gpu.lock'):
            raise ValueError('worker requires inherited project GPU lease')
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.legacy.unlimited_host_cgroup(PREFIX_UNIT if phase == 'prefix' else ARMS_UNIT)
        base.launch(worker, args)
    else:
        run_phase(args, p, phase)


if __name__ == '__main__':
    main()
