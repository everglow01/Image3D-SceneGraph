"""Two observed trajectories and isolated event renders; no legacy gate reclassification."""
from __future__ import annotations

import argparse
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
import subprocess
import sys
import time
from unittest.mock import patch

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission import run_b as previous
from . import analyze, render_probe
from .observer import MinimalObserver, Endpoint, snapshots, sha256, write_new

PROJECT = previous.PROJECT
REL = 'experiment/mcmc_shard_imbalance/questions/q1_post_relocation_survival/experiments/e003_minimal_survival'
MODULE = REL.replace('/', '.') + '.run'
RUNS = PROJECT / REL / 'runs'
UNIT = 'image3d-minimal-survival-v1.service'
PREFLIGHT_UNIT = 'image3d-minimal-survival-preflight-v1.service'
PROFILE = 'mcmc_minimal_survival_v1'
CASES = ('T0', 'T1')
FILES = tuple(REL + '/' + n for n in ('run.py', 'observer.py', 'analyze.py', 'render_probe.py',
                                      'tests/test_minimal.py', 'tests/test_minimal_remote.py'))
DEPENDENCIES = tuple(dict.fromkeys((*previous.DEPENDENCIES, *FILES)))
legacy, connected, read = previous.legacy, previous.connected, previous.read
FLAGS = ('active_state_unchanged', 'state_negative_rejected', 'sampling_return_unchanged', 'training_collectives_unchanged',
         'mapping_and_trajectory_verified', 'zero_target_verified', 'forbidden_decode_rejected',
         'coordinated_rejection_verified', 'rng_negative_rejected', 'terminal_before_evaluation')
COUNTS = {'training_camera_forwards': 8, 'loss_backwards': 8, 'optimizer_group_steps': 48,
          'synthetic_updates': 8, 'isolated_render_forwards': 4}
CONTRACT = {'profile': PROFILE, 'cases': list(CASES), 'start': 12000, 'event': 12100, 'end': 12200,
    'owner_size': 1500000, 'nominal_iterations': 30000, 'parent_checkpoint_sha256': connected.PARENT_SHA,
    'core': legacy.CORE, 'new_updates_max': 400, 'training_camera_forwards_max': 800,
    'loss_backwards_max': 800, 'optimizer_group_steps_max': 4800, 'global_relocation_events_max': 4,
    'per_owner_relocation_calls_max': 8, 'contribution_forwards_max': 0, 'isolated_render_forwards_max': 64,
    'render_cameras': 8, 'render_repeats_per_state': 2, 'render_after_all_training': True,
    'suite_timeout_seconds': 3600, 'collective_timeout_seconds': 120,
    'max_output_bytes': 12 * 1024**3, 'disk_admission_bytes': 16 * 1024**3,
    'max_per_rank_reserved_bytes': 18 * 1024**3, 'host_limits': None,
    'preflight_timeout_seconds': 900, 'atol': snapshots.ATOL, 'rtol': snapshots.RTOL,
    'old_gate1': 'not_reclassified', 'E007': 'calibration_blocked_not_reclassified',
    'full_numerical_noninterference_proven': False, 'cross_run_slot_pairing': False,
    'reuse_bins': [[1, 1], [2, 3], [4, None]], 'initial_alpha_edges': [.005, .01, .05],
    'interpretation': 'descriptive_observed_trajectories_not_causal',
    'preflight_budget': {'synthetic_updates': 8, 'training_camera_forwards': 16, 'loss_backwards': 16,
        'optimizer_group_steps': 96, 'isolated_render_forwards': 8, 'real_image_decodes': 0}}


def output_root(path, preparing=False):
    path = Path(path).absolute()
    result = snapshots.safe_file(RUNS, path.relative_to(RUNS))
    if result.parent != RUNS or not result.name.startswith('survival-') or (preparing and result.exists()):
        raise ValueError('new independent survival-* output root required')
    return result


def execution_commit():
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *DEPENDENCIES], cwd=PROJECT,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *DEPENDENCIES], cwd=PROJECT, check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()


def expected_runtime():
    if sha256(previous.A_ROOT / 'protocol.json') != previous.A_SHA:
        raise ValueError('frozen runtime source changed')
    return read(previous.A_ROOT / 'protocol.json')['source_inventory']['runtime']


def verify_preflight(path):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    proof = read(path)
    if (proof['status'] != 'passed' or proof['profile'] != PROFILE or proof['implementation_commit'] != execution_commit()
            or proof['budget'] != CONTRACT['preflight_budget'] or proof['runtime'] != expected_runtime()
            or proof['code_sha256'] != {n: sha256(PROJECT / n) for n in DEPENDENCIES}
            or set(proof['workers']) != {'0', '1'}):
        raise ValueError('missing or stale independent minimal-observer preflight')
    for rank, row in proof['workers'].items():
        file = snapshots.safe_file(RUNS, Path(row['path']).relative_to(RUNS))
        if sha256(file) != row['sha256']:
            raise ValueError('preflight evidence changed')
        value = read(file)
        if (value['status'] != 'passed' or value['rank'] != int(rank) or value['counts'] != COUNTS
                or value['real_image_decodes'] != 0 or value['invariants'] != dict.fromkeys(FLAGS, True)):
            raise ValueError('preflight coverage differs')
    return {'path': str(path), 'sha256': sha256(path)}


def reserve(root, extra):
    if extra < 0 or previous.output_bytes(root) + 2 * extra + 2 * 1024**2 > CONTRACT['max_output_bytes']:
        raise ValueError('minimal survival output budget exceeded')
    if shutil.disk_usage(root).free < 2 * extra + 4 * 1024**3:
        raise ValueError('minimal survival disk reserve')


def prepare(root, preflight):
    commit = execution_commit()
    source = previous.source_identity()
    dataset = read(Path(source['native']['pair']['replay']) / 'dataset.json')
    cameras = render_probe.camera_contract(dataset, source['config']['effective_config'])
    parent_size = sum(r['bytes'] for r in source['parent_metadata']['files'].values())
    # SH3 fp32 six-parameter states: two owners, before/after/endpoint for each trajectory.
    parameter_bytes = 2 * CONTRACT['owner_size'] * (3 + 3 + 4 + 1 + 3 * 16) * 4
    estimated = parent_size + 6 * parameter_bytes + 2 * 1024**3
    if estimated > CONTRACT['max_output_bytes'] or shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
        raise ValueError('insufficient frozen-suite storage budget')
    root.mkdir(exist_ok=False)
    write_new(root / 'protocol.json', {**CONTRACT, 'implementation_commit': commit, 'source_identity': source,
        'camera_contract': cameras, 'estimated_output_bytes': estimated, 'preflight': preflight,
        'code_sha256': {n: sha256(PROJECT / n) for n in DEPENDENCIES}, 'status': 'prepared_not_executed'})


def protocol_at(root, digest):
    if not digest or sha256(snapshots.safe_file(root, 'protocol.json')) != digest:
        raise ValueError('explicit minimal survival protocol SHA required')
    p = read(root / 'protocol.json')
    if any(p.get(k) != v for k, v in CONTRACT.items()) or set(p['code_sha256']) != set(DEPENDENCIES):
        raise ValueError('minimal survival contract differs')
    if any(sha256(snapshots.safe_file(PROJECT, n)) != s for n, s in p['code_sha256'].items()):
        raise ValueError('implementation changed')
    return p


def verify_inputs(p):
    if (execution_commit() != p['implementation_commit'] or verify_preflight(p['preflight']['path']) != p['preflight']
            or previous.source_identity() != p['source_identity']):
        raise ValueError('frozen source or preflight changed')
    dataset = read(Path(p['source_identity']['native']['pair']['replay']) / 'dataset.json')
    if render_probe.camera_contract(dataset, p['source_identity']['config']['effective_config']) != p['camera_contract']:
        raise ValueError('frozen Train cameras changed')


def prime_package(p, rank):
    import torch
    import gsplat
    torch.cuda.set_device(rank)
    source = p['source_identity']['native']
    package = Path(source['pair']['source_package'])
    if Path(gsplat.__file__).resolve().parent != package.resolve():
        raise ValueError('original frozen gsplat required')
    spec = find_spec('gsplat.csrc')
    if spec is None or not isinstance(spec.loader, ExtensionFileLoader) or not spec.origin:
        raise ValueError('precompiled extension required, no JIT fallback')
    binary = snapshots.safe_file(package, Path(spec.origin).relative_to(package))
    if (sha256(binary) != source['pair']['source_hashes'].get(str(binary.relative_to(package)))
            or Path(import_module('gsplat.csrc').__file__).resolve() != binary.resolve()):
        raise ValueError('extension identity changed')
    import_module('gsplat.rendering')
    import_module('gsplat.strategy.ops')
    actual = {'rank': rank, 'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(rank)}
    if actual != expected_runtime()[str(rank)]:
        raise ValueError('runtime changed')


def worker(local_rank, rank, world_size, args):
    import torch
    from scripts import run_mcmc_first_divergence as native
    from image3d_scenegraph.gaussian import trainer, runtime
    from image3d_scenegraph.gaussian.config import ResolvedGaussianConfig
    from image3d_scenegraph.gaussian.initialization import load_frozen_initialization
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    if (world_size, local_rank) != (2, rank):
        raise ValueError('two local ranks required')
    p = protocol_at(args.output_root, args.protocol_sha256)
    verify_inputs(p)
    prime_package(p, rank)
    source, config = p['source_identity']['native'], p['source_identity']['config']
    replay = Path(source['pair']['replay'])
    dataset = read(replay / 'dataset.json')
    stage = args.output_root / ('renders' if args.render_worker else '') / args.case
    endpoint = Endpoint(stage, rank, dataset['splits']['train'], lambda n: reserve(args.output_root, n))
    failure = stage / f'failure-rank-{rank}'
    failure.mkdir(exist_ok=False)
    with memory_telemetry(stage / 'memory', rank, rank), failure_record(failure, PROFILE, args.case):
        if args.render_worker:
            with endpoint.installed():
                views = runtime.load_training_views(dataset, replay, split='train',
                    longest_edge=int(config['effective_config']['resolution']['longest_edge']), device=torch.device('cpu'))
                # Reject any real pixel decode except the protocol-selected Train set.
                endpoint.train_ids = set(p['camera_contract']['image_ids'])
                measured = render_probe.probe(args.output_root / args.case, rank, views,
                                              p['camera_contract']['image_ids'], config['effective_config'], audit_root=stage)
                endpoint.consensus('render-finish', lambda: write_new(stage / f'worker-rank-{rank}.json',
                    {**measured, 'case': args.case, 'loaded_train_ids': sorted(endpoint.loaded),
                     'protocol_sha256': args.protocol_sha256}))
            return
        connected.parent_metadata(source, args.output_root / 'training')
        resolved = ResolvedGaussianConfig(config['requested_profile'], config['effective_config'], config['effective_config_hash'])
        provenance = trainer.training_provenance(dataset_hash=dataset['dataset_hash'],
                                                effective_config_hash=resolved.effective_config_hash, world_size=2)
        if asdict(provenance) != native.lifecycle.parent_provenance(source, 'baseline'):
            raise ValueError('native provenance changed')
        bundle = source['replay_record']
        initialization = load_frozen_initialization(replay / bundle['initialization_path'], replay / bundle['initialization_diagnostics_path'],
                                                    expected_sha256=bundle['initialization_sha256'])
        observer = MinimalObserver(stage, rank, lambda n: reserve(args.output_root, n))
        try:
            with endpoint.installed(), observer.installed():
                trainer.train_gaussians(contract=dataset, dataset_root=replay, initialization=initialization,
                    resolved_config=resolved, run_dir=args.output_root / 'training', attempt_id=args.case, attempt_kind='resume',
                    parent_attempt_id='train-001', resume_iteration=12000, local_rank=rank, world_rank=rank, world_size=2,
                    stop_after_iteration=12200, save_intermediate_previews=False, cancel_requested=lambda: (stage / 'cancel').exists())
        except snapshots.ProbeComplete:
            def finish():
                if not endpoint.completed or endpoint.calls != 1:
                    raise ValueError('terminal did not stop before evaluation')
                write_new(stage / f'worker-rank-{rank}.json', {'status': 'terminal_complete', 'case': args.case,
                    'rank': rank, 'new_updates': 200, 'validation_evaluated': False, 'native_training_complete': False,
                    'endpoint_manifest_sha256': endpoint.manifest, 'loaded_train_ids': sorted(endpoint.loaded),
                    'observation': observer.finish(), 'protocol_sha256': args.protocol_sha256})
            endpoint.consensus('terminal-finish', finish)
        else:
            raise ValueError('native trainer crossed terminal boundary')


class StageMonitor:
    def __init__(self, root, stage, deadline):
        self.root, self.stage, self.deadline = root, stage, deadline
        self.peaks = [0, 0]

    def __call__(self, elapsed, *, final=False):
        if time.monotonic() >= self.deadline:
            return 'suite_timeout'
        try:
            reserve(self.root, 0)
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

    def record(self):
        return {'per_rank_peak_reserved_bytes': self.peaks}


def verify_training(root, name, p, digest):
    import numpy as np
    stage = root / name
    descriptions = {}
    for rank in (0, 1):
        value = read(stage / f'worker-rank-{rank}.json')
        expected = {'status': 'terminal_complete', 'case': name, 'rank': rank, 'new_updates': 200,
                    'validation_evaluated': False, 'native_training_complete': False, 'protocol_sha256': digest}
        if any(value.get(k) != v for k, v in expected.items()) or not value['loaded_train_ids']:
            raise ValueError('invalid training terminal')
        observed = value['observation']
        if (observed['posts'] != 200 or observed['relocations'] != [12100, 12200]
                or observed['extra_training_forwards'] != 0 or observed['extra_backwards'] != 0
                or observed['full_numerical_noninterference_proven'] is not False
                or set(observed['manifests']) != {'event-before', 'event-after', 'cohort', 'cohort-final'}
                or sha256(stage / f'trajectory-rank-{rank}/index.json') != observed['index_sha256']):
            raise ValueError('minimal observation evidence incomplete')
        for label, sha in {**observed['manifests'], 'endpoint': value['endpoint_manifest_sha256']}.items():
            folder = stage / f'{label}-rank-{rank}'
            if sha256(folder / 'manifest.json') != sha:
                raise ValueError('training snapshot binding changed')
            payload = analyze.state(folder)
            if label in ('event-before', 'event-after'):
                phase = 'after_Adam_before_relocation' if label == 'event-before' else 'after_relocation_before_noise'
                if (payload['owner'], payload['event'], payload['phase']) != (rank, 12100, phase):
                    raise ValueError('event snapshot phase or owner differs')
            if label in ('event-before', 'event-after', 'endpoint'):
                params = payload['parameters']
                shapes = {'means': (1500000, 3), 'scales': (1500000, 3), 'quats': (1500000, 4),
                          'opacities': (1500000,), 'sh0': (1500000, 1, 3), 'shN': (1500000, 15, 3)}
                if set(params) != set(shapes) or any(a.shape != shapes[n] or a.dtype != np.float32 or not np.isfinite(a).all() for n, a in params.items()):
                    raise ValueError('invalid model snapshot')
        descriptions[str(rank)] = analyze.cohort(stage, rank)
    current = previous.metrics(root, name, {'source_identity': {'native': p['source_identity']['native']}})
    health = {str(r): previous.admission.jsonl(previous.native_artifacts(root, name) / f'mcmc-rank-{r}.jsonl') for r in (0, 1)}
    for rank, description in descriptions.items():
        for iteration, phase, dead_key, cutoff_key in ((12100, 'after_strategy_before_validation', 'initial_dead_count', 'initial_below_cutoff'),
                (12200, 'before_strategy_after_optimizer', 'final_dead_count', 'final_below_cutoff')):
            h = next(r['health'] for r in health[rank] if r['iteration'] == iteration and r['phase'] == phase)
            if h['dead_count'] != description[dead_key] or h['below_render_cutoff_count'] != description[cutoff_key]:
                raise ValueError('cohort and native health disagree')
    return {'owners': descriptions, 'native_metrics': current['metrics'], 'camera_sequence': current['camera_sequence']}


def verify_render(root, name, p, digest):
    results = {}
    for rank in (0, 1):
        value = read(root / 'renders' / name / f'worker-rank-{rank}.json')
        ids = p['camera_contract']['image_ids'][rank::2]
        expected = {'status': 'measured', 'case': name, 'rank': rank, 'protocol_sha256': digest,
                    'image_ids': ids, 'camera_forwards': 16, 'backwards': 0, 'optimizer_steps': 0,
                    'atol': snapshots.ATOL, 'rtol': snapshots.RTOL,
                    'discrete_selection_causality_proven': False, 'long_term_quality_effect_proven': False}
        if (any(value.get(k) != v for k, v in expected.items()) or set(value['loaded_train_ids']) != set(ids)
                or set(value['effects']) != set(ids) or set(value['repeat_checks']) != {f'{s}/{i}' for s in ('before', 'after') for i in ids}
                or value['interpretation_usable'] != all(r['usable'] for r in value['repeat_checks'].values())):
            raise ValueError('isolated render evidence incomplete')
        for row in value['repeat_checks'].values():
            if row['usable'] != (render_probe.repeat_usable(row['pixels']) and not row['loss']['mismatch']):
                raise ValueError('render repeat classification differs')
        results[str(rank)] = value
    return results


def bind_complete(root, stage, name, p, digest, monitor, rendering):
    data = verify_render(root, name, p, digest) if rendering else verify_training(root, name, p, digest)
    files = [f for f in stage.rglob('*') if f.is_file()]
    if not rendering:
        files += [f for f in previous.native_artifacts(root, name).rglob('*') if f.is_file()]
        files.append(root / 'training/attempts' / name / 'attempt.json')
    exit_record = read(stage / 'case.exit.json')
    if exit_record['returncode'] != 0 or exit_record['resource_failure'] is not None:
        raise ValueError('stage did not finish cleanly')
    write_new(stage / 'complete.json', {'case': name, 'mode': 'render' if rendering else 'training',
        'protocol_sha256': digest, 'resources': monitor.record(),
        'files': {str(f.relative_to(root)): sha256(f) for f in files}})
    return data


def verify_complete(root, stage, name, digest, rendering=False):
    value = read(stage / 'complete.json')
    if (value['case'], value['mode'], value['protocol_sha256']) != (name, 'render' if rendering else 'training', digest):
        raise ValueError('completion identity differs')
    required = {str((stage / f'worker-rank-{r}.json').relative_to(root)) for r in (0, 1)}
    required.add(str((stage / 'case.exit.json').relative_to(root)))
    if not required <= value['files'].keys():
        raise ValueError('completion bindings incomplete')
    for rel, sha in value['files'].items():
        path = snapshots.safe_file(root, rel)
        native = previous.native_artifacts(root, name)
        attempt = root / 'training/attempts' / name / 'attempt.json'
        if (not (path.is_relative_to(stage) or (not rendering and (path.is_relative_to(native) or path == attempt)))
                or sha256(path) != sha):
            raise ValueError('completion evidence changed or escaped stage')
    exit_record = read(stage / 'case.exit.json')
    if exit_record['returncode'] != 0 or exit_record['resource_failure'] is not None:
        raise ValueError('failed stage cannot be complete')
    return value


def run(args, p):
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    legacy.unlimited_host_cgroup(UNIT)
    root, digest = args.output_root, args.protocol_sha256
    with (root / 'execution.once').open('x'):
        pass
    connected.publish_status(root / 'started.json', {'pid': os.getpid(),
        'proc_start': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], 'protocol_sha256': digest, 'started_unix': time.time()})
    terminal = {'status': 'failed', 'exit_code': 1, 'protocol_sha256': digest}
    deadline = time.monotonic() + CONTRACT['suite_timeout_seconds']
    def expired(signum, frame):
        raise TimeoutError('minimal survival suite exceeded one hour')
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(CONTRACT['suite_timeout_seconds'])
    try:
        with FileLease(PROJECT / 'outputs/.gpu.lock') as lease:
            require_idle_gpu()
            verify_inputs(p)
            if shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
                raise ValueError('insufficient disk admission')
            source = p['source_identity']['native']
            parent = Path(source['parent_training']) / 'attempts/train-001'
            target = root / 'training/attempts/train-001'
            target.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(parent / 'attempt.json', target / 'attempt.json')
            shutil.copytree(parent / 'checkpoints/iteration_000012000', target / 'checkpoints/iteration_000012000')
            for path in target.rglob('*'):
                path.chmod(0o555 if path.is_dir() else 0o444)
            target.chmod(0o555)
            connected.parent_metadata(source, root / 'training')
            outcomes = {'training': {}, 'render': {}}
            durations = []
            for rendering in (False, True):
                for name in CASES:
                    if time.monotonic() + 60 >= deadline:
                        raise TimeoutError('insufficient time for next stage')
                    if not rendering and durations and time.monotonic() + max(durations) + 300 >= deadline:
                        raise TimeoutError('remaining fixed suite cannot fit budget')
                    if rendering:
                        for case in CASES:
                            verify_complete(root, root / case, case, digest)
                    require_idle_gpu()
                    subprocess.run(['nvidia-smi', '-L'], check=True)
                    stage = root / ('renders' if rendering else '') / name
                    stage.mkdir(parents=True, exist_ok=False)
                    monitor = StageMonitor(root, stage, deadline)
                    command = ['/usr/bin/env', '-u', 'PYTHONPATH', sys.executable, '-m', MODULE,
                        '--render-worker' if rendering else '--worker', '--case', name, '--output-root', str(root),
                        '--authorize-execution', '--protocol-sha256', digest, '--lease-fd', str(lease.fileno())]
                    started = time.monotonic()
                    legacy.run_stage(stage, 'case', command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
                    data = bind_complete(root, stage, name, p, digest, monitor, rendering)
                    outcomes['render' if rendering else 'training'][name] = data
                    if not rendering:
                        durations.append(time.monotonic() - started)
                        connected.parent_metadata(source, root / 'training')
            if outcomes['training']['T0']['camera_sequence'] != outcomes['training']['T1']['camera_sequence']:
                raise ValueError('training camera sequences differ')
            for name in CASES:
                verify_complete(root, root / name, name, digest)
                verify_complete(root, root / 'renders' / name, name, digest, True)
            verify_inputs(p)
            report = {'status': 'descriptive_trajectories_complete', 'protocol_sha256': digest,
                'implementation_commit': p['implementation_commit'], **outcomes,
                'descriptive_summary': analyze.summarize(outcomes['training']),
                'complete_sha256': {str(folder.relative_to(root)): sha256(folder / 'complete.json')
                    for name in CASES for folder in (root / name, root / 'renders' / name)},
                'full_numerical_noninterference_proven': False, 'old_gate1': 'not_reclassified',
                'E007': 'calibration_blocked_not_reclassified', 'cross_run_slot_pairing': False,
                'long_term_causality_proven': False, 'new_training_updates': 400}
            reserve(root, len(json.dumps(report).encode()) + 4096)
            write_new(root / 'comparison.json', report)
            terminal.update(status='completed', exit_code=0, comparison_sha256=sha256(root / 'comparison.json'))
    except BaseException as exc:
        terminal['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        connected.publish_status(root / 'execution.json', {**terminal, 'ended_unix': time.time()})


def monitor(root, digest):
    start = time.monotonic()
    while time.monotonic() - start < CONTRACT['suite_timeout_seconds'] + 300:
        if (root / 'execution.json').exists():
            value = read(root / 'execution.json')
            if value['protocol_sha256'] != digest or {'completed': 0, 'failed': 1}.get(value['status']) != value['exit_code']:
                raise ValueError('terminal identity differs')
            if value['status'] == 'completed' and sha256(root / 'comparison.json') != value['comparison_sha256']:
                raise ValueError('comparison changed')
            print(json.dumps(value), flush=True)
            return value['exit_code']
        if (root / 'started.json').exists():
            s = read(root / 'started.json')
            if s['protocol_sha256'] != digest:
                raise ValueError('executor identity changed')
            try:
                alive = Path(f'/proc/{s["pid"]}/stat').read_text().split()[21] == s['proc_start']
            except FileNotFoundError:
                alive = False
            if not alive and not (root / 'execution.json').exists():
                raise RuntimeError('executor disappeared without terminal')
        elif time.monotonic() - start > 300:
            raise RuntimeError('executor did not start')
        print(json.dumps({'status': 'waiting_not_acceptance', 'completed_training': [n for n in CASES if (root / n / 'complete.json').exists()],
                          'completed_renders': [n for n in CASES if (root / 'renders' / n / 'complete.json').exists()]}), flush=True)
        time.sleep(30)
    raise TimeoutError('monitor expired without stopping executor')


def distributed_rank(rank, function, args):
    import torch
    from gsplat.distributed import _distributed_worker
    init = torch.distributed.init_process_group
    def bounded(*a, **kw):
        kw['timeout'] = timedelta(seconds=CONTRACT['collective_timeout_seconds'])
        return init(*a, **kw)
    with patch.object(torch.distributed, 'init_process_group', bounded):
        return _distributed_worker(rank, 2, function, args, None, True)


def launch(function, args):
    import torch
    from gsplat.distributed import _find_free_port
    if torch.cuda.device_count() != 2 or 'OMPI_COMM_WORLD_SIZE' in os.environ:
        raise ValueError('exactly two local GPUs required')
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', str(_find_free_port())
    torch.multiprocessing.spawn(distributed_rank, args=(function, args), nprocs=2, join=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'worker', 'render-worker', 'monitor'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--authorize-prepare', action='store_true')
    parser.add_argument('--authorize-execution', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != 'i-94B8D131' or str(PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        raise SystemExit('仅指定远端可执行；本机只允许轻量检查')
    if not (args.authorize_prepare if args.prepare else True if args.monitor else args.authorize_execution):
        raise SystemExit('未获Q1最小观测显式授权')
    if args.case is not None and not (args.worker or args.render_worker):
        parser.error('--case仅允许内部worker，不开放跳过固定次序')
    args.output_root = output_root(args.output_root, args.prepare)
    if args.prepare:
        if args.preflight_report is None:
            parser.error('必须提供独立新版结构前置报告')
        prepare(args.output_root, verify_preflight(args.preflight_report))
        return
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.monitor:
        raise SystemExit(monitor(args.output_root, args.protocol_sha256))
    if args.run:
        run(args, p)
        return
    if args.case is None or args.lease_fd is None or os.readlink(f'/proc/self/fd/{args.lease_fd}') != str(PROJECT / 'outputs/.gpu.lock'):
        raise ValueError('worker requires inherited GPU lease')
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    legacy.unlimited_host_cgroup(UNIT)
    launch(worker, args)


if __name__ == '__main__':
    main()
