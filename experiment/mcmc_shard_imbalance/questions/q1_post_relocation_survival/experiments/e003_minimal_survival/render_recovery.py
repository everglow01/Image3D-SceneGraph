"""Render-only recovery from pinned completed trajectories; no training entrypoint."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from . import run as base, render_probe

PROJECT, RUNS, read, sha256, write_new, snapshots = base.PROJECT, base.RUNS, base.read, base.sha256, base.write_new, base.snapshots
MODULE = base.REL.replace('/', '.') + '.render_recovery'
UNIT = 'image3d-survival-render-recovery-v1.service'
PREFLIGHT_UNIT = 'image3d-survival-cold-render-v1.service'
SOURCE = RUNS / 'survival-20261027-v1'
SOURCE_COMMIT = '3d5a7df2822bd5dfda617ceb890c3d48f39bb004'
SOURCE_SHA = '7390dacff982e1add40cb3dd99b6b1c4943eb771aa0edf86ceee120b27286c97'
SOURCE_COMPLETE = {'T0': '7e3d49761582a67dfdbd75d5ff19a1bd953b306f17a7cb1285fd656f9089d089',
                   'T1': 'f97903a431e757eb1bef5f591f6b3f144e3d314b7677fa3054a31cc5eefa7124'}
FILES = tuple(base.REL + '/' + n for n in ('render_recovery.py', 'tests/test_render_recovery.py', 'tests/test_cold_render_remote.py'))
DEPENDENCIES = tuple(dict.fromkeys((*base.DEPENDENCIES, *FILES)))
CONTRACT = {'profile': 'minimal_survival_render_recovery_v1', 'cases': list(base.CASES), 'source_commit': SOURCE_COMMIT,
    'source_protocol_sha256': SOURCE_SHA, 'source_complete_sha256': SOURCE_COMPLETE,
    'new_training_updates': 0, 'loss_backwards': 0, 'optimizer_steps': 0, 'strategy_calls': 0,
    'camera_forwards': 64, 'suite_timeout_seconds': 1800, 'collective_timeout_seconds': 120,
    'max_output_bytes': 1024**3, 'disk_admission_bytes': 5 * 1024**3, 'host_limits': None,
    'max_per_rank_reserved_bytes': 18 * 1024**3, 'preflight_timeout_seconds': 900,
    'preflight_camera_forwards': 8, 'atol': snapshots.ATOL, 'rtol': snapshots.RTOL,
    'original_suite_status': 'failed_at_isolated_render_not_reclassified',
    'full_numerical_noninterference_proven': False, 'long_term_causality_proven': False}


def execution_commit():
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *DEPENDENCIES], cwd=PROJECT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *DEPENDENCIES], cwd=PROJECT, check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()


def source_identity():
    if sha256(snapshots.safe_file(SOURCE, 'protocol.json')) != SOURCE_SHA:
        raise ValueError('historical protocol changed')
    p = read(SOURCE / 'protocol.json')
    if p['implementation_commit'] != SOURCE_COMMIT or set(p['code_sha256']) != set(base.DEPENDENCIES):
        raise ValueError('historical dependency identity differs')
    # Old artifacts bind the old Git objects, not today's repaired renderer.
    for name, expected in p['code_sha256'].items():
        snapshots.safe_file(PROJECT, name)
        blob = subprocess.check_output(['git', 'show', f'{SOURCE_COMMIT}:{name}'], cwd=PROJECT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('historical execution code differs')
    native = p['source_identity']['native']
    package = Path(native['pair']['source_package'])
    if base.legacy.file_hashes(package) != native['pair']['source_hashes']:
        raise ValueError('original renderer package changed')
    replay = Path(native['pair']['replay'])
    if sha256(replay / 'dataset.json') != native['replay_record']['dataset_sha256']:
        raise ValueError('original replay dataset changed')
    for image in p['camera_contract']['images']:
        path = (replay / image['path']).resolve()
        if not path.is_relative_to(replay.resolve()) or sha256(path) != image['sha256']:
            raise ValueError('selected Train image changed')
    terminal = read(SOURCE / 'execution.json')
    if terminal['status'] != 'failed' or terminal['exit_code'] != 1 or terminal['protocol_sha256'] != SOURCE_SHA:
        raise ValueError('historical failed terminal changed')
    for name, expected in SOURCE_COMPLETE.items():
        if sha256(SOURCE / name / 'complete.json') != expected:
            raise ValueError('historical training completion changed')
        base.verify_complete(SOURCE, SOURCE / name, name, SOURCE_SHA)
    return {'protocol_sha256': SOURCE_SHA, 'complete_sha256': SOURCE_COMPLETE,
            'failed_terminal_sha256': sha256(SOURCE / 'execution.json'), 'camera_contract': p['camera_contract']}


def verify_audit(folder, rank, result):
    initialization = read(folder / f'initialization-rank-{rank}.json')
    if (initialization != result['initialization'] or initialization['rank'] != rank
            or initialization['warmup_forward_calls'] != 0 or initialization['warmup_backward_calls'] != 0
            or not initialization['initialization_outside_measurement'] or initialization['rng_restored']):
        raise ValueError('render initialization audit differs')
    keys = {'python', 'numpy', 'torch', 'cuda'}
    if set(initialization['rng_before']) != keys or set(initialization['rng_after']) != keys:
        raise ValueError('initialization RNG coverage incomplete')
    changed = [k for k in initialization['rng_before'] if initialization['rng_before'][k] != initialization['rng_after'][k]]
    if initialization['changed_rng_components'] != changed:
        raise ValueError('initialization RNG classification differs')
    count = len(result['image_ids'])
    expected = {f'rng-{s}-{r}-{i}-{part}-rank-{rank}.json' for s in ('before', 'after') for r in range(2)
                for i in range(count) for part in ('load', 'render', 'loss', 'export')}
    if set(result['rng_audit_sha256']) != expected or result['camera_forwards'] != count * 4:
        raise ValueError('render audit coverage or forward budget differs')
    for name, digest in result['rng_audit_sha256'].items():
        path = snapshots.safe_file(folder, name)
        if sha256(path) != digest:
            raise ValueError('RNG phase audit changed')
        row = read(path)
        if (row['rank'] != rank or name != f'rng-{row["phase"]}-rank-{rank}.json'
                or set(row['before']) != keys or row['before'] != row['after']
                or row['changed_rng_components'] or row['error'] is not None or row['rng_restored']):
            raise ValueError('render phase consumed RNG or failed')


def verify_preflight(path):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    p = read(path)
    if (p['status'] != 'passed' or p['profile'] != CONTRACT['profile'] or p['implementation_commit'] != execution_commit()
            or p['code_sha256'] != {n: sha256(PROJECT / n) for n in DEPENDENCIES}
            or p['runtime'] != base.expected_runtime() or p['camera_forwards'] != 8
            or p['new_training_updates'] != 0 or p['real_image_decodes'] != 0 or p['loss_backwards'] != 0
            or p['optimizer_steps'] != 0 or set(p['workers']) != {'0', '1'}):
        raise ValueError('missing or stale cold-render preflight')
    for rank, item in p['workers'].items():
        file = snapshots.safe_file(RUNS, Path(item['path']).relative_to(RUNS))
        if sha256(file) != item['sha256']:
            raise ValueError('cold preflight evidence changed')
        result = read(file)
        if (result['rank'] != int(rank) or result['status'] != 'measured' or result['camera_forwards'] != 4
                or result['backwards'] != 0 or result['optimizer_steps'] != 0
                or result['initialization']['backend_loaded_before'] or not result['cold_before_first_render']
                or result['training_calls'] != 0 or not result['negative_rng_guard_passed']
                or not result['model_unchanged'] or not result['interpretation_usable'] or result['synthetic_image_decodes'] != 1
                or result['atol'] != snapshots.ATOL or result['rtol'] != snapshots.RTOL):
            raise ValueError('cold process boundary not verified')
        expected = {f'{phase}/{i}' for phase in ('before', 'after') for i in result['image_ids']}
        if set(result['repeat_checks']) != expected or not all(
                row['usable'] and render_probe.repeat_usable(row['pixels']) and not row['loss']['mismatch']
                for row in result['repeat_checks'].values()):
            raise ValueError('cold render repeat evidence incomplete')
        verify_audit(file.parent, int(rank), result)
        negative = result['negative_rng_audit']
        bad = snapshots.safe_file(file.parent, negative['path'])
        if sha256(bad) != negative['sha256']:
            raise ValueError('negative RNG audit changed')
        negative_row = read(bad)
        if (negative_row['changed_rng_components'] != ['python'] or negative_row['rng_restored']
                or set(negative_row['before']) != {'python', 'numpy', 'torch', 'cuda'}
                or [k for k in negative_row['before'] if negative_row['before'][k] != negative_row['after'][k]] != ['python']):
            raise ValueError('RNG rejection fixture missing')
    return {'path': str(path), 'sha256': sha256(path)}


def output_root(path, preparing=False):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    if path.parent != RUNS or not path.name.startswith('render-recovery-') or (preparing and path.exists()):
        raise ValueError('new independent render-recovery-* directory required')
    return path


def reserve(root, extra=0):
    if extra < 0 or base.previous.output_bytes(root) + 2 * extra + 1024**2 > CONTRACT['max_output_bytes']:
        raise ValueError('render recovery output budget')
    if shutil.disk_usage(root).free < 4 * 1024**3 + 2 * extra:
        raise ValueError('render recovery disk reserve')


def prepare(root, preflight):
    source = source_identity()
    if shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
        raise ValueError('insufficient render recovery disk admission')
    root.mkdir(exist_ok=False)
    write_new(root / 'protocol.json', {**CONTRACT, 'source': source, 'camera_contract': source['camera_contract'],
        'implementation_commit': execution_commit(), 'code_sha256': {n: sha256(PROJECT / n) for n in DEPENDENCIES},
        'preflight': preflight, 'status': 'prepared_render_only'})


def protocol_at(root, digest):
    if not digest or sha256(snapshots.safe_file(root, 'protocol.json')) != digest:
        raise ValueError('explicit recovery protocol SHA required')
    p = read(root / 'protocol.json')
    if (any(p.get(k) != v for k, v in CONTRACT.items()) or set(p['code_sha256']) != set(DEPENDENCIES)
            or any(sha256(snapshots.safe_file(PROJECT, n)) != h for n, h in p['code_sha256'].items())):
        raise ValueError('render recovery contract/code changed')
    return p


def verify_inputs(p):
    if (execution_commit() != p['implementation_commit'] or source_identity() != p['source']
            or verify_preflight(p['preflight']['path']) != p['preflight']
            or p['camera_contract'] != p['source']['camera_contract']):
        raise ValueError('render recovery source or cold preflight changed')


def worker(local_rank, rank, world_size, args):
    import torch
    from image3d_scenegraph.gaussian import runtime
    from image3d_scenegraph.gaussian.absgrad_resources import memory_telemetry, failure_record
    if (world_size, local_rank) != (2, rank):
        raise ValueError('exactly two local ranks required')
    p = protocol_at(args.output_root, args.protocol_sha256)
    verify_inputs(p)
    historical = read(SOURCE / 'protocol.json')
    base.prime_package(historical, rank)
    replay = Path(historical['source_identity']['native']['pair']['replay'])
    config = historical['source_identity']['config']['effective_config']
    dataset = read(replay / 'dataset.json')
    if render_probe.camera_contract(dataset, config) != p['camera_contract']:
        raise ValueError('original selected Train cameras changed')
    stage = args.output_root / 'renders' / args.case
    endpoint = base.Endpoint(stage, rank, p['camera_contract']['image_ids'], lambda n: reserve(args.output_root, n))
    failure = stage / f'failure-rank-{rank}'
    failure.mkdir(exist_ok=False)
    with memory_telemetry(stage / 'memory', rank, rank), failure_record(failure, CONTRACT['profile'], args.case), endpoint.installed():
        views = runtime.load_training_views(dataset, replay, split='train',
            longest_edge=int(config['resolution']['longest_edge']), device=torch.device('cpu'))
        result = render_probe.probe(SOURCE / args.case, rank, views, p['camera_contract']['image_ids'], config, audit_root=stage)
        endpoint.consensus('render-finish', lambda: write_new(stage / f'worker-rank-{rank}.json',
            {**result, 'case': args.case, 'loaded_train_ids': sorted(endpoint.loaded),
             'protocol_sha256': args.protocol_sha256, 'source_complete_sha256': SOURCE_COMPLETE[args.case]}))


class Monitor(base.StageMonitor):
    def __call__(self, elapsed, *, final=False):
        try:
            reserve(self.root)
        except ValueError as exc:
            return str(exc)
        return super().__call__(elapsed, final=final)


def run(args, p):
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    base.legacy.unlimited_host_cgroup(UNIT)
    root, digest = args.output_root, args.protocol_sha256
    with (root / 'execution.once').open('x'):
        pass
    base.connected.publish_status(root / 'started.json', {'pid': os.getpid(),
        'proc_start': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], 'protocol_sha256': digest, 'started_unix': time.time()})
    terminal = {'status': 'failed', 'exit_code': 1, 'protocol_sha256': digest}
    deadline = time.monotonic() + CONTRACT['suite_timeout_seconds']
    def expired(signum, frame):
        raise TimeoutError('render-only recovery exceeded thirty minutes')
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(CONTRACT['suite_timeout_seconds'])
    try:
        with FileLease(PROJECT / 'outputs/.gpu.lock') as lease:
            require_idle_gpu()
            verify_inputs(p)
            if shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
                raise ValueError('insufficient render-only disk admission')
            results = {}
            for name in base.CASES:
                if time.monotonic() + 30 >= deadline:
                    raise TimeoutError('no remaining render budget')
                require_idle_gpu()
                subprocess.run(['nvidia-smi', '-L'], check=True)
                stage = root / 'renders' / name
                stage.mkdir(parents=True, exist_ok=False)
                monitor = Monitor(root, stage, deadline)
                command = ['/usr/bin/env', '-u', 'PYTHONPATH', sys.executable, '-m', MODULE, '--worker', '--case', name,
                    '--output-root', str(root), '--protocol-sha256', digest, '--authorize-render-only', '--lease-fd', str(lease.fileno())]
                base.legacy.run_stage(stage, 'case', command, cwd=PROJECT, monitor=monitor, pass_fds=(lease.fileno(),), host_group=None)
                for rank in (0, 1):
                    result = read(stage / f'worker-rank-{rank}.json')
                    if result['source_complete_sha256'] != SOURCE_COMPLETE[name]:
                        raise ValueError('render used a different historical trajectory')
                    verify_audit(stage, rank, result)
                results[name] = base.bind_complete(root, stage, name, p, digest, monitor, True)
            for name in base.CASES:
                base.verify_complete(root, root / 'renders' / name, name, digest, True)
            verify_inputs(p)
            report = {'status': 'render_only_complete', 'protocol_sha256': digest, 'source': p['source'], 'results': results,
                'complete_sha256': {n: sha256(root / 'renders' / n / 'complete.json') for n in base.CASES},
                'new_training_updates': 0, 'original_suite_status': CONTRACT['original_suite_status'],
                'full_numerical_noninterference_proven': False, 'long_term_causality_proven': False}
            reserve(root, len(json.dumps(report).encode()))
            write_new(root / 'comparison.json', report)
            terminal.update(status='completed', exit_code=0, comparison_sha256=sha256(root / 'comparison.json'))
    except BaseException as exc:
        terminal['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        base.connected.publish_status(root / 'execution.json', {**terminal, 'ended_unix': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'worker', 'monitor'):
        group.add_argument('--' + name, action='store_true')
    parser.add_argument('--authorize-prepare', action='store_true')
    parser.add_argument('--authorize-render-only', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--case', choices=base.CASES)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if socket.gethostname() != 'i-94B8D131' or str(PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        raise SystemExit('仅指定远端可执行只读补渲染')
    if not (args.authorize_prepare if args.prepare else True if args.monitor else args.authorize_render_only):
        raise SystemExit('未获独立只读补渲染授权')
    if args.case is not None and not args.worker:
        parser.error('不允许跳过固定补渲染次序')
    args.output_root = output_root(args.output_root, args.prepare)
    if args.prepare:
        if args.preflight_report is None:
            parser.error('必须提供新版冷进程前置报告')
        prepare(args.output_root, verify_preflight(args.preflight_report))
        return
    p = protocol_at(args.output_root, args.protocol_sha256)
    if args.monitor:
        # Read-only monitor may wait longer than the bounded 30-minute worker; it never stops or restarts it.
        raise SystemExit(base.monitor(args.output_root, args.protocol_sha256))
    if args.run:
        run(args, p)
        return
    if args.case is None or args.lease_fd is None or os.readlink(f'/proc/self/fd/{args.lease_fd}') != str(PROJECT / 'outputs/.gpu.lock'):
        raise ValueError('render worker requires inherited GPU lease')
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.legacy.unlimited_host_cgroup(UNIT)
    base.launch(worker, args)


if __name__ == '__main__':
    main()
