"""CPU-only adoption of a pinned prefix payload; never rerun or reclassify it."""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from . import run as base

PROJECT, RUNS, read, sha256, write_new, snapshots = base.PROJECT, base.RUNS, base.read, base.sha256, base.write_new, base.snapshots
MODULE = base.REL.replace('/', '.') + '.prefix_recovery'
PROFILE = 'source_adam_prefix_recovery_v1'
UNIT = 'image3d-source-adam-recovery-v1.service'
PREFLIGHT_UNIT = 'image3d-source-adam-recovery-cpu-preflight-v1.service'
SOURCE = RUNS / 'adam-intervention-20261029-v2'
SOURCE_COMMIT = 'a269eb1df1153756f20ae531c60982f239053fb6'
SOURCE_SHA = '0332814e20f2dd4fcbbe0234e201a9fc22633ac52d169472b4a74fd0a64e793c'
SOURCE_PREFLIGHT_SHA = '6c649c14cfa03324149c53f96aaec7d0dc274764fe26f39c68451a59170ffb14'
CHECKPOINT_HASH = 'ed3257af8a07625b8abeacfe6df4016d24c9db9e8a4cca95336fa11888f326f9'
STATE_HASHES = {'0': '22300bebffa170eb2e70420ba5ffa3c46ac4c9ed6f4385fadcee3be912624afd',
                '1': '1f88f16a47ad2881a77a71ab943c7d73b051d63163714d9ceddf9efdb4a2e5ec'}
HISTORICAL_FILES = tuple(dict.fromkeys((*base.base.DEPENDENCIES, *base.FILES)))
CPU_DEPENDENCIES = tuple(n for n in base.DEPENDENCIES if '/tests/' not in n or n in base.RECOVERY_FILES)
CONTRACT = {'profile': PROFILE, 'new_training_updates': 0, 'forward_calls': 0, 'backward_calls': 0,
    'optimizer_steps': 0, 'strategy_calls': 0, 'timeout_seconds': 1800, 'preflight_timeout_seconds': 900,
    'output_limit_bytes': 8 * 1024**3, 'disk_admission_bytes': 12 * 1024**3, 'disk_reserve_bytes': 4 * 1024**3,
    'host_limits': None, 'original_suite_status': 'failed', 'original_child_exit_code': None,
    'original_execution_reclassified': False, 'arms_authorized': False}
CPU_FLAGS = ('real_runner_success_exit_record', 'real_runner_failure_exit_record', 'real_runner_rejection_record',
             'native_checkpoint_cpu_decode', 'lineage_copy_verified', 'learning_rate_formula_verified',
             'json_representation_verified', 'original_files_unchanged', 'cuda_not_initialized', 'no_training_calls')


def output_root(path, preparing=False):
    path = base.output_root(path, preparing)
    if not path.name.startswith('adam-intervention-recovery-'):
        raise ValueError('independent adam-intervention-recovery-* directory required')
    return path


def contract(p):
    r = p.get('prefix_recovery')
    if not isinstance(r, dict) or any(r.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError('invalid zero-training recovery contract')
    source = r.get('source', {})
    if (source.get('root') != str(SOURCE) or source.get('commit') != SOURCE_COMMIT
            or source.get('protocol_sha256') != SOURCE_SHA or source.get('checkpoint_hash') != CHECKPOINT_HASH
            or source.get('full_state_sha256') != STATE_HASHES):
        raise ValueError('recovery source identity differs')
    return r


def verify_historical_code(p):
    hashes = p['code_sha256']
    if p['implementation_commit'] != SOURCE_COMMIT or set(hashes) != set(HISTORICAL_FILES):
        raise ValueError('historical code set differs')
    mutable = {base.REL + '/' + n for n in ('run.py', 'analyze.py')}
    for name, expected in hashes.items():
        snapshots.safe_file(PROJECT, name)
        old = subprocess.check_output(['git', 'show', f'{SOURCE_COMMIT}:{name}'], cwd=PROJECT)
        if hashlib.sha256(old).hexdigest() != expected:
            raise ValueError('historical Git object differs')
        if name not in mutable and sha256(PROJECT / name) != expected:
            raise ValueError('frozen GPU probe/training dependency changed; CPU recovery is insufficient')
        if name == base.REL + '/analyze.py':
            original, current = ast.parse(old), ast.parse((PROJECT / name).read_text())
            for function in ('exact', 'optimizer_options', 'gate0', 'adam_prediction', 'first_step', 'trajectory'):
                a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == function)
                b = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == function)
                if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
                    raise ValueError('frozen numerical analysis changed: ' + function)


def missing_original_records():
    for name in ('prefix/case.exit.json', 'prefix/complete.json', 'gate0.json', 'A', 'B', 'arms-protocol.json', 'arms.once'):
        if snapshots.safe_file(SOURCE, name).exists():
            raise ValueError('original failed execution boundary changed: ' + name)


def checkpoint_state_digest(folder, rank, *, count=1500000):
    """Decode already-hash-verified native components on CPU; instantiate no model/optimizer."""
    import torch
    from image3d_scenegraph.gaussian import trainer
    def component(name):
        content = snapshots.safe_file(folder, name + '.bin').read_bytes()
        shard = trainer._checkpoint_rank_bytes(content, rank, 2)
        return torch.load(io.BytesIO(shard), map_location='cpu', weights_only=False)
    model = component('model')
    if model['max_sh_degree'] != 3:
        raise ValueError('prefix SH degree differs')
    s = model['state_dict']
    params = {'means': s['means'], 'scales': s['log_scales'], 'quats': s['quats'], 'opacities': s['opacity_logits'],
              'sh0': s['sh_coeffs'][:, :1], 'shN': s['sh_coeffs'][:, 1:]}
    shapes = {'means': (count, 3), 'scales': (count, 3), 'quats': (count, 4), 'opacities': (count,),
              'sh0': (count, 1, 3), 'shN': (count, 15, 3)}
    if any(tuple(v.shape) != shapes[n] or v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for n, v in params.items()):
        raise ValueError('invalid checkpoint parameters')
    optim = component('optimizer')
    if set(optim) != set(snapshots.PARAMETERS):
        raise ValueError('checkpoint optimizer groups differ')
    adam = {}
    for name, saved in optim.items():
        if len(saved['param_groups']) != 1 or len(saved['param_groups'][0]['params']) != 1:
            raise ValueError('checkpoint parameter mapping differs')
        group = saved['param_groups'][0]
        key = group['params'][0]
        if set(saved['state']) != {key}:
            raise ValueError('unexpected checkpoint optimizer state')
        adam[name] = {'state': saved['state'][key], 'param_groups': [{k: v for k, v in group.items() if k != 'params'}]}
    dense, rng = component('densification'), component('rng')
    state = {'parameters': params, 'adam': adam, 'strategy': dense['strategy_state'],
             'camera': {'order': list(dense['camera_order']), 'cursor': int(dense['camera_cursor'])}, 'rng': rng,
             'optimizer_classes': dict.fromkeys(snapshots.PARAMETERS, 'torch.optim.adam.Adam')}
    def cpu_only(value):
        if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
            raise ValueError('CPU decoder reached a non-CPU tensor')
        if isinstance(value, dict):
            for item in value.values():
                cpu_only(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                cpu_only(item)
    cpu_only(state)
    return snapshots.digest_state(state)


def source_paths(p):
    files = []
    for file in SOURCE.rglob('*'):
        if file.is_symlink():
            raise ValueError('symlink in original prefix')
        if file.is_file():
            files.append(file)
    preflight = Path(p['preflight']['path'])
    files.append(preflight)
    for row in read(preflight)['workers'].values():
        file = Path(row['path'])
        files.append(file)
        files.extend(snapshots.safe_file(file.parent, name) for name in read(file)['evidence_sha256'])
    return files


def lineage_manifest():
    from image3d_scenegraph.gaussian.checkpoint import _COMPONENT_FILES
    copies = {}
    for attempt, iteration in (('train-001', 12000), ('prefix', 12100)):
        folder = SOURCE / 'training/attempts' / attempt
        checkpoint = folder / 'checkpoints' / f'iteration_{iteration:09d}'
        names = {'checkpoint.json', *_COMPONENT_FILES.values()}
        if {f.name for f in checkpoint.iterdir()} != names:
            raise ValueError('unexpected checkpoint entries')
        for file in [folder / 'attempt.json', *(checkpoint / name for name in sorted(names))]:
            file = snapshots.safe_file(SOURCE, file.relative_to(SOURCE))
            if not file.is_file():
                raise ValueError('missing checkpoint file')
            copies[str(file.relative_to(SOURCE))] = {'source': str(file.relative_to(PROJECT)),
                                                    'sha256': sha256(file), 'bytes': file.stat().st_size}
    return copies


def source_identity():
    if sha256(snapshots.safe_file(SOURCE, 'protocol.json')) != SOURCE_SHA:
        raise ValueError('original protocol changed')
    p = read(SOURCE / 'protocol.json')
    verify_historical_code(p)
    if any(p.get(k) != v for k, v in base.CONTRACT.items()) or base.base.previous.source_identity() != p['source_identity']:
        raise ValueError('original runtime/source contract differs')
    missing_original_records()
    terminal = read(SOURCE / 'prefix-execution.json')
    if (terminal['status'] != 'failed' or terminal['exit_code'] != 1 or terminal['phase'] != 'prefix'
            or terminal['protocol_sha256'] != SOURCE_SHA
            or terminal['error'] != "AttributeError: 'Monitor' object has no attribute 'record'"):
        raise ValueError('original failure must remain unchanged')
    preflight = Path(p['preflight']['path'])
    if sha256(preflight) != SOURCE_PREFLIGHT_SHA or p['preflight']['sha256'] != SOURCE_PREFLIGHT_SHA:
        raise ValueError('historical GPU preflight changed')
    base.preflight_evidence(preflight, SOURCE_COMMIT, p['code_sha256'], base.base.expected_runtime())
    meta = base.checkpoint(SOURCE, p)
    if meta['checkpoint_hash'] != CHECKPOINT_HASH:
        raise ValueError('original complete checkpoint changed')
    base.base.connected.parent_metadata(p['source_identity']['native'], SOURCE / 'training')
    owners = {}
    for rank in (0, 1):
        base.verify_worker(SOURCE, 'prefix', rank, p)
        owner = base.analyze.gate0(SOURCE / 'prefix', rank)
        if base.canonical_report(owner) != base.canonical_report(read(SOURCE / 'prefix' / f'gate0-rank-{rank}.json')):
            raise ValueError('original Gate 0 values changed')
        owners[str(rank)] = json.loads(base.canonical_report(owner))
        memory = read(SOURCE / 'prefix/memory' / f'rank-{rank}.json')
        if not memory['finished'] or memory['peak_reserved_bytes'] > base.CONTRACT['max_per_rank_reserved_bytes']:
            raise ValueError('prefix telemetry is incomplete or outside budget')
        recorded = read(SOURCE / 'prefix' / f'prefix-state-rank-{rank}.json')['full_state_sha256']
        folder = SOURCE / 'training/attempts/prefix/checkpoints/iteration_000012100'
        if recorded != STATE_HASHES[str(rank)] or checkpoint_state_digest(folder, rank) != recorded:
            raise ValueError('full prefix checkpoint state differs')
    bound = {str(f.relative_to(PROJECT)): sha256(snapshots.safe_file(PROJECT, f.relative_to(PROJECT))) for f in source_paths(p)}
    copies = lineage_manifest()
    return {'root': str(SOURCE), 'commit': SOURCE_COMMIT, 'protocol_sha256': SOURCE_SHA,
            'historical_gpu_preflight': {'path': str(preflight), 'sha256': SOURCE_PREFLIGHT_SHA, 'scope': 'original_GPU_probe_not_new_CPU_recovery'},
            'checkpoint_hash': CHECKPOINT_HASH, 'full_state_sha256': STATE_HASHES, 'bound_files': bound,
            'copy_files': copies, 'gate0_owners': owners, 'original_failed_terminal_sha256': sha256(SOURCE / 'prefix-execution.json')}


def verify_source(source):
    missing_original_records()
    if sha256(SOURCE / 'protocol.json') != SOURCE_SHA or sha256(SOURCE / 'prefix-execution.json') != source['original_failed_terminal_sha256']:
        raise ValueError('original failure/protocol changed')
    p = read(SOURCE / 'protocol.json')
    required = {str(f.relative_to(PROJECT)) for f in source_paths(p)}
    if not required or set(source['bound_files']) != required or source['copy_files'] != lineage_manifest():
        raise ValueError('missing or changed recovery source/lineage bindings')
    for name, digest in source['bound_files'].items():
        if sha256(snapshots.safe_file(PROJECT, name)) != digest:
            raise ValueError('bound source changed: ' + name)
    for name, row in source['copy_files'].items():
        if row['source'] != str((SOURCE / name).relative_to(PROJECT)) or source['bound_files'].get(row['source']) != row['sha256']:
            raise ValueError('copy origin escaped frozen source')
    p = read(SOURCE / 'protocol.json')
    verify_historical_code(p)
    if base.base.previous.source_identity() != p['source_identity']:
        raise ValueError('original native inputs changed')
    return p


def verify_cpu_preflight(path):
    path = snapshots.safe_file(RUNS, Path(path).absolute().relative_to(RUNS))
    proof = read(path)
    if (proof['profile'] != PROFILE or proof['status'] != 'passed'
            or any(proof['code_sha256'].get(n) != sha256(PROJECT / n) for n in CPU_DEPENDENCIES)
            or proof['flags'] != dict.fromkeys(CPU_FLAGS, True)
            or any(proof[k] != 0 for k in ('new_training_updates', 'forward_calls', 'backward_calls', 'optimizer_steps'))
            or proof['cuda_initialized'] or not proof['evidence_sha256']):
        raise ValueError('new zero-training CPU preflight required')
    folder = snapshots.safe_file(RUNS, Path(proof['evidence_root']).absolute().relative_to(RUNS))
    required = {'runner-success/stage.exit.json', 'runner-failure/stage.exit.json', 'runner-rejected/stage.exit.json',
                'checkpoint-proof.json', 'formula-proof.json'}
    if not required <= proof['evidence_sha256'].keys():
        raise ValueError('CPU preflight evidence incomplete')
    for name, digest in proof['evidence_sha256'].items():
        if sha256(snapshots.safe_file(folder, name)) != digest:
            raise ValueError('CPU preflight evidence changed')
    for case, code in (('success', 0), ('failure', 7), ('rejected', 0)):
        row = read(folder / ('runner-' + case) / 'stage.exit.json')
        if row['returncode'] != code or set(row['resources']) != {'per_rank_peak_reserved_bytes'}:
            raise ValueError('real runner exit record differs')
        if (row['resource_failure'] is not None) != (case == 'rejected'):
            raise ValueError('runner failure classification differs')
    checkpoint = read(folder / 'checkpoint-proof.json')
    if (not checkpoint['lineage_copy_verified'] or not checkpoint['source_unchanged'] or checkpoint['cuda_initialized']
            or set(checkpoint['ranks']) != {'0', '1'} or not all(x['matches'] for x in checkpoint['ranks'].values())):
        raise ValueError('CPU checkpoint proof incomplete')
    if read(folder / 'formula-proof.json') != {'native_formula_exact': True, 'json_exact': True, 'cuda_initialized': False}:
        raise ValueError('CPU formula proof differs')
    return {'path': str(path), 'sha256': sha256(path), 'kind': 'zero_training_CPU_interface_validation'}


def verify_inputs(p):
    r = contract(p)
    base.execution_commit()  # Require committed execution files, not an unchanged repository HEAD.
    if (p['code_sha256'] != {n: sha256(PROJECT / n) for n in base.DEPENDENCIES}
            or verify_cpu_preflight(r['cpu_preflight']['path']) != r['cpu_preflight']):
        raise ValueError('recovery code or CPU preflight changed')
    old = verify_source(r['source'])
    if p['source_identity'] != old['source_identity'] or p['historical_gpu_preflight'] != r['source']['historical_gpu_preflight']:
        raise ValueError('recovery native provenance differs')


def reserve(root, extra=0):
    if extra < 0 or base.base.previous.output_bytes(root) + extra + 1024**2 > CONTRACT['output_limit_bytes']:
        raise ValueError('recovery output budget')
    if shutil.disk_usage(root).free < extra + CONTRACT['disk_reserve_bytes']:
        raise ValueError('recovery disk reserve')


def prepare(root, preflight):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('recovery prepare requires CPU-only execution')
    source = source_identity()
    if torch.cuda.is_initialized():
        raise ValueError('source validation unexpectedly initialized CUDA')
    size = sum(x['bytes'] for x in source['copy_files'].values())
    if size + 1024**3 > CONTRACT['output_limit_bytes'] or shutil.disk_usage(PROJECT).free < CONTRACT['disk_admission_bytes']:
        raise ValueError('checkpoint copy exceeds recovery admission')
    old = read(SOURCE / 'protocol.json')
    root.mkdir(exist_ok=False)
    write_new(root / 'protocol.json', {**base.CONTRACT, 'implementation_commit': base.execution_commit(),
        'code_sha256': {n: sha256(PROJECT / n) for n in base.DEPENDENCIES}, 'source_identity': old['source_identity'],
        'historical_gpu_preflight': source['historical_gpu_preflight'],
        'prefix_recovery': {**CONTRACT, 'source': source, 'cpu_preflight': preflight},
        'status': 'prepared_zero_training_recovery', 'new_prefix_updates': 0})


def copy_lineage(root, copies):
    # Metadata last in each checkpoint; downstream admission also requires recovery-complete.json.
    for relative, row in sorted(copies.items(), key=lambda item: (Path(item[0]).name == 'checkpoint.json', item[0])):
        source = snapshots.safe_file(PROJECT, row['source'])
        target = snapshots.safe_file(root, relative)
        if sha256(source) != row['sha256'] or source.stat().st_size != row['bytes']:
            raise ValueError('checkpoint source changed before copy')
        reserve(root, row['bytes'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 4 * 1024**2)
            writer.flush()
            os.fsync(writer.fileno())
        if sha256(target) != row['sha256'] or target.stat().st_size != row['bytes']:
            raise ValueError('copied checkpoint differs')
        target.chmod(0o444)
    for folder in sorted((root / 'training').rglob('*'), reverse=True):
        if folder.is_dir():
            folder.chmod(0o555)
    # Keep attempts/ writable for future separately authorized A/B attempts.
    (root / 'training/attempts').chmod(0o755)
    (root / 'training').chmod(0o755)


def verify_copies(root, source):
    for name, row in source['copy_files'].items():
        f = snapshots.safe_file(root, name)
        if sha256(f) != row['sha256'] or f.stat().st_size != row['bytes']:
            raise ValueError('copied lineage changed')


def verify_binding(root, p):
    r = contract(p)
    verify_inputs(p)
    value = read(snapshots.safe_file(root, 'recovery-complete.json'))
    if (value['status'] != 'prefix_payload_rebound' or value['protocol_sha256'] != sha256(root / 'protocol.json')
            or value['original_suite_status'] != 'failed' or value['original_child_exit_code'] is not None
            or any(value[k] != 0 for k in ('new_training_updates', 'forward_calls', 'backward_calls', 'optimizer_steps', 'strategy_calls'))
            or value['cuda_initialized'] or value['original_exit_record_fabricated'] or value['arms_authorized']
            or value['checkpoint_hash'] != CHECKPOINT_HASH
            or value['full_state_sha256'] != STATE_HASHES or value['source_binding_sha256'] != hashlib.sha256(base.canonical_report(r['source']).encode()).hexdigest()):
        raise ValueError('invalid recovered prefix proof')
    if value['gate0_sha256'] != sha256(root / 'gate0.json'):
        raise ValueError('recovered Gate 0 changed')
    verify_copies(root, r['source'])
    for name in ('prefix/complete.json', 'prefix/case.exit.json', 'prefix-execution.json'):
        if (root / name).exists():
            raise ValueError('recovery must not fabricate original completion records')
    terminal = read(root / 'recovery-execution.json')
    if (terminal['status'] != 'completed' or terminal['exit_code'] != 0 or terminal['phase'] != 'recovery'
            or terminal['protocol_sha256'] != sha256(root / 'protocol.json')
            or terminal['report_sha256'] != sha256(root / 'recovery-complete.json')):
        raise ValueError('recovery itself did not complete')
    return value


def recover(root, p):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('recovery requires CUDA hidden and uninitialized')
    base.base.legacy.unlimited_host_cgroup(UNIT)
    lock = (root / 'recovery.lock').open('a')
    with lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / 'recovery.once').open('x'):
            pass
        terminal = {'status': 'failed', 'exit_code': 1, 'phase': 'recovery', 'protocol_sha256': sha256(root / 'protocol.json')}
        base.base.connected.publish_status(root / 'recovery-started.json', {'pid': os.getpid(),
            'proc_start': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], 'protocol_sha256': terminal['protocol_sha256']})
        def expired(signum, frame):
            raise TimeoutError('zero-training recovery deadline exceeded')
        old = signal.signal(signal.SIGALRM, expired)
        signal.alarm(CONTRACT['timeout_seconds'])
        try:
            verify_inputs(p)
            r = contract(p)
            if shutil.disk_usage(root).free < CONTRACT['disk_admission_bytes']:
                raise ValueError('recovery disk admission')
            copy_lineage(root, r['source']['copy_files'])
            copied = base.checkpoint(root, p)
            if copied['checkpoint_hash'] != CHECKPOINT_HASH:
                raise ValueError('rebound native checkpoint changed')
            base.base.connected.parent_metadata(p['source_identity']['native'], root / 'training')
            actual = {str(rank): checkpoint_state_digest(root / 'training/attempts/prefix/checkpoints/iteration_000012100', rank) for rank in (0, 1)}
            if actual != STATE_HASHES or torch.cuda.is_initialized():
                raise ValueError('rebound CPU state differs or initialized CUDA')
            gate = {'status': 'structural_verified_review_required', 'owners': r['source']['gate0_owners'],
                'effective_intervention_available': any(x['effective_intervention_available'] for x in r['source']['gate0_owners'].values()),
                'checkpoint_hash': CHECKPOINT_HASH, 'source_protocol_sha256': sha256(root / 'protocol.json'),
                'arms_started': False, 'moment_magnitude_automatically_admitted': False,
                'origin': 'readonly_reconstruction_from_pinned_prefix_payload', 'original_suite_status': 'failed',
                'original_child_exit_code': None}
            write_new(root / 'gate0.json', gate)
            verify_source(r['source'])
            verify_copies(root, r['source'])
            report = {'status': 'prefix_payload_rebound', 'protocol_sha256': sha256(root / 'protocol.json'),
                'source_binding_sha256': hashlib.sha256(base.canonical_report(r['source']).encode()).hexdigest(),
                'full_state_sha256': actual, 'checkpoint_hash': CHECKPOINT_HASH, 'gate0_sha256': sha256(root / 'gate0.json'),
                'original_suite_status': 'failed', 'original_child_exit_code': None, 'original_exit_record_fabricated': False,
                'new_training_updates': 0, 'forward_calls': 0, 'backward_calls': 0, 'optimizer_steps': 0,
                'strategy_calls': 0, 'cuda_initialized': False, 'arms_authorized': False}
            reserve(root, len(json.dumps(report).encode()))
            write_new(root / 'recovery-complete.json', report)
            terminal.update(status='completed', exit_code=0, report='recovery-complete.json', report_sha256=sha256(root / 'recovery-complete.json'))
        except BaseException as exc:
            terminal['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            base.base.connected.publish_status(root / 'recovery-execution.json', {**terminal, 'ended_unix': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'recover', 'monitor'):
        group.add_argument('--' + name, action='store_true')
    parser.add_argument('--authorize-recovery', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if socket.gethostname() != 'i-94B8D131' or str(PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        raise SystemExit('仅指定远端执行独立CPU恢复')
    if not args.monitor and not args.authorize_recovery:
        raise SystemExit('需要独立零训练恢复授权')
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise ValueError('CUDA was already initialized')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = output_root(args.output_root, args.prepare)
    if args.prepare:
        if args.preflight_report is None:
            parser.error('需要新版零训练CPU前置报告')
        prepare(root, verify_cpu_preflight(args.preflight_report))
        return
    p = base.protocol_at(root, args.protocol_sha256)
    contract(p)
    if args.monitor:
        raise SystemExit(base.monitor(root, args.protocol_sha256, 'recovery'))
    recover(root, p)


if __name__ == '__main__':
    main()
