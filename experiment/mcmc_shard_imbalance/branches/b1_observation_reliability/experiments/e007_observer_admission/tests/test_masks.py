"""Local standard-library checks; importing these tests never loads tensor libraries."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission import masks, run


def test_phase_a_only_and_frozen_budget():
    assert run.CONTRACT['phase'] == 'A_only'
    assert run.CONTRACT['cases'] == [x[0] for x in run.connected.CASES]
    assert run.CONTRACT['cuda_sigmoid_calls'] == run.CONTRACT['cpu_sigmoid_calls'] == 10 * 2 * 2
    assert all(run.CONTRACT[k] == 0 for k in ('new_training_updates', 'rasterizer_forwards', 'backwards', 'strategy_calls'))
    assert run.CONTRACT['suite_timeout_seconds'] == 3600
    assert run.CONTRACT['max_output_bytes'] == 1024**3 and run.CONTRACT['host_limits'] is None
    assert [len(x) for x in masks.PAIRS.values()] == [3, 3, 4]
    assert len(masks.MASK_KEYS) * (10 * 2 * 2 + 10 * 2) == 360


def test_fresh_import_and_help_are_notorch():
    code = f'''import contextlib,importlib,io,sys
r=importlib.import_module({run.MODULE!r})
sys.argv=['phase-a','--help']
with contextlib.redirect_stdout(io.StringIO()) as out:
 try:r.main()
 except SystemExit as exc:assert exc.code==0
assert '--monitor' in out.getvalue()
assert '--case' not in out.getvalue()
assert not ({{'torch','numpy','gsplat'}} & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=run.PROJECT, check=True)


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker', 'monitor'])
def test_wrong_host_rejected_without_writes(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'local')
    monkeypatch.setattr(sys, 'argv', ['phase-a', '--' + mode, '--output-root', str(tmp_path / 'out')])
    with pytest.raises(SystemExit, match='仅指定远端'):
        run.main()
    assert not (tmp_path / 'out').exists() and 'torch' not in sys.modules


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker'])
def test_authorization_before_source_or_writes(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(run, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    monkeypatch.setattr(sys, 'argv', ['phase-a', '--' + mode, '--output-root', str(tmp_path / 'out')])
    with pytest.raises(SystemExit, match='未获显式阶段A授权'):
        run.main()
    assert not (tmp_path / 'out').exists()


def test_output_rejects_sources_existing_and_symlink(tmp_path, monkeypatch):
    root = tmp_path / 'runs'
    root.mkdir()
    monkeypatch.setattr(run, 'RUNS', root)
    assert run.output_root(root / 'new', True) == root / 'new'
    (root / 'old').mkdir()
    (root / 'link').symlink_to(tmp_path, target_is_directory=True)
    for name in (root, root / 'old', root / '../escape', tmp_path / 'outside', root / 'link/new'):
        with pytest.raises(ValueError):
            run.output_root(name, True)


def test_protocol_binds_phase_source_budget_and_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PROJECT', tmp_path)
    monkeypatch.setattr(run, 'DEPENDENCIES', ('one.py',))
    (tmp_path / 'one.py').write_text('frozen')
    original = {**copy.deepcopy(run.CONTRACT), 'code_sha256': {'one.py': run.sha256(tmp_path / 'one.py')}}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(original))
    assert run.protocol_at(tmp_path, run.sha256(path))['phase'] == 'A_only'
    with pytest.raises(ValueError, match='explicit protocol'):
        run.protocol_at(tmp_path, 'bad')
    for update in ({'phase': 'B'}, {'cases': []}, {'max_output_bytes': 2**50}, {'code_sha256': {}}):
        path.write_text(json.dumps({**original, **update}))
        with pytest.raises(ValueError):
            run.protocol_at(tmp_path, run.sha256(path))
    path.write_text(json.dumps(original))
    (tmp_path / 'one.py').write_text('changed')
    with pytest.raises(ValueError, match='dependency changed'):
        run.protocol_at(tmp_path, run.sha256(path))


def test_disk_and_output_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10 * 1024**3))
    run.reserve(tmp_path, 100)
    with pytest.raises(ValueError, match='budget'):
        run.reserve(tmp_path, run.CONTRACT['max_output_bytes'])
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=4 * 1024**3))
    with pytest.raises(ValueError, match='reserve'):
        run.reserve(tmp_path, 1)


def test_monitor_reports_timeout_and_incomplete_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(run.time, 'monotonic', lambda: 10)
    assert run.AnalysisMonitor(tmp_path, 9)(0) == 'phase_A_timeout'
    monitor = run.AnalysisMonitor(tmp_path, 20)
    assert monitor(0, final=True) == 'missing_memory_telemetry'
    (tmp_path / 'memory').mkdir()
    for rank in (0, 1):
        (tmp_path / 'memory' / f'rank-{rank}.json').write_text(json.dumps({'peak_reserved_bytes': 1, 'finished': True}))
    assert monitor(0, final=True) is None
    (tmp_path / 'memory/rank-1.json').write_text(json.dumps({'peak_reserved_bytes': 19 * 1024**3, 'finished': True}))
    assert monitor(0, final=True) == 'gpu_reserved_limit'


def row(changed=0):
    return {k: {'xor_count': changed if k == 'dead_after' else 0} for k in masks.MASK_KEYS}


def test_direction_does_not_confuse_counts_masks_or_causality():
    pairs = {'owner-0/cpu_float64': {'same': {'a': row()}, 'varied': {'a': row(2)}, 'native': {'a': row()}}}
    result = masks.direction(pairs, [row()])
    assert result['varied_death_mask_difference'] and not result['method_death_mask_disagreement']
    assert 'not_executed_strategy' in result['scope']
    pairs['owner-0/cpu_float64']['same']['a'] = row(1)
    with pytest.raises(ValueError, match='same-input'):
        masks.direction(pairs, [row()])


def test_bitset_roundtrip_rejects_padding_and_counts(tmp_path):
    path = tmp_path / 'slots.bits.gz'
    path.write_bytes(gzip.compress(bytes([0b10000001, 0b00000001]), mtime=0))
    identity = {'path': path.name, 'bytes': path.stat().st_size, 'sha256': run.sha256(path),
                'slot_count': 9, 'xor_count': 3, 'encoding': 'gzip_numpy_packbits_little_zero_based_slots'}
    masks.verify_bitset(tmp_path, identity)
    for change in ({'slot_count': 8}, {'xor_count': 2}, {'path': '../escape'}, {'slot_count': 0}, {'sha256': 'wrong'}):
        with pytest.raises(ValueError):
            masks.verify_bitset(tmp_path, {**identity, **change})


def test_source_wrong_sha_rejected_before_arrays(tmp_path, monkeypatch):
    imported_before = set(sys.modules)
    monkeypatch.setattr(run, 'SOURCE', tmp_path)
    (tmp_path / 'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='source identity'):
        run.source_inventory()
    assert not ({'numpy', 'torch'} & (set(sys.modules) - imported_before))


def test_worker_requires_inherited_lease_before_torch(tmp_path):
    with pytest.raises(ValueError, match='inherited GPU lease'):
        run.worker(SimpleNamespace(lease_fd=None, output_root=tmp_path), {})
    assert 'torch' not in sys.modules


def test_atomic_terminal_rejects_overwrite(tmp_path):
    path = tmp_path / 'execution.json'
    run.connected.publish_status(path, {'status': 'failed'})
    with pytest.raises(FileExistsError):
        run.connected.publish_status(path, {'status': 'completed'})
    assert json.loads(path.read_text())['status'] == 'failed'


def test_readonly_monitor_returns_failure_without_gpu_lock(tmp_path):
    (tmp_path / 'execution.json').write_text(json.dumps({'status': 'failed', 'exit_code': 1, 'protocol_sha256': 'abc'}))
    assert run.monitor(tmp_path, 'abc') == 1
    assert set(p.name for p in tmp_path.iterdir()) == {'execution.json'}


def test_source_chain_and_selected_opacity_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'SOURCE', tmp_path)
    monkeypatch.setattr(run.connected, 'CASES', (('same-0', 0),))
    protocol = {**copy.deepcopy(run.connected.CONTRACT), 'implementation_commit': 'frozen'}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    source_sha = run.sha256(tmp_path / 'protocol.json')
    monkeypatch.setattr(run, 'SOURCE_SHA', source_sha)
    stage = tmp_path / 'same-0'
    stage.mkdir()
    exit_path = stage / 'case.exit.json'
    exit_path.write_text(json.dumps({'returncode': 0, 'resource_failure': None}))
    complete_files = {'case.exit.json': run.sha256(exit_path)}
    leaves = {}
    for rank in (0, 1):
        manifests = {}
        for phase in ('P0', 'P3'):
            folder = stage / f'rank-{rank}' / phase
            folder.mkdir(parents=True)
            file = folder / 'opacity.npy'
            file.write_bytes(b'opaque-file-identity-only')
            tree = {'rank': rank, 'phase': phase}
            record = {'tree': tree, 'tree_sha256': run.snapshots.digest_state(tree), 'files': {}}
            path = folder / 'manifest.json'
            path.write_text(json.dumps(record))
            manifests[phase] = run.sha256(path)
            leaves[(rank, phase)] = {('parameters', 'opacities'): {
                'dtype': 'float32', 'shape': [1500000], 'file': file.name,
                'bytes': file.stat().st_size, 'sha256': run.sha256(file)}}
            for key, value in {'torch': 'frozen', 'cuda': 'frozen', 'gpu': 'frozen', 'rank': rank}.items():
                leaves[(rank, phase)][('runtime', key)] = {'value': value}
        worker = {'rank': rank, 'iteration': 12001, 'new_updates': 1, 'calls': {'strategy': 0},
                  'validation_evaluated': False, 'original_backward_calls': 0, 'manifests': manifests}
        path = stage / f'worker-rank-{rank}.json'
        path.write_text(json.dumps(worker))
        complete_files[path.name] = run.sha256(path)
    # Array schema/numerics have separate remote tests; this fixture checks the signed chain.
    monkeypatch.setattr(run.snapshots, 'validate_stage_schema', lambda rec, phase: leaves[(rec['tree']['rank'], phase)])
    complete = {'case': 'same-0', 'status': 'probe_complete', 'protocol_sha256': source_sha, 'files': complete_files}
    (stage / 'complete.json').write_text(json.dumps(complete))
    comparison = {'protocol_sha256': source_sha, 'same_input_gate_passed': True,
                  'status': 'analysis_complete', 'implementation_commit': 'frozen',
                  'complete_sha256': {'same-0': run.sha256(stage / 'complete.json')},
                  'point_and_opacity_summary': {'same-0': {str(r): dict.fromkeys(masks.DEATH_KEYS, 0) for r in (0, 1)}}}
    (tmp_path / 'comparison.json').write_text(json.dumps(comparison))
    monkeypatch.setattr(run, 'COMPARISON_SHA', run.sha256(tmp_path / 'comparison.json'))
    inventory = run.source_inventory()
    assert set(inventory['cases']['same-0']) == {'0', '1'}
    assert len([p for p in inventory['bound_files'] if p.endswith('opacity.npy')]) == 4
    file = stage / 'rank-1/P3/opacity.npy'
    before = file.read_bytes()
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='opacity input changed'):
        run.source_inventory()
    file.write_bytes(before)
    (stage / 'worker-rank-0.json').write_text('{}')
    with pytest.raises(ValueError, match='source identity changed'):
        run.source_inventory()


def test_verify_result_rejects_incomplete_references(tmp_path):
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps({'status': 'analysis_complete', 'protocol_sha256': 'abc',
        'stage_B_authorized': False, 'long_term_causality_proven': False,
        'pairs': {}, 'method_comparison': {}, 'files': {}}))
    with pytest.raises(ValueError, match='incomplete'):
        run.verify_result(tmp_path, 'abc')
