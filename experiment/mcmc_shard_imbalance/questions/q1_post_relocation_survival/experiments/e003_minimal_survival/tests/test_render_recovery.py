"""NoTorch checks for initialization boundaries and the zero-training recovery path."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e003_minimal_survival import render_probe, render_recovery as recovery


def test_fresh_import_and_no_training_contract():
    subprocess.run([sys.executable, '-B', '-c', f"import importlib,sys; importlib.import_module({recovery.MODULE!r}); assert not ({{'torch','numpy','gsplat'}} & sys.modules.keys())"], check=True, cwd=recovery.PROJECT)
    result = subprocess.run([sys.executable, '-B', '-m', recovery.MODULE, '--help'], capture_output=True, text=True, check=True, cwd=recovery.PROJECT)
    assert '--authorize-render-only' in result.stdout and '--authorize-execution' not in result.stdout
    for name in ('new_training_updates', 'loss_backwards', 'optimizer_steps', 'strategy_calls'):
        assert recovery.CONTRACT[name] == 0
    assert recovery.CONTRACT['camera_forwards'] == 64 and recovery.CONTRACT['preflight_camera_forwards'] == 8
    tree = ast.parse(Path(recovery.__file__).read_text())
    assert not ({'train_gaussians', 'load_frozen_initialization', 'copytree', 'copyfile', 'optimizers', 'backward'} &
                {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


def test_initialization_is_outside_guard_and_has_no_restore(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(render_probe, 'initialize_backend', lambda: calls.append('initialize') or {'warmup_forward_calls': 0, 'warmup_backward_calls': 0})
    value = render_probe.initialize_render(tmp_path, 0)
    assert calls == ['initialize'] and value['initialization_outside_measurement'] and not value['rng_restored']
    assert recovery.read(tmp_path / 'initialization-rank-0.json') == value
    tree = ast.parse(Path(render_probe.__file__).read_text())
    probe = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'probe')
    init_line = next(n.lineno for n in ast.walk(probe) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'initialize_render')
    guard_lines = [n.lineno for n in ast.walk(probe) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'unchanged_rng']
    assert init_line < min(guard_lines)
    assert not ({'setstate', 'set_rng_state', 'set_rng_state_all', 'manual_seed'} &
                {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


def test_phase_guard_records_python_consumption_without_hiding_it(tmp_path, monkeypatch):
    def state():
        return {'python': hashlib.sha256(repr(random.getstate()).encode()).hexdigest(), 'numpy': 'n', 'torch': 't', 'cuda': 'c'}
    monkeypatch.setattr(render_probe, 'rng_components', state)
    original = random.getstate()
    try:
        with pytest.raises(ValueError, match='load: observer consumed RNG'):
            with render_probe.audited_rng(tmp_path, 0, 'load'):
                random.random()
        assert random.getstate() != original
        record = recovery.read(tmp_path / 'rng-load-rank-0.json')
        assert record['changed_rng_components'] == ['python'] and not record['rng_restored']
        with render_probe.audited_rng(tmp_path, 0, 'render'):
            pass
        assert recovery.read(tmp_path / 'rng-render-rank-0.json')['changed_rng_components'] == []
        with pytest.raises(RuntimeError, match='original failure'):
            with render_probe.audited_rng(tmp_path, 0, 'loss'):
                raise RuntimeError('original failure')
        assert 'original failure' in recovery.read(tmp_path / 'rng-loss-rank-0.json')['error']
    finally:
        random.setstate(original)


def test_historical_code_uses_git_objects_not_current_worktree(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(recovery, 'PROJECT', tmp_path)
    monkeypatch.setattr(recovery, 'SOURCE', source)
    monkeypatch.setattr(recovery.base, 'DEPENDENCIES', ('old.py',))
    (tmp_path / 'old.py').write_text('repaired current renderer')
    old_hash = hashlib.sha256(b'old code').hexdigest()
    package, replay = tmp_path / 'package', tmp_path / 'replay'
    package.mkdir()
    replay.mkdir()
    (replay / 'dataset.json').write_text('{}')
    p = {'implementation_commit': recovery.SOURCE_COMMIT, 'code_sha256': {'old.py': old_hash},
         'camera_contract': {'image_ids': ['1', '2'], 'images': []},
         'source_identity': {'native': {'pair': {'source_package': str(package), 'source_hashes': {}, 'replay': str(replay)},
                                       'replay_record': {'dataset_sha256': recovery.sha256(replay / 'dataset.json')}}}}
    (source / 'protocol.json').write_text(json.dumps(p))
    monkeypatch.setattr(recovery, 'SOURCE_SHA', recovery.sha256(source / 'protocol.json'))
    (source / 'execution.json').write_text(json.dumps({'status': 'failed', 'exit_code': 1, 'protocol_sha256': recovery.SOURCE_SHA}))
    completions = {}
    for n in ('T0', 'T1'):
        (source / n).mkdir()
        (source / n / 'complete.json').write_text(json.dumps({'case': n}))
        completions[n] = recovery.sha256(source / n / 'complete.json')
    monkeypatch.setattr(recovery, 'SOURCE_COMPLETE', completions)
    commands = []
    monkeypatch.setattr(recovery.subprocess, 'check_output', lambda cmd, **kw: commands.append(cmd) or b'old code')
    monkeypatch.setattr(recovery.base, 'verify_complete', lambda *args: None)
    assert recovery.source_identity()['complete_sha256'] == completions
    assert commands == [['git', 'show', f'{recovery.SOURCE_COMMIT}:old.py']]
    (source / 'T0/complete.json').write_text('changed')
    with pytest.raises(ValueError, match='completion changed'):
        recovery.source_identity()


def proof_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'PROJECT', tmp_path)
    monkeypatch.setattr(recovery, 'RUNS', tmp_path)
    monkeypatch.setattr(recovery, 'DEPENDENCIES', ('code.py',))
    monkeypatch.setattr(recovery, 'execution_commit', lambda: 'repaired-commit')
    runtime = {'0': {'rank': 0}, '1': {'rank': 1}}
    monkeypatch.setattr(recovery.base, 'expected_runtime', lambda: runtime)
    (tmp_path / 'code.py').write_text('fixed')
    workers = {}
    for rank in (0, 1):
        components = dict.fromkeys(('python', 'numpy', 'torch', 'cuda'), 'same')
        init = {'rank': rank, 'rng_before': components, 'rng_after': components, 'changed_rng_components': [],
                'warmup_forward_calls': 0, 'warmup_backward_calls': 0, 'backend_loaded_before': False,
                'initialization_outside_measurement': True, 'rng_restored': False}
        (tmp_path / f'initialization-rank-{rank}.json').write_text(json.dumps(init))
        hashes = {}
        for stage in ('before', 'after'):
            for repeat in (0, 1):
                for part in ('load', 'render', 'loss', 'export'):
                    phase = f'{stage}-{repeat}-0-{part}'
                    file = tmp_path / f'rng-{phase}-rank-{rank}.json'
                    file.write_text(json.dumps({'rank': rank, 'phase': phase, 'before': components, 'after': components,
                                                'changed_rng_components': [], 'error': None, 'rng_restored': False}))
                    hashes[file.name] = recovery.sha256(file)
        negative = tmp_path / f'negative-rank-{rank}.json'
        negative.write_text(json.dumps({'changed_rng_components': ['python'], 'rng_restored': False,
                                       'before': components, 'after': {**components, 'python': 'changed'}}))
        result = {'rank': rank, 'status': 'measured', 'camera_forwards': 4, 'backwards': 0, 'optimizer_steps': 0,
                  'initialization': init, 'cold_before_first_render': True, 'training_calls': 0,
                  'negative_rng_guard_passed': True, 'model_unchanged': True, 'interpretation_usable': True,
                  'image_ids': [str(rank)], 'rng_audit_sha256': hashes,
                  'synthetic_image_decodes': 1, 'atol': recovery.snapshots.ATOL, 'rtol': recovery.snapshots.RTOL,
                  'repeat_checks': {f'{phase}/{rank}': {'usable': True, 'loss': {'mismatch': False},
                      'pixels': {'rgb': {'status': 'bitwise_equal'}, 'alpha': {'status': 'bitwise_equal'}}} for phase in ('before', 'after')},
                  'negative_rng_audit': {'path': negative.name, 'sha256': recovery.sha256(negative)}}
        file = tmp_path / f'preflight-rank-{rank}.json'
        file.write_text(json.dumps(result))
        workers[str(rank)] = {'path': str(file), 'sha256': recovery.sha256(file)}
    proof = {'status': 'passed', 'profile': recovery.CONTRACT['profile'], 'implementation_commit': 'repaired-commit',
             'runtime': runtime, 'camera_forwards': 8, 'new_training_updates': 0, 'real_image_decodes': 0,
             'loss_backwards': 0, 'optimizer_steps': 0, 'workers': workers,
             'code_sha256': {'code.py': recovery.sha256(tmp_path / 'code.py')}}
    file = tmp_path / 'preflight.json'
    file.write_text(json.dumps(proof))
    return file, proof


def test_cold_preflight_rejects_warmed_or_stale_evidence(tmp_path, monkeypatch):
    file, proof = proof_fixture(tmp_path, monkeypatch)
    assert recovery.verify_preflight(file)['sha256'] == recovery.sha256(file)
    for change in ({'profile': recovery.base.PROFILE}, {'implementation_commit': 'old'}, {'new_training_updates': 8},
                   {'workers': {'0': proof['workers']['0']}}):
        file.write_text(json.dumps({**proof, **change}))
        with pytest.raises(ValueError):
            recovery.verify_preflight(file)
    worker = Path(proof['workers']['0']['path'])
    value = recovery.read(worker)
    value['initialization']['backend_loaded_before'] = True
    worker.write_text(json.dumps(value))
    proof['workers']['0']['sha256'] = recovery.sha256(worker)
    file.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='cold process'):
        recovery.verify_preflight(file)


def test_audit_rejects_missing_stage_or_changed_file(tmp_path, monkeypatch):
    file, proof = proof_fixture(tmp_path, monkeypatch)
    worker = Path(proof['workers']['0']['path'])
    value = recovery.read(worker)
    damaged = copy.deepcopy(value)
    damaged['rng_audit_sha256'].pop(next(iter(damaged['rng_audit_sha256'])))
    with pytest.raises(ValueError, match='coverage'):
        recovery.verify_audit(tmp_path, 0, damaged)
    path = tmp_path / next(iter(value['rng_audit_sha256']))
    path.write_text('{}')
    with pytest.raises(ValueError, match='audit changed'):
        recovery.verify_audit(tmp_path, 0, value)


def test_output_and_protocol_cannot_turn_recovery_into_training(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'RUNS', tmp_path)
    monkeypatch.setattr(recovery, 'PROJECT', tmp_path)
    monkeypatch.setattr(recovery, 'DEPENDENCIES', ('code.py',))
    assert recovery.output_root(tmp_path / 'render-recovery-new', True) == tmp_path / 'render-recovery-new'
    with pytest.raises(ValueError):
        recovery.output_root(tmp_path / 'survival-20261027-v1', True)
    (tmp_path / 'code.py').write_text('fixed')
    p = {**copy.deepcopy(recovery.CONTRACT), 'code_sha256': {'code.py': recovery.sha256(tmp_path / 'code.py')}}
    file = tmp_path / 'protocol.json'
    file.write_text(json.dumps(p))
    assert recovery.protocol_at(tmp_path, recovery.sha256(file))['new_training_updates'] == 0
    for change in ({'new_training_updates': 400}, {'camera_forwards': 128}, {'source_protocol_sha256': 'other'}):
        file.write_text(json.dumps({**p, **change}))
        with pytest.raises(ValueError):
            recovery.protocol_at(tmp_path, recovery.sha256(file))


@pytest.mark.parametrize('failed', ['T0', 'T1', None])
def test_recovery_only_renders_and_never_retries(failed, tmp_path, monkeypatch):
    from image3d_scenegraph import gpu_lease
    class Lease:
        def __init__(self, *a):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def fileno(self):
            return 99
    monkeypatch.setattr(gpu_lease, 'FileLease', Lease)
    monkeypatch.setattr(gpu_lease, 'require_idle_gpu', lambda: None)
    monkeypatch.setattr(recovery.base.legacy, 'unlimited_host_cgroup', lambda *a: None)
    monkeypatch.setattr(recovery, 'verify_inputs', lambda p: None)
    monkeypatch.setattr(recovery, 'verify_audit', lambda *a: None)
    monkeypatch.setattr(recovery, 'reserve', lambda *a: None)
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda p: SimpleNamespace(free=8*1024**3))
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: None)
    root = tmp_path / 'render-recovery-new'
    root.mkdir()
    seen = []
    def execute(stage, label, command, **kw):
        assert command[command.index('-m')+1] == recovery.MODULE and '--authorize-render-only' in command
        name = command[command.index('--case')+1]
        seen.append(name)
        if name == failed:
            raise RuntimeError('synthetic rendering failure')
        for rank in (0, 1):
            (stage / f'worker-rank-{rank}.json').write_text(json.dumps({'source_complete_sha256': recovery.SOURCE_COMPLETE[name]}))
    monkeypatch.setattr(recovery.base.legacy, 'run_stage', execute)
    def complete(root, stage, name, p, digest, monitor, rendering):
        assert rendering
        (stage / 'complete.json').write_text('{}')
        return {'render_fixture': True}
    monkeypatch.setattr(recovery.base, 'bind_complete', complete)
    monkeypatch.setattr(recovery.base, 'verify_complete', lambda *a: None)
    args = SimpleNamespace(output_root=root, protocol_sha256='sha')
    if failed:
        with pytest.raises(RuntimeError, match='synthetic'):
            recovery.run(args, {'source': {}})
        assert seen == (['T0'] if failed == 'T0' else ['T0', 'T1'])
        assert not (root / 'comparison.json').exists()
    else:
        recovery.run(args, {'source': {}})
        assert seen == ['T0', 'T1']
        assert recovery.read(root / 'comparison.json')['new_training_updates'] == 0
    assert not (root / 'training').exists()
    with pytest.raises(FileExistsError):
        recovery.run(args, {'source': {}})


def test_recovery_resource_bounds_and_explicit_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda _: SimpleNamespace(free=8*1024**3))
    recovery.reserve(tmp_path, 100)
    for extra in (-1, 1024**3):
        with pytest.raises(ValueError, match='output budget'):
            recovery.reserve(tmp_path, extra)
    monkeypatch.setattr(sys, 'argv', ['recovery', '--run', '--output-root', str(tmp_path / 'render-recovery-new')])
    monkeypatch.setattr(recovery.socket, 'gethostname', lambda: 'laptop')
    with pytest.raises(SystemExit, match='指定远端'):
        recovery.main()
    monkeypatch.setattr(recovery.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(recovery, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    with pytest.raises(SystemExit, match='独立只读补渲染授权'):
        recovery.main()
    assert not (tmp_path / 'render-recovery-new').exists()
