"""Local standard-library admission checks; tensor execution is remote opt-in only."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission import admission, observer, run_b


def metric_row(value=.01):
    return {'metrics': {k: value for k in admission.metric_names()}, 'camera_sequence': [['train0', 'train1']] * 200}


def test_frozen_suite_and_no_phase_a_training_entry():
    c = run_b.CONTRACT
    assert tuple(c['cases']) == ('C0', 'C1', 'C2', 'U0', 'O0', 'O1', 'U1', 'U2', 'O2')
    assert c['new_updates_max'] == 9 * 200 == 1800
    assert c['training_camera_forwards_max'] == c['loss_backwards_max'] == 3600
    assert c['contribution_forwards_max'] == 3 * 200 * 2 == 1200
    assert c['optimizer_group_steps_max'] == 1800 * 2 * 6
    assert c['max_output_bytes'] == 30 * 1024**3 and c['host_limits'] is None
    assert c['preflight_budget']['synthetic_updates'] == 8
    assert run_b.phase_a.CONTRACT['new_training_updates'] == 0


def test_fresh_import_help_loads_no_tensor_libraries():
    code = f'''import contextlib,importlib,io,sys
r=importlib.import_module({run_b.MODULE!r})
sys.argv=['admission','--help']
with contextlib.redirect_stdout(io.StringIO()):
 try:r.main()
 except SystemExit as exc:assert exc.code==0
assert not ({{'numpy','torch','gsplat'}} & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=run_b.PROJECT, check=True)


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker', 'monitor'])
def test_host_gate_precedes_tensor_import_or_writes(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(run_b.socket, 'gethostname', lambda: 'local')
    monkeypatch.setattr(sys, 'argv', ['b', '--' + mode, '--output-root', str(tmp_path / 'new')])
    with pytest.raises(SystemExit, match='仅指定远端'):
        run_b.main()
    assert not (tmp_path / 'new').exists() and 'torch' not in sys.modules


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker'])
def test_phase_b_needs_its_own_authorization(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(run_b.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(run_b, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    monkeypatch.setattr(sys, 'argv', ['b', '--' + mode, '--output-root', str(tmp_path)])
    with pytest.raises(SystemExit, match='阶段B显式授权'):
        run_b.main()


def test_output_is_separate_from_a_and_existing_data(tmp_path, monkeypatch):
    root = tmp_path / 'runs'
    root.mkdir()
    monkeypatch.setattr(run_b.phase_a, 'RUNS', root)
    assert run_b.output_root(root / 'admission-new', True) == root / 'admission-new'
    for path in (root / 'masks-old', root / '../escape'):
        with pytest.raises(ValueError):
            run_b.output_root(path, True)


def test_calibration_is_independent_and_ceilings_are_not_relaxed():
    c = {n: metric_row() for n in admission.CALIBRATION}
    frozen = admission.calibrate(c)
    assert frozen['status'] == 'calibrated'
    original = copy.deepcopy(frozen)
    assert admission.check_case('U0', metric_row(), frozen)['passed']
    outside = metric_row()
    outside['metrics']['loss/12100'] = .1
    assert not admission.check_case('O0', outside, frozen)['passed']
    assert frozen == original
    for bad in ({'C0': c['C0'], 'C1': c['C1'], 'U0': c['C2']}, dict(reversed(list(c.items())))):
        with pytest.raises(ValueError, match='only C0'):
            admission.calibrate(bad)
    c['C2']['metrics']['owner0/12100/relocation_fraction'] += .002
    assert admission.calibrate(c)['status'] == 'calibration_blocked'
    with pytest.raises(ValueError, match='successful calibration'):
        admission.check_case('U0', metric_row(), admission.calibrate(c))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), None, True])
def test_nonfinite_or_nonnumeric_metric_is_rejected(value):
    c = {n: metric_row() for n in admission.CALIBRATION}
    c['C0']['metrics']['loss/12100'] = value
    with pytest.raises(ValueError, match='metric'):
        admission.calibrate(c)


def test_missing_metric_camera_change_and_stop_classification():
    c = {n: metric_row() for n in admission.CALIBRATION}
    frozen = admission.calibrate(c)
    row = metric_row()
    row['metrics'].pop('loss/12200')
    with pytest.raises(ValueError, match='coverage'):
        admission.check_case('O0', row, frozen)
    row = metric_row()
    row['camera_sequence'] = []
    with pytest.raises(ValueError, match='cameras differ'):
        admission.check_case('U1', row, frozen)
    assert admission.classify_stop('O0', ['U0']) == 'observed_outside_envelope_holdout_coverage_incomplete'
    assert admission.classify_stop('O2', ['U0', 'U1', 'U2']) == 'observer_suspect'
    assert admission.classify_stop('U1', ['U0']) == 'baseline_unstable_or_insufficient'


def progress_and_health():
    rows = []
    for i in range(12001, 12201):
        rows.append({'iteration': i, 'world_size': 2, 'gaussian_count': 3000000,
            'nominal_iterations': 30000, 'requested_stop_iteration': 12200, 'mcmc_noise_applied': True,
            'batch_view_ids': ['0', '1'], 'loss': .2, 'reconstruction_loss': .1,
            'opacity_regularization': .05, 'scale_regularization': .05})
    health = {}
    for r in (0, 1):
        health[str(r)] = []
        for i in (12100, 12200):
            rows[i - 12001].update(mcmc_refinement=True, mcmc_relocated_count=30, mcmc_added_count=0)
            for phase in admission.HEALTH_PHASES:
                h = {'gaussian_count': 1500000, 'min_opacity': .005, 'dead_count': 10 + 10*r,
                     'alive_count': 1500000 - 10 - 10*r, 'dead_fraction': (10 + 10*r) / 1500000}
                h.update(dict.fromkeys(('opacity_nonfinite_count', 'log_scale_nan_count', 'log_scale_positive_inf_count',
                    'log_scale_negative_inf_count', 'activated_scale_nonfinite_count', 'origin_distance_nonfinite_count'), 0))
                health[str(r)].append({'rank': r, 'iteration': i, 'phase': phase, 'health': h})
    return rows, health


def test_native_metrics_keep_owners_separate_and_check_relocation_accounting():
    rows, health = progress_and_health()
    result = admission.extract(rows, health, ['0', '1'])
    assert result['metrics']['owner0/12100/relocation_fraction'] == 10/1500000
    assert result['metrics']['owner1/12100/relocation_fraction'] == 20/1500000
    assert result['metrics']['loss/12200'] == .1
    rows[-1]['mcmc_relocated_count'] = 29
    with pytest.raises(ValueError, match='relocation accounting'):
        admission.extract(rows, health, ['0', '1'])


@pytest.mark.parametrize('change', ['missing', 'extra', 'validation', 'topology', 'noise', 'holdout_image', 'health'])
def test_native_window_rejects_invalid_lifecycle(change):
    rows, health = progress_and_health()
    if change == 'missing':
        rows.pop()
    elif change == 'extra':
        rows.append(rows[-1])
    elif change == 'validation':
        rows[0]['event'] = 'validation'
    elif change == 'topology':
        rows[0]['gaussian_count'] += 1
    elif change == 'noise':
        rows[0]['mcmc_noise_applied'] = False
    elif change == 'holdout_image':
        rows[0]['batch_view_ids'][0] = 'test'
    elif change == 'health':
        health['1'].pop()
    with pytest.raises(ValueError):
        admission.extract(rows, health, ['0', '1'])


def test_unobserved_context_has_no_intermediate_hooks():
    tree = ast.parse(Path(observer.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Endpoint')
    text = ast.unparse(cls)
    for forbidden in ('register_step', '_build_mcmc_strategy', '_next_camera_batch', '_multinomial_sample', 'rasterization'):
        assert forbidden not in text
    assert 'evaluate_views' in text and '_ViewSource' in text
    runner = Path(run_b.__file__).read_text()
    assert "if args.case.startswith('O') else None" in runner


def test_budget_and_failed_monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(run_b.shutil, 'disk_usage', lambda _: SimpleNamespace(free=50 * 1024**3))
    run_b.reserve(tmp_path, 100)
    with pytest.raises(ValueError, match='output budget'):
        run_b.reserve(tmp_path, 16 * 1024**3)
    (tmp_path / 'execution.json').write_text(json.dumps({'status': 'admission_blocked', 'exit_code': 2, 'protocol_sha256': 'x'}))
    assert run_b.monitor(tmp_path, 'x') == 2
    assert not (tmp_path / 'execution.once').exists()


def test_protocol_rejects_phase_a_and_modified_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(run_b, 'PROJECT', tmp_path)
    monkeypatch.setattr(run_b, 'DEPENDENCIES', ('b.py',))
    (tmp_path / 'b.py').write_text('frozen')
    value = {**copy.deepcopy(run_b.CONTRACT), 'code_sha256': {'b.py': run_b.sha256(tmp_path / 'b.py')}}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(value))
    assert run_b.protocol_at(tmp_path, run_b.sha256(path))['phase'] == 'B'
    for change in ({'phase': 'A_only'}, {'cases': ['C0', 'O0']}, {'ratio_noise_ceiling': .1}, {'code_sha256': {}}):
        path.write_text(json.dumps({**value, **change}))
        with pytest.raises(ValueError):
            run_b.protocol_at(tmp_path, run_b.sha256(path))


def test_preflight_requires_bound_passing_reports_from_both_ranks(tmp_path, monkeypatch):
    monkeypatch.setattr(run_b, 'RUNS', tmp_path)
    monkeypatch.setattr(run_b, 'PROJECT', tmp_path)
    monkeypatch.setattr(run_b, 'DEPENDENCIES', ('b.py',))
    monkeypatch.setattr(run_b, 'execution_commit', lambda: 'commit')
    (tmp_path / 'b.py').write_text('frozen')
    a_root = tmp_path / 'a'
    a_root.mkdir()
    runtime = {str(r): {'rank': r, 'torch': 'frozen', 'cuda': 'frozen', 'gpu': 'frozen'} for r in (0, 1)}
    (a_root / 'protocol.json').write_text(json.dumps({'source_inventory': {'runtime': runtime}}))
    monkeypatch.setattr(run_b, 'A_ROOT', a_root)
    monkeypatch.setattr(run_b, 'A_SHA', run_b.sha256(a_root / 'protocol.json'))
    workers = {}
    for rank in (0, 1):
        file = tmp_path / f'worker-{rank}.json'
        file.write_text(json.dumps({'status': 'passed', 'rank': rank,
            'counts': {'training_camera_forwards': 8, 'contribution_forwards': 4, 'loss_backwards': 8,
                       'optimizer_group_steps': 48, 'synthetic_updates': 8, 'reference_contribution_forwards': 1},
            'coordinated_rejection_checks': 1, 'real_image_decodes': 0,
            'active_state_unchanged_by_observation': True, 'terminal_before_evaluation': True}))
        workers[str(rank)] = {'path': str(file), 'sha256': run_b.sha256(file)}
    path = tmp_path / 'preflight.json'
    proof = {'status': 'passed', 'profile': run_b.CONTRACT['profile'], 'implementation_commit': 'commit',
             'budget': run_b.CONTRACT['preflight_budget'], 'runtime': runtime,
             'code_sha256': {'b.py': run_b.sha256(tmp_path / 'b.py')}, 'workers': workers}
    path.write_text(json.dumps(proof))
    assert run_b.verify_preflight(path)['sha256'] == run_b.sha256(path)
    for change in ({'implementation_commit': 'stale'}, {'budget': {}}, {'workers': {'0': workers['0']}}, {'runtime': {}}):
        path.write_text(json.dumps({**proof, **change}))
        with pytest.raises(ValueError):
            run_b.verify_preflight(path)
    path.write_text(json.dumps(proof))
    (tmp_path / 'worker-1.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        run_b.verify_preflight(path)


@pytest.mark.parametrize('blocked', ['C2', 'U0', 'O0'])
def test_suite_stops_at_first_block_and_never_runs_q1(blocked, tmp_path, monkeypatch):
    from image3d_scenegraph import gpu_lease
    class Lease:
        def __init__(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            pass
        def fileno(self):
            return 99
    monkeypatch.setattr(gpu_lease, 'FileLease', Lease)
    monkeypatch.setattr(gpu_lease, 'require_idle_gpu', lambda: None)
    monkeypatch.setattr(run_b.legacy, 'unlimited_host_cgroup', lambda unit: None)
    monkeypatch.setattr(run_b, 'verify_inputs', lambda p: None)
    monkeypatch.setattr(run_b.connected, 'parent_metadata', lambda *args: None)
    monkeypatch.setattr(run_b, 'verify_worker', lambda *args: None)
    monkeypatch.setattr(run_b.subprocess, 'run', lambda *args, **kw: None)
    monkeypatch.setattr(run_b.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 1024**3))
    parent = tmp_path / 'parent/attempts/train-001'
    (parent / 'checkpoints/iteration_000012000').mkdir(parents=True)
    (parent / 'attempt.json').write_text('{}')
    (parent / 'checkpoints/iteration_000012000/value').write_text('parent')
    root = tmp_path / 'admission-new'
    root.mkdir()
    launched = []
    def stage(folder, name, command, **kwargs):
        launched.append(folder.name)
        (folder / 'case.exit.json').write_text('{}')
        attempt = root / 'training/attempts' / folder.name
        attempt.mkdir()
        (attempt / 'attempt.json').write_text('{}')
    monkeypatch.setattr(run_b.legacy, 'run_stage', stage)
    def metric(root, name, protocol):
        value = metric_row()
        if name == blocked:
            value['metrics']['loss/12100'] = .1
        return value
    monkeypatch.setattr(run_b, 'metrics', metric)
    monkeypatch.setattr(run_b, 'completed', lambda root, name, digest: {})
    monkeypatch.setattr(run_b, 'compare', lambda *args: pytest.fail('blocked suite must not analyze Q1'))
    args = SimpleNamespace(output_root=root, protocol_sha256='abc')
    p = {'source_identity': {'native': {'parent_training': str(tmp_path / 'parent')}}}
    try:
        with pytest.raises(SystemExit) as exc:
            run_b.run(args, p)
        assert exc.value.code == 2
        assert launched == list(admission.CASES[:admission.CASES.index(blocked) + 1])
        end = json.loads((root / 'execution.json').read_text())
        assert end['status'] == 'admission_blocked' and end['exit_code'] == 2
        assert not (root / 'comparison.json').exists()
        with pytest.raises(FileExistsError):
            run_b.run(args, p)
    finally:
        for file in (root / 'training/attempts/train-001').rglob('*'):
            file.chmod(0o755 if file.is_dir() else 0o644)
        (root / 'training/attempts/train-001').chmod(0o755)


def test_output_accounting_handles_atomic_telemetry_rename(tmp_path, monkeypatch):
    (tmp_path / 'stable').write_bytes(b'123')
    (tmp_path / 'telemetry.tmp').write_bytes(b'456')
    original = Path.lstat
    def disappearing(path):
        if path.name == 'telemetry.tmp':
            raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', disappearing)
    assert run_b.output_bytes(tmp_path) == 3
    (tmp_path / 'link').symlink_to(tmp_path / 'stable')
    with pytest.raises(ValueError, match='nonregular'):
        run_b.output_bytes(tmp_path)
