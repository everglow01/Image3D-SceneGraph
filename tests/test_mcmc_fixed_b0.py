"""Standard-library-only checks; actual arrays/CUDA are remote opt-in tests."""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from scripts import mcmc_fixed_b0_analysis as analysis
from scripts import run_mcmc_fixed_b0 as runner


def test_pairs_and_scope():
    assert analysis.REPEATS == 5 and len(analysis.PAIRS) == 10
    assert len({(r, i, j) for r in (0, 1) for i, j in analysis.PAIRS}) == 20
    assert all(0 <= i < j < 5 for i, j in analysis.PAIRS)
    assert runner.CONTRACT['backward_calls'] == 10
    assert runner.CONTRACT['new_optimizer_updates'] == runner.CONTRACT['new_forward_calls'] == 0
    assert runner.CONTRACT['old_gate1'] == 'not_reclassified'
    assert runner.native.lifecycle.core_hash() == runner.native.lifecycle.CORE


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker', 'compare'])
def test_local_and_authorization_guards(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: 'local')
    monkeypatch.setattr(sys, 'argv', ['probe', '--output-root', str(tmp_path / 'new'), '--' + mode])
    with pytest.raises(SystemExit, match='本机禁止'):
        runner.main()
    assert not (tmp_path / 'new').exists() and 'torch' not in sys.modules
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(runner, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    with pytest.raises(SystemExit, match='未获显式授权'):
        runner.main()


def test_ulp_signed_zero_and_subnormals():
    assert analysis.scalar_ulp(-0., 0.) == 0
    assert analysis.scalar_ulp(1., 1. + 2**-23) == 1
    assert analysis.scalar_ulp(-1., -1. - 2**-23) == 1
    assert analysis.scalar_ulp(-2**-149, 2**-149) == 2
    for value in (math.nan, math.inf):
        with pytest.raises(ValueError, match='finite'):
            analysis.scalar_ulp(value, value)


def test_control_tie_and_no_candidate():
    assert analysis.nearest_control([1., 2.], [(4, [1., 3.]), (2, [1., 1.])]) == (1., 2, [1., 1.])
    with pytest.raises(ValueError, match='no eligible'):
        analysis.nearest_control([1.], [])


def test_adam_scalar_and_unsupported_configs():
    config = {'betas': (.9, .999), 'lr': .01, 'eps': 1e-8}
    result = analysis.adam_scalar(1., 1., 0., 0., 0, config)
    assert result['parameter'] == pytest.approx(1 - .01 / (1 + 1e-8))
    assert result['exp_avg'] == pytest.approx(.1)
    assert result['exp_avg_sq'] == pytest.approx(.001)
    assert analysis.adam_scalar(1., 0., 0., 0., 12000, config)['update'] == 0
    for key in ('amsgrad', 'maximize', 'weight_decay'):
        with pytest.raises(ValueError, match='unsupported'):
            analysis.adam_scalar(1., 1., 0., 0., 1, {**config, key: True})


def test_typed_decoder_without_arrays():
    tree = {'type': 'dict', 'items': [[{'type': 'str', 'value': 'x'},
            {'type': 'tuple', 'items': [{'type': 'NoneType', 'value': None}, {'type': 'bool', 'value': False}]}]]}
    assert analysis.decode(tree, Path('.')) == {'x': (None, False)}
    with pytest.raises(ValueError, match='nonfinite'):
        analysis.decode({'type': 'nonfinite', 'value': 'nan'}, Path('.'))


def test_input_binding_tamper(tmp_path, monkeypatch):
    path = tmp_path / 'bound'
    path.write_bytes(b'first')
    monkeypatch.setattr(runner, 'PROJECT', tmp_path)
    monkeypatch.setattr(runner, 'file_hashes', lambda _: {})
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: Mock(free=100 * runner.snapshots.GIB))
    p = {'core': runner.native.lifecycle.CORE, 'package': str(tmp_path), 'package_hashes': {},
         'bound_files': {str(path): runner.sha256_file(path)}}
    monkeypatch.setattr(runner, 'read', lambda _: {'pair': {'source_package': str(tmp_path), 'source_hashes': {}}})
    runner.verify_inputs(p)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bound file changed'):
        runner.verify_inputs(p)


def test_worker_missing_reordered_and_changed_trials(tmp_path, monkeypatch):
    trials = [{'trial': i, 'manifest_sha256': 'x', 'input_sha256': 'same', 'rng_sha256': 'same'} for i in range(5)]
    record = {'rank': 0, 'status': 'replay_complete', 'original_backward_calls': 5, 'new_optimizer_updates': 0,
              'new_forward_calls': 0, 'source_manifest_sha256': 'x', 'trials': trials}
    monkeypatch.setattr(runner, 'sha256_file', lambda _: 'x')
    monkeypatch.setattr(runner.snapshots, 'load_snapshot', lambda _: {})
    monkeypatch.setattr(runner.boundary, 'validate_boundary_schema', lambda *_: None)
    monkeypatch.setattr(runner, 'read', lambda _: record)
    assert runner.verify_worker(tmp_path, 0) == record
    for changed in (trials[:-1], trials[::-1], [trials[0]] * 5):
        record['trials'] = changed
        with pytest.raises(ValueError, match='trial'):
            runner.verify_worker(tmp_path, 0)
    record['trials'] = copy.deepcopy(trials)
    record['trials'][2]['input_sha256'] = 'changed'
    with pytest.raises(ValueError, match='not fixed'):
        runner.verify_worker(tmp_path, 0)


def test_protocol_requires_sha_and_fixed_contract(tmp_path):
    runner.write_new(tmp_path / 'protocol.json', {'profile': 'wrong'})
    with pytest.raises(ValueError, match='explicit protocol'):
        runner.protocol_at(tmp_path, None)
    with pytest.raises(ValueError, match='contract differs'):
        runner.protocol_at(tmp_path, runner.sha256_file(tmp_path / 'protocol.json'))


def test_monitor_has_no_training_budget(tmp_path, monkeypatch):
    monitor = runner.Monitor(tmp_path)
    assert monitor.record()['new_optimizer_updates'] == 0
    assert monitor(21601) == 'replay_timeout'
    assert monitor(0, final=True) == 'missing_memory_telemetry'
    (tmp_path / 'memory').mkdir()
    for r in (0, 1):
        runner.write_new(tmp_path / f'memory/rank-{r}.json', {'peak_reserved_bytes': 19 * runner.snapshots.GIB})
    assert monitor(0) == 'gpu_reserved_limit'


def test_no_extra_compute_or_rng_restore_in_runner():
    tree = ast.parse(Path(runner.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attrs = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
    assert attrs.count('backward') == 1
    assert not {'forward', 'apply', 'step', 'train_gaussians', 'manual_seed', 'set_rng_state'} & set(attrs)
    replay = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'replay_five')
    loops = [n for n in ast.walk(replay) if isinstance(n, ast.For)]
    assert len(loops) == 1 and ast.unparse(loops[0].iter) == 'range(analysis.REPEATS)'
    source = Path(runner.__file__).read_text()
    assert 'host_group=None' in source and 'require_idle_gpu()' in source and 'FileLease(' in source


def test_protocol_complete_bindings_and_point_identity(tmp_path, monkeypatch):
    bound = {str(runner.PROJECT / name): 'x' for name in runner.DEPENDENCIES}
    bound.update({str(runner.PROJECT / 'outputs/analysis/first-divergence-t1-20261021-v1' / name): 'x' for name in ('protocol.json', 'comparison.json')})
    bound.update({str(runner.SOURCE / 'protocol.json'): runner.SOURCE_SHA,
                  str(runner.SOURCE / 'comparison.json'): runner.COMPARISON_SHA})
    for branch in ('a', 'b'):
        bound[str(runner.SOURCE / branch / 'complete.json')] = 'x'
        for rank in (0, 1):
            for stage in runner.boundary.ORDER:
                folder = runner.SOURCE / branch / f'rank-{rank}' / stage
                bound[str(folder / 'manifest.json')] = 'x'
                bound[str(folder / 'value.npy')] = 'tensor-sha'
    points = []
    for owner, slot, group in analysis.ANOMALIES:
        points += [{'owner': owner, 'slot': slot, 'parameter': group, 'role': 'prior_anomaly'},
                   {'owner': owner, 'slot': slot + 1, 'parameter': group, 'role': 'matched_control', 'matched_to': slot}]
    p = {**runner.CONTRACT, 'bound_files': bound, 'points': points, 'disk_admission_bytes': 20 * runner.snapshots.GIB}
    monkeypatch.setattr(runner, 'sha256_file', lambda _: 'digest')
    monkeypatch.setattr(runner, 'read', lambda path: p if path == tmp_path / 'protocol.json' else {'files': {'value.npy': {'sha256': 'tensor-sha'}}})
    assert runner.protocol_at(tmp_path, 'digest') == p
    key = str(runner.SOURCE / 'a/rank-0/B0/value.npy')
    del bound[key]
    with pytest.raises(ValueError, match='tensor binding'):
        runner.protocol_at(tmp_path, 'digest')
    bound[key] = 'tensor-sha'
    p['points'][1]['matched_to'] = 12
    with pytest.raises(ValueError, match='matched controls'):
        runner.protocol_at(tmp_path, 'digest')


def test_source_and_output_path_escape(tmp_path, monkeypatch):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (tmp_path / 'symlink').symlink_to(outside, target_is_directory=True)
    for name in ('../escape', '/absolute', 'symlink/file'):
        with pytest.raises(ValueError, match='unsafe'):
            runner.snapshots.safe_file(tmp_path, name)
