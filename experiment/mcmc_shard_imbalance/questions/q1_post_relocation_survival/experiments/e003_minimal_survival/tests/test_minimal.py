"""Local standard-library/mocked checks; never run the model or array analysis."""
from __future__ import annotations

import ast
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e003_minimal_survival import run, analyze, render_probe, observer
from scripts.mcmc_lifecycle_observer import reuse_record, survival_summary


def test_fresh_import_and_help_are_lightweight():
    code = f"""
import importlib, subprocess, sys
m=importlib.import_module({run.MODULE!r})
assert not ({{'numpy','torch','gsplat'}} & sys.modules.keys())
"""
    subprocess.run([sys.executable, '-B', '-c', code], cwd=run.PROJECT, check=True)
    result = subprocess.run([sys.executable, '-B', '-m', run.MODULE, '--help'], cwd=run.PROJECT, capture_output=True, text=True, check=True)
    assert '--render-worker' in result.stdout and '--authorize-execution' in result.stdout


def test_frozen_budget_and_scope():
    assert run.CASES == ('T0', 'T1')
    c = run.CONTRACT
    assert (c['new_updates_max'], c['training_camera_forwards_max'], c['loss_backwards_max']) == (400, 800, 800)
    assert (c['optimizer_group_steps_max'], c['isolated_render_forwards_max'], c['contribution_forwards_max']) == (4800, 64, 0)
    assert c['suite_timeout_seconds'] == 3600 and c['collective_timeout_seconds'] == 120
    assert c['host_limits'] is None and not c['full_numerical_noninterference_proven']
    assert c['E007'] == 'calibration_blocked_not_reclassified'
    assert sum(run.COUNTS[k] * 2 for k in ('training_camera_forwards',)) == c['preflight_budget']['training_camera_forwards']
    assert c['preflight_budget']['synthetic_updates'] == run.COUNTS['synthetic_updates']
    assert c['preflight_budget']['isolated_render_forwards'] == 2 * run.COUNTS['isolated_render_forwards']


def test_independent_output_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'RUNS', tmp_path)
    assert run.output_root(tmp_path / 'survival-one', True) == tmp_path / 'survival-one'
    for p in (tmp_path, tmp_path / 'admission-old', tmp_path / 'nested/survival-x', tmp_path / '../survival-x'):
        with pytest.raises(ValueError):
            run.output_root(p, True)
    (tmp_path / 'survival-existing').mkdir()
    with pytest.raises(ValueError):
        run.output_root(tmp_path / 'survival-existing', True)
    (tmp_path / 'survival-link').symlink_to(tmp_path / 'survival-existing')
    with pytest.raises(ValueError):
        run.output_root(tmp_path / 'survival-link')


def test_host_and_authorization_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run', '--prepare', '--output-root', str(tmp_path / 'survival-x')])
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'laptop')
    with pytest.raises(SystemExit, match='指定远端'):
        run.main()
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(run, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    with pytest.raises(SystemExit, match='显式授权'):
        run.main()
    assert not (tmp_path / 'survival-x').exists()


def test_camera_selection_frozen_independent_of_input_order():
    ids = list(map(str, range(20)))
    selected = render_probe.select_cameras(ids)
    assert len(selected) == 8 and selected == render_probe.select_cameras(reversed(ids))
    for bad in (ids[:7], ids + ['1']):
        with pytest.raises(ValueError):
            render_probe.select_cameras(bad)
    with pytest.raises(ValueError):
        render_probe.select_cameras(ids, 3)
    dataset = {'splits': {'train': ids}, 'images': [{'image_id': i, 'path': i} for i in ids], 'normalization': {}}
    config = {'loss': {'clamp_render': True}, 'resolution': {'longest_edge': 2048}}
    assert [r['image_id'] for r in render_probe.camera_contract(dataset, config)['images']] == selected


def test_source_groups_empty_and_invalid():
    record = reuse_record(8, [0, 1, 2], [5, 5, 6])
    assert record['group_sizes'] == {'target': 3, 'source': 2, 'uninvolved': 3}
    assert record['reuse_histogram'] == {2: 1, 1: 1}
    assert reuse_record(8, [], [])['group_sizes'] == {'target': 0, 'source': 0, 'uninvolved': 8}
    for targets, sources in (([0, 0], [5, 6]), ([0], [0]), ([0], [8]), ([0], [])):
        with pytest.raises(ValueError):
            reuse_record(8, targets, sources)
    assert survival_summary([], [], [], [])['redeath_fraction_of_initially_alive'] is None
    value = survival_summary([False, True, True], [2, 3, -1], [True, False, True], [3, -1, -1])
    assert value['not_alive_after_relocation'] == 1 and value['initially_alive_ever_died'] == 1
    assert value['redeath_fraction_of_initially_alive'] == .5


def test_trajectory_schema_rejects_missing_reordered_or_wrong_phase():
    def row(i):
        return {'iteration': i, 'phase': 'after_Adam_before_strategy', 'alive': {'slot_count': 8}, 'below_cutoff': {'slot_count': 8}}
    value = {'owner': 0, 'event': 2, 'end': 4, 'initial': {'alive': {'slot_count': 8}, 'below_cutoff': {'slot_count': 8}},
             'rows': [row(3), row(4)]}
    analyze.validate_rows(value, 0, 2, 4, 8)
    for change in ({'owner': 1}, {'event': 3}, {'rows': [row(3)]}, {'rows': [row(4), row(3)]},
                   {'rows': [row(3), {**row(4), 'phase': 'after_second_relocation'}]}):
        with pytest.raises(ValueError):
            analyze.validate_rows({**value, **change}, 0, 2, 4, 8)


def test_bitset_padding_and_count_rejection(tmp_path):
    file = tmp_path / 'mask.bits.gz'
    file.write_bytes(gzip.compress(b'\x05', mtime=0))
    row = {'path': file.name, 'bytes': file.stat().st_size, 'sha256': run.sha256(file), 'slot_count': 3,
           'xor_count': 2, 'encoding': 'gzip_numpy_packbits_little_zero_based_slots'}
    analyze.masks.verify_bitset(tmp_path, row)
    for change in ({'xor_count': 1}, {'slot_count': 2}, {'sha256': 'changed'}):
        with pytest.raises(ValueError):
            analyze.masks.verify_bitset(tmp_path, {**row, **change})


def test_minimal_observer_has_no_optimizer_or_render_hooks():
    tree = ast.parse(Path(observer.__file__).read_text())
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not ({'register_step_pre_hook', 'register_step_post_hook', 'backward', 'rasterization'} & attrs)
    assert '_render_visible_training_view' not in strings and 'contribution' not in strings
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MinimalObserver')
    assert not cls.bases


def test_protocol_cannot_relabel_old_gate_or_change_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PROJECT', tmp_path)
    monkeypatch.setattr(run, 'DEPENDENCIES', ('code.py',))
    (tmp_path / 'code.py').write_text('fixed')
    value = {**copy.deepcopy(run.CONTRACT), 'code_sha256': {'code.py': run.sha256(tmp_path / 'code.py')}}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(value))
    assert run.protocol_at(tmp_path, run.sha256(path))['profile'] == run.PROFILE
    for change in ({'cases': ['T0']}, {'full_numerical_noninterference_proven': True}, {'host_limits': 12},
                   {'isolated_render_forwards_max': 128}, {'code_sha256': {}}):
        path.write_text(json.dumps({**value, **change}))
        with pytest.raises(ValueError):
            run.protocol_at(tmp_path, run.sha256(path))


def test_independent_preflight_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'RUNS', tmp_path)
    monkeypatch.setattr(run, 'PROJECT', tmp_path)
    monkeypatch.setattr(run, 'DEPENDENCIES', ('code.py',))
    monkeypatch.setattr(run, 'execution_commit', lambda: 'new-commit')
    runtime = {'0': {'rank': 0}, '1': {'rank': 1}}
    monkeypatch.setattr(run, 'expected_runtime', lambda: runtime)
    (tmp_path / 'code.py').write_text('fixed')
    workers = {}
    for r in (0, 1):
        path = tmp_path / f'rank-{r}.json'
        path.write_text(json.dumps({'status': 'passed', 'rank': r, 'counts': run.COUNTS,
                                   'real_image_decodes': 0, 'invariants': dict.fromkeys(run.FLAGS, True)}))
        workers[str(r)] = {'path': str(path), 'sha256': run.sha256(path)}
    proof = {'status': 'passed', 'profile': run.PROFILE, 'implementation_commit': 'new-commit', 'runtime': runtime,
             'workers': workers, 'budget': run.CONTRACT['preflight_budget'], 'code_sha256': {'code.py': run.sha256(tmp_path / 'code.py')}}
    file = tmp_path / 'preflight.json'
    file.write_text(json.dumps(proof))
    assert run.verify_preflight(file)['sha256'] == run.sha256(file)
    for change in ({'profile': run.previous.CONTRACT['profile']}, {'implementation_commit': '539dc88'},
                   {'workers': {'0': workers['0']}}, {'budget': {}}, {'runtime': {}}):
        file.write_text(json.dumps({**proof, **change}))
        with pytest.raises(ValueError):
            run.verify_preflight(file)
    file.write_text(json.dumps(proof))
    (tmp_path / 'rank-1.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        run.verify_preflight(file)


def test_repeat_variation_is_not_survival_gate():
    assert render_probe.repeat_usable({'rgb': {'status': 'numerically_equivalent'}, 'alpha': {'status': 'bitwise_equal'}})
    assert not render_probe.repeat_usable({'rgb': {'status': 'numerically_inconsistent'}})
    assert not render_probe.repeat_usable({'rgb': {'status': 'structure_mismatch'}})
    assert not render_probe.repeat_usable({})
    assert not render_probe.repeat_usable({'rgb': {'status': 'bitwise_equal'}})


def test_budget_monitor_and_failed_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 1024**3))
    run.reserve(tmp_path, 100)
    with pytest.raises(ValueError, match='budget'):
        run.reserve(tmp_path, 7 * 1024**3)
    monkeypatch.setattr(run.time, 'monotonic', lambda: 20)
    assert run.StageMonitor(tmp_path, tmp_path, 10)(0) == 'suite_timeout'
    monitor = run.StageMonitor(tmp_path, tmp_path, 30)
    assert monitor(0, final=True) == 'missing_telemetry'
    (tmp_path / 'execution.json').write_text(json.dumps({'status': 'failed', 'exit_code': 1, 'protocol_sha256': 'sha'}))
    assert run.monitor(tmp_path, 'sha') == 1
    assert not (tmp_path / 'execution.once').exists()


@pytest.mark.parametrize('failure', ['T0', 'T1', 'render/T0', None])
def test_two_trajectories_then_render_and_stop_without_retry(failure, tmp_path, monkeypatch):
    from image3d_scenegraph import gpu_lease
    class Lease:
        def __init__(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def fileno(self):
            return 99
    monkeypatch.setattr(gpu_lease, 'FileLease', Lease)
    monkeypatch.setattr(gpu_lease, 'require_idle_gpu', lambda: None)
    monkeypatch.setattr(run.legacy, 'unlimited_host_cgroup', lambda unit: None)
    monkeypatch.setattr(run, 'verify_inputs', lambda p: None)
    monkeypatch.setattr(run.connected, 'parent_metadata', lambda *a: None)
    monkeypatch.setattr(run, 'verify_complete', lambda *a: None)
    monkeypatch.setattr(run, 'reserve', lambda *a: None)
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: None)
    parent = tmp_path / 'parent/attempts/train-001'
    (parent / 'checkpoints/iteration_000012000').mkdir(parents=True)
    (parent / 'attempt.json').write_text('{}')
    root = tmp_path / 'survival-test'
    root.mkdir()
    seen = []
    def stage(folder, label, command, **kwargs):
        mode = 'render/' if '--render-worker' in command else ''
        case = command[command.index('--case') + 1]
        seen.append(mode + case)
        if seen[-1] == failure:
            raise RuntimeError('synthetic execution failure')
    monkeypatch.setattr(run.legacy, 'run_stage', stage)
    def bind(root, folder, name, p, digest, monitor, rendering):
        (folder / 'complete.json').write_text('{}')
        return {'interpretation_usable': False} if rendering else {'camera_sequence': [['0', '1']]}
    monkeypatch.setattr(run, 'bind_complete', bind)
    monkeypatch.setattr(run.analyze, 'summarize', lambda data: {'test_fixture': True})
    p = {'source_identity': {'native': {'parent_training': str(tmp_path / 'parent')}}, 'implementation_commit': 'commit'}
    args = SimpleNamespace(output_root=root, protocol_sha256='sha')
    if failure:
        with pytest.raises(RuntimeError, match='synthetic'):
            run.run(args, p)
        expected = ['T0', 'T1', 'render/T0']
        assert seen == expected[:expected.index(failure) + 1]
        assert not (root / 'comparison.json').exists()
        assert run.read(root / 'execution.json')['exit_code'] == 1
    else:
        run.run(args, p)
        assert seen == ['T0', 'T1', 'render/T0', 'render/T1']
        report = run.read(root / 'comparison.json')
        assert report['status'] == 'descriptive_trajectories_complete'
        assert not report['full_numerical_noninterference_proven']
        assert report['E007'] == 'calibration_blocked_not_reclassified'
    with pytest.raises(FileExistsError):
        run.run(args, p)
    assert len(seen) <= 4


def test_completion_rejects_missing_and_escaped_bindings(tmp_path):
    stage = tmp_path / 'T0'
    stage.mkdir()
    p = stage / 'complete.json'
    base = {'case': 'T0', 'mode': 'training', 'protocol_sha256': 'sha', 'files': {}}
    p.write_text(json.dumps(base))
    with pytest.raises(ValueError, match='incomplete'):
        run.verify_complete(tmp_path, stage, 'T0', 'sha')
    (stage / 'case.exit.json').write_text(json.dumps({'returncode': 1, 'resource_failure': None}))
    for r in (0, 1):
        (stage / f'worker-rank-{r}.json').write_text('{}')
    files = {str(f.relative_to(tmp_path)): run.sha256(f) for f in stage.iterdir() if f.name != 'complete.json'}
    p.write_text(json.dumps({**base, 'files': files}))
    with pytest.raises(ValueError, match='failed stage'):
        run.verify_complete(tmp_path, stage, 'T0', 'sha')
    (tmp_path / 'other.json').write_text('{}')
    files['other.json'] = run.sha256(tmp_path / 'other.json')
    p.write_text(json.dumps({**base, 'files': files}))
    with pytest.raises(ValueError, match='escaped'):
        run.verify_complete(tmp_path, stage, 'T0', 'sha')


def test_descriptive_direction_is_not_forced_to_agree():
    assert analyze.direction([.1, .2]) == 'same_positive_direction'
    assert analyze.direction([-.1, -.2]) == 'same_negative_direction'
    assert analyze.direction([.1, -.1]) == 'direction_not_consistent'
    assert analyze.direction([.1, 0]) == 'direction_not_consistent'
    assert analyze.direction([None, .1]) == 'insufficient_denominator'
    assert analyze.direction([0, 0]) == 'zero_in_both'
    with pytest.raises(ValueError, match='both fixed'):
        analyze.summarize({'T0': {}})
