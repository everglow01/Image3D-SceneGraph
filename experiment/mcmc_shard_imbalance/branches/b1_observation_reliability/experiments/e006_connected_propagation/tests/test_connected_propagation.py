"""Local standard-library checks; no tensor libraries or GPU execution."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e006_connected_propagation import run, probe, analyze


def test_frozen_cases_and_accounting():
    cases = run.CASES
    assert len(cases) == len(dict(cases)) == 10
    assert [v for k, v in cases if k.startswith('same')] == [0, 0, 0]
    assert [v for k, v in cases if k.startswith('varied')] == [1, 2, 3, 4]
    assert sum(v is None for _, v in cases) * 2 == run.CONTRACT['original_rasterizer_backwards'] == 6
    assert sum(v is not None for _, v in cases) * 2 == run.CONTRACT['injected_returns'] == 14
    assert len(cases) * 2 == run.CONTRACT['camera_forwards'] == run.CONTRACT['loss_backwards'] == 20
    assert len(cases) * 2 * len(probe.snapshots.PARAMETERS) == run.CONTRACT['optimizer_group_steps'] == 120
    assert run.CONTRACT['global_updates'] == 10 and run.CONTRACT['strategy_posts'] == 0
    assert run.CONTRACT['suite_timeout_seconds'] == 21600 and run.CONTRACT['host_limits'] is None
    assert len(analyze.SAME_PAIRS) == len(analyze.NATIVE_PAIRS) == 3
    assert analyze.VARIED_PAIRS == tuple((f'varied-{i}', 'same-0') for i in range(1, 5))


def test_fresh_import_and_help_do_not_load_tensor_libraries():
    code = f'''import contextlib,importlib,io,sys
r=importlib.import_module({run.MODULE!r})
sys.argv=['probe','--help']
with contextlib.redirect_stdout(io.StringIO()) as text:
 try:r.main()
 except SystemExit as exc:assert exc.code==0
assert '--monitor' in text.getvalue()
assert not ({{'torch','numpy','gsplat'}} & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=run.PROJECT, check=True)


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker', 'compare', 'monitor'])
def test_wrong_host_before_tensor_import_or_writes(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'local')
    monkeypatch.setattr(sys, 'argv', ['probe', '--output-root', str(tmp_path / 'new'), '--' + mode])
    with pytest.raises(SystemExit, match='仅指定远端'):
        run.main()
    assert not (tmp_path / 'new').exists() and 'torch' not in sys.modules


@pytest.mark.parametrize('mode', ['prepare', 'run', 'worker', 'compare'])
def test_authorization_precedes_protocol(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(run, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    monkeypatch.setattr(sys, 'argv', ['probe', '--output-root', str(tmp_path), '--' + mode])
    with pytest.raises(SystemExit, match='未获显式'):
        run.main()
    if mode == 'run':
        monkeypatch.setattr(sys, 'argv', [*sys.argv, '--authorize-execution'])
        with pytest.raises(SystemExit, match='未获显式'):
            run.main()


def test_output_boundaries(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    monkeypatch.setattr(run, 'RUNS', runs)
    assert run.output_root(runs / 'new', True) == runs / 'new'
    for path in (tmp_path / 'outside', runs, runs / '../escape'):
        with pytest.raises(ValueError):
            run.output_root(path, True)
    (runs / 'existing').mkdir()
    with pytest.raises(ValueError):
        run.output_root(runs / 'existing', True)
    (runs / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        run.output_root(runs / 'link/child')


def test_protocol_rejects_old_profile_missing_binding_and_wrong_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(run.legacy, 'verify', lambda: None)
    monkeypatch.setattr(run, 'PROJECT', tmp_path)
    monkeypatch.setattr(run.fixed, 'DEPENDENCIES', ('frozen.py',))
    monkeypatch.setattr(run, 'FILES', ('new.py',))
    for name in ('frozen.py', 'new.py'):
        (tmp_path / name).write_text(name)
    p = {**copy.deepcopy(run.CONTRACT), 'checkpoint_bytes': 1, 'snapshot_bytes_per_case': 2,
         'disk_admission_bytes': run.admission_bytes(1, 2),
         'bound_files': {str(tmp_path / n): run.sha256(tmp_path / n) for n in ('frozen.py', 'new.py')}}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='explicit protocol'):
        run.protocol_at(tmp_path, 'wrong')
    assert run.protocol_at(tmp_path, run.sha256(path))['profile'] == run.PROFILE
    for change in ({'profile': run.fixed.PROFILE}, {'cases': []}, {'bound_files': {}}, {'disk_admission_bytes': 1}):
        path.write_text(json.dumps({**p, **change}))
        with pytest.raises(ValueError):
            run.protocol_at(tmp_path, run.sha256(path))
    path.write_text(json.dumps(p))
    (tmp_path / 'new.py').write_text('changed')
    with pytest.raises(ValueError, match='dependency'):
        run.protocol_at(tmp_path, run.sha256(path))


def test_disk_budget_covers_ten_snapshots_and_one_parent():
    g = run.snapshots.GIB
    assert run.admission_bytes(0, 0) == 20 * g
    assert run.admission_bytes(2 * g, 7 * g) == 80 * g


def test_exact_and_numeric_gates_are_separate():
    probe.require_exact({'x': {'status': 'bitwise_equal'}})
    for rows in ({}, {'x': {'status': 'numerically_equivalent'}}, {'x': {'status': 'structure_mismatch'}}):
        with pytest.raises(ValueError):
            probe.require_exact(rows)
    assert analyze.acceptable({'P2': {'x': {'status': 'numerically_equivalent'}}})
    assert not analyze.acceptable({})
    assert not analyze.acceptable({'P3': {}})
    for status in ('numerically_inconsistent', 'missing_gradient', 'structure_mismatch'):
        assert not analyze.acceptable({'P2': {'x': {'status': status}}})


def test_same_gate_checks_all_three_pairs_and_two_ranks(tmp_path, monkeypatch):
    calls = []
    def compare(root, a, b, rank):
        calls.append((a, b, rank))
        return {'P2': {'gradient': {'status': 'bitwise_equal'}}, 'P3': {'parameter': {'status': 'bitwise_equal'}}}
    monkeypatch.setattr(analyze, 'compare_stages', compare)
    monkeypatch.setattr(analyze, 'updates', lambda *a: {'means': {'status': 'bitwise_equal'}})
    assert analyze.same_gate(tmp_path)['passed']
    assert len(calls) == 6
    monkeypatch.setattr(analyze, 'updates', lambda *a: {'means': {'status': 'numerically_inconsistent'}})
    assert not analyze.same_gate(tmp_path)['passed']
    monkeypatch.setattr(analyze, 'updates', lambda *a: {'means': {'status': 'bitwise_equal'}})
    monkeypatch.setattr(analyze, 'compare_stages', lambda *a: {'P2': {'gradient': {'status': 'numerically_inconsistent'}}})
    assert not analyze.same_gate(tmp_path)['passed']


def test_p3_capture_raises_before_strategy(tmp_path, monkeypatch):
    p = probe.PropagationProbe(tmp_path, 0, tmp_path, None, None, ['1'])
    p.stages = ['P0', 'P1', 'P2']
    p.order = ['P0', 'P1', 'B0', 'B1', 'P2']
    monkeypatch.setattr(probe.snapshots.FirstDivergenceProbe, 'capture', lambda self, stage, extra=None: self.stages.append(stage))
    monkeypatch.setattr(probe, 'require_finite', lambda _: None)
    monkeypatch.setattr(p, 'consensus', lambda label, action: action())
    with pytest.raises(probe.snapshots.ProbeComplete):
        p.capture('P3')
    assert p.stages == list(probe.STAGES) and p.order == list(probe.ORDER)
    assert p.calls['strategy'] == 0


def completed_probe(tmp_path):
    p = probe.PropagationProbe(tmp_path, 0, tmp_path, None, None, ['1'])
    p.stages, p.order, p.steps = list(probe.STAGES), list(probe.ORDER), list(probe.snapshots.PARAMETERS)
    p.calls = {'restore': 1, 'batch': 1, 'render': 1, 'backward': 1, 'strategy': 0}
    p.forward_calls = p.dispatch_calls = p.native_calls = 1
    p.loaded_train_ids = ['1']
    p.manifests = dict.fromkeys(probe.STAGES, 'sha')
    p.boundary_manifests = dict.fromkeys(('B0', 'B1'), 'sha')
    return p


@pytest.mark.parametrize('change', ['steps', 'order', 'strategy', 'native_calls', 'forward_calls', 'loaded_train_ids'])
def test_finish_rejects_budget_and_holdout_violations(tmp_path, change):
    p = completed_probe(tmp_path)
    if change == 'steps':
        p.steps = p.steps[:-1]
    elif change == 'order':
        p.order.append('P4')
    elif change == 'strategy':
        p.calls['strategy'] = 1
    elif change == 'loaded_train_ids':
        p.loaded_train_ids = ['test-image']
    else:
        setattr(p, change, 2)
    with pytest.raises(ValueError, match='incomplete'):
        p.finish()


def test_finish_and_worker_validation(tmp_path, monkeypatch):
    p = completed_probe(tmp_path)
    p.finish()
    (tmp_path / 'phases-rank-0.jsonl').write_text('\n'.join(json.dumps({'stage': s, 'iteration': 12001, 'manifest_sha256': 'sha'}) for s in probe.STAGES))
    monkeypatch.setattr(run, 'sha256', lambda _: 'sha')
    monkeypatch.setattr(run, 'require_finite', lambda _: {})
    monkeypatch.setattr(run.snapshots, 'validate_stage_schema', lambda *_: None)
    monkeypatch.setattr(run.legacy.boundary, 'validate_boundary_schema', lambda *_: None)
    assert run.verify_worker(tmp_path, 'native-0', 0)['calls']['strategy'] == 0
    with pytest.raises(ValueError, match='identity'):
        run.verify_worker(tmp_path, 'same-0', 0)
    with pytest.raises(FileExistsError):
        p.finish()


def test_case_monitor_deadline_and_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(run.time, 'monotonic', lambda: 10)
    assert run.CaseMonitor(tmp_path, 'native-0', 9)(0) == 'suite_timeout'
    monitor = run.CaseMonitor(tmp_path, 'native-0', 20)
    assert monitor(0, final=True) == 'missing_memory_telemetry'
    (tmp_path / 'memory').mkdir()
    (tmp_path / 'memory/rank-0.json').write_text(json.dumps({'peak_reserved_bytes': 19 * 1024**3}))
    assert monitor(0) == 'gpu_reserved_limit'


def test_monitor_success_failure_and_protocol(tmp_path):
    p = tmp_path / 'execution.json'
    (tmp_path / 'comparison.json').write_text('{}')
    for code, status in enumerate(('completed', 'failed', 'downstream_repeatability_blocked')):
        p.write_text(json.dumps({'protocol_sha256': 'sha', 'exit_code': code, 'status': status,
                                'comparison_sha256': run.sha256(tmp_path / 'comparison.json')}))
        assert run.monitor(tmp_path, 'sha') == code
    with pytest.raises(ValueError, match='protocol'):
        run.monitor(tmp_path, 'different')


def test_monitor_reports_disappeared_execution(tmp_path):
    (tmp_path / 'started.json').write_text(json.dumps({'protocol_sha256': 'sha', 'pid': -1, 'proc_start': '0'}))
    with pytest.raises(RuntimeError, match='disappeared'):
        run.monitor(tmp_path, 'sha')


def test_child_exit_does_not_skip_panel_terminal_marker():
    for code in (0, 2):
        command = f"bash -c 'exit {code}'; rc=$?; printf 'GP-JOB-END exit=%s\\n' \"$rc\"; exit \"$rc\""
        result = subprocess.run(['bash', '-c', command], text=True, capture_output=True)
        assert result.returncode == code and result.stdout.strip() == f'GP-JOB-END exit={code}'


def test_no_rng_restore_no_p4_and_remote_guard_before_import():
    tree = ast.parse(Path(probe.__file__).read_text())
    attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {'manual_seed', 'set_rng_state', '_restore_rng', 'step_post_backward'} & attrs
    remote = Path(__file__).with_name('test_connected_propagation_remote.py').read_text()
    assert remote.index('pytest.skip') < remote.index('from experiment.')
    assert 'IMAGE3D_AUTHORIZE_CONNECTED_PROPAGATION_TESTS' in remote


def test_status_publication_is_complete_and_exclusive(tmp_path):
    path = tmp_path / 'execution.json'
    run.publish_status(path, {'status': 'failed', 'exit_code': 1})
    assert json.loads(path.read_text())['exit_code'] == 1
    assert not path.with_suffix('.pending').exists()
    with pytest.raises(FileExistsError):
        run.publish_status(path, {'status': 'completed', 'exit_code': 0})
    assert json.loads(path.read_text())['status'] == 'failed'


@pytest.mark.parametrize('gate_passed', [False, True])
def test_suite_stops_before_later_cases_and_never_restarts(tmp_path, monkeypatch, gate_passed):
    from types import SimpleNamespace
    from image3d_scenegraph import gpu_lease
    (tmp_path / 'outputs').mkdir()
    source = tmp_path / 'source'
    checkpoint = source / 'attempts/train-001/checkpoints/iteration_000012000'
    checkpoint.mkdir(parents=True)
    (source / 'attempts/train-001/attempt.json').write_text('{}')
    (checkpoint / 'immutable.bin').write_bytes(b'parent')
    native_source = tmp_path / 'native'
    native_source.mkdir()
    (native_source / 'protocol.json').write_text('{}')
    root = tmp_path / 'new'
    root.mkdir()
    monkeypatch.setattr(run, 'PROJECT', tmp_path)
    monkeypatch.setattr(run.fixed, 'SOURCE', native_source)
    monkeypatch.setattr(run.legacy, 'unlimited_host_cgroup', lambda _: None)
    monkeypatch.setattr(run, 'verify_inputs', lambda _: None)
    monkeypatch.setattr(run, 'parent_metadata', lambda *a: {})
    monkeypatch.setattr(gpu_lease, 'require_idle_gpu', lambda: None)
    monkeypatch.setattr(run.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(run.signal, 'alarm', lambda *a: 0)
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(run.analyze, 'same_gate', lambda _: {'passed': gate_passed})
    executed = []
    def stage(folder, mode, command, **kwargs):
        executed.append(folder.name)
        assert command[command.index('--case') + 1] == folder.name
        assert kwargs['host_group'] is None and len(kwargs['pass_fds']) == 1
        (folder / 'case.exit.json').write_text('{}')
        for rank in (0, 1):
            (folder / f'worker-rank-{rank}.json').write_text('{}')
            (folder / f'phases-rank-{rank}.jsonl').write_text('{}')
    monkeypatch.setattr(run.legacy, 'run_stage', stage)
    def compare(folder, p, digest):
        assert len(executed) == 10
        (folder / 'comparison.json').write_text('{}')
    monkeypatch.setattr(run, 'compare', compare)
    args = SimpleNamespace(output_root=root, protocol_sha256='sha')
    p = {'disk_admission_bytes': 0, 'parent_training': str(source)}
    if gate_passed:
        run.run(args, p)
        assert executed == [name for name, _ in run.CASES]
        assert json.loads((root / 'execution.json').read_text())['exit_code'] == 0
    else:
        with pytest.raises(SystemExit) as failure:
            run.run(args, p)
        assert failure.value.code == 2
        assert executed == ['native-0', 'same-0', 'same-1', 'same-2']
        assert json.loads((root / 'execution.json').read_text())['status'] == 'downstream_repeatability_blocked'
        assert not (root / 'comparison.json').exists()
    with pytest.raises(FileExistsError):
        run.run(args, p)
    assert (checkpoint / 'immutable.bin').read_bytes() == b'parent'


def test_monitor_rejects_changed_comparison(tmp_path):
    (tmp_path / 'comparison.json').write_text('{}')
    (tmp_path / 'execution.json').write_text(json.dumps({'status': 'completed', 'exit_code': 0,
        'protocol_sha256': 'sha', 'comparison_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='comparison changed'):
        run.monitor(tmp_path, 'sha')


@pytest.mark.parametrize('local_rejection', [False, True])
def test_consensus_propagates_local_or_peer_rejection_without_tensor_library(tmp_path, monkeypatch, local_rejection):
    from types import SimpleNamespace
    class Flag:
        def __init__(self, values):
            self.value = values[0]
        def item(self):
            return self.value
    collectives = []
    def all_reduce(flag, op):
        collectives.append(flag.value)
        flag.value = 1
    fake = SimpleNamespace(tensor=lambda values, **kw: Flag(values), int32='int32',
        distributed=SimpleNamespace(is_initialized=lambda: True, all_reduce=all_reduce, ReduceOp=SimpleNamespace(MAX='max')))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    p = probe.PropagationProbe(tmp_path, 0, tmp_path, None, None, ['train'])
    def action():
        if local_rejection:
            raise ValueError('wrong source')
        return 'local success'
    with pytest.raises(ValueError, match='failed on a rank'):
        p.consensus('B0', action)
    assert collectives == [int(local_rejection)]
    r = json.loads((tmp_path / 'rejected-B0-rank-0.json').read_text())
    assert r['error'] == ('wrong source' if local_rejection else 'peer rejected')


def test_git_identity_requires_tracked_clean_execution_files(monkeypatch):
    calls = []
    monkeypatch.setattr(run.subprocess, 'run', lambda command, **kw: calls.append(command))
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **kw: 'a' * 40 + '\n')
    assert run.execution_commit() == 'a' * 40
    assert calls[0][:3] == ['git', 'ls-files', '--error-unmatch']
    assert calls[1][:4] == ['git', 'diff', '--quiet', 'HEAD']
    assert set(run.FILES) <= set(calls[0])
    monkeypatch.setattr(run, 'execution_commit', lambda: 'b' * 40)
    with pytest.raises(ValueError, match='commit changed'):
        run.verify_inputs({'implementation_commit': 'a' * 40})
