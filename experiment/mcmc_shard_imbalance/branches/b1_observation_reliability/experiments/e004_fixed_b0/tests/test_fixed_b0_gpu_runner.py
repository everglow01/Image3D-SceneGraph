"""Standard-library-only checks; actual arrays/CUDA are remote opt-in tests."""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import sys
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0 import tensor_analysis as analysis
from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0 import run as runner


def test_pairs_and_scope():
    assert analysis.REPEATS == 5 and len(analysis.PAIRS) == 10
    assert len({(r, i, j) for r in (0, 1) for i, j in analysis.PAIRS}) == 20
    assert all(0 <= i < j < 5 for i, j in analysis.PAIRS)
    assert runner.CONTRACT['backward_calls'] == 10
    assert runner.CONTRACT['new_optimizer_updates'] == runner.CONTRACT['new_forward_calls'] == 0
    assert runner.CONTRACT['old_gate1'] == 'not_reclassified'
    assert runner.legacy.core_hash() == runner.legacy.CORE


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
    p = {'core': runner.legacy.CORE, 'package': str(tmp_path), 'package_hashes': {},
         'bound_files': {str(path): runner.sha256_file(path)}}
    monkeypatch.setattr(runner, 'read', lambda _: {'pair': {'source_package': str(tmp_path), 'source_hashes': {}}})
    runner.verify_inputs(p)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bound file changed'):
        runner.verify_inputs(p)


def test_worker_missing_reordered_and_changed_trials(tmp_path, monkeypatch):
    trials = [{'trial': i, 'manifest_sha256': 'x', 'input_sha256': 'same', 'rng_sha256': 'same'} for i in range(5)]
    record = {'rank': 0, 'status': 'replay_complete', 'original_backward_calls': 5, 'new_optimizer_updates': 0,
              'new_forward_calls': 0, 'source_manifest_sha256': 'x', 'trials': trials, 'initialization_sha256': 'x'}
    monkeypatch.setattr(runner, 'sha256_file', lambda _: 'x')
    monkeypatch.setattr(runner.snapshots, 'load_snapshot', lambda _: {})
    monkeypatch.setattr(runner.boundary, 'validate_boundary_schema', lambda *_: None)
    initialization = {'method': runner.CONTRACT['initialization'], 'warmup_forward_calls': 0,
                      'warmup_backward_calls': 0, 'rng_after_sha256': 'same'}
    monkeypatch.setattr(runner, 'read', lambda p: initialization if p.name == 'initialization.json' else record)
    assert runner.verify_worker(tmp_path, 0) == record
    for changed in (trials[:-1], trials[::-1], [trials[0]] * 5):
        record['trials'] = changed
        with pytest.raises(ValueError, match='trial'):
            runner.verify_worker(tmp_path, 0)
    record['trials'] = trials
    initialization['warmup_backward_calls'] = 1
    with pytest.raises(ValueError, match='boundary differs'):
        runner.verify_worker(tmp_path, 0)
    initialization['warmup_backward_calls'] = 0
    record['initialization_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='audit changed'):
        runner.verify_worker(tmp_path, 0)
    record['initialization_sha256'] = 'x'
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
    init_call = next(n for n in ast.walk(replay) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'initialize_backend')
    assert init_call.lineno < loops[0].lineno
    source = Path(runner.__file__).read_text()
    assert 'host_group=None' in source and 'require_idle_gpu()' in source and 'FileLease(' in source


def test_protocol_complete_bindings_and_point_identity(tmp_path, monkeypatch):
    bound = {str(runner.PROJECT / name): 'digest' for name in runner.DEPENDENCIES}
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
    script_key = str(runner.PROJECT / runner.EXECUTION_FILES[0])
    bound[script_key] = 'tampered'
    with pytest.raises(ValueError, match='execution dependency'):
        runner.protocol_at(tmp_path, 'digest')
    bound[script_key] = 'digest'
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


def test_backend_initialization_audits_import_without_kernel_or_rng_restore(tmp_path, monkeypatch):
    binary = tmp_path / 'csrc.so'
    binary.write_bytes(b'pinned-test-binary-not-executed')
    expected = runner.sha256_file(binary)
    monkeypatch.setattr(runner, 'SOURCE', tmp_path)
    (tmp_path / 'protocol.json').write_text('{}')
    monkeypatch.setattr(runner, 'SOURCE_SHA', runner.sha256_file(tmp_path / 'protocol.json'))
    monkeypatch.setattr(runner, 'read', lambda _: {'pair': {'source_package': str(tmp_path), 'source_hashes': {'csrc.so': expected}}})
    spec = SimpleNamespace(origin=str(binary), loader=runner.ExtensionFileLoader('gsplat.csrc', str(binary)))
    monkeypatch.setattr(runner, 'find_spec', lambda _: spec)
    kernel = Mock(side_effect=AssertionError('initialization must not launch kernels'))
    extension = SimpleNamespace(__file__=str(binary), rasterize_to_pixels_3dgs_bwd=kernel)
    state = {'python': 0, 'numpy': 0, 'torch': 0, 'cuda': [0, 0]}
    monkeypatch.setattr(runner, 'rng_state', lambda: copy.deepcopy(state))
    imported = []
    def load(name):
        imported.append(name)
        if name == 'gsplat.csrc':
            return extension
        assert name == 'gsplat.cuda._backend'
        state['python'] = 1
        return SimpleNamespace(_C=extension)
    monkeypatch.setattr(runner, 'import_module', load)
    result = runner.initialize_backend()
    assert imported == ['gsplat.csrc', 'gsplat.cuda._backend']
    assert result['changed_rng_components'] == ['python'] and state['python'] == 1
    assert result['rng_after_sha256'] == runner.digest_state(state)
    assert result['extension_sha256'] == expected
    assert result['warmup_backward_calls'] == result['warmup_forward_calls'] == 0
    assert runner.initialize_backend()['changed_rng_components'] == []
    kernel.assert_not_called()
    monkeypatch.setattr(runner, 'import_module', lambda n: extension if n == 'gsplat.csrc' else SimpleNamespace(_C=object()))
    with pytest.raises(ValueError, match='loaded backend identity'):
        runner.initialize_backend()
    monkeypatch.setattr(runner, 'import_module', load)
    imported.clear()
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity differs'):
        runner.initialize_backend()
    assert imported == []
    monkeypatch.setattr(runner, 'find_spec', lambda _: None)
    with pytest.raises(ValueError, match='no JIT fallback'):
        runner.initialize_backend()
    assert imported == [] and 'torch' not in sys.modules


def test_workspace_identity_and_scientific_contract():
    from scripts import run_mcmc_fixed_b0 as original
    assert runner.PROFILE != original.PROFILE
    assert runner.UNIT != original.UNIT
    assert runner.RUNS_ROOT == runner.PROJECT / runner.EXPERIMENT_REL / 'runs'
    actual = {k: v for k, v in runner.CONTRACT.items()
              if k not in {'profile', 'legacy_source_manifest_sha256'}}
    assert actual == {k: v for k, v in original.CONTRACT.items() if k != 'profile'}
    assert set(runner.EXECUTION_FILES) <= set(runner.DEPENDENCIES)
    assert set(runner.legacy.LEGACY_FILES) <= set(runner.DEPENDENCIES)
    assert runner.CONTRACT['legacy_source_manifest_sha256'] == runner.legacy.MANIFEST_SHA256


def test_compute_function_ast_unchanged():
    from scripts import mcmc_fixed_b0_analysis as old_analysis
    from scripts import run_mcmc_fixed_b0 as old_runner
    def definitions(module):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert definitions(analysis) == definitions(old_analysis)
    old, new = definitions(old_runner), definitions(runner)
    for name in ('validate_context', 'load_context', 'storage_ids', 'initialize_backend',
                 'replay_five', 'verify_worker', 'worker', 'compare'):
        assert new[name] == old[name], name
    legacy_defs = definitions(runner.legacy)
    from scripts import run_mcmc_stability_ablation as resources
    from scripts import run_mcmc_lifecycle_trace as lifecycle
    assert legacy_defs['unlimited_host_cgroup'] == definitions(resources)['unlimited_host_cgroup']
    assert legacy_defs['core_hash'] == definitions(lifecycle)['core_hash']
    for module in (runner, analysis):
        assert not any(isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('scripts')
                       for n in ast.walk(ast.parse(Path(module.__file__).read_text())))


def test_pure_scalar_results_match_legacy():
    from scripts import mcmc_fixed_b0_analysis as original
    for a, b in ((0., -0.), (-2**-149, 2**-149), (1., 1. + 2**-23)):
        assert analysis.scalar_ulp(a, b) == original.scalar_ulp(a, b)
    for step, m, v, g in ((0, 0., 0., .1), (12000, -2e-14, 6e-20, 4e-9), (12000, 0., 0., 7e-15)):
        group = {'betas': (.9, .999), 'lr': .001, 'eps': 1e-15}
        assert analysis.adam_scalar(.5, g, m, v, step, group) == original.adam_scalar(.5, g, m, v, step, group)


def test_frozen_bridge_dependency_tamper_missing_and_symlink(tmp_path):
    manifest = runner.legacy.MANIFEST
    for name in manifest['files']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((runner.PROJECT / name).read_bytes())
    assert runner.legacy.verify(tmp_path) == manifest
    name = next(iter(manifest['files']))
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b'\n# changed\n')
    with pytest.raises(ValueError, match='dependency differs'):
        runner.legacy.verify(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        runner.legacy.verify(tmp_path)
    path.symlink_to(runner.PROJECT / name)
    with pytest.raises(ValueError, match='unsafe'):
        runner.legacy.verify(tmp_path)


def test_frozen_bridge_manifest_and_escape(tmp_path, monkeypatch):
    import hashlib
    import json
    path = tmp_path / 'gpu-source.json'
    path.write_text('{}')
    monkeypatch.setattr(runner.legacy, 'MANIFEST_PATH', path)
    with pytest.raises(ValueError, match='manifest differs'):
        runner.legacy.verify()
    source = copy.deepcopy(runner.legacy.MANIFEST)
    source['files'] = {'../escape': {'bytes': 0, 'sha256': '0' * 64}}
    path.write_text(json.dumps(source))
    monkeypatch.setattr(runner.legacy, 'MANIFEST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='unsafe'):
        runner.legacy.verify(tmp_path)


def test_legacy_bridge_only_delegates_source_validation(monkeypatch):
    expected = {'validated': True}
    check = Mock(return_value=expected)
    frozen = SimpleNamespace(verify_source=check)
    for name in ('main', 'run', 'worker', 'replay_five', 'prepare', 'compare'):
        setattr(frozen, name, Mock(side_effect=AssertionError('old execution forbidden')))
    monkeypatch.setattr(runner.legacy, '_load_source', lambda: frozen)
    assert runner.verify_source() == expected
    check.assert_called_once_with()


def test_workspace_output_guards(tmp_path, monkeypatch):
    runs = tmp_path / 'experiment/runs'
    monkeypatch.setattr(runner, 'RUNS_ROOT', runs)
    assert runner.checked_output_root(runs / 'new', preparing=True) == runs / 'new'
    for path in (runs, tmp_path / 'outputs/analysis/old', runner.SOURCE, runs / '../escape'):
        with pytest.raises(ValueError):
            runner.checked_output_root(path, preparing=True)
    runs.mkdir(parents=True)
    (runs / 'used').mkdir()
    with pytest.raises(FileExistsError):
        runner.checked_output_root(runs / 'used', preparing=True)
    (runs / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe'):
        runner.checked_output_root(runs / 'link/new')


def test_old_profile_rejected(tmp_path):
    from scripts import run_mcmc_fixed_b0 as original
    runner.write_new(tmp_path / 'protocol.json', original.CONTRACT)
    with pytest.raises(ValueError, match='contract differs'):
        runner.protocol_at(tmp_path, runner.sha256_file(tmp_path / 'protocol.json'))


def test_new_module_worker_command_and_resource_controls(tmp_path, monkeypatch):
    captured = {}
    class StopAtLaunch(Exception):
        pass
    class Lease:
        def __init__(self, path):
            captured['lease_path'] = path
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def fileno(self):
            return 123
    def stage(*args, **kwargs):
        captured['command'] = args[2]
        captured.update(kwargs)
        raise StopAtLaunch
    monkeypatch.setattr(runner, 'unlimited_host_cgroup', lambda unit: captured.update(unit=unit))
    monkeypatch.setattr(runner, 'verify_inputs', lambda _: None)
    monkeypatch.setattr(runner, 'FileLease', Lease)
    monkeypatch.setattr(runner, 'require_idle_gpu', lambda: captured.update(idle_checked=True))
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: Mock(free=30 * runner.snapshots.GIB))
    monkeypatch.setattr(runner, 'run_stage', stage)
    args = SimpleNamespace(output_root=tmp_path, protocol_sha256='bound-sha')
    with pytest.raises(StopAtLaunch):
        runner.run(args, {'disk_admission_bytes': 20 * runner.snapshots.GIB})
    cmd = captured['command']
    assert cmd[:6] == ['/usr/bin/env', '-u', 'PYTHONPATH', sys.executable, '-m', runner.MODULE]
    assert '--worker' in cmd and '--authorize-execution' in cmd
    assert cmd[cmd.index('--protocol-sha256') + 1] == 'bound-sha'
    assert cmd[cmd.index('--lease-fd') + 1] == '123'
    assert captured['host_group'] is None and captured['pass_fds'] == (123,)
    assert captured['cwd'] == runner.PROJECT and captured['idle_checked']
    assert captured['unit'] == runner.UNIT
    assert captured['lease_path'] == runner.PROJECT / 'outputs/.gpu.lock'


def test_worker_rejects_missing_lease_before_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setattr(runner, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph'))
    monkeypatch.setattr(runner, 'RUNS_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'protocol_at', lambda *_: {})
    monkeypatch.setattr(runner, 'unlimited_host_cgroup', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['run', '--output-root', str(tmp_path / 'new'),
                                    '--worker', '--authorize-execution', '--protocol-sha256', 'x'])
    with pytest.raises(ValueError, match='inherited GPU lease'):
        runner.main()
    assert 'torch' not in sys.modules


def test_fresh_import_and_help_do_not_load_tensor_libraries():
    import subprocess
    command = ('import importlib,sys; '
               f'importlib.import_module({runner.MODULE!r}); '
               'assert not {"torch", "numpy", "gsplat"}.intersection(sys.modules)')
    result = subprocess.run([sys.executable, '-B', '-c', command], cwd=runner.PROJECT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-B', '-m', runner.MODULE, '--help'],
                            cwd=runner.PROJECT, capture_output=True, text=True)
    assert result.returncode == 0 and '--protocol-sha256' in result.stdout, result.stderr


def test_old_remote_opt_in_does_not_enable_workspace_preflight(monkeypatch):
    import importlib.util
    import os
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: 'i-94B8D131')
    monkeypatch.setenv('IMAGE3D_AUTHORIZE_FIXED_B0_TESTS', '1')
    monkeypatch.delenv('IMAGE3D_AUTHORIZE_FIXED_B0_WORKSPACE_TESTS', raising=False)
    path = Path(__file__).with_name('test_fixed_b0_gpu_remote.py')
    spec = importlib.util.spec_from_file_location('fixed_b0_workspace_gate_check', path)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(pytest.skip.Exception):
        spec.loader.exec_module(module)
    assert os.environ.get('IMAGE3D_AUTHORIZE_FIXED_B0_WORKSPACE_TESTS') is None
    assert 'torch' not in sys.modules
