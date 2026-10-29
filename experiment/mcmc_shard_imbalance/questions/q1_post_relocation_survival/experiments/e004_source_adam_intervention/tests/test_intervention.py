"""Local stdlib/mock checks only; numerical/model checks remain remote opt-in."""
from __future__ import annotations

from contextlib import ExitStack
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import run, analyze, probe


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_fresh_import_and_help_are_lightweight(self):
        module = run.MODULE
        code = f"import sys; import {module}; assert not {{'torch','numpy','gsplat'}} & sys.modules.keys()"
        subprocess.run([sys.executable, '-B', '-c', code], cwd=run.PROJECT, check=True, capture_output=True)
        help_result = subprocess.run([sys.executable, '-B', '-m', module, '--help'], cwd=run.PROJECT,
                                     check=True, capture_output=True, text=True)
        self.assertIn('--acknowledge-gate0-review', help_result.stdout)
        self.assertIn('--authorize-prefix', help_result.stdout)
        self.assertIn('--authorize-arms', help_result.stdout)
        self.assertNotIn('--authorize-execution', help_result.stdout)

    def test_frozen_budget_and_scope(self):
        c = run.CONTRACT
        self.assertEqual(c['prefix_updates'] + 2 * c['arm_updates'], c['total_updates'])
        self.assertEqual(c['total_updates'], 140)
        self.assertEqual((c['camera_forwards'], c['loss_backwards'], c['optimizer_group_steps']), (280, 280, 1680))
        self.assertEqual(c['extra_render_forwards'], 0)
        self.assertEqual(c['global_relocations'], 1)
        self.assertIsNone(c['host_limits'])
        self.assertTrue(c['requires_separate_gate0_review'])
        self.assertFalse(c['long_term_root_cause_proven'])
        self.assertEqual(run.PREFLIGHT_COUNTS['global_updates'], 8)
        self.assertEqual(len(run.DEPENDENCIES), len(set(run.DEPENDENCIES)))
        self.assertTrue(set(run.base.DEPENDENCIES) <= set(run.DEPENDENCIES))

    def test_new_root_only(self):
        with patch.object(run, 'RUNS', self.folder):
            self.assertEqual(run.output_root(self.folder / 'adam-intervention-fixture', True), self.folder / 'adam-intervention-fixture')
            for path in (self.folder, self.folder / 'survival-old', self.folder / 'nested/adam-intervention-x', self.folder / '../adam-intervention-x'):
                with self.assertRaises(ValueError):
                    run.output_root(path, True)
            path = self.folder / 'adam-intervention-existing'
            path.mkdir()
            with self.assertRaises(ValueError):
                run.output_root(path, True)

    def test_explicit_options_and_step_not_reinitialized(self):
        group = {'betas': (.9, .999), 'lr': .01, 'eps': 1e-15, 'weight_decay': 0,
                 'amsgrad': False, 'maximize': False, 'capturable': False, 'differentiable': False,
                 'fused': None, 'foreach': None}
        self.assertIs(analyze.optimizer_options(group), group)
        for key, value in [('betas', (1., .999)), ('lr', float('nan')), ('eps', 0), ('weight_decay', .01),
                           ('amsgrad', True), ('maximize', True), ('fused', True), ('capturable', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze.optimizer_options({**group, key: value})
        text = Path(probe.__file__).read_text()
        tree = ast.parse(text)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'restore_source_moments')
        assignments = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign))]
        self.assertTrue(any('active[key][self.ids] = expected' in x for x in assignments))
        self.assertFalse(any("active['step']" in x for x in assignments))
        self.assertIn('self.protected_digest() != protected', ast.unparse(fn))

    def test_no_hidden_rng_restore_or_extra_backward(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [ast.unparse(n.func) for n in calls]
        self.assertFalse(any(n.endswith(('setstate', 'set_rng_state', 'manual_seed', 'autograd.grad')) for n in names))
        self.assertEqual(names.count('tensor_backward'), 1)
        self.assertEqual(names.count('sample'), 1)
        self.assertEqual(names.count('noise'), 1)
        sampled = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'sampled')
        self.assertNotIn('consensus', ast.unparse(sampled))
        self.assertIn('return result', ast.unparse(sampled))
        run_tree = ast.parse(Path(run.__file__).read_text())
        main = next(n for n in ast.walk(run_tree) if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertIn('authorize_prepare_arms', ast.unparse(main))
        self.assertIn('acknowledge_gate0_review', ast.unparse(main))

    def preflight_fixture(self):
        workers = {}
        for rank in (0, 1):
            folder = self.folder / f'rank-{rank}'
            self.put(folder / 'evidence.json', {'checkpoint_verified': True})
            for name in run.preflight_required(rank):
                value = {'placeholder': True}
                if name.endswith('/manifest.json'):
                    leaf = self.put((folder / name).parent / 'array.npy', {'synthetic_mock_only': True})
                    value = {'files': {'array.npy': {'sha256': run.sha256(leaf)}}}
                elif '/worker-rank-' in name:
                    case = Path(name).parts[0]
                    n = 4 if case == 'prefix' else 2
                    value = {'status': 'measured', 'rank': rank, 'case': case, 'new_updates': n,
                             'calls': {**dict.fromkeys(('batch', 'render', 'backward', 'raster_forward', 'raster_backward', 'noise', 'strategy'), n),
                                       'optimizer': n * 6, 'checkpoint': int(case == 'prefix')},
                             'validation_evaluated': False, 'native_training_complete': False}
                elif name.startswith('initialization-'):
                    value = {'warmup_forward_calls': 0, 'warmup_backward_calls': 0}
                self.put(folder / name, value)
            bound = {str(f.relative_to(folder)): run.sha256(f) for f in folder.rglob('*') if f.is_file()}
            file = self.put(folder / 'worker.json', {'rank': rank, 'flags': dict.fromkeys(run.FLAGS, True),
                'counts': {'updates': 8, 'forward': 8, 'backward': 8, 'adam': 48},
                'noise_equivalence_fixture_calls': 2, 'evidence_sha256': bound})
            workers[str(rank)] = {'path': str(file), 'sha256': run.sha256(file)}
        runtime = {'0': {'rank': 0}, '1': {'rank': 1}}
        file = self.put(self.folder / 'preflight.json', {'status': 'passed', 'profile': run.PROFILE,
            'implementation_commit': 'f' * 40, 'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES},
            'runtime': runtime, 'counts': run.PREFLIGHT_COUNTS, 'workers': workers})
        stack = ExitStack()
        stack.enter_context(patch.object(run, 'RUNS', self.folder))
        stack.enter_context(patch.object(run, 'execution_commit', return_value='f' * 40))
        stack.enter_context(patch.object(run.base, 'expected_runtime', return_value=runtime))
        self.addCleanup(stack.close)
        return file

    def test_new_preflight_binding_and_negative_fields(self):
        file = self.preflight_fixture()
        self.assertEqual(run.verify_preflight(file)['sha256'], run.sha256(file))
        original = run.read(file)
        for key, value in [('profile', 'mcmc_minimal_survival_v1'), ('implementation_commit', 'a' * 40),
                           ('workers', {'0': original['workers']['0']}), ('counts', {**run.PREFLIGHT_COUNTS, 'global_updates': 9})]:
            self.put(file, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.verify_preflight(file)
        self.put(file, original)
        worker_path = Path(original['workers']['0']['path'])
        worker = run.read(worker_path)
        for flag in run.FLAGS:
            changed = copy.deepcopy(worker)
            changed['flags'][flag] = False
            self.put(worker_path, changed)
            original['workers']['0']['sha256'] = run.sha256(worker_path)
            self.put(file, original)
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                run.verify_preflight(file)
        self.put(worker_path, worker)
        original['workers']['0']['sha256'] = run.sha256(worker_path)
        self.put(file, original)
        self.put(worker_path.parent / 'evidence.json', {'tampered': True})
        with self.assertRaises(ValueError):
            run.verify_preflight(file)

    def test_preflight_cannot_drop_bound_checkpoint_or_snapshot(self):
        file = self.preflight_fixture()
        report = run.read(file)
        path = Path(report['workers']['0']['path'])
        original = run.read(path)
        for missing in ('A/opacity-before-1-rank-0/manifest.json',
                        'training/attempts/prefix/checkpoints/iteration_000000004/optimizer.bin',
                        'A/opacity-before-1-rank-0/array.npy'):
            worker = copy.deepcopy(original)
            worker['evidence_sha256'].pop(missing)
            self.put(path, worker)
            report['workers']['0']['sha256'] = run.sha256(path)
            self.put(file, report)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                run.verify_preflight(file)

    def test_gate0_requires_review_and_nonzero_signal(self):
        root = self.folder
        self.put(root / 'protocol.json', {'scope': 'prefix'})
        self.put(root / 'prefix/complete.json', {'completed': True})
        owners = {'0': {'effective_intervention_available': True}, '1': {'effective_intervention_available': False}}
        gate = {'status': 'structural_verified_review_required', 'effective_intervention_available': True, 'checkpoint_hash': 'c' * 64,
                'owners': owners, 'source_protocol_sha256': run.sha256(root / 'protocol.json'),
                'arms_started': False, 'moment_magnitude_automatically_admitted': False}
        file = self.put(root / 'gate0.json', gate)
        p = {'implementation_commit': 'f' * 40, 'code_sha256': {}}
        with patch.object(run, 'verify_inputs'), patch.object(run, 'verify_complete'), patch.object(run, 'checkpoint', return_value={'checkpoint_hash': 'c' * 64}), patch.object(run.analyze, 'gate0', side_effect=lambda stage, rank: owners[str(rank)]):
            for digest, reviewed in ((None, True), (run.sha256(file), False), ('a' * 64, True)):
                with self.assertRaises(ValueError):
                    run.prepare_arms(root, p, digest, reviewed)
            self.put(file, {**gate, 'effective_intervention_available': False})
            with self.assertRaises(ValueError):
                run.prepare_arms(root, p, run.sha256(file), True)
            self.put(file, gate)
            run.prepare_arms(root, p, run.sha256(file), True)
            digest = run.sha256(root / 'arms-protocol.json')
            self.assertTrue(run.verify_arms(root, p, digest)['review_acknowledged'])
            self.put(root / 'prefix/complete.json', {'changed': True})
            with self.assertRaises(ValueError):
                run.verify_arms(root, p, digest)

    def test_protocol_rejects_changed_code_and_contract(self):
        p = {**run.CONTRACT, 'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES}}
        file = self.put(self.folder / 'protocol.json', p)
        self.assertEqual(run.protocol_at(self.folder, run.sha256(file)), p)
        for field, value in [('total_updates', 141), ('requires_separate_gate0_review', False), ('code_sha256', {})]:
            self.put(file, {**p, field: value})
            with self.assertRaises(ValueError):
                run.protocol_at(self.folder, run.sha256(file))

    def test_reserve_and_stage_resource_failures(self):
        with patch.object(run.base.previous, 'output_bytes', return_value=0), patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)):
            run.reserve(self.folder)
            for extra in (-1, run.CONTRACT['max_output_bytes']):
                with self.assertRaises(ValueError):
                    run.reserve(self.folder, extra)
        monitor = run.Monitor(self.folder, self.folder, 100)
        with patch.object(run.time, 'monotonic', return_value=0), patch.object(run, 'reserve'):
            self.assertEqual(monitor(0, final=True), 'missing_telemetry')
            for rank in (0, 1):
                self.put(self.folder / f'memory/rank-{rank}.json', {'peak_reserved_bytes': 1024, 'finished': True})
            self.assertIsNone(monitor(0, final=True))
            self.put(self.folder / 'memory/rank-1.json', {'peak_reserved_bytes': 19 * 1024**3, 'finished': True})
            self.assertEqual(monitor(0), 'gpu_reserved_limit')

    def run_fixture(self, phase, fail_case=None):
        root = self.folder / 'adam-intervention-fixture'
        root.mkdir()
        self.put(root / 'protocol.json', {'scope': phase})
        parent = self.folder / 'parent/attempts/train-001'
        self.put(parent / 'attempt.json', {'parent': True})
        self.put(parent / 'checkpoints/iteration_000012000/checkpoint.json', {'hash': 'frozen'})
        args = SimpleNamespace(output_root=root, protocol_sha256=run.sha256(root / 'protocol.json'), arms_protocol_sha256='a' * 64)
        p = {'source_identity': {'native': {'parent_training': str(self.folder / 'parent')}}}
        order = []
        def stage(folder, label, command, **kwargs):
            case = command[command.index('--case') + 1]
            order.append(case)
            self.assertIn('--authorize-prefix' if phase == 'prefix' else '--authorize-arms', command)
            if case == fail_case:
                raise RuntimeError('injected case failure')
        class Lease:
            def __init__(self, path):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def fileno(self):
                return 9
        module = SimpleNamespace(FileLease=Lease, require_idle_gpu=lambda: None)
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {'image3d_scenegraph.gpu_lease': module}))
        for target, name in ((run, 'verify_inputs'), (run, 'verify_arms'), (run, 'verify_complete'), (run, 'bind_complete'),
                             (run.base.legacy, 'unlimited_host_cgroup'), (run.base.connected, 'parent_metadata'),
                             (run.subprocess, 'run'), (run.signal, 'signal'), (run.signal, 'alarm'), (run, 'reserve')):
            stack.enter_context(patch.object(target, name))
        stack.enter_context(patch.object(run.base.legacy, 'run_stage', side_effect=stage))
        stack.enter_context(patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)))
        stack.enter_context(patch.object(run.analyze, 'gate0', return_value={'effective_intervention_available': True}))
        stack.enter_context(patch.object(run, 'checkpoint', return_value={'checkpoint_hash': 'c' * 64}))
        stack.enter_context(patch.object(run.analyze, 'compare', return_value={'status': 'paired_short_window_measured'}))
        if phase == 'arms':
            def bind(root, case, p, resources):
                for rank in (0, 1):
                    self.put(root / case / f'worker-rank-{rank}.json', {'noise_schedule': []})
            stack.enter_context(patch.object(run, 'bind_complete', side_effect=bind))
        self.addCleanup(stack.close)
        return root, args, p, order

    def test_prefix_stops_for_review_without_any_arm(self):
        root, args, p, order = self.run_fixture('prefix')
        run.run_phase(args, p, 'prefix')
        self.assertEqual(order, ['prefix'])
        self.assertFalse((root / 'A').exists())
        self.assertFalse((root / 'arms-protocol.json').exists())
        self.assertFalse(run.read(root / 'gate0.json')['arms_started'])
        with self.assertRaises(FileExistsError):
            run.run_phase(args, p, 'prefix')

    def test_arms_order_and_failure_stop(self):
        root, args, p, order = self.run_fixture('arms', fail_case='A')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            run.run_phase(args, p, 'arms')
        self.assertEqual(order, ['A'])
        self.assertFalse((root / 'B').exists())
        self.assertEqual(run.read(root / 'arms-execution.json')['exit_code'], 1)
        with self.assertRaises(FileExistsError):
            run.run_phase(args, p, 'arms')

    def test_arms_complete_in_fixed_order(self):
        root, args, p, order = self.run_fixture('arms')
        run.run_phase(args, p, 'arms')
        self.assertEqual(order, ['A', 'B'])
        self.assertEqual(run.read(root / 'arms-execution.json')['exit_code'], 0)

    def test_host_and_authorization_checked_before_actions(self):
        with patch.object(sys, 'argv', ['run', '--prefix', '--output-root', str(self.folder)]), patch.object(run.socket, 'gethostname', return_value='not-authorized-host'):
            with self.assertRaises(SystemExit):
                run.main()
        with patch.object(run, 'PROJECT', Path('/usr/local/3dgs_new/Image3D-SceneGraph')), patch.object(run.socket, 'gethostname', return_value='i-94B8D131'):
            for mode in ('--prefix', '--arms', '--prepare-arms'):
                with patch.object(sys, 'argv', ['run', mode, '--output-root', str(self.folder)]), self.assertRaises(SystemExit):
                    run.main()

    def test_old_gate_not_reclassified_and_algebra_not_training(self):
        text = Path(analyze.__file__).read_text()
        tree = ast.parse(text)
        algebra = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'adam_prediction')
        calls = [ast.unparse(n.func) for n in ast.walk(algebra) if isinstance(n, ast.Call)]
        self.assertFalse(any('backward' in n or n.endswith('.step') for n in calls))
        self.assertIn("'first_loss_is_pre_intervention_update': True", text)
        self.assertIn("'long_term_root_cause_proven': False", text)
        self.assertIn("'algorithm_improvement_automatically_declared': False", text)


if __name__ == '__main__':
    unittest.main()
