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

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import run, analyze, probe, moment_split


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
        code = f"import sys; import {module}; import {module.rsplit('.', 1)[0]}.moment_split; assert not {{'torch','numpy','gsplat'}} & sys.modules.keys()"
        subprocess.run([sys.executable, '-B', '-c', code], cwd=run.PROJECT, check=True, capture_output=True)
        help_result = subprocess.run([sys.executable, '-B', '-m', module, '--help'], cwd=run.PROJECT,
                                     check=True, capture_output=True, text=True)
        self.assertIn('--acknowledge-gate0-review', help_result.stdout)
        self.assertIn('--authorize-prefix', help_result.stdout)
        self.assertIn('--authorize-arms', help_result.stdout)
        self.assertIn('--prepare-split', help_result.stdout)
        self.assertIn('--authorize-split', help_result.stdout)
        self.assertNotIn('--authorize-execution', help_result.stdout)

    def test_empty_moment_digest_preserves_canonical_metadata(self):
        class EmptyTensor:
            dtype = 'torch.float32'
            shape = (0, 3)
            def numel(self):
                return 0
            def detach(self):
                return self
            def reshape(self, *args):
                return self
            def contiguous(self):
                return self
            def view(self, *args):
                raise RuntimeError('self.stride(-1) must be 1 to view Float as Byte, but got 0')
        fake_torch = SimpleNamespace(Tensor=EmptyTensor, uint8='torch.uint8')
        with patch.dict(sys.modules, {'torch': fake_torch}):
            empty = EmptyTensor()
            with self.assertRaisesRegex(RuntimeError, 'stride'):
                probe.digest_state(empty)
            digests = set()
            for shape in ((0,), (0, 3), (0, 4), (0, 1, 3), (0, 15, 3)):
                empty.shape = shape
                expected = probe.digest_state(('tensor', str(empty.dtype), list(shape)))
                self.assertEqual(probe.moment_digest(empty), expected)
                digests.add(expected)
            self.assertEqual(len(digests), 5)
            before = probe.moment_digest(empty)
            empty.dtype = 'torch.float64'
            self.assertNotEqual(probe.moment_digest(empty), before)

    def test_nonempty_moment_digest_delegates_without_conversion(self):
        value = SimpleNamespace(numel=lambda: 3)
        with patch.object(probe, 'digest_state', return_value='original-digest') as digest:
            self.assertEqual(probe.moment_digest(value), 'original-digest')
            digest.assert_called_once_with(value)
        tree = ast.parse(Path(probe.__file__).read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'restore_source_moments')
        calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count('moment_digest'), 5)
        self.assertNotIn('digest_state', calls)

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
        if phase == 'split':
            p['moment_split'] = {}
        order = []
        def stage(folder, label, command, **kwargs):
            case = command[command.index('--case') + 1]
            order.append(case)
            self.assertIn('--authorize-' + phase, command)
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
        if phase in ('arms', 'split'):
            def bind(root, case, p, resources):
                return {str(rank): {'noise_schedule': [], 'analysis': {'case': case, 'rank': rank}} for rank in (0, 1)}
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
        run.verify_inputs.assert_called_once_with(p)
        run.verify_arms.assert_called_once_with(root, p, args.arms_protocol_sha256)
        run.verify_complete.assert_not_called()
        self.assertEqual(run.bind_complete.call_count, 2)
        run.analyze.compare.assert_called_once_with(root, prefix=root / 'prefix',
            arms={case: {str(rank): {'case': case, 'rank': rank} for rank in (0, 1)} for case in ('A', 'B')})

    def test_failed_arm_acceptance_stops_before_next_arm(self):
        root, args, p, order = self.run_fixture('arms')
        run.bind_complete.side_effect = ValueError('invalid first arm evidence')
        with self.assertRaisesRegex(ValueError, 'invalid first arm evidence'):
            run.run_phase(args, p, 'arms')
        self.assertEqual(order, ['A'])
        self.assertFalse((root / 'B').exists())
        run.analyze.compare.assert_not_called()

    def test_workers_keep_live_checks_without_shared_admission(self):
        tree = ast.parse(Path(run.__file__).read_text())
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        calls = {ast.unparse(n.func) for n in ast.walk(worker) if isinstance(n, ast.Call)}
        self.assertFalse({'verify_inputs', 'verify_arms', 'verify_complete'} & calls)
        self.assertTrue({'protocol_at', 'InterventionProbe', 'probe.installed', 'trainer.train_gaussians'} <= calls)
        probe_tree = ast.parse(Path(probe.__file__).read_text())
        initialize = next(n for n in ast.walk(probe_tree) if isinstance(n, ast.FunctionDef) and n.name == 'initialize_arm')
        live_calls = {ast.unparse(n.func) for n in ast.walk(initialize) if isinstance(n, ast.Call)}
        self.assertTrue({'self.validate_live', 'self.restore_source_moments'} <= live_calls)

    def test_fixed_moment_selection_and_split_budget(self):
        for case in ('A', 'B', 'V', 'M', 'O'):
            selected = {(name, key) for name in run.snapshots.PARAMETERS for key in analyze.MOMENTS
                        if probe.retains_moment(case, name, key)}
            if case == 'A':
                expected = set()
            elif case == 'B':
                expected = {(n, k) for n in run.snapshots.PARAMETERS for k in analyze.MOMENTS}
            elif case == 'O':
                expected = {('opacities', k) for k in analyze.MOMENTS}
            else:
                expected = {(n, 'exp_avg_sq' if case == 'V' else 'exp_avg') for n in run.snapshots.PARAMETERS}
            self.assertEqual(selected, expected)
        with self.assertRaises(ValueError):
            probe.retains_moment('unknown', 'opacities', 'exp_avg')
        self.assertEqual(moment_split.CONTRACT['arms'], ['V', 'M', 'O'])
        self.assertEqual(moment_split.CONTRACT['total_updates'], 60)
        self.assertEqual(moment_split.CONTRACT['optimizer_group_steps'], 720)
        self.assertEqual(moment_split.CONTRACT['prefix_updates'], 0)
        self.assertEqual(moment_split.CONTRACT['extra_render_forwards'], 0)

    def test_split_order_admission_once_and_reference_reuse(self):
        root, args, p, order = self.run_fixture('split')
        with patch.object(moment_split, 'verify_checkpoint_copies') as copies, \
                patch.object(moment_split, 'verify_schedule') as schedule, \
                patch.object(moment_split, 'compare', return_value={'status': 'moment_split_short_window_measured'}) as compare:
            run.run_phase(args, p, 'split')
            self.assertEqual(order, ['V', 'M', 'O'])
            self.assertFalse((root / 'A').exists())
            self.assertFalse((root / 'B').exists())
            copies.assert_called_once_with(root, p)
            run.verify_inputs.assert_called_once_with(p)
            run.verify_arms.assert_not_called()
            run.verify_complete.assert_not_called()
            self.assertEqual(schedule.call_count, 3)
            compare.assert_called_once()
            self.assertEqual(run.read(root / 'split-execution.json')['exit_code'], 0)

    def test_split_failure_stops_and_cannot_run_as_old_arms(self):
        root, args, p, order = self.run_fixture('split', fail_case='M')
        with self.assertRaisesRegex(ValueError, 'own execution mode'):
            run.run_phase(args, p, 'arms')
        self.assertFalse((root / 'arms.once').exists())
        with patch.object(moment_split, 'verify_checkpoint_copies'), patch.object(moment_split, 'verify_schedule'):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                run.run_phase(args, p, 'split')
        self.assertEqual(order, ['V', 'M'])
        self.assertFalse((root / 'O').exists())

    def test_split_contract_requires_pinned_reference_and_new_root(self):
        binding = {'reference_root': str(moment_split.REFERENCE),
            'reference_protocol_sha256': moment_split.REFERENCE_PROTOCOL_SHA,
            'reference_comparison_sha256': moment_split.REFERENCE_COMPARISON_SHA,
            'checkpoint_hash': moment_split.prefix_recovery.CHECKPOINT_HASH}
        p = {**moment_split.CONTRACT, 'moment_split': binding}
        moment_split.contract(p)
        for key in binding:
            changed = copy.deepcopy(p)
            changed['moment_split'][key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                moment_split.contract(changed)
        for key, value in (('arms', ['A', 'B']), ('total_updates', 100), ('prefix_recovery', {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                moment_split.contract({**p, key: value})
        with patch.object(moment_split, 'reference_inputs') as reference:
            with self.assertRaises(ValueError):
                moment_split.prepare(self.folder / 'adam-intervention-recovery-old')
            reference.assert_not_called()

    def test_split_reference_reads_only_bound_inputs(self):
        reference = self.folder / 'reference'
        prefix = self.folder / 'original/prefix'
        bound = {}
        for rank in (0, 1):
            for name in ('pre-relocation', 'cohort'):
                file = self.put(prefix / f'{name}-rank-{rank}/manifest.json', {'fixture': name})
                bound[str(file.relative_to(self.folder))] = run.sha256(file)
            file = self.put(prefix / f'prefix-state-rank-{rank}.json', {'fixture': 'state'})
            bound[str(file.relative_to(self.folder))] = run.sha256(file)
        protocol = self.put(reference / 'protocol.json', {'prefix_recovery': {'source': {'bound_files': bound, 'copy_files': {}}},
            'source_identity': {'native': {'bound_files': {}, 'pair': {'replay': str(self.folder / 'replay'),
                'source_package': str(self.folder / 'package'), 'source_hashes': {}}}}})
        protocol_sha = run.sha256(protocol)
        comparison = self.put(reference / 'comparison.json', {'status': 'paired_short_window_measured'})
        proof = self.put(reference / 'recovery-complete.json', {'checkpoint_hash': moment_split.prefix_recovery.CHECKPOINT_HASH,
            'full_state_sha256': moment_split.prefix_recovery.STATE_HASHES})
        for phase, report in (('recovery', proof), ('arms', comparison)):
            self.put(reference / f'{phase}-execution.json', {'status': 'completed', 'exit_code': 0,
                'protocol_sha256': protocol_sha, 'report': report.name, 'report_sha256': run.sha256(report)})
        for case in ('A', 'B'):
            files = {}
            for rank in (0, 1):
                worker = self.put(reference / case / f'worker-rank-{rank}.json', {'case': case, 'rank': rank,
                    'protocol_sha256': protocol_sha, 'new_updates': 20, 'status': 'measured',
                    'restored_digest': moment_split.prefix_recovery.STATE_HASHES[str(rank)], 'schedule': [1], 'noise_schedule': [2]})
                snapshot = self.put(reference / case / f'opacity-before-1-rank-{rank}/manifest.json', {'fixture': 'opacity'})
                files.update({str(f.relative_to(reference)): run.sha256(f) for f in (worker, snapshot)})
            self.put(reference / case / 'complete.json', {'case': case, 'protocol_sha256': protocol_sha, 'files': files})
        self.put(prefix.parent / 'unrelated-log.json', {'not_an_input': True})
        with patch.object(run, 'PROJECT', self.folder), patch.object(moment_split, 'REFERENCE', reference), \
                patch.object(moment_split, 'PREFIX', prefix), patch.object(moment_split, 'REFERENCE_PROTOCOL_SHA', protocol_sha), \
                patch.object(moment_split, 'REFERENCE_COMPARISON_SHA', run.sha256(comparison)):
            result = moment_split.reference_inputs()
            self.assertNotIn('original/unrelated-log.json', result['binding']['input_sha256'])
            self.put(prefix / 'cohort-rank-0/manifest.json', {'changed': True})
            with self.assertRaisesRegex(ValueError, 'input changed'):
                moment_split.reference_inputs()

    def test_split_reference_schedule_mismatch_rejected(self):
        reference = {str(r): {'restored_digest': str(r), 'schedule': [1, 2], 'noise_schedule': ['rng']} for r in (0, 1)}
        moment_split.verify_schedule(copy.deepcopy(reference), reference)
        for key in ('restored_digest', 'schedule', 'noise_schedule'):
            current = copy.deepcopy(reference)
            current['1'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                moment_split.verify_schedule(current, reference)

    def test_split_comparison_uses_saved_reference_and_excludes_first_loss(self):
        def arm(value):
            groups = {g: {'risk': value, 'family_equal_risk': value} for g in ('S', 'T', 'U')}
            return {str(r): {'survival': {'horizons': {h: {'groups': groups} for h in ('1', '5', '10', '20')}}} for r in (0, 1)}
        fields = ('loss', 'reconstruction_loss', 'l1', 'ssim', 'opacity_regularization', 'scale_regularization')
        rows = [{'iteration': i, 'batch_view_ids': ['a', 'b'], **dict.fromkeys(fields, 1.)} for i in range(12101, 12121)]
        reference = {'protocol': {'implementation_commit': 'old'}, 'comparison': {
            'arms': {'A': arm(.5), 'B': arm(.1)}, 'native_loss_before_each_update': {'A': rows, 'B': rows}}}
        completed = {case: {rank: {'analysis': data} for rank, data in arm(.2).items()} for case in probe.SPLIT_CASES}
        for case in probe.SPLIT_CASES:
            new_rows = copy.deepcopy(rows)
            new_rows[0]['loss'] = 100.
            for row in new_rows[1:]:
                row['loss'] = .8
            self.put(self.folder / case / 'native-tail-rank-0.json', new_rows)
        with patch.object(moment_split, 'first_step_mechanism', return_value={}):
            result = moment_split.compare(self.folder, completed, reference)
        self.assertEqual(set(result['arms']), {'A', 'B', 'V', 'M', 'O'})
        self.assertIs(result['arms']['A'], reference['comparison']['arms']['A'])
        self.assertAlmostEqual(result['paired_loss_changes']['V']['A']['mean_12102_through_12120']['loss'], -.2)
        self.assertEqual(result['paired_loss_changes']['V']['A']['per_step'][0]['difference']['loss'], 99.)
        self.assertFalse(result['reference_arms_reexecuted'])

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
