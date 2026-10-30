"""Stdlib integration checks for real process finalization and zero-training recovery."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import run, analyze, prefix_recovery as recovery


def runner_fixture(root, case):
    """Exercise the real shared runner, not a mock of the interface that failed."""
    root.mkdir()
    (root / 'memory').mkdir()
    for rank in (0, 1):
        run.write_new(root / 'memory' / f'rank-{rank}.json', {'finished': True, 'peak_reserved_bytes': (rank + 1) * 1024})
    monitor = run.Monitor(root, root, time.monotonic() + 30)
    if case == 'rejected':
        class Rejected(run.Monitor):
            def __call__(self, elapsed, *, final=False):
                super().__call__(elapsed, final=final)
                return 'fixture_resource_rejected'
        monitor = Rejected(root, root, time.monotonic() + 30)
    code = 7 if case == 'failure' else 0
    try:
        run.base.legacy.run_stage(root, 'stage', [sys.executable, '-I', '-c', f'raise SystemExit({code})'], cwd=root, monitor=monitor, host_group=None)
    except RuntimeError:
        if case == 'success':
            raise
    else:
        if case != 'success':
            raise AssertionError('runner must reject failed/rejected child')
    row = run.read(root / 'stage.exit.json')
    assert row['returncode'] == code
    assert row['resources'] == {'per_rank_peak_reserved_bytes': [1024, 2048]}
    assert (row['resource_failure'] is not None) == (case == 'rejected')
    return row


class PrefixRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_real_runner_success_failure_and_rejection(self):
        # Tiny local child processes do not need the remote experiment's 4 GiB reserve.
        with patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)):
            for case in ('success', 'failure', 'rejected'):
                with self.subTest(case=case):
                    runner_fixture(self.root / case, case)

    def schedule_fixture(self):
        config = {'iterations': 30000, 'sh_schedule': {'initial_degree': 0, 'max_degree': 3, 'increase_every_iterations': 1000},
                  'learning_rate': {'position': {'initial': .01, 'final': .001, 'delay_multiplier': .5},
                                    'scaling': .02, 'rotation': .03, 'opacity': .05, 'feature': .1}}
        rows, schedule, noise = [], [], []
        for iteration in (12099, 12100):
            rates = analyze.learning_rates(config, 2., iteration)
            row = {'iteration': iteration, 'batch_view_ids': ['a', 'b'], 'sh_degree': 3}
            if iteration == 12100:
                row['mcmc_health_by_rank'] = [{'rank': r, 'position_lr': rates['means']} for r in (0, 1)]
            rows.append(row)
            schedule.append({'iteration': iteration, 'image_id': 'a', 'sh_degree': 3, 'learning_rates': rates})
            noise.append({'iteration': iteration})
        return config, rows, schedule, noise

    def test_native_lr_schema_without_top_level_lr(self):
        cfg, rows, schedule, noise = self.schedule_fixture()
        result = analyze.verify_schedule(rows, schedule, noise, 0, cfg, 2., {'a', 'b'})
        self.assertEqual(result, {'checked_steps': 2, 'diagnostic_steps': [12100]})
        self.assertTrue(all('position_lr' not in row for row in rows))
        self.assertAlmostEqual(analyze.learning_rates(cfg, 2., 0)['means'], .01)
        self.assertAlmostEqual(analyze.learning_rates(cfg, 2., 30000)['means'], .002)
        for mutation in ('ordinary_rate', 'nested_rate', 'duplicate_owner', 'missing_owner', 'missing_diagnostic', 'sh'):
            r, s = copy.deepcopy(rows), copy.deepcopy(schedule)
            if mutation == 'ordinary_rate':
                s[0]['learning_rates']['means'] *= 1.00000001
            elif mutation == 'nested_rate':
                r[-1]['mcmc_health_by_rank'][0]['position_lr'] *= 1.00000001
            elif mutation == 'duplicate_owner':
                r[-1]['mcmc_health_by_rank'][1]['rank'] = 0
            elif mutation == 'missing_owner':
                r[-1]['mcmc_health_by_rank'].pop()
            elif mutation == 'missing_diagnostic':
                del r[-1]['mcmc_health_by_rank']
            else:
                r[0]['sh_degree'] = s[0]['sh_degree'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze.verify_schedule(r, s, noise, 0, cfg, 2., {'a', 'b'})

    def test_json_report_is_exact_not_tolerant(self):
        original = {'betas': (.9, .999), 'nested': {'count': 2, 'value': 1e-20}}
        self.assertEqual(run.canonical_report(original), run.canonical_report(json.loads(json.dumps(original))))
        for changed in ({'betas': [.9, .999]}, {'betas': [.9, .998], 'nested': original['nested']},
                        {'betas': [.9, .999], 'nested': {'count': 2, 'value': 1.0000000001e-20}}):
            self.assertNotEqual(run.canonical_report(original), run.canonical_report(changed))
        with self.assertRaises(ValueError):
            run.canonical_report({'value': float('nan')})

    def test_fresh_import_help_and_no_training_calls(self):
        code = f"import sys; import {recovery.MODULE}; assert not {{'torch','numpy','gsplat'}} & sys.modules.keys()"
        subprocess.run([sys.executable, '-B', '-c', code], cwd=run.PROJECT, check=True, capture_output=True)
        help_result = subprocess.run([sys.executable, '-B', '-m', recovery.MODULE, '--help'], cwd=run.PROJECT, check=True, capture_output=True, text=True)
        self.assertIn('--authorize-recovery', help_result.stdout)
        self.assertNotIn('--arms', help_result.stdout)
        tree = ast.parse(Path(recovery.__file__).read_text())
        names = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        forbidden = ('train_gaussians', 'backward', 'step', 'manual_seed', 'set_rng_state', '_restore_rng', 'prepare_arms', 'run_phase', 'launch')
        self.assertFalse(any(name.split('.')[-1] in forbidden for name in names))
        for key in ('new_training_updates', 'forward_calls', 'backward_calls', 'optimizer_steps', 'strategy_calls'):
            self.assertEqual(recovery.CONTRACT[key], 0)
        self.assertIsNone(recovery.CONTRACT['original_child_exit_code'])

    def test_new_recovery_root_only(self):
        with patch.object(run, 'RUNS', self.root):
            valid = self.root / 'adam-intervention-recovery-test'
            self.assertEqual(recovery.output_root(valid, True), valid)
            for name in ('adam-intervention-20261029-v2', 'other', 'nested/adam-intervention-recovery-x'):
                with self.assertRaises(ValueError):
                    recovery.output_root(self.root / name, True)
            valid.mkdir()
            with self.assertRaises(ValueError):
                recovery.output_root(valid, True)

    def test_no_prefix_training_for_recovered_root(self):
        with patch.object(run.base.legacy, 'unlimited_host_cgroup') as unit:
            with self.assertRaisesRegex(ValueError, 'cannot be trained again'):
                run.run_phase(SimpleNamespace(output_root=self.root), {'prefix_recovery': {}}, 'prefix')
            unit.assert_not_called()
        self.assertFalse((self.root / 'prefix.once').exists())

    def test_proof_kind_cannot_replace_normal_completion(self):
        with patch.object(run, 'verify_complete', side_effect=ValueError('missing original completion')):
            with self.assertRaises(ValueError):
                run.prefix_proof(self.root, {})
        self.put(self.root / 'recovery-complete.json', {'new_proof': True})
        with patch.object(recovery, 'verify_binding') as verify:
            proof = run.prefix_proof(self.root, {'prefix_recovery': {}})
            verify.assert_called_once()
            self.assertEqual(proof['kind'], 'recovered_payload')
            self.assertEqual(proof['path'], 'recovery-complete.json')

    def test_recovered_prefix_arms_bind_new_proof_and_require_review(self):
        source = {'root': str(recovery.SOURCE), 'commit': recovery.SOURCE_COMMIT, 'protocol_sha256': recovery.SOURCE_SHA,
                  'checkpoint_hash': recovery.CHECKPOINT_HASH, 'full_state_sha256': recovery.STATE_HASHES}
        p = {'prefix_recovery': {**recovery.CONTRACT, 'source': source}, 'implementation_commit': 'c' * 40, 'code_sha256': {}}
        self.put(self.root / 'protocol.json', p)
        self.put(self.root / 'recovery-complete.json', {'separate_recovery_proof': True})
        owners = {str(rank): {'effective_intervention_available': True, 'betas': (.9, .999)} for rank in (0, 1)}
        gate = {'status': 'structural_verified_review_required', 'owners': owners,
                'source_protocol_sha256': run.sha256(self.root / 'protocol.json'), 'checkpoint_hash': recovery.CHECKPOINT_HASH,
                'arms_started': False, 'moment_magnitude_automatically_admitted': False, 'effective_intervention_available': True}
        file = self.put(self.root / 'gate0.json', gate)
        source['gate0_owners'] = owners
        with patch.object(run, 'verify_inputs') as inputs, patch.object(recovery, 'verify_binding') as binding, \
                patch.object(run, 'checkpoint', side_effect=AssertionError('duplicate checkpoint scan')), \
                patch.object(run.analyze, 'gate0', side_effect=AssertionError('duplicate Gate 0 analysis')):
            with self.assertRaises(ValueError):
                run.prepare_arms(self.root, p, run.sha256(file), False)
            run.prepare_arms(self.root, p, run.sha256(file), True)
            inputs.assert_called_once_with(p)
            binding.assert_called_once_with(self.root, p)
            arms = run.verify_arms(self.root, p, run.sha256(self.root / 'arms-protocol.json'))
            self.assertEqual(arms['prefix_proof']['kind'], 'recovered_payload')
            self.assertEqual(binding.call_count, 2)
            inputs.assert_called_once_with(p)
            self.assertFalse((self.root / 'prefix/complete.json').exists())
            self.assertFalse((self.root / 'A').exists())
            self.put(file, {**gate, 'effective_intervention_available': False})
            with self.assertRaises(ValueError):
                run.verify_arms(self.root, p, run.sha256(self.root / 'arms-protocol.json'))

    def test_copy_lineage_does_not_link_or_overwrite(self):
        project = self.root / 'project'
        source = self.put(project / 'source/optimizer.bin', {'frozen': True})
        destination = self.root / 'destination'
        destination.mkdir()
        copies = {'training/attempts/prefix/checkpoints/iteration_000012100/optimizer.bin':
                  {'source': 'source/optimizer.bin', 'sha256': run.sha256(source), 'bytes': source.stat().st_size}}
        before = source.read_bytes()
        with patch.object(recovery, 'PROJECT', project), patch.object(recovery, 'reserve'):
            recovery.copy_lineage(destination, copies)
            recovery.verify_copies(destination, {'copy_files': copies})
            target = destination / next(iter(copies))
            self.assertNotEqual(target.stat().st_ino, source.stat().st_ino)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(target.stat().st_mode & 0o222, 0)
            with self.assertRaises((FileExistsError, PermissionError)):
                recovery.copy_lineage(destination, copies)
        target.chmod(0o644)
        target.write_text('tampered')
        with self.assertRaises(ValueError):
            recovery.verify_copies(destination, {'copy_files': copies})

    def test_source_scan_hashes_payload_once_and_rejects_changes(self):
        source = self.root / 'source'
        protocol = self.put(source / 'protocol.json', {'original': True})
        terminal = self.put(source / 'prefix-execution.json', {'status': 'failed'})
        payload = self.put(source / 'model.bin', {'value': 1})
        files = (protocol, terminal, payload)
        hashes = {str(f.relative_to(self.root)): run.sha256(f) for f in files}
        copies = {'model.bin': {'source': 'source/model.bin', 'sha256': hashes['source/model.bin'], 'bytes': payload.stat().st_size}}
        binding = {'bound_files': hashes, 'copy_files': copies, 'original_failed_terminal_sha256': run.sha256(terminal)}
        with patch.object(recovery, 'SOURCE', source), patch.object(recovery, 'PROJECT', self.root), \
                patch.object(recovery, 'SOURCE_SHA', run.sha256(protocol)), \
                patch.object(recovery, 'source_paths', return_value=files), \
                patch.object(recovery, 'lineage_manifest', return_value=copies) as lineage, \
                patch.object(recovery, 'sha256', wraps=run.sha256) as digest:
            self.assertEqual(recovery.verify_source(binding), {'original': True})
            lineage.assert_called_once_with(hashes)
            self.assertEqual(sum(call.args[0] == payload for call in digest.call_args_list), 1)
            self.put(payload, {'value': 2})
            with self.assertRaisesRegex(ValueError, 'bound source'):
                recovery.verify_source(binding)

    def test_contract_rejects_wrong_source_or_nonzero_budget(self):
        source = {'root': str(recovery.SOURCE), 'commit': recovery.SOURCE_COMMIT, 'protocol_sha256': recovery.SOURCE_SHA,
                  'checkpoint_hash': recovery.CHECKPOINT_HASH, 'full_state_sha256': recovery.STATE_HASHES}
        p = {'prefix_recovery': {**recovery.CONTRACT, 'source': source}}
        recovery.contract(p)
        for field, value in [('new_training_updates', 1), ('original_child_exit_code', 0), ('arms_authorized', True), ('source', {})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.contract({'prefix_recovery': {**p['prefix_recovery'], field: value}})

    def test_historical_code_uses_git_objects(self):
        with patch.object(recovery, 'HISTORICAL_FILES', ('example.py',)), patch.object(recovery, 'PROJECT', self.root), \
                patch.object(recovery.subprocess, 'check_output', return_value=b'old bytes') as git:
            self.put(self.root / 'example.py', {'not': 'the old execution'})
            with self.assertRaises(ValueError):
                recovery.verify_historical_code({'implementation_commit': recovery.SOURCE_COMMIT, 'code_sha256': {'example.py': '0' * 64}})
            self.assertEqual(git.call_args.args[0], ['git', 'show', recovery.SOURCE_COMMIT + ':example.py'])

    def test_recovery_completes_without_original_exit_and_cannot_retry(self):
        from contextlib import ExitStack
        project = self.root / 'project'
        input_file = self.put(project / 'input/model.bin', {'fixture': True})
        destination = self.root / 'adam-intervention-recovery-fixture'
        destination.mkdir()
        copy_name = 'training/attempts/prefix/checkpoints/iteration_000012100/model.bin'
        source = {'root': str(recovery.SOURCE), 'commit': recovery.SOURCE_COMMIT, 'protocol_sha256': recovery.SOURCE_SHA,
            'checkpoint_hash': recovery.CHECKPOINT_HASH, 'full_state_sha256': recovery.STATE_HASHES,
            'copy_files': {copy_name: {'source': 'input/model.bin', 'sha256': run.sha256(input_file), 'bytes': input_file.stat().st_size}},
            'gate0_owners': {'0': {'effective_intervention_available': True}, '1': {'effective_intervention_available': False}}}
        p = {'prefix_recovery': {**recovery.CONTRACT, 'source': source}, 'source_identity': {'native': {}}}
        self.put(destination / 'protocol.json', p)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))}))
            stack.enter_context(patch.dict(recovery.os.environ, {'CUDA_VISIBLE_DEVICES': ''}))
            stack.enter_context(patch.object(recovery, 'PROJECT', project))
            inputs = stack.enter_context(patch.object(recovery, 'verify_inputs'))
            source_check = stack.enter_context(patch.object(recovery, 'verify_source'))
            stack.enter_context(patch.object(recovery, 'reserve'))
            stack.enter_context(patch.object(recovery.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)))
            stack.enter_context(patch.object(run.base.legacy, 'unlimited_host_cgroup'))
            stack.enter_context(patch.object(run.base.connected, 'parent_metadata'))
            stack.enter_context(patch.object(run, 'checkpoint', return_value={'checkpoint_hash': recovery.CHECKPOINT_HASH}))
            stack.enter_context(patch.object(recovery, 'checkpoint_state_digest', side_effect=lambda folder, rank: recovery.STATE_HASHES[str(rank)]))
            stack.enter_context(patch.object(recovery.signal, 'signal'))
            stack.enter_context(patch.object(recovery.signal, 'alarm'))
            before = input_file.read_bytes()
            recovery.recover(destination, p)
            inputs.assert_called_once_with(p)
            source_check.assert_called_once_with(source)
            proof = recovery.verify_binding(destination, p)
            inputs.assert_called_once_with(p)
            self.assertEqual(proof['status'], 'prefix_payload_rebound')
            self.assertIsNone(proof['original_child_exit_code'])
            self.assertEqual(input_file.read_bytes(), before)
            for name in ('prefix', 'A', 'B', 'arms-protocol.json', 'training/attempts/A', 'training/attempts/B'):
                self.assertFalse((destination / name).exists())
            with self.assertRaises(FileExistsError):
                recovery.recover(destination, p)
            original = run.read(destination / 'recovery-complete.json')
            for key, value in [('forward_calls', 1), ('original_child_exit_code', 0), ('original_exit_record_fabricated', True)]:
                self.put(destination / 'recovery-complete.json', {**original, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    recovery.verify_binding(destination, p)
            self.put(destination / 'recovery-complete.json', original)
            self.put(destination / 'gate0.json', {'tampered': True})
            with self.assertRaises(ValueError):
                recovery.verify_binding(destination, p)

    def test_cpu_preflight_requires_current_complete_zero_training_evidence(self):
        evidence = self.root / 'evidence'
        for case, code in (('success', 0), ('failure', 7), ('rejected', 0)):
            self.put(evidence / ('runner-' + case) / 'stage.exit.json', {'returncode': code,
                'resource_failure': 'fixture' if case == 'rejected' else None,
                'resources': {'per_rank_peak_reserved_bytes': [1, 2]}})
        self.put(evidence / 'checkpoint-proof.json', {'lineage_copy_verified': True, 'source_unchanged': True,
            'cuda_initialized': False, 'ranks': {'0': {'matches': True}, '1': {'matches': True}}})
        self.put(evidence / 'formula-proof.json', {'native_formula_exact': True, 'json_exact': True, 'cuda_initialized': False})
        proof = {'profile': recovery.PROFILE, 'status': 'passed', 'implementation_commit': 'c' * 40,
            'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES},
            'flags': dict.fromkeys(recovery.CPU_FLAGS, True), 'new_training_updates': 0, 'forward_calls': 0,
            'backward_calls': 0, 'optimizer_steps': 0, 'cuda_initialized': False, 'evidence_root': str(evidence),
            'evidence_sha256': {str(f.relative_to(evidence)): run.sha256(f) for f in evidence.rglob('*') if f.is_file()}}
        path = self.put(self.root / 'preflight.json', proof)
        with patch.object(recovery, 'RUNS', self.root), patch.object(run, 'execution_commit', return_value='c' * 40):
            self.assertEqual(recovery.verify_cpu_preflight(path)['kind'], 'zero_training_CPU_interface_validation')
            unrelated = {**proof, 'implementation_commit': 'a' * 40,
                'code_sha256': {**proof['code_sha256'], run.REL + '/tests/test_intervention.py': '0' * 64}}
            self.put(path, unrelated)
            self.assertEqual(recovery.verify_cpu_preflight(path)['kind'], 'zero_training_CPU_interface_validation')
            changed = {**proof['code_sha256'], run.REL + '/run.py': '0' * 64}
            missing = {n: digest for n, digest in proof['code_sha256'].items() if n != run.REL + '/prefix_recovery.py'}
            for field, value in [('profile', run.PROFILE), ('code_sha256', changed), ('code_sha256', missing), ('forward_calls', 1),
                                 ('cuda_initialized', True), ('evidence_sha256', {})]:
                self.put(path, {**proof, field: value})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    recovery.verify_cpu_preflight(path)
            self.put(path, proof)
            self.put(evidence / 'runner-failure/stage.exit.json', {'returncode': 0})
            with self.assertRaises(ValueError):
                recovery.verify_cpu_preflight(path)

    def test_original_failure_markers_must_not_be_fabricated(self):
        with patch.object(recovery, 'SOURCE', self.root):
            recovery.missing_original_records()
            self.put(self.root / 'prefix/case.exit.json', {'returncode': 0})
            with self.assertRaises(ValueError):
                recovery.missing_original_records()


if __name__ == '__main__':
    unittest.main()
