"""Opt-in CPU-only recovery preflight: native files and real runner, zero model calls."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import socket
from unittest.mock import patch

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_PREFIX_RECOVERY_CPU_TESTS') != '1':
    pytest.skip('仅指定远端显式授权执行零训练CPU恢复前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import run, analyze, prefix_recovery as recovery  # noqa: E402
from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention.tests.test_prefix_recovery import runner_fixture  # noqa: E402


def test_cpu_recovery_checkpoint_and_interfaces(tmp_path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or str(run.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        pytest.fail('CPU恢复前置必须隐藏CUDA并使用指定仓库')
    name = os.environ.get('IMAGE3D_PREFIX_RECOVERY_CPU_PREFLIGHT_REPORT')
    if not name:
        pytest.fail('必须指定新的CPU前置报告')
    report = run.snapshots.safe_file(run.RUNS, Path(name).absolute().relative_to(run.RUNS))
    if report.exists() or not report.parent.is_dir() or not tmp_path.resolve().is_relative_to(run.RUNS.resolve()):
        pytest.fail('报告与basetemp必须为新runs目录')
    run.base.legacy.unlimited_host_cgroup(recovery.PREFLIGHT_UNIT)
    import random
    import numpy as np
    import torch
    from image3d_scenegraph.gaussian import trainer, training_math
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance, CheckpointState, create_attempt, write_checkpoint, load_checkpoint
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise AssertionError('model/optimizer/training/CUDA/RNG restore forbidden in CPU recovery preflight')
    frozen_inputs = [recovery.SOURCE / 'protocol.json', recovery.SOURCE / 'prefix-execution.json']
    frozen_inputs += [recovery.SOURCE / 'prefix' / f'{name}-rank-{rank}.json' for name in ('worker', 'native-tail', 'prefix-state') for rank in (0, 1)]
    frozen_hashes = {str(f): run.sha256(f) for f in frozen_inputs}
    with ExitStack() as stack:
        for obj, attr in ((trainer, 'train_gaussians'), (trainer, '_restore_rng'), (GaussianModel, '__init__'),
                          (torch.Tensor, 'backward'), (torch.optim.Adam, 'step'), (torch.cuda, '_lazy_init')):
            stack.enter_context(patch.object(obj, attr, forbidden))
        for case in ('success', 'failure', 'rejected'):
            runner_fixture(tmp_path / ('runner-' + case), case)
        provenance = CheckpointProvenance(*('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64))
        original = tmp_path / 'original'
        destination = tmp_path / 'destination'
        destination.mkdir()
        expected = {}
        for attempt, iteration in (('train-001', 12000), ('prefix', 12100)):
            create_attempt(original, attempt_id=attempt, kind='fresh' if attempt == 'train-001' else 'resume',
                provenance=provenance, parent_attempt_id=None if attempt == 'train-001' else 'train-001',
                resume_iteration=None if attempt == 'train-001' else 12000)
            components = {n: [] for n in ('model', 'optimizer', 'densification', 'rng')}
            for rank in (0, 1):
                means = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10 + rank
                coeffs = torch.arange(384, dtype=torch.float32).reshape(8, 16, 3) / 1000
                model = {'max_sh_degree': 3, 'state_dict': {'means': means, 'log_scales': torch.full((8, 3), -3.),
                    'quats': torch.ones(8, 4), 'opacity_logits': torch.full((8,), -.5), 'sh_coeffs': coeffs}}
                s = model['state_dict']
                params = {'means': means, 'scales': s['log_scales'], 'quats': s['quats'], 'opacities': s['opacity_logits'],
                          'sh0': coeffs[:, :1], 'shN': coeffs[:, 1:]}
                optim, adam = {}, {}
                for key, value in params.items():
                    state = {'step': torch.tensor(float(iteration)), 'exp_avg': torch.full_like(value, .01), 'exp_avg_sq': torch.full_like(value, .02)}
                    group = {'lr': .05, 'betas': (.9, .999), 'eps': 1e-15, 'weight_decay': 0, 'amsgrad': False}
                    optim[key] = {'state': {0: state}, 'param_groups': [{**group, 'params': [0]}]}
                    adam[key] = {'state': state, 'param_groups': [group]}
                dense = {'strategy_state': {'binoms': torch.ones(2, 2)}, 'camera_order': [1, 0], 'camera_cursor': 2}
                rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                       'cuda': [torch.tensor([rank, 1], dtype=torch.uint8)]}  # Opaque synthetic bytes, never restored.
                if attempt == 'prefix':
                    expected[str(rank)] = run.snapshots.digest_state({'parameters': params, 'adam': adam,
                        'strategy': dense['strategy_state'], 'camera': {'order': [1, 0], 'cursor': 2}, 'rng': rng,
                        'optimizer_classes': dict.fromkeys(run.snapshots.PARAMETERS, 'torch.optim.adam.Adam')})
                for field, value in (('model', model), ('optimizer', optim), ('densification', dense), ('rng', rng)):
                    components[field].append(trainer._torch_bytes(value))
            payload = {k: trainer._pack_checkpoint_shards(v, 2) for k, v in components.items()}
            write_checkpoint(original, attempt_id=attempt, iteration=iteration, purpose='final', provenance=provenance,
                state=CheckpointState(**payload, scheduler=run.json.dumps({'iteration': iteration, 'nominal_iterations': 30000,
                    'requested_stop_iteration': iteration}).encode(), metric_history=({'iteration': iteration},)))
        before = {str(f.relative_to(run.PROJECT)): run.sha256(f) for f in original.rglob('*') if f.is_file()}
        copies = {str(Path('training') / f.relative_to(original)):
                  {'source': str(f.relative_to(run.PROJECT)), 'sha256': run.sha256(f), 'bytes': f.stat().st_size}
                  for f in original.rglob('*') if f.is_file()}
        recovery.copy_lineage(destination, copies)
        recovery.verify_copies(destination, {'copy_files': copies})
        loaded = load_checkpoint(destination / 'training', 'prefix', 12100, expected_provenance=provenance)
        assert loaded.record.checkpoint_hash == load_checkpoint(original, 'prefix', 12100, expected_provenance=provenance).record.checkpoint_hash
        create_attempt(destination / 'training', attempt_id='lineage-check-only', kind='resume', provenance=provenance,
                       parent_attempt_id='prefix', resume_iteration=12100)  # Descriptor only; no training entry.
        ranks = {}
        for rank in (0, 1):
            value = recovery.checkpoint_state_digest(destination / 'training/attempts/prefix/checkpoints/iteration_000012100', rank, count=8)
            assert value == expected[str(rank)]
            ranks[str(rank)] = {'matches': True, 'digest': value}
        assert all(run.sha256(run.PROJECT / n) == h for n, h in before.items())
        run.write_new(tmp_path / 'checkpoint-proof.json', {'ranks': ranks, 'lineage_copy_verified': True,
            'source_unchanged': True, 'cuda_initialized': torch.cuda.is_initialized()})
        assert run.sha256(recovery.SOURCE / 'protocol.json') == recovery.SOURCE_SHA
        source = run.read(recovery.SOURCE / 'protocol.json')
        dataset = run.read(Path(source['source_identity']['native']['pair']['replay']) / 'dataset.json')
        cfg = source['source_identity']['config']['effective_config']
        radius = analyze.scene_scale(dataset)
        assert radius == trainer._training_scene_scale(dataset)
        for rank in (0, 1):
            rows = run.read(recovery.SOURCE / 'prefix' / f'native-tail-rank-{rank}.json')
            worker = run.read(recovery.SOURCE / 'prefix' / f'worker-rank-{rank}.json')
            result = analyze.verify_schedule(rows, worker['schedule'], worker['noise_schedule'], rank, cfg, radius, set(map(str, dataset['splits']['train'])))
            assert result['checked_steps'] == 100
            for iteration in (12001, 12050, 12100):
                lr = cfg['learning_rate']['position']
                native = radius * training_math.exponential_learning_rate(float(lr['initial']), float(lr['final']), iteration,
                    int(cfg['iterations']), delay_multiplier=float(lr['delay_multiplier']))
                assert analyze.learning_rates(cfg, radius, iteration)['means'] == native
        original_json = {'betas': (.9, .999), 'value': 1e-30}
        assert run.canonical_report(original_json) == run.canonical_report(run.json.loads(run.json.dumps(original_json)))
        assert run.canonical_report(original_json) != run.canonical_report({**original_json, 'value': 1.0000000001e-30})
        run.write_new(tmp_path / 'formula-proof.json', {'native_formula_exact': True, 'json_exact': True, 'cuda_initialized': False})
    assert not torch.cuda.is_initialized()
    assert all(run.sha256(Path(name)) == digest for name, digest in frozen_hashes.items())
    run.write_new(tmp_path / 'original-inputs-proof.json', frozen_hashes)
    evidence = {str(f.relative_to(tmp_path)): run.sha256(f) for f in tmp_path.rglob('*') if f.is_file()}
    run.write_new(report, {'profile': recovery.PROFILE, 'status': 'passed', 'implementation_commit': run.execution_commit(),
        'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES}, 'flags': dict.fromkeys(recovery.CPU_FLAGS, True),
        'new_training_updates': 0, 'forward_calls': 0, 'backward_calls': 0, 'optimizer_steps': 0, 'cuda_initialized': False,
        'evidence_root': str(tmp_path), 'evidence_sha256': evidence})
    recovery.verify_cpu_preflight(report)
