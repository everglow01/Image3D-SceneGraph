"""Opt-in synthetic two-rank native-checkpoint/Adam preflight, never real training."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_SOURCE_ADAM_TESTS') != '1':
    pytest.skip('仅指定远端显式授权运行来源Adam独立前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import run, analyze  # noqa: E402
from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention.probe import InterventionProbe, moment_digest  # noqa: E402
from scripts.mcmc_lifecycle_observer import digest_state, unchanged_rng  # noqa: E402


def analysis_fixtures():
    import numpy as np
    labels = np.array([1, 0, 0, 1, 0, 2], dtype=np.int8)
    alpha = np.array([.1, .1, .1, .005, .005, .1], dtype=np.float32)
    alive0 = alpha > .005
    alive1 = np.array([False, True, True, False, False, True])
    alive2 = np.array([True, False, True, True, True, False])
    def masks(alive, cutoff):
        return {'alive': {'slot_count': 6, 'value': alive}, 'below_cutoff': {'slot_count': 6, 'value': cutoff}}
    index = {'owner': 0, 'event': 4, 'end': 6, 'initial': masks(alive0, alpha < 1 / 255),
             'rows': [{'iteration': 5, 'phase': 'after_Adam_before_strategy', **masks(alive1, np.array([True, False, False, False, False, False]))},
                      {'iteration': 6, 'phase': 'after_Adam_before_strategy', **masks(alive2, ~alive2)}]}
    cohort = {'owner': 0, 'event': 4, 'labels': labels, 'targets': np.array([1, 2, 4]),
              'sources': np.array([0, 0, 3]), 'initial_alpha': alpha}
    def snapshot(folder):
        if folder.name.startswith('cohort-'):
            return cohort
        if folder.name.startswith('final-opacity'):
            return {'alpha': np.where(alive2, .1, .001).astype(np.float32)}
        return {'logits': np.ones(6, dtype=np.float32), 'gradient': np.ones(6, dtype=np.float32),
                'exp_avg': np.ones(6, dtype=np.float32), 'exp_avg_sq': np.ones(6, dtype=np.float32)}
    with patch.object(analyze, 'state', side_effect=snapshot), patch.object(analyze, 'read', return_value=index), patch.object(analyze.previous, 'bits', side_effect=lambda folder, row: row['value']):
        result = analyze.trajectory(Path('/fixture/prefix'), Path('/fixture/A'), 0, 4, 6, 6)
        last = result['horizons']['2']
        assert last['groups']['S']['at_risk'] == 1 and last['groups']['S']['initial_not_alive'] == 1
        assert last['groups']['S']['risk'] == 1 and last['groups']['S']['currently_dead'] == 0
        assert last['groups']['T']['family_equal_risk'] == .5 and last['target_family_outcomes']['mixed'] == 1
        assert last['groups']['U']['risk'] == 1
        assert result['individual_target_timing_family_equal']['both_source_first'] == .5
        assert result['individual_target_timing_family_equal']['source_only'] == .5
        with patch.object(analyze, 'read', return_value={**index, 'rows': index['rows'][:-1]}), pytest.raises(ValueError):
            analyze.trajectory(Path('/fixture/prefix'), Path('/fixture/A'), 0, 4, 6, 6)
    group = {'betas': (.9, .999), 'lr': .01, 'eps': 1e-15}
    a = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError):
        analyze.adam_prediction(a, a, a, -a, 1, group)


def empty_digest_fixtures(device):
    import numpy as np
    import torch
    with unchanged_rng():
        for target in (torch.device('cpu'), device):
            for shape in ((0,), (0, 3), (0, 4), (0, 1, 3), (0, 15, 3)):
                for value in (torch.from_numpy(np.empty(shape, dtype=np.float32)).to(target),
                              torch.empty_strided(shape, (0,) * len(shape), dtype=torch.float32, device=target)):
                    expected = digest_state(('tensor', str(value.dtype), list(value.shape)))
                    assert moment_digest(value) == expected
            nonempty = torch.arange(6, dtype=torch.float32, device=target).reshape(2, 3).t()
            assert moment_digest(nonempty) == digest_state(nonempty)
            assert moment_digest(nonempty + 1) != moment_digest(nonempty)


def synthetic_worker(local_rank, rank, world_size, args):
    import random
    import numpy as np
    import torch
    from gsplat.strategy import MCMCStrategy, ops
    from image3d_scenegraph.gaussian import trainer, runtime
    from image3d_scenegraph.gaussian.checkpoint import CheckpointProvenance, create_attempt, load_checkpoint
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.render import RenderCamera
    assert world_size == 2 and local_rank == rank
    root, training = Path(args.root), Path(args.root) / 'training'
    run.base.prime_package({'source_identity': args.source}, rank)
    initialization = run.initialize_backend()
    assert initialization['warmup_forward_calls'] == initialization['warmup_backward_calls'] == 0
    run.write_new(root / f'initialization-rank-{rank}.json', initialization)
    device = torch.device(f'cuda:{rank}')
    empty_digest_fixtures(device)
    cfg = {'loss': {'l1_weight': .8, 'ssim_weight': .2, 'clamp_render': True,
                    'opacity_regularization': .01, 'scale_regularization': .01},
           'densification': {'start_iteration': 1, 'end_iteration': 100, 'every_iterations': 4},
           'strategy': {'mcmc_min_opacity': .005, 'mcmc_noise_lr': 1.},
           'learning_rate': {'position': {'initial': .00001, 'final': .00001, 'delay_multiplier': 1.},
                            'scaling': .00001, 'rotation': .00001, 'opacity': .00001, 'feature': .00001}}
    points = torch.stack((torch.linspace(-.15, .15, 8, device=device) + .03 * rank,
                          torch.zeros(8, device=device), torch.linspace(2., 2.4, 8, device=device)), -1)
    model = GaussianModel.from_points(points, torch.full((8, 3), .5, device=device),
                                     torch.full((8,), .02, device=device), initial_opacity=.2, max_sh_degree=3)
    with torch.no_grad():
        if rank == 0:
            model.params['opacities'][:2] = torch.logit(torch.tensor(.001, device=device))
    optimizers = model.optimizers(cfg['learning_rate'])
    # Fixture state, not uncounted optimizer updates. Rank 1 exercises zero local targets.
    for name, opt in optimizers.items():
        p = model.params[name]
        opt.state[p] = {'step': torch.tensor(0.), 'exp_avg': torch.full_like(p, .0001),
                        'exp_avg_sq': torch.full_like(p, .0002)}
    provenance = CheckpointProvenance(*('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64))
    if rank == 0:
        create_attempt(training, attempt_id='seed', kind='fresh', provenance=provenance)
    torch.distributed.barrier()
    strategy = trainer._build_mcmc_strategy(MCMCStrategy, cfg, 8)
    seed_state = strategy.initialize_state()
    trainer._write_streaming_checkpoint(training, attempt_id='seed', iteration=0, provenance=provenance,
        model=model, optimizers=optimizers, strategy_state=seed_state, camera_order=[], camera_cursor=0, history=[],
        world_rank=rank, world_size=2, execution_budget={'nominal_iterations': 30000, 'requested_stop_iteration': 0})
    views = [runtime.TrainingView(RenderCamera(str(i), torch.eye(4, device=device),
        torch.tensor([[52., 0., 32.], [0., 52., 32.], [0., 0., 1.]], device=device), 64, 64),
        torch.zeros((64, 64, 3), device=device)) for i in (0, 1)]
    totals = {'updates': 0, 'forward': 0, 'backward': 0, 'adam': 0}
    proofs = {}
    sample_returns = []
    sample = ops._multinomial_sample
    def sampled(*a, **kw):
        value = sample(*a, **kw)
        sample_returns.append((value, value.detach().clone()))
        return value
    def forbidden(*a, **kw):
        raise AssertionError('native evaluation/export/real training entry is forbidden in this synthetic fixture')
    with patch.object(ops, '_multinomial_sample', sampled), patch.object(trainer, 'train_gaussians', forbidden), patch.object(trainer, 'evaluate_views', forbidden):
        for case in ('prefix', 'A', 'B'):
            parent, start, end = ('seed', 0, 4) if case == 'prefix' else ('prefix', 4, 6)
            if rank == 0:
                create_attempt(training, attempt_id=case, kind='resume', provenance=provenance,
                               parent_attempt_id=parent, resume_iteration=start)
                (root / case).mkdir()
            torch.distributed.barrier()
            probe = InterventionProbe(root / case, rank, case, ['0', '1'], lambda n: None,
                prefix=root / 'prefix', start=0, event=4, end=6, count=8, training=training, provenance=provenance)
            pure_checks = []
            for method in ('capture', 'record_masks', 'optimizer_before', 'optimizer_after'):
                original = getattr(probe, method)
                def pure(*a, _original=original, **kw):
                    before = digest_state({'state': probe.full_state(), 'gradients': {n: p.grad for n, p in probe.params.items()}})
                    result = _original(*a, **kw)
                    assert digest_state({'state': probe.full_state(), 'gradients': {n: p.grad for n, p in probe.params.items()}}) == before
                    pure_checks.append(1)
                    return result
                setattr(probe, method, pure)
            with probe.installed():
                strategy = trainer._build_mcmc_strategy(MCMCStrategy, cfg, 8)
                strategy.check_sanity(model.params, optimizers)
                loaded = load_checkpoint(training, parent, start, expected_provenance=provenance)
                model = trainer._load_model(trainer._checkpoint_rank_bytes(loaded.state.model, rank, 2), device)
                optimizers = model.optimizers(cfg['learning_rate'])
                payload = trainer._torch_load(trainer._checkpoint_rank_bytes(loaded.state.optimizer, rank, 2), device)
                for name, opt in optimizers.items():
                    opt.load_state_dict(payload[name])
                dense = trainer._torch_load(trainer._checkpoint_rank_bytes(loaded.state.densification, rank, 2), device)
                state = dense['strategy_state']
                order, cursor = dense['camera_order'], dense['camera_cursor']
                trainer._restore_rng(trainer._checkpoint_rank_bytes(loaded.state.rng, rank, 2))
                strategy.check_sanity(model.params, optimizers)
                rows = list(loaded.state.metric_history)
                probe.loaded.update(('0', '1'))  # Synthetic tensors, no image files decoded.
                with pytest.raises(ValueError, match='decode forbidden'):
                    runtime._ViewSource.load(SimpleNamespace(camera=SimpleNamespace(image_id='Test-image')))
                with pytest.raises(ValueError, match='decode forbidden'):
                    runtime._ViewSource.load(SimpleNamespace(camera=SimpleNamespace(image_id='Validation-image')))
                with pytest.raises(ValueError, match='checkpoint/export'):
                    trainer.save_model_snapshot(model, root / 'forbidden-export')
                dispatch = ops._multinomial_sample
                def same_return(*a, **kw):
                    value = dispatch(*a, **kw)
                    assert value is sample_returns[-1][0] and torch.equal(value, sample_returns[-1][1])
                    return value
                with patch.object(ops, '_multinomial_sample', same_return):
                    for iteration in range(start + 1, end + 1):
                        order, cursor, indices = trainer._next_camera_batch(2, order, cursor, 2)
                        trainer._update_position_learning_rate(optimizers['means'], cfg, iteration, 30000)
                        for opt in optimizers.values():
                            opt.zero_grad(set_to_none=True)
                        view, rendered = trainer._render_visible_training_view(model, views, indices[rank], 3, distributed=True)
                        strategy.step_pre_backward(model.params, optimizers, state, iteration, rendered.metadata)
                        reconstruction, terms = trainer.l1_ssim_loss(rendered.image.clamp(0, 1), view.image, l1_weight=.8, ssim_weight=.2)
                        regularizers = trainer._mcmc_regularizers(model, cfg)
                        loss = reconstruction + sum(regularizers)
                        (loss / 2).backward()
                        model.validate_gradients()
                        for opt in optimizers.values():
                            opt.step()
                        strategy.step_post_backward(params=model.params, optimizers=optimizers, state=state,
                            step=iteration, info=rendered.metadata, lr=optimizers['means'].param_groups[0]['lr'])
                        rows.append({'iteration': iteration, 'world_size': 2, 'gaussian_count': 16,
                            'nominal_iterations': 30000, 'requested_stop_iteration': end, 'mcmc_noise_applied': True,
                            'batch_view_ids': [str(i) for i in indices], 'loss': float(loss.detach()),
                            'reconstruction_loss': float(reconstruction.detach()), 'opacity_regularization': float(regularizers[0].detach()),
                            'scale_regularization': float(regularizers[1].detach()), **terms})
                with pytest.raises(run.snapshots.ProbeComplete):
                    trainer.evaluate_views(model, None, cfg, progress_events=rows)
            proof = probe.finish()
            run.write_new(root / case / f'worker-rank-{rank}.json', proof)
            assert pure_checks
            totals['updates'] += proof['new_updates']
            totals['forward'] += proof['calls']['raster_forward']
            totals['backward'] += proof['calls']['raster_backward']
            totals['adam'] += proof['calls']['optimizer']
            proofs[case] = proof
    assert totals == {'updates': 8, 'forward': 8, 'backward': 8, 'adam': 48}
    assert proofs['prefix']['calls']['sample'] == int(rank == 0)
    assert proofs['A']['schedule'] == proofs['B']['schedule'] and proofs['A']['noise_schedule'] == proofs['B']['noise_schedule']
    gate = analyze.gate0(root / 'prefix', rank)
    assert gate['target_count'] == (2 if rank == 0 else 0)
    analysis_fixtures()
    original_state = analyze.state
    post_folder = root / 'prefix' / f'post-relocation-rank-{rank}'
    post = original_state(post_folder)
    bad = copy.deepcopy(post)
    bad['adam']['opacities']['state']['exp_avg'] = np.array(bad['adam']['opacities']['state']['exp_avg'], copy=True)
    bad['adam']['opacities']['state']['exp_avg'][0] += 1
    def corrupt(folder):
        return bad if folder == post_folder else original_state(folder)
    with patch.object(analyze, 'state', side_effect=corrupt), pytest.raises(ValueError):
        analyze.gate0(root / 'prefix', rank)
    bad = copy.deepcopy(post)
    bad['parameters']['means'] = np.array(bad['parameters']['means'], copy=True)
    bad['parameters']['means'][0, 0] += 1
    with patch.object(analyze, 'state', side_effect=corrupt), pytest.raises(ValueError):
        analyze.gate0(root / 'prefix', rank)
    for case in ('A', 'B'):
        assert analyze.first_step(root / case, rank)['interpretation_usable']
        survival = analyze.trajectory(root / 'prefix', root / case, rank, 4, 6, 8)
        assert sum(g['count'] for g in survival['horizons']['2']['groups'].values()) == 8
    intervention = run.read(root / 'B' / f'intervention-rank-{rank}.json')
    assert intervention['protected_before'] == intervention['protected_after']
    # Two disposable position-noise calls, not extra training/render/backward calls.
    saved_rng = trainer._rng_bytes()
    before_params = {n: p.detach().clone() for n, p in model.params.items()}
    first_params = {n: p.clone() for n, p in before_params.items()}
    second_params = {n: p.clone() for n, p in before_params.items()}
    try:
        ops.inject_noise_to_position(first_params, {}, {}, scaler=100.)
        trainer._restore_rng(saved_rng)
        ops.inject_noise_to_position(second_params, {'unused_moments': object()}, {}, scaler=100.)
        assert digest_state(first_params) == digest_state(second_params)
    finally:
        trainer._restore_rng(saved_rng)  # Explicit fixture cleanup only, not a measurement guard.
    saved = random.getstate()
    try:
        with pytest.raises(ValueError, match='consumed RNG'):
            with unchanged_rng():
                random.random()
    finally:
        random.setstate(saved)  # Negative fixture cleanup.
    rejection = run.base.Endpoint(root, rank, ['0', '1'], lambda n: None)
    def fail_rank_zero():
        if rank == 0:
            raise ValueError('deliberate coordinated rejection')
    with pytest.raises(ValueError, match='prerequisite failed'):
        rejection.consensus('negative', fail_rank_zero)
    torch.distributed.barrier()
    summary = analyze.compare(root, event=4, end=6, count=8)
    assert all(m['interpretation_usable'] for m in summary['mechanism'].values())
    files = {str(f.relative_to(root)): run.sha256(f) for f in root.rglob('*')
             if f.is_file() and not f.name.startswith('preflight-worker-')}
    run.write_new(root / f'preflight-worker-{rank}.json', {'rank': rank, 'flags': dict.fromkeys(run.FLAGS, True),
        'counts': totals, 'noise_equivalence_fixture_calls': 2, 'evidence_sha256': files})


def test_native_prefix_and_source_adam_arms(tmp_path):
    import torch
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    if str(run.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
        pytest.fail('必须使用指定远端仓库')
    name = os.environ.get('IMAGE3D_SOURCE_ADAM_PREFLIGHT_REPORT')
    if not name:
        pytest.fail('必须指定新的来源Adam前置报告')
    path = run.snapshots.safe_file(run.RUNS, Path(name).absolute().relative_to(run.RUNS))
    if path.exists() or not path.parent.is_dir() or not tmp_path.resolve().is_relative_to(run.RUNS.resolve()):
        pytest.fail('必须使用新runs报告与basetemp，不能覆盖')
    run.base.legacy.unlimited_host_cgroup(run.PREFLIGHT_UNIT)
    with FileLease(run.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        assert torch.cuda.device_count() == 2
        source = run.base.previous.source_identity()
        runtime = {str(r): {'rank': r, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(r)} for r in (0, 1)}
        assert runtime == run.base.expected_runtime()
        run.base.launch(synthetic_worker, SimpleNamespace(root=str(tmp_path), source=source))
    workers = {str(r): {'path': str(tmp_path / f'preflight-worker-{r}.json'),
                       'sha256': run.sha256(tmp_path / f'preflight-worker-{r}.json')} for r in (0, 1)}
    run.write_new(path, {'status': 'passed', 'profile': run.PROFILE, 'implementation_commit': run.execution_commit(),
        'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES}, 'runtime': runtime,
        'counts': run.PREFLIGHT_COUNTS, 'workers': workers})
    run.verify_preflight(path)
