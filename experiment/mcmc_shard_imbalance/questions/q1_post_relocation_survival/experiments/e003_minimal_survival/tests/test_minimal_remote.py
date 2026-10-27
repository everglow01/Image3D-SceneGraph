"""Explicit remote-only synthetic preflight. Never enable on a developer laptop."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
import inspect
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_MINIMAL_SURVIVAL_TESTS') != '1':
    pytest.skip('仅显式授权指定远端执行Q1最小观测前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e003_minimal_survival import run, analyze, render_probe  # noqa: E402
from scripts.mcmc_lifecycle_observer import digest_state, unchanged_rng, reuse_record  # noqa: E402

if str(run.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
    raise RuntimeError('前置必须使用指定仓库')


def synthetic_worker(local_rank, rank, world_size, args):
    import random
    import numpy as np
    import torch
    from gsplat.strategy import MCMCStrategy, ops
    from gsplat.cuda._wrapper import _RasterizeToPixels
    from image3d_scenegraph.gaussian import trainer, runtime
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.render import RenderCamera
    assert world_size == 2 and rank == local_rank
    run.prime_package({'source_identity': args.source}, rank)
    device = torch.device(f'cuda:{rank}')
    root = Path(args.root)
    cfg = {'loss': {'l1_weight': .8, 'ssim_weight': .2, 'clamp_render': True,
                    'opacity_regularization': .01, 'scale_regularization': .01},
           'densification': {'start_iteration': 1, 'end_iteration': 5, 'every_iterations': 2},
           'strategy': {'mcmc_min_opacity': .005, 'mcmc_noise_lr': 0.},
           'learning_rate': {'position': {'initial': .00001}, 'scaling': .00001,
                            'rotation': .00001, 'opacity': .00001, 'feature': .00001}}
    views = [runtime.TrainingView(RenderCamera(str(i), torch.eye(4, device=device),
                torch.tensor([[52., 0., 32.], [0., 52., 32.], [0., 0., 1.]], device=device), 64, 64),
                torch.zeros((64, 64, 3), device=device)) for i in range(2)]
    counts = dict.fromkeys(run.COUNTS, 0)
    forward_calls, backward_calls, sampling_calls = [], [], []
    original_forward, original_backward, original_sample = _RasterizeToPixels.forward, _RasterizeToPixels.backward, ops._multinomial_sample
    native_sequences, observation_collectives = {}, []
    sequence, sampled_returns = [], []

    def forward(*a, **kw):
        forward_calls.append(1)
        return original_forward(*a, **kw)

    def backward(*a, **kw):
        backward_calls.append(1)
        return original_backward(*a, **kw)

    def sample(*a, **kw):
        result = original_sample(*a, **kw)
        sampling_calls.append(a[1] if len(a) > 1 else kw['n'])
        sampled_returns.append((result, result.detach().clone()))
        return result

    def collective(name, original):
        def called(*a, **kw):
            frame = inspect.currentframe().f_back
            own = frame.f_code.co_name == 'consensus' and frame.f_code.co_filename.endswith('/e006_connected_propagation/probe.py')
            if own:
                observation_collectives.append(name)
            else:
                sequence.append(name)
            return original(*a, **kw)
        return called

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(_RasterizeToPixels, 'forward', staticmethod(forward)))
        hooks.enter_context(patch.object(_RasterizeToPixels, 'backward', staticmethod(backward)))
        hooks.enter_context(patch.object(ops, '_multinomial_sample', sample))
        for name in ('broadcast', 'all_reduce', 'all_gather', 'all_gather_object', 'all_to_all', 'all_to_all_single', 'barrier'):
            hooks.enter_context(patch.object(torch.distributed, name, collective(name, getattr(torch.distributed, name))))
        for name in ('C', 'T'):
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            torch.cuda.manual_seed_all(7)
            points = torch.stack((torch.linspace(-.15, .15, 8, device=device) + .03 * rank,
                                  torch.zeros(8, device=device), torch.linspace(2., 2.4, 8, device=device)), -1)
            model = GaussianModel.from_points(points, torch.full((8, 3), .5, device=device),
                torch.full((8,), .02, device=device), initial_opacity=.2, max_sh_degree=1)
            with torch.no_grad():
                model.params['opacities'][:2] = torch.logit(torch.tensor(.001, device=device))
            optimizers = model.optimizers(cfg['learning_rate'])
            stage = root / name
            stage.mkdir(exist_ok=True)
            endpoint = run.Endpoint(stage, rank, ['0', '1'], lambda n: None, start=0, end=4, owner_size=8)
            endpoint.loaded.update(('0', '1'))  # Direct synthetic tensors; no image decoder.
            observer = run.MinimalObserver(stage, rank, lambda n: None, window=(0, 2, 4), owner_size=8) if name == 'T' else None
            state = None

            def active():
                return digest_state({'parameters': dict(model.params.items()),
                    'optimizers': {n: o.state_dict() for n, o in optimizers.items()},
                    'gradients': {n: p.grad for n, p in model.params.items()}, 'strategy': state, 'rng': run.snapshots.rng_state()})

            pure_checks = []
            if observer:
                for method in ('before_post', 'before_relocation', 'record_sample', 'after_relocation'):
                    original = getattr(observer, method)
                    def checked(*a, _original=original, **kw):
                        before = active()
                        result = _original(*a, **kw)
                        assert active() == before
                        pure_checks.append(1)
                        return result
                    setattr(observer, method, checked)
            handles = []
            def stepped(*unused):
                counts['optimizer_group_steps'] += 1
            for opt in optimizers.values():
                handles.append(opt.register_step_post_hook(stepped))
            rows, health, sequence = [], [], []
            order, cursor = [], 0
            start_sample = len(sampling_calls)
            try:
                with endpoint.installed(), observer.installed() if observer else nullcontext():
                    with pytest.raises(ValueError, match='decode forbidden'):
                        runtime._ViewSource.load(SimpleNamespace(camera=SimpleNamespace(image_id='Test-image')))
                    with pytest.raises(ValueError, match='decode forbidden'):
                        runtime._ViewSource.load(SimpleNamespace(camera=SimpleNamespace(image_id='Validation-image')))
                    strategy = trainer._build_mcmc_strategy(MCMCStrategy, cfg, 8)
                    state = strategy.initialize_state()
                    strategy.check_sanity(model.params, optimizers)
                    # Verify the observer returns the exact original sampling tensor, not a replacement.
                    dispatched = ops._multinomial_sample
                    def check_sample(*a, **kw):
                        value = dispatched(*a, **kw)
                        assert value is sampled_returns[-1][0] and torch.equal(value, sampled_returns[-1][1])
                        return value
                    with patch.object(ops, '_multinomial_sample', check_sample):
                        for iteration in range(1, 5):
                            order, cursor, indices = trainer._next_camera_batch(2, order, cursor, 2)
                            for opt in optimizers.values():
                                opt.zero_grad(set_to_none=True)
                            _, rendered = trainer._render_visible_training_view(model, views, indices[rank], 0, distributed=True)
                            counts['training_camera_forwards'] += 1
                            strategy.step_pre_backward(model.params, optimizers, state, iteration, rendered.metadata)
                            regularizers = trainer._mcmc_regularizers(model, cfg)
                            loss = rendered.image.square().mean() + sum(regularizers)
                            (loss / 2).backward()
                            counts['loss_backwards'] += 1
                            model.validate_gradients()
                            for opt in optimizers.values():
                                opt.step()
                            if iteration in (2, 4):
                                health.append({'iteration': iteration, 'phase': 'before', 'health': trainer._mcmc_parameter_health(model, cfg)})
                            strategy.step_post_backward(params=model.params, optimizers=optimizers, state=state,
                                step=iteration, info=rendered.metadata, lr=.00001)
                            if iteration in (2, 4):
                                health.append({'iteration': iteration, 'phase': 'after', 'health': trainer._mcmc_parameter_health(model, cfg)})
                            rows.append({'iteration': iteration, 'world_size': 2, 'gaussian_count': 16,
                                'nominal_iterations': 30000, 'requested_stop_iteration': 4, 'mcmc_noise_applied': True,
                                'batch_view_ids': [str(i) for i in indices], 'loss': float(loss.detach()),
                                'reconstruction_loss': float(rendered.image.detach().square().mean()),
                                'opacity_regularization': float(regularizers[0].detach()), 'scale_regularization': float(regularizers[1].detach())})
                            counts['synthetic_updates'] += 1
                    endpoint.terminate(model, None, cfg, progress_events=rows)
                    raise AssertionError('endpoint crossed evaluation')
            except run.snapshots.ProbeComplete:
                assert endpoint.completed and endpoint.calls == 1
            finally:
                for handle in handles:
                    handle.remove()
            native_sequences[name] = sequence.copy()
            assert len(sampling_calls) == start_sample + 1 and sampling_calls[start_sample] == 2
            if observer:
                assert observer.finish()['posts'] == 4 and len(pure_checks) == 7
                result = analyze.cohort(stage, rank, event=2, end=4, count=8)
                assert result['groups']['target']['count'] == 2
                assert result['initial_dead_count'] == health[1]['health']['dead_count']
                assert result['final_dead_count'] == health[2]['health']['dead_count']
                # Empty-cohort record fixture, without another strategy/optimizer call.
                empty_root = root / 'empty'
                empty_root.mkdir(exist_ok=True)
                empty = run.MinimalObserver(empty_root, rank, lambda n: None, window=(0, 2, 4), owner_size=8)
                assert bool((model.params['opacities'].sigmoid() > .005).all())
                before = active()
                empty.params, empty.iteration, empty.posts = model.params, 2, 2
                empty.before_relocation()
                empty.after_relocation(0)
                for step in (3, 4):
                    empty.before_post(model.params, step)
                    empty.posts += 1
                assert active() == before
                zero = analyze.cohort(empty_root, rank, event=2, end=4, count=8)
                assert zero['groups']['target']['count'] == zero['groups']['source']['count'] == 0
                assert zero['groups']['target']['redeath_fraction_of_initially_alive'] is None
                assert zero['groups']['uninvolved']['count'] == 8
                assert reuse_record(8, [], [])['unique_source_count'] == 0
            else:
                assert not list(stage.glob('*cohort*')) and not list(stage.glob('*event-before*'))
        assert native_sequences['C'] == native_sequences['T'] and observation_collectives
        assert len(forward_calls) == len(backward_calls) == 8
        loaded_models = []
        load = render_probe.model_from_snapshot
        def capture_model(folder, device):
            value = load(folder, device)
            loaded_models.append((value, digest_state(dict(value.params.items()))))
            return value
        active_before = active()
        with patch.object(render_probe, 'model_from_snapshot', capture_model):
            rendered = render_probe.probe(root / 'T', rank, views, ['0', '1'], cfg, sh_degree=0)
        assert active() == active_before
        assert all(digest_state(dict(m.params.items())) == digest and all(p.grad is None for p in m.params.values()) for m, digest in loaded_models)
        assert rendered['camera_forwards'] == 4 and rendered['interpretation_usable']
        counts['isolated_render_forwards'] = rendered['camera_forwards']
        assert len(forward_calls) == 12 and len(backward_calls) == 8
    assert counts == run.COUNTS
    # Prove the same state guard rejects a deliberate write to this disposable synthetic model.
    original_value = model.params['means'][0, 0].detach().clone()
    try:
        with pytest.raises(AssertionError):
            baseline = active()
            with torch.no_grad():
                model.params['means'][0, 0].add_(1.)
            assert active() == baseline
    finally:
        with torch.no_grad():
            model.params['means'][0, 0].copy_(original_value)
    assert active() == baseline
    rejected = run.Endpoint(root, rank, ['0'], lambda n: None)
    def fail_one_rank():
        if rank == 0:
            raise ValueError('synthetic single-rank rejection')
    with pytest.raises(ValueError, match='prerequisite failed'):
        rejected.consensus('synthetic-rejection', fail_one_rank)
    assert (root / f'rejected-synthetic-rejection-rank-{rank}.json').is_file()
    saved = random.getstate()
    try:
        with pytest.raises(ValueError, match='consumed RNG'):
            with unchanged_rng():
                random.random()
    finally:
        random.setstate(saved)  # Only this deliberate negative fixture is restored.
    run.write_new(root / f'preflight-rank-{rank}.json', {'status': 'passed', 'rank': rank,
        'counts': counts, 'invariants': dict.fromkeys(run.FLAGS, True), 'real_image_decodes': 0,
        'native_collective_sequence': native_sequences, 'observation_collectives': observation_collectives})


def test_bounded_minimal_preflight(tmp_path):
    import torch
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    report_name = os.environ.get('IMAGE3D_MINIMAL_SURVIVAL_PREFLIGHT_REPORT')
    if not report_name:
        pytest.fail('必须指定独立新版前置报告路径')
    report = run.snapshots.safe_file(run.RUNS, Path(report_name).absolute().relative_to(run.RUNS))
    if report.exists() or not report.parent.is_dir() or not tmp_path.resolve().is_relative_to(run.RUNS.resolve()):
        pytest.fail('前置报告与basetemp必须位于独立新runs目录，不得覆盖')
    run.legacy.unlimited_host_cgroup(run.PREFLIGHT_UNIT)
    with FileLease(run.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        assert torch.cuda.device_count() == 2
        source = run.previous.source_identity()
        runtime = {str(r): {'rank': r, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(r)} for r in (0, 1)}
        assert runtime == run.expected_runtime()
        run.launch(synthetic_worker, SimpleNamespace(root=str(tmp_path), source=source))
    workers = {str(r): {'path': str(tmp_path / f'preflight-rank-{r}.json'),
                       'sha256': run.sha256(tmp_path / f'preflight-rank-{r}.json')} for r in (0, 1)}
    assert all(run.read(Path(v['path']))['status'] == 'passed' for v in workers.values())
    run.write_new(report, {'status': 'passed', 'profile': run.PROFILE, 'implementation_commit': run.execution_commit(),
        'budget': run.CONTRACT['preflight_budget'], 'runtime': runtime, 'workers': workers,
        'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES}})
    run.verify_preflight(report)
