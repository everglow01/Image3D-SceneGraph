"""Explicit remote-only structural admission; synthetic data, no production training."""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_OBSERVER_ADMISSION_TESTS') != '1':
    pytest.skip('仅显式授权指定远端执行阶段B结构前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission import admission, cohort_analysis, observer, run_b  # noqa: E402

if str(run_b.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
    raise RuntimeError('前置必须使用指定仓库')


def synthetic_worker(local_rank, rank, world_size, args):
    import torch
    from gsplat.strategy import MCMCStrategy, ops
    from gsplat.cuda._wrapper import _RasterizeToPixels
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.runtime import TrainingView
    from image3d_scenegraph.gaussian.render import RenderCamera
    from scripts.mcmc_lifecycle_observer import digest_state
    from unittest.mock import patch
    assert world_size == 2 and rank == local_rank
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    admission.START, admission.EVENT, admission.END, admission.OWNER_SIZE = 0, 2, 4, 8
    counts = {'training_camera_forwards': 0, 'contribution_forwards': 0, 'loss_backwards': 0,
              'optimizer_group_steps': 0, 'synthetic_updates': 0, 'reference_contribution_forwards': 0}
    cfg = {'loss': {'opacity_regularization': .01, 'scale_regularization': .01},
           'densification': {'start_iteration': 1, 'end_iteration': 5, 'every_iterations': 2},
           'strategy': {'mcmc_min_opacity': .005, 'mcmc_noise_lr': 0.},
           'learning_rate': {'position': {'initial': .00001}, 'scaling': .00001,
                            'rotation': .00001, 'opacity': .00001, 'feature': .00001}}
    camera = RenderCamera(str(rank), torch.eye(4, device=device),
        torch.tensor([[52., 0., 32.], [0., 52., 32.], [0., 0., 1.]], device=device), 64, 64)
    views = [TrainingView(RenderCamera(str(i), camera.camera_from_normalized, camera.intrinsic, 64, 64),
                          torch.zeros((64, 64, 3), device=device)) for i in range(2)]
    sampler_calls = []
    original_sample = ops._multinomial_sample
    original_backward, original_forward = _RasterizeToPixels.backward, _RasterizeToPixels.forward
    backward_calls, forward_calls = [], []
    def sample(*a, **k):
        sampler_calls.append(a[1] if len(a) > 1 else k['n'])
        return original_sample(*a, **k)
    def backward(*a, **k):
        backward_calls.append(1)
        return original_backward(*a, **k)
    def forward(*a, **k):
        forward_calls.append(1)
        return original_forward(*a, **k)
    for name in ('C', 'O'):
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        points = torch.stack((torch.linspace(-.15, .15, 8, device=device) + .03 * rank,
                              torch.zeros(8, device=device), torch.linspace(2., 2.4, 8, device=device)), -1)
        model = GaussianModel.from_points(points, torch.full((8, 3), .5, device=device),
                                          torch.full((8,), .02, device=device), initial_opacity=.2, max_sh_degree=1)
        with torch.no_grad():
            model.params['opacities'][:2] = torch.logit(torch.tensor(.001, device=device))
        optimizers = model.optimizers(cfg['learning_rate'])
        root = Path(args.root) / name
        root.mkdir(exist_ok=True)
        endpoint = observer.Endpoint(root, rank, ['0', '1'], lambda n: None, start=0, end=4, owner_size=8)
        endpoint.loaded.update(('0', '1'))  # Synthetic tensors are supplied directly; no image decoder is called.
        trace = observer.CohortObserver(root, rank, cfg, lambda n: None, window=(0, 2, 4), expected_count=8) if name == 'O' else None
        pure_checks = []
        if trace:
            for method in ('contribution', 'observe_gradients', 'observe_update'):
                original = getattr(trace, method)
                def checked(*a, _original=original, **k):
                    # Observe only the active model/optimizer/gradient/RNG, not the observer's private buffers.
                    def active():
                        return digest_state({'parameters': dict(model.params.items()),
                            'optimizers': {n: o.state_dict() for n, o in optimizers.items()},
                            'gradients': {n: p.grad for n, p in model.params.items()}, 'rng': run_b.snapshots.rng_state()})
                    before = active()
                    result = _original(*a, **k)
                    assert active() == before
                    pure_checks.append(1)
                    return result
                setattr(trace, method, checked)
        handles = []
        def step_count(*unused):
            counts['optimizer_group_steps'] += 1
        for opt in optimizers.values():
            handles.append(opt.register_step_post_hook(step_count))
        rows, health = [], []
        order, cursor = [], 0
        sample_start = len(sampler_calls)
        try:
            with patch.object(ops, '_multinomial_sample', sample), patch.object(_RasterizeToPixels, 'forward', staticmethod(forward)), patch.object(_RasterizeToPixels, 'backward', staticmethod(backward)), endpoint.installed(), trace.installed() if trace else nullcontext():
                from image3d_scenegraph.gaussian.runtime import _ViewSource
                with pytest.raises(ValueError, match='decode forbidden'):
                    _ViewSource.load(SimpleNamespace(camera=SimpleNamespace(image_id='Test-image')))
                strategy = trainer._build_mcmc_strategy(MCMCStrategy, cfg, 8)
                strategy_state = strategy.initialize_state()
                strategy.check_sanity(model.params, optimizers)
                for iteration in range(1, 5):
                    order, cursor, indices = trainer._next_camera_batch(2, order, cursor, 2)
                    for opt in optimizers.values():
                        opt.zero_grad(set_to_none=True)
                    view, rendered = trainer._render_visible_training_view(model, views, indices[rank], 0, distributed=True)
                    counts['training_camera_forwards'] += 1
                    if trace and iteration == 3:
                        from gsplat.rendering import rasterization
                        from image3d_scenegraph.gaussian.render import NEAR_PLANE_NORMALIZED, FAR_PLANE_NORMALIZED
                        with torch.no_grad():
                            values = list(model.activated()[:4])
                            values.append(torch.nn.functional.one_hot(trace.labels + 3 * rank, 6).float())
                            full = []
                            for value in values:
                                gathered = [torch.empty_like(value) for _ in range(2)]
                                torch.distributed.all_gather(gathered, value.contiguous())
                                full.append(torch.cat(gathered))
                            ref, alpha, _ = rasterization(*full, view.camera.camera_from_normalized[None],
                                view.camera.intrinsic[None], width=64, height=64, sh_degree=None, packed=False,
                                backgrounds=None, render_mode='RGB', distributed=False,
                                near_plane=NEAR_PLANE_NORMALIZED, far_plane=FAR_PLANE_NORMALIZED)
                            counts['reference_contribution_forwards'] += 1
                            actual = next(r for r in admission.jsonl(root / f'diagnostic-rank-{rank}.jsonl')
                                          if r.get('event') == 'contribution' and r['iteration'] == iteration)
                            torch.testing.assert_close(torch.tensor(actual['weight_sum'], device=device, dtype=torch.float64),
                                                       ref[0].double().sum((0, 1)), atol=2e-6, rtol=2e-4)
                            from scripts.mcmc_lifecycle_observer import verify_alpha
                            verify_alpha(ref[0], alpha[0], rendered.alpha)
                    strategy.step_pre_backward(model.params, optimizers, strategy_state, iteration, rendered.metadata)
                    regularizers = trainer._mcmc_regularizers(model, cfg)
                    loss = rendered.image.square().mean() + sum(regularizers)
                    (loss / 2).backward()
                    counts['loss_backwards'] += 1
                    model.validate_gradients()
                    for opt in optimizers.values():
                        opt.step()
                    if iteration in (2, 4):
                        health.append({'rank': rank, 'iteration': iteration, 'phase': admission.HEALTH_PHASES[0],
                                       'health': trainer._mcmc_parameter_health(model, cfg)})
                    strategy.step_post_backward(params=model.params, optimizers=optimizers, state=strategy_state,
                        step=iteration, info=rendered.metadata, lr=.00001)
                    if iteration in (2, 4):
                        health.append({'rank': rank, 'iteration': iteration, 'phase': admission.HEALTH_PHASES[1],
                                       'health': trainer._mcmc_parameter_health(model, cfg)})
                    rows.append({'iteration': iteration, 'world_size': 2, 'gaussian_count': 16,
                        'nominal_iterations': 30000, 'requested_stop_iteration': 4, 'mcmc_noise_applied': True,
                        'batch_view_ids': [str(i) for i in indices], 'loss': float(loss.detach()),
                        'reconstruction_loss': float(rendered.image.detach().square().mean()),
                        'opacity_regularization': float(regularizers[0].detach()), 'scale_regularization': float(regularizers[1].detach())})
                    counts['synthetic_updates'] += 1
                endpoint.terminate(model, None, cfg, progress_events=rows)
                raise AssertionError('endpoint did not stop before evaluation')
        except run_b.snapshots.ProbeComplete:
            assert endpoint.completed and endpoint.calls == 1
        finally:
            for handle in handles:
                handle.remove()
        assert len(sampler_calls) == sample_start + 1 and sampler_calls[sample_start] == 2
        if trace:
            info = trace.finish()
            counts['contribution_forwards'] += info['contribution_forwards']
            assert len(pure_checks) == 16
            # Independent reconstruction of first death/recovery from saved bitsets is part of Q1 analysis.
            report = cohort_analysis.cohort(root, rank)
            assert report['groups']['target']['count'] == 2 and not report['cross_run_slot_pairing']
            value = run_b.snapshots.load_snapshot(root / f'cohort-initial-rank-{rank}')
            assert ('labels',) in run_b.snapshots.snapshot_leaves(value)
            with pytest.raises(ValueError, match='conservation'):
                from scripts.mcmc_lifecycle_observer import verify_alpha
                verify_alpha(torch.ones((1, 1, 6), device=device), torch.ones((1, 1, 1), device=device), torch.ones((1, 1, 1), device=device))
        else:
            assert not list(root.glob('*cohort*')) and not list(root.glob('audit-*'))
        assert len(rows) == 4
        torch.distributed.barrier()
    assert len(backward_calls) == 8 and len(forward_calls) == 13
    assert counts == {'training_camera_forwards': 8, 'contribution_forwards': 4, 'loss_backwards': 8,
                      'optimizer_group_steps': 48, 'synthetic_updates': 8, 'reference_contribution_forwards': 1}
    reject = observer.Endpoint(Path(args.root), rank, ['0'], lambda n: None)
    def action():
        if rank == 0:
            raise ValueError('synthetic one-rank rejection')
    with pytest.raises(ValueError, match='prerequisite failed'):
        reject.consensus('synthetic-rejection', action)
    assert (Path(args.root) / f'rejected-synthetic-rejection-rank-{rank}.json').is_file()
    run_b.write_new(Path(args.root) / f'preflight-rank-{rank}.json', {'status': 'passed', 'rank': rank,
        'counts': counts, 'coordinated_rejection_checks': 1, 'real_image_decodes': 0,
        'active_state_unchanged_by_observation': True, 'terminal_before_evaluation': True})


def test_bounded_structural_preflight(tmp_path):
    import torch
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    report_name = os.environ.get('IMAGE3D_OBSERVER_ADMISSION_PREFLIGHT_REPORT')
    if not report_name:
        pytest.fail('必须提供独立的新前置报告路径')
    report_path = run_b.snapshots.safe_file(run_b.RUNS, Path(report_name).absolute().relative_to(run_b.RUNS))
    if report_path.exists() or not report_path.parent.is_dir():
        pytest.fail('前置报告不得覆盖，父目录需预先创建')
    if not tmp_path.resolve().is_relative_to(run_b.RUNS.resolve()):
        pytest.fail('使用本实验runs独立目录作为--basetemp')
    with FileLease(run_b.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        assert torch.cuda.device_count() == 2
        import gsplat
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        from importlib.util import find_spec
        assert run_b.sha256(run_b.phase_a.SOURCE / 'protocol.json') == run_b.phase_a.SOURCE_SHA
        source = run_b.read(run_b.phase_a.SOURCE / 'protocol.json')
        package = Path(source['package'])
        assert Path(gsplat.__file__).resolve().parent == package.resolve()
        assert run_b.legacy.file_hashes(package) == source['package_hashes']
        spec = find_spec('gsplat.csrc')
        assert spec is not None and isinstance(spec.loader, ExtensionFileLoader)
        assert Path(import_module('gsplat.csrc').__file__).resolve() == Path(spec.origin).resolve()
        assert run_b.sha256(run_b.A_ROOT / 'protocol.json') == run_b.A_SHA
        runtime = {str(r): {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(r), 'rank': r} for r in (0, 1)}
        assert runtime == run_b.read(run_b.A_ROOT / 'protocol.json')['source_inventory']['runtime']
        from scripts.mcmc_lifecycle_observer import regularizer_gradient, unchanged_rng
        import random
        saved_rng = random.getstate()
        try:
            with pytest.raises(ValueError, match='consumed RNG'):
                with unchanged_rng():
                    random.random()
        finally:
            random.setstate(saved_rng)  # Clean up only this deliberate negative test, never a training observer.
        for name, shape in (('opacities', (3,)), ('scales', (3, 3))):
            parameter = torch.linspace(-7, 1, 3 if name == 'opacities' else 9, dtype=torch.float64).reshape(shape).requires_grad_()
            activation = parameter.sigmoid() if name == 'opacities' else parameter.exp()
            reference = torch.autograd.grad(.01 * activation.abs().mean() / 2, parameter)[0]
            config = {'loss': {'opacity_regularization': .01, 'scale_regularization': .01}}
            torch.testing.assert_close(regularizer_gradient(name, parameter, config, 2), reference, atol=1e-15, rtol=1e-12)
        run_b.launch(synthetic_worker, SimpleNamespace(root=str(tmp_path)))
    workers = {str(r): run_b.read(tmp_path / f'preflight-rank-{r}.json') for r in (0, 1)}
    assert all(v['status'] == 'passed' for v in workers.values())
    budget = {'synthetic_updates': 8, 'training_camera_forwards': 16, 'contribution_forwards': 8,
              'reference_contribution_forwards': 2, 'loss_backwards': 16, 'analytic_autograd_calls': 2,
              'optimizer_group_steps': 96, 'coordinated_rejection_checks': 1, 'rng_rejection_checks': 1, 'real_image_decodes': 0}
    assert budget == run_b.CONTRACT['preflight_budget']
    run_b.write_new(report_path, {'status': 'passed', 'profile': run_b.CONTRACT['profile'],
        'implementation_commit': run_b.execution_commit(), 'budget': budget, 'runtime': runtime,
        'code_sha256': {n: run_b.sha256(run_b.PROJECT / n) for n in run_b.DEPENDENCIES},
        'workers': {str(r): {'path': str(tmp_path / f'preflight-rank-{r}.json'),
            'sha256': run_b.sha256(tmp_path / f'preflight-rank-{r}.json')} for r in (0, 1)}})
