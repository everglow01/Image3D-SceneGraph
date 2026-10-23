"""Explicit remote-only synthetic graph/Adam and two-rank rejection checks."""
from __future__ import annotations

import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_CONNECTED_PROPAGATION_TESTS') != '1':
    pytest.skip('仅明确授权指定远端运行连通传播前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e006_connected_propagation import run, probe, analyze  # noqa: E402

if str(run.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
    raise RuntimeError('远端前置必须位于指定项目目录')


def test_synthetic_injection_graph_and_real_adam_p3_stop(tmp_path):
    import torch
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    if torch.cuda.device_count() != 2:
        pytest.fail('需要正好两张空闲GPU')
    counts = {'synthetic_scalar_forwards': 0, 'loss_backwards': 0, 'injected_returns': 0,
              'optimizer_group_steps': 0, 'real_rasterizer_forwards': 0, 'real_rasterizer_backwards': 0,
              'strategy_posts': 0, 'coordinated_rejection_checks': 0}
    with FileLease(run.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        torch.multiprocessing.spawn(rejection_worker, args=(str(tmp_path), (tmp_path / 'rendezvous').as_uri()), nprocs=2, join=True)
        counts['coordinated_rejection_checks'] += 1
        for rank in (0, 1):
            torch.cuda.set_device(rank)
            device = f'cuda:{rank}'
            shapes = [(1, 2, 2), (1, 2, 3), (1, 2, 3), (1, 2)]
            values = [.25, .5, .75, 1.]
            folder = tmp_path / f'source-{rank}'
            payload = {'outputs': {k: torch.full(shape, v, device=device) for k, shape, v in zip(probe.OUTPUTS[:4], shapes, values)},
                       'rng': run.legacy.rng_state()}
            payload['outputs']['v_backgrounds'] = None
            run.snapshots.save_snapshot(folder, payload)
            ctx = SimpleNamespace(saved_tensors=tuple(torch.zeros(s, device=device) for s in shapes) + (None,) * 6)
            scale = torch.ones((), device=device)
            copied = probe.injected_result(folder, ctx, rank, (scale, None))
            counts['injected_returns'] += 1
            for x, expected in zip(copied[:4], values):
                assert torch.all(x == expected)
            with pytest.raises(ValueError, match='camera rank'):
                probe.injected_result(folder, ctx, 1 - rank, (scale, None))
            with pytest.raises(ValueError, match='aliases'):
                probe.validate_result((*ctx.saved_tensors[:4], *(None,) * 8), ctx, rank, (scale, None))
            with pytest.raises(ValueError, match='shape'):
                probe.validate_result((copied[0].reshape(-1), *copied[1:]), ctx, rank, (scale, None))
            with pytest.raises(ValueError, match='signature'):
                probe.validate_result(copied[:-1], ctx, rank, (scale, None))
            nonfinite = list(copied)
            nonfinite[0] = torch.full_like(copied[0], float('nan'))
            with pytest.raises(ValueError, match='finiteness'):
                probe.validate_result(tuple(nonfinite), ctx, rank, (scale, None))

            class FrozenBoundary(torch.autograd.Function):
                @staticmethod
                def forward(ctx, a, b, c, d, *unused):
                    counts['synthetic_scalar_forwards'] += 1
                    ctx.save_for_backward(a, b, c, d, *(None,) * 6)
                    return a.sum() + b.sum() + c.sum() + d.sum()

                @staticmethod
                def backward(ctx, gradient):
                    counts['injected_returns'] += 1
                    result = probe.injected_result(folder, ctx, rank, (gradient, None))
                    assert all(torch.equal(a, b) for a, b in zip(result[:4], copied[:4]))
                    return result

            params = {name: torch.nn.Parameter(torch.ones(2, device=device)) for name in run.snapshots.PARAMETERS}
            optimizers = {name: torch.optim.Adam([p], lr=.01) for name, p in params.items()}
            inputs = (torch.stack((params['means'], params['quats']), -1)[None],
                      torch.stack([params['scales']] * 3, -1)[None],
                      torch.stack([params['sh0'] + params['shN']] * 3, -1)[None], params['opacities'][None])
            FrozenBoundary.apply(*inputs, *(None,) * 8).backward()
            counts['loss_backwards'] += 1
            expected_gradients = {'means': .25, 'quats': .25, 'scales': 1.5, 'opacities': 1., 'sh0': 2.25, 'shN': 2.25}
            for name, p in params.items():
                torch.testing.assert_close(p.grad, torch.full_like(p, expected_gradients[name]))
            out = tmp_path / f'adam-{rank}'
            (out / f'rank-{rank}').mkdir(parents=True)
            instance = probe.PropagationProbe(out, rank, tmp_path, None, folder, ['synthetic'], expected_count=2)
            instance.params, instance.optimizers, instance.state = params, optimizers, {'synthetic': True}
            instance.camera = {'order': [0, 1], 'cursor': 0}
            instance.stages = ['P0', 'P1', 'P2']
            instance.order = ['P0', 'P1', 'B0', 'B1', 'P2']
            completed = []
            handles = []
            def after(name):
                def hook(*args):
                    completed.append(name)
                    counts['optimizer_group_steps'] += 1
                    if len(completed) == len(run.snapshots.PARAMETERS):
                        instance.capture('P3')
                return hook
            try:
                for name, opt in optimizers.items():
                    handles.append(opt.register_step_post_hook(after(name)))
                with pytest.raises(run.snapshots.ProbeComplete):
                    for opt in optimizers.values():
                        opt.step()
                    counts['strategy_posts'] += 1
            finally:
                for h in handles:
                    h.remove()
            assert completed == list(run.snapshots.PARAMETERS)
            for name, p in params.items():
                g = expected_gradients[name]
                torch.testing.assert_close(p, torch.full_like(p, 1 - .01 * g / (abs(g) + 1e-8)), rtol=1e-6, atol=1e-7)
                assert optimizers[name].state[p]['step'].item() == 1
            run.snapshots.validate_stage_schema(probe.require_finite(out / f'rank-{rank}/P3'), 'P3')
            tensor = next(folder.glob('*.npy'))
            with tensor.open('r+b') as f:
                f.seek(-1, 2)
                value = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([value[0] ^ 1]))
            with pytest.raises(ValueError, match='changed'):
                probe.injected_result(folder, ctx, rank, (scale, None))
    assert counts == run.CONTRACT['preflight_budget']
    run.write_new(tmp_path / 'preflight-counts.json', counts)


def rejection_worker(rank, folder, rendezvous):
    import datetime
    import torch
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                                         timeout=datetime.timedelta(seconds=30))
    try:
        p = probe.PropagationProbe(Path(folder), rank, Path(folder), None, None, ['synthetic'])
        def reject_one():
            if rank == 0:
                raise ValueError('synthetic source mismatch')
        with pytest.raises(ValueError, match='failed on a rank'):
            p.consensus('B0', reject_one)
        assert (Path(folder) / f'rejected-B0-rank-{rank}.json').is_file()
    finally:
        torch.distributed.destroy_process_group()


def test_cpu_reference_updates_and_opacity_threshold():
    import numpy as np
    a, b = np.array([1., 1.001, 0.]), np.array([1., 1., 0.])
    report = run.snapshots.compare_arrays(a, b)
    assert report['num_mismatch'] == int((~np.isclose(a, b, atol=2e-6, rtol=2e-4)).sum())
    logits = np.array([-1000., 0., 1000., np.log(.005 / .995)])
    actual = analyze.opacity(logits)
    np.testing.assert_allclose(actual, [0., .5, 1., .005], rtol=1e-14, atol=1e-16)
    assert np.isfinite(actual).all()
    # Subtract in float64: compare update differences, not merely large parameter values.
    before = np.array([1000.], dtype=np.float32)
    x, y = before + np.float32(.001), before + np.float32(.002)
    assert run.snapshots.compare_arrays(x, y)['num_mismatch'] == 0
    assert run.snapshots.compare_arrays(x.astype(float) - before, y.astype(float) - before)['num_mismatch'] == 1
