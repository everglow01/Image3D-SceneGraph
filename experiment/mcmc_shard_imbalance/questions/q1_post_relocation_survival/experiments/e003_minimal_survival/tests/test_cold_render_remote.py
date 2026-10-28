"""Fresh-process SH3 render preflight with synthetic PNG decoding and zero training."""
from __future__ import annotations

import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_COLD_RENDER_TESTS') != '1':
    pytest.skip('仅显式授权指定远端执行独立冷进程渲染前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e003_minimal_survival import render_recovery as recovery, render_probe  # noqa: E402


def cold_worker(local_rank, rank, world_size, args):
    import random
    import sys
    import numpy as np
    import torch
    from gsplat.cuda._wrapper import _RasterizeToPixels
    from image3d_scenegraph.gaussian import trainer, runtime
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.render import RenderCamera
    assert world_size == 2 and rank == local_rank
    historical = recovery.read(recovery.SOURCE / 'protocol.json')
    recovery.base.prime_package(historical, rank)
    root = Path(args.root)
    source, audit = root / 'fixtures', root / 'audit'
    source.mkdir(exist_ok=True)
    device = torch.device(f'cuda:{rank}')
    points = torch.stack((torch.linspace(-.15, .15, 8, device=device) + .02 * rank,
                          torch.zeros(8, device=device), torch.linspace(2., 2.4, 8, device=device)), -1)
    model = GaussianModel.from_points(points, torch.full((8, 3), .5, device=device),
        torch.full((8,), .02, device=device), initial_opacity=.2, max_sh_degree=3)
    for phase in ('before', 'after'):
        if phase == 'after':
            with torch.no_grad():
                model.params['opacities'].sub_(.2)  # Synthetic second state, not an optimizer/relocation call.
        recovery.snapshots.save_snapshot(source / f'event-{phase}-rank-{rank}', {'parameters': dict(model.params.items())})
    intrinsic = np.array([[52., 0., 32.], [0., 52., 32.], [0., 0., 1.]])
    views = runtime.TrainingViews([runtime._ViewSource(
        RenderCamera(str(i), torch.eye(4), torch.tensor(intrinsic, dtype=torch.float32), 64, 64),
        root / f'{i}.png', intrinsic.copy(), {'state': 'none'}, torch.device('cpu')) for i in (0, 1)])
    cfg = {'loss': {'clamp_render': True, 'l1_weight': .8, 'ssim_weight': .2}}
    calls = []
    original = _RasterizeToPixels.forward
    def forward(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    def forbidden(*a, **kw):
        raise AssertionError('training/backward/optimizer forbidden in cold-render preflight')
    load_model = render_probe.model_from_snapshot
    models = []
    def loaded(*a, **kw):
        value = load_model(*a, **kw)
        models.append((value, recovery.snapshots.digest_state(dict(value.params.items()))))
        return value
    initialize = render_probe.initialize_backend
    def initialize_without_warmup():
        assert not calls and 'gsplat.cuda._backend' not in sys.modules
        value = initialize()
        assert not calls and 'gsplat.cuda._backend' in sys.modules
        return value
    assert 'gsplat.cuda._backend' not in sys.modules
    endpoint = recovery.base.Endpoint(audit, rank, ['0', '1'], lambda n: None)
    with patch.object(_RasterizeToPixels, 'forward', staticmethod(forward)), \
            patch.object(_RasterizeToPixels, 'backward', staticmethod(forbidden)), \
            patch.object(trainer, 'train_gaussians', forbidden), \
            patch.object(torch.optim.Adam, 'step', forbidden), \
            patch.object(GaussianModel, 'optimizers', forbidden), \
            patch.object(render_probe, 'model_from_snapshot', loaded), \
            patch.object(render_probe, 'initialize_backend', initialize_without_warmup), endpoint.installed():
        result = render_probe.probe(source, rank, views, ['0', '1'], cfg, audit_root=audit, sh_degree=3)
    assert len(calls) == result['camera_forwards'] == 4 and result['interpretation_usable']
    assert endpoint.loaded == {str(rank)}
    assert all(recovery.snapshots.digest_state(dict(m.params.items())) == digest and all(p.grad is None for p in m.params.values()) for m, digest in models)
    saved = random.getstate()
    try:
        with pytest.raises(ValueError, match='consumed RNG'):
            with render_probe.audited_rng(audit, rank, 'negative-test'):
                random.random()
    finally:
        random.setstate(saved)  # Clean up the explicit negative fixture, never a measurement.
    negative = audit / f'rng-negative-test-rank-{rank}.json'
    assert recovery.read(negative)['changed_rng_components'] == ['python']
    result.update(cold_before_first_render=True, training_calls=0, negative_rng_guard_passed=True,
                  model_unchanged=True, synthetic_image_decodes=1,
                  negative_rng_audit={'path': negative.name, 'sha256': recovery.sha256(negative)})
    recovery.verify_audit(audit, rank, result)
    recovery.write_new(audit / f'preflight-rank-{rank}.json', result)


def test_fresh_process_first_render(tmp_path):
    import torch
    from PIL import Image
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    name = os.environ.get('IMAGE3D_COLD_RENDER_PREFLIGHT_REPORT')
    if not name:
        pytest.fail('必须指定新版冷进程前置报告')
    path = recovery.snapshots.safe_file(recovery.RUNS, Path(name).absolute().relative_to(recovery.RUNS))
    if path.exists() or not path.parent.is_dir() or not tmp_path.resolve().is_relative_to(recovery.RUNS.resolve()):
        pytest.fail('报告与basetemp必须为本实验新runs目录，不能覆盖')
    recovery.base.legacy.unlimited_host_cgroup(recovery.PREFLIGHT_UNIT)
    with FileLease(recovery.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        assert torch.cuda.device_count() == 2
        recovery.source_identity()
        runtime = {str(r): {'rank': r, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(r)} for r in (0, 1)}
        assert runtime == recovery.base.expected_runtime()
        for i in (0, 1):
            Image.new('RGB', (64, 64), color=(0, 0, 0)).save(tmp_path / f'{i}.png')
        recovery.base.launch(cold_worker, SimpleNamespace(root=str(tmp_path)))
    workers = {str(r): {'path': str(tmp_path / 'audit' / f'preflight-rank-{r}.json'),
                       'sha256': recovery.sha256(tmp_path / 'audit' / f'preflight-rank-{r}.json')} for r in (0, 1)}
    recovery.write_new(path, {'status': 'passed', 'profile': recovery.CONTRACT['profile'],
        'implementation_commit': recovery.execution_commit(), 'runtime': runtime,
        'camera_forwards': 8, 'new_training_updates': 0, 'real_image_decodes': 0,
        'loss_backwards': 0, 'optimizer_steps': 0, 'workers': workers,
        'code_sha256': {n: recovery.sha256(recovery.PROJECT / n) for n in recovery.DEPENDENCIES}})
    recovery.verify_preflight(path)
