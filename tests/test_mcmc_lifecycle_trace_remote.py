"""Explicitly opt-in remote tensor/CUDA checks; never import Torch on a local machine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import socket
from types import SimpleNamespace

import pytest

if socket.gethostname() != "i-94B8D131" or os.environ.get("IMAGE3D_AUTHORIZE_TRACE_TESTS") != "1":
    pytest.skip("仅显式授权远端运行张量/CUDA检查；本机不导入Torch", allow_module_level=True)

from scripts import mcmc_lifecycle_observer as observer  # noqa: E402


def test_regularizer_separation_against_independent_autograd():
    import torch
    config = {"loss": {"opacity_regularization": .01, "scale_regularization": .01}}
    for name, shape in (("opacities", (4,)), ("scales", (4, 3))):
        param = torch.linspace(-7, 1, torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape).requires_grad_()
        activated = param.sigmoid() if name == "opacities" else param.exp()
        image = (activated * torch.linspace(.1, .7, param.numel()).reshape(shape)).square().mean()
        regularizer = .01 * activated.abs().mean()
        total = torch.autograd.grad((image + regularizer) / 2, param, retain_graph=True)[0]
        expected_image = torch.autograd.grad(image / 2, param, retain_graph=True)[0]
        expected_regularizer = torch.autograd.grad(regularizer / 2, param)[0]
        analytic = observer.regularizer_gradient(name, param, config, 2)
        torch.testing.assert_close(analytic, expected_regularizer, atol=1e-15, rtol=1e-12)
        torch.testing.assert_close(total - analytic, expected_image, atol=1e-15, rtol=1e-12)


def test_canonical_tensor_state_and_rng_guard():
    import torch
    values = torch.tensor([[1., 2.], [3., 4.]])
    assert observer.digest_state(values) == observer.digest_state(values.t().contiguous().t())
    assert observer.digest_state(values) != observer.digest_state(values.double())
    assert observer.digest_state(values) != observer.digest_state(values.flatten())
    before = random.getstate()
    try:
        with pytest.raises(ValueError, match="consumed RNG"):
            with observer.unchanged_rng():
                random.random()
    finally:
        random.setstate(before)


def test_alpha_conservation_rejects_wrong_owner_mass():
    import torch
    weights = torch.tensor([[[.1, .2, 0., 0., .3, 0.]]])
    alpha = torch.tensor([[[.6]]])
    observer.verify_alpha(weights, alpha, alpha)
    with pytest.raises(ValueError, match="conservation"):
        observer.verify_alpha(weights * 2, alpha, alpha)


def test_sampling_wrapper_calls_real_sampler_once_and_keeps_rng(tmp_path, monkeypatch):
    import torch
    from gsplat.strategy import ops
    from image3d_scenegraph.gaussian import trainer

    def fake_build(*args, **kwargs):
        return SimpleNamespace(step_pre_backward=lambda *a, **k: None,
                               step_post_backward=lambda *a, **k: None, _relocate_gs=lambda *a: 0)
    monkeypatch.setattr(trainer, "_build_mcmc_strategy", fake_build)
    original = ops._multinomial_sample
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(ops, "_multinomial_sample", counted)
    trace = observer.LifecycleObserver(tmp_path, 0, {}, False, window=(0, 1, 0), expected_count=4)
    trace.sample_context = (torch.tensor([0, 2]), torch.tensor([1, 3]))
    torch.manual_seed(7)
    expected = original(torch.tensor([.3, .7]), 2, replacement=True)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(7)
    with trace.installed():
        actual = ops._multinomial_sample(torch.tensor([.3, .7]), 2, replacement=True)
    assert calls == [1]
    assert torch.equal(actual, expected) and torch.equal(torch.get_rng_state(), expected_rng)


def contribution_worker(local_rank, rank, world_size, args):
    import torch
    from gsplat.rendering import rasterization
    from image3d_scenegraph.gaussian.model import GaussianModel
    from image3d_scenegraph.gaussian.render import RenderCamera, render_gaussians, NEAR_PLANE_NORMALIZED, FAR_PLANE_NORMALIZED

    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    z = torch.linspace(2., 2.75, 16, device=device)
    points = torch.stack((torch.linspace(-.15, .15, 16, device=device), torch.zeros_like(z), z), -1)
    models = [GaussianModel.from_points(points[r * 8:(r + 1) * 8], torch.full((8, 3), .5, device=device),
              torch.full((8,), .02, device=device), initial_opacity=.3, max_sh_degree=0) for r in (0, 1)]
    model = models[rank]
    camera = RenderCamera(str(rank), torch.eye(4, device=device),
                          torch.tensor([[52., 0., 32.], [0., 52., 32.], [0., 0., 1.]], device=device), 64, 64)
    trace = observer.LifecycleObserver(args.root, rank, {}, True, window=(0, 1, 2), expected_count=8)
    trace.iteration = 2
    trace.labels = torch.arange(8, device=device) % 3
    actual = render_gaussians(model, camera, sh_degree=0, distributed=True)
    trace.contribution(model, SimpleNamespace(camera=camera), actual)
    row = json.loads((args.root / f"diagnostic-rank-{rank}.jsonl").read_text())
    arrays = [m.activated() for m in models]
    means, quats, scales, opacity = [torch.cat([a[i] for a in arrays]) for i in range(4)]
    labels = torch.cat([torch.arange(8, device=device) % 3 + 3*r for r in (0, 1)])
    colors = torch.nn.functional.one_hot(labels, 6).float()
    reference, alpha, _ = rasterization(means, quats, scales, opacity, colors,
        camera.camera_from_normalized[None], camera.intrinsic[None], width=64, height=64,
        packed=False, sh_degree=None, distributed=False, backgrounds=None,
        near_plane=NEAR_PLANE_NORMALIZED, far_plane=FAR_PLANE_NORMALIZED)
    torch.testing.assert_close(torch.tensor(row["weight_sum"], device=device, dtype=torch.float64),
                               reference[0].double().sum((0, 1)), atol=observer.ATOL, rtol=observer.RTOL)
    observer.verify_alpha(reference[0], alpha[0], actual.alpha)
    torch.distributed.barrier()


def test_distributed_owner_cohort_contribution_matches_full_model(tmp_path):
    import torch
    from gsplat.distributed import cli
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    from scripts.run_mcmc_stability_ablation import PROJECT

    if torch.cuda.device_count() != 2:
        pytest.skip("需要两张空闲GPU")
    with FileLease(PROJECT / "outputs/.gpu.lock"):
        require_idle_gpu()
        cli(contribution_worker, SimpleNamespace(root=Path(tmp_path)), verbose=True)
