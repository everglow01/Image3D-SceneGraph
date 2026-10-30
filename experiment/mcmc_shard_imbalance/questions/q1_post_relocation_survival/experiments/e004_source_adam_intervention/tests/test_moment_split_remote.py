"""Opt-in remote CPU test of actual moment writes and one native Adam step."""
from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_MOMENT_SPLIT_CPU_TESTS') != '1':
    pytest.skip('仅指定远端显式授权执行两矩拆分CPU数值测试', allow_module_level=True)

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import run, analyze, probe  # noqa: E402


def test_selected_source_moments_and_native_adam(tmp_path):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or str(run.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph'
            or not tmp_path.resolve().is_relative_to(run.RUNS.resolve())):
        pytest.fail('CPU测试必须隐藏CUDA且使用新的实验runs basetemp')
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    shapes = {'means': (8, 3), 'scales': (8, 3), 'quats': (8, 4), 'opacities': (8,), 'sh0': (8, 1, 3), 'shN': (8, 15, 3)}
    counts = 0
    def forbidden(*args, **kwargs):
        raise AssertionError('no backward or CUDA in CPU moment fixture')
    with patch.object(torch.Tensor, 'backward', forbidden), patch.object(torch.cuda, '_lazy_init', forbidden):
        for empty in (False, True):
            for case in ('A', 'B', 'V', 'M', 'O'):
                root = tmp_path / f'{case}-{int(empty)}'
                root.mkdir()
                prefix = root / 'prefix'
                prefix.mkdir()
                p = probe.InterventionProbe(root, 0, case, ['fixture'], lambda n: None, prefix=prefix, count=8)
                p.params = {name: torch.nn.Parameter(torch.full(shape, -.5)) for name, shape in shapes.items()}
                p.optimizers = {name: torch.optim.Adam([p.params[name]], lr=.05, eps=1e-15, foreach=False)
                                for name in run.snapshots.PARAMETERS}
                p.state, p.camera = {}, {'order': [0], 'cursor': 0}
                p.labels = torch.tensor([1, 1, 0, 0, 2, 2, 2, 2] if not empty else [2] * 8, dtype=torch.int8)
                p.ids = torch.where(p.labels == 1)[0]
                original = {}
                for name, opt in p.optimizers.items():
                    value = p.params[name]
                    state = {'step': torch.tensor(12100.), 'exp_avg': torch.full_like(value, .003),
                             'exp_avg_sq': torch.full_like(value, .002)}
                    opt.state[value] = state
                    original[name] = {'state': {k: v.clone() for k, v in state.items()}}
                run.snapshots.save_snapshot(prefix / 'pre-relocation-rank-0', {'adam': original})
                for name, opt in p.optimizers.items():
                    for key in analyze.MOMENTS:
                        opt.state[p.params[name]][key][p.ids] = 0
                protected = p.protected_digest()
                p.restore_source_moments()
                assert p.protected_digest() == protected
                for name, opt in p.optimizers.items():
                    value = p.params[name]
                    state = opt.state[value]
                    for key in analyze.MOMENTS:
                        selected = (case == 'B' or (case == 'V' and key == 'exp_avg_sq')
                                    or (case == 'M' and key == 'exp_avg') or (case == 'O' and name == 'opacities'))
                        expected = original[name]['state'][key].clone()
                        if not selected:
                            expected[p.ids] = 0
                        assert torch.equal(state[key], expected)
                    assert state['step'].item() == 12100
                    value.grad = torch.full_like(value, .0001)
                    prediction = analyze.adam_prediction(value.detach().numpy().copy(), value.grad.numpy().copy(),
                        state['exp_avg'].numpy().copy(), state['exp_avg_sq'].numpy().copy(), 12100,
                        {k: v for k, v in opt.param_groups[0].items() if k != 'params'})
                    opt.step()
                    counts += 1
                    for key, actual in (('logits', value.detach().numpy()), ('exp_avg', state['exp_avg'].numpy()),
                                        ('exp_avg_sq', state['exp_avg_sq'].numpy())):
                        assert np.allclose(actual, prediction[key], atol=run.snapshots.ATOL, rtol=run.snapshots.RTOL)
                    assert state['step'].item() == 12101
    assert counts == 60 and not torch.cuda.is_initialized()
    print('CPU Adam参数组step=60；forward=0；backward=0；真实训练=0；五条件及空来源通过')
