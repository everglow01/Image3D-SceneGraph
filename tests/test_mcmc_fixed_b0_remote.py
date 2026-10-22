"""Explicit remote-only preflight: analytical one-pixel context, no forward/training."""
from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_FIXED_B0_TESTS') != '1':
    pytest.skip('仅明确授权远端执行固定B0真实张量检查', allow_module_level=True)

from scripts import run_mcmc_fixed_b0 as runner  # noqa: E402
from scripts import mcmc_fixed_b0_analysis as analysis  # noqa: E402


def one_pixel(rank):
    import torch
    def t(values, integer=False):
        return torch.tensor(values, device=f'cuda:{rank}', dtype=torch.int32 if integer else torch.float32)
    return {'inputs': {'means2d': t([[[.5, .5]]]), 'conics': t([[[1., 0., 1.]]]),
        'colors': t([[[.2, .4, .6]]]), 'opacities': t([[.5]]), 'backgrounds': None, 'masks': None,
        'isect_offsets': t([[[0]]], True), 'flatten_ids': t([0], True), 'render_alphas': t([[[[.5]]]]),
        'last_ids': t([[[0]]], True)}, 'image_gradients': {'rgb': t([[[[1., 1., 1.]]]]), 'alpha': t([[[[.25]]]])},
        'settings': {'width': 1, 'height': 1, 'tile_size': 16, 'absgrad': False,
                     'needs_input_grad': (True,) * 4 + (False,) * 8}, 'rng': runner.rng_state()}


def test_original_backward_five_calls_per_gpu_and_exact_context(tmp_path):
    import numpy as np
    import torch
    from gsplat.cuda._wrapper import _RasterizeToPixels
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    if torch.cuda.device_count() != 2:
        pytest.fail('需要正好两张空闲GPU')
    original = _RasterizeToPixels.backward
    calls = []
    def counted(ctx, rgb, alpha):
        calls.append(torch.cuda.current_device())
        return original(ctx, rgb, alpha)
    with FileLease(runner.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        for rank in (0, 1):
            torch.cuda.set_device(rank)
            folder, out = tmp_path / f'input-{rank}', tmp_path / f'output-{rank}'
            runner.snapshots.save_snapshot(folder, one_pixel(rank))
            with patch.object(_RasterizeToPixels, 'backward', staticmethod(counted)):
                result = runner.replay_five(folder, out, rank)
            assert result['original_backward_calls'] == 5
            for trial in range(5):
                directory = out / f'trial-{trial}'
                record = runner.snapshots.load_snapshot(directory)
                values = analysis.decode(record['tree'], directory)['outputs']
                for name in ('v_means2d', 'v_conics'):
                    assert np.all(values[name] == 0)
                np.testing.assert_allclose(values['v_colors'], .5, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(values['v_opacities'], 1.45, rtol=1e-6, atol=1e-7)
            wrong = 1 - rank
            with pytest.raises(ValueError, match='camera GPU'):
                runner.load_context(folder, wrong)
        assert calls == [0] * 5 + [1] * 5
        check_replay_rejects_input_mutation_alias_and_file_tamper(tmp_path)
        payload = one_pixel(0)
        payload['inputs']['means2d'] = payload['inputs']['means2d'].transpose(1, 2)
        bad = tmp_path / 'bad-shape'
        runner.snapshots.save_snapshot(bad, payload)
        with pytest.raises(ValueError, match='input structure'):
            runner.load_context(bad, 0)
        payload = one_pixel(0)
        payload['inputs']['means2d'] = torch.full((1, 1, 4), .5, device='cuda:0')[..., ::2]
        noncontiguous = tmp_path / 'noncontiguous'
        runner.snapshots.save_snapshot(noncontiguous, payload)
        with pytest.raises(ValueError, match='layout identity'):
            runner.load_context(noncontiguous, 0)
    runner.write_new(tmp_path / 'preflight-call-counts.json', {'synthetic_backward_calls': len(calls),
                     'per_rank': [5, 5], 'forward_calls': 0, 'not_part_of_real_replay_budget': True})


def test_repeat_statistics_independent_cpu_reference(tmp_path):
    import numpy as np
    arrays = [np.array([0., -0., 1., -1., 1e-12, 0.], dtype=np.float32) for _ in range(5)]
    arrays[1][1] = 0.
    arrays[2][2] = np.nextafter(np.float32(1), np.float32(2))
    arrays[3][4] = 2e-12
    arrays[4][5] = .01
    report = analysis.repeated_statistics(arrays, tmp_path, 'synthetic')
    strict, bitwise, failure = (np.zeros(6, dtype=np.uint8) for _ in range(3))
    for i, j in analysis.PAIRS:
        x, y = arrays[i], arrays[j]
        p = report['pairs'][f'{i}-{j}']
        assert p['num_mismatch'] == np.count_nonzero(~np.isclose(x.astype(float), y.astype(float), atol=2e-6, rtol=2e-4))
        expected = [analysis.scalar_ulp(float(a), float(b)) for a, b in zip(x, y)]
        assert p['max_ulp'] == max(expected)
        assert p['ulp_histogram'] == {str(u): expected.count(u) for u in set(expected)}
        strict += x != y
        bitwise += x.view(np.uint32) != y.view(np.uint32)
        failure += ~np.isclose(x.astype(float), y.astype(float), atol=2e-6, rtol=2e-4)
    for name, expected in (('strict_frequency', strict), ('bitwise_frequency', bitwise), ('failure_frequency', failure)):
        actual = np.load(tmp_path / report['array_files'][name]['file'], allow_pickle=False)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.load(tmp_path / 'synthetic-range.npy', allow_pickle=False),
                                  np.ptp(np.stack(arrays).astype(float), axis=0))
    assert report['observed_max_range'] == float(np.float32(.01))
    with pytest.raises(FileExistsError):
        analysis.repeated_statistics(arrays, tmp_path, 'synthetic')
    arrays[0][0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        analysis.repeated_statistics(arrays, tmp_path, 'nonfinite')


def test_control_selection_visibility_eligibility_and_tie(monkeypatch):
    import numpy as np
    monkeypatch.setattr(analysis, 'OWNER_SIZE', 6)
    monkeypatch.setattr(analysis, 'ANOMALIES', ((0, 0, 'quats'), (1, 0, 'quats'), (1, 1, 'shN')))
    features = np.zeros((6, 4))
    visible = np.ones((2, 6), dtype=bool)
    visible[1, 2] = False
    monkeypatch.setattr(analysis, 'geometry', lambda *_: (features, visible, None))
    class FakeSource:
        def get(self, branch, owner, stage, key, group):
            value = np.zeros((6, 2), dtype=np.float32)
            if branch == 'b':
                value[3] = 1.  # rejected by tolerance, not strict equality
                value[4] = 1e-8  # allowed despite non-bitwise equality
            return value
    selected = analysis.select_points(FakeSource())
    assert [x['slot'] for x in selected if x['role'] == 'matched_control'] == [1, 4, 5]


def check_replay_rejects_input_mutation_alias_and_file_tamper(tmp_path):
    import torch
    from gsplat.cuda._wrapper import _RasterizeToPixels
    torch.cuda.set_device(0)
    folder = tmp_path / 'input'
    runner.snapshots.save_snapshot(folder, one_pixel(0))
    def mutated(ctx, rgb, alpha):
        ctx.saved_tensors[0].add_(1)
        return tuple(x.clone() for x in ctx.saved_tensors[:4]) + (None,) * 8
    with patch.object(_RasterizeToPixels, 'backward', staticmethod(mutated)):
        with pytest.raises(ValueError, match='changed fixed inputs'):
            runner.replay_five(folder, tmp_path / 'mutated', 0)
    def aliased(ctx, rgb, alpha):
        return ctx.saved_tensors[:4] + (None,) * 8
    with patch.object(_RasterizeToPixels, 'backward', staticmethod(aliased)):
        with pytest.raises(ValueError, match='aliases live'):
            runner.replay_five(folder, tmp_path / 'aliased', 0)
    def rng_consumed(ctx, rgb, alpha):
        torch.rand(1, device='cuda:0')
        return tuple(x.clone() for x in ctx.saved_tensors[:4]) + (None,) * 8
    with patch.object(_RasterizeToPixels, 'backward', staticmethod(rng_consumed)):
        with pytest.raises(ValueError, match='consumed RNG'):
            runner.replay_five(folder, tmp_path / 'rng-consumed', 0)
    path = next(folder.glob('*.npy'))
    with path.open('r+b') as f:
        f.seek(-1, 2)
        old = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match='file changed'):
        runner.load_context(folder, 0)
