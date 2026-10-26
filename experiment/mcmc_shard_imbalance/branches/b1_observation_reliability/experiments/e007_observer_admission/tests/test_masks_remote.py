"""Remote opt-in synthetic mask checks; no model, image, rasterizer or training."""
from __future__ import annotations

import gzip
import os
import socket

import pytest

if socket.gethostname() != 'i-94B8D131' or os.environ.get('IMAGE3D_AUTHORIZE_OBSERVER_MASK_TESTS') != '1':
    pytest.skip('仅显式授权指定远端运行阶段A张量前置', allow_module_level=True)

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission import masks, run  # noqa: E402

if str(run.PROJECT) != '/usr/local/3dgs_new/Image3D-SceneGraph':
    raise RuntimeError('远端前置必须位于指定项目目录')


def test_two_gpu_diagnostics_thresholds_and_readonly_source(tmp_path):
    import numpy as np
    import torch
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    with FileLease(run.PROJECT / 'outputs/.gpu.lock'):
        require_idle_gpu()
        assert torch.cuda.device_count() == 2
        direct = np.array([0., .005, np.nextafter(.005, 0.), np.nextafter(.005, 1.), 1 / 255, 1.])
        exact = masks.activated_masks(direct, direct)
        assert exact['dead_after'].tolist() == [True, True, True, False, True, False]
        assert not exact['below_cutoff_after'][4]
        for bad in (np.array([np.nan]), np.array([-1.]), np.array([np.inf])):
            with pytest.raises(ValueError):
                masks.activated_masks(bad, bad)
        sigmoid_calls = 0
        for rank in (0, 1):
            torch.cuda.set_device(rank)
            path = tmp_path / f'opacity-{rank}.npy'
            np.save(path, np.array([-8., -7., -6., -5., -4., 0., 1., 8.], dtype=np.float32), allow_pickle=False)
            digest = run.sha256(path)
            before = np.load(path, mmap_mode='r', allow_pickle=False)
            after = np.array(before[::-1], copy=True)
            result = masks.diagnostics(before, after, rank)
            sigmoid_calls += 2
            for method in masks.METHODS:
                r = result[method]
                assert r['dead_before'].sum() == r['dead_after'].sum() == 3
                assert np.logical_xor(r['dead_before'], r['dead_after']).sum() == 6
                assert r['crossed_to_dead'].sum() == r['crossed_to_alive'].sum() == 3
                bitset = masks.save_xor(tmp_path, f'r{rank}-{method}', r['dead_before'], r['dead_after'], lambda n: None)
                masks.verify_bitset(tmp_path, bitset)
                decoded = np.unpackbits(np.frombuffer(gzip.decompress((tmp_path / bitset['path']).read_bytes()), dtype=np.uint8), bitorder='little')[:before.size]
                assert np.array_equal(decoded, np.logical_xor(r['dead_before'], r['dead_after']))
            assert run.sha256(path) == digest and not before.flags.writeable
            folder = tmp_path / f'snapshot-{rank}'
            manifest_sha = run.snapshots.save_snapshot(folder, {'parameters': {'opacities': np.array(before, copy=True)}})
            loaded = masks.opacity_leaf(folder, manifest_sha, 8)
            assert np.array_equal(loaded, before) and not loaded.flags.writeable
            with pytest.raises(ValueError, match='manifest changed'):
                masks.opacity_leaf(folder, 'wrong', 8)
            with pytest.raises(ValueError, match='finite float32'):
                masks.opacity_leaf(folder, manifest_sha, 7)
            file = next(folder.glob('*.npy'))
            with file.open('r+b') as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
            with pytest.raises(ValueError, match='file changed'):
                masks.opacity_leaf(folder, manifest_sha, 8)
            with pytest.raises(ValueError, match='finite float32'):
                masks.validate_logits(before.astype(np.float64), count=8)
            with pytest.raises(ValueError, match='finite float32'):
                masks.validate_logits(before, count=7)
        assert sigmoid_calls == run.CONTRACT['preflight_budget']['cuda_sigmoid_calls'] == 4
