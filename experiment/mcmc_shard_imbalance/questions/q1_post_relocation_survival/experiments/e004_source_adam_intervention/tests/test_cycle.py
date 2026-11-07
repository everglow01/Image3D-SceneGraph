"""Cycle endpoint, source-pool and protocol regressions; no CUDA required."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention import cycle, moment_split, probe, run


def rows(start=4, end=8, count=8):
    return [{'iteration': i, 'world_size': 2, 'gaussian_count': count * 2,
        'nominal_iterations': 30000, 'requested_stop_iteration': end,
        'mcmc_noise_applied': True, 'batch_view_ids': ['0', '1'],
        **dict.fromkeys(('loss', 'reconstruction_loss', 'l1', 'ssim',
                        'opacity_regularization', 'scale_regularization'), .1)}
        for i in range(start + 1, end)]


def test_pool_uses_weights_not_just_alive_count():
    equal = cycle.source_pool([.5, .5, 0, .005])
    unequal = cycle.source_pool([.9, .1, 0, .005])
    assert equal['eligible_count'] == unequal['eligible_count'] == 2
    assert equal['dead_target_count'] == 2
    assert equal['effective_sample_size'] == pytest.approx(2)
    assert unequal['effective_sample_size'] == pytest.approx(1 / .82)
    assert equal['expected_unique_sources'] == pytest.approx(1.5)
    assert unequal['expected_unique_sources'] == pytest.approx(1.18)


@pytest.mark.parametrize('values,unique', [([1, 0, 0], 1), ([.2, .3], 0), ([0, .005], 0)])
def test_pool_degenerate_draws(values, unique):
    assert cycle.source_pool(values)['expected_unique_sources'] == unique


@pytest.mark.parametrize('values', [[float('nan')], [float('inf')], [-.1], [1.1]])
def test_pool_rejects_invalid_alpha(values):
    with pytest.raises(ValueError):
        cycle.source_pool(values)


def test_native_history_keeps_requested_12200_but_ends_before_strategy():
    history = rows()
    assert cycle.native_rows(history, 4, 8, 8) == history
    for broken in (history[:-1], history + [dict(history[-1], iteration=8)],
                   [dict(history[0], requested_stop_iteration=7), *history[1:]],
                   [dict(history[0], mcmc_noise_applied=False), *history[1:]]):
        with pytest.raises(ValueError):
            cycle.native_rows(broken, 4, 8, 8)


def test_finish_boundary_does_not_create_final_native_row_or_call_strategy(tmp_path):
    (tmp_path / 'A').mkdir()
    p = probe.InterventionProbe(tmp_path / 'A', 0, 'A', ['0', '1'], lambda n: None,
        start=0, event=4, end=8, count=8, training=tmp_path / 'training', cycle_cameras=['0', '1'])
    native = p.training / 'attempts/A/artifacts/progress.jsonl'
    native.parent.mkdir(parents=True)
    native.write_text('\n'.join(json.dumps(r) for r in rows()))
    p.initial_masks, p.rows = {}, [{'iteration': i} for i in range(5, 9)]
    p.consensus = lambda label, fn: fn()
    with patch.object(cycle, 'evaluate_endpoint', return_value=[{'image_id': '0'}]) as evaluate:
        cycle.finish_boundary(p)
    assert evaluate.call_count == 1 and p.completed
    assert p.calls['terminal'] == 1 and p.calls['strategy'] == p.calls['noise'] == 0
    saved = json.loads((p.root / 'native-tail-rank-0.json').read_text())
    assert [r['iteration'] for r in saved] == [5, 6, 7]
    assert json.loads((p.trace / 'index.json').read_text())['end'] == 8


def test_probe_finish_counts_last_adam_without_last_strategy(tmp_path):
    (tmp_path / 'A').mkdir()
    p = probe.InterventionProbe(tmp_path / 'A', 0, 'A', ['0', '1'], lambda n: None,
        start=0, event=4, end=8, count=8, cycle_cameras=['0', '1'])
    p.completed, p.calls = True, cycle.expected_calls(4)
    p.schedule, p.noise_schedule = [{}] * 4, [{}] * 3
    p.evaluation_forwards = 1
    assert p.finish()['new_updates'] == 4
    p.calls['strategy'] += 1
    with pytest.raises(ValueError):
        p.finish()


def test_prepare_freezes_A_O_V_and_views(tmp_path):
    root = tmp_path / 'adam-intervention-cycle-test'
    reference = {'binding': {'checkpoint_files': {}}, 'protocol': {'source_identity': {'fixture': True}}}
    with patch.object(run, 'execution_commit', return_value='commit'), patch.object(run, 'DEPENDENCIES', ()), \
            patch.object(moment_split, 'reference_inputs', return_value=reference), \
            patch.object(moment_split.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
            patch.object(moment_split.prefix_recovery, 'copy_lineage'), \
            patch.object(cycle, 'camera_contract', return_value={'image_ids': ['0', '1', '2', '3']}):
        moment_split.prepare(root, full_cycle=True)
    p = run.read(root / 'protocol.json')
    assert p['arms'] == ['A', 'O', 'V']
    assert (p['arm_updates'], p['total_updates'], p['camera_forwards'], p['loss_backwards'],
            p['optimizer_group_steps'], p['extra_render_forwards']) == (100, 300, 600, 600, 3600, 12)
    assert p['cycle_camera_contract']['image_ids'] == ['0', '1', '2', '3']
    assert p['global_relocations'] == 0
    with pytest.raises(ValueError):
        moment_split.prepare(tmp_path / 'adam-intervention-opacity-only-wrong', full_cycle=True)


def test_cycle_protocol_cannot_change_arm_or_endpoint():
    p = {**cycle.CONTRACT, 'moment_split': {
        'reference_root': str(moment_split.REFERENCE),
        'reference_protocol_sha256': moment_split.REFERENCE_PROTOCOL_SHA,
        'reference_comparison_sha256': moment_split.REFERENCE_COMPARISON_SHA,
        'checkpoint_hash': moment_split.prefix_recovery.CHECKPOINT_HASH}}
    moment_split.contract(p)
    for key, value in [('arms', ['A', 'O']), ('end', 12201), ('extra_render_forwards', 0)]:
        broken = copy.deepcopy(p)
        broken[key] = value
        with pytest.raises(ValueError):
            moment_split.contract(broken)


def test_installed_hook_stops_before_next_native_strategy(tmp_path):
    import torch
    from unittest.mock import Mock
    from image3d_scenegraph.gaussian import trainer
    (tmp_path / 'A').mkdir()
    p = probe.InterventionProbe(tmp_path / 'A', 0, 'A', ['0', '1'], lambda n: None,
        start=0, event=4, end=8, count=8, cycle_cameras=['0', '1'])
    native_post = Mock()
    strategy = SimpleNamespace(check_sanity=lambda params, optimizers: None,
        step_post_backward=native_post, _relocate_gs=Mock(), _add_new_gs=Mock())
    p.consensus = lambda label, fn: fn()
    p.capture, p.record_masks = Mock(), Mock(return_value={})
    p.params = {'means': torch.zeros(8, 3), 'opacities': torch.zeros(8)}
    p.state, p.step_names = {}, list(run.snapshots.PARAMETERS)
    with patch.object(trainer, '_build_mcmc_strategy', return_value=strategy),             patch.object(cycle, 'finish_boundary') as finish:
        with p.installed():
            built = trainer._build_mcmc_strategy(None, {})
            p.iteration = 7
            built.step_post_backward(params=p.params, state=p.state, step=7)
            assert native_post.call_count == 1 and finish.call_count == 0
            p.iteration = 8
            with pytest.raises(run.snapshots.ProbeComplete):
                built.step_post_backward(params=p.params, state=p.state, step=8)
    assert native_post.call_count == 1 and finish.call_count == 1
    assert [r['iteration'] for r in p.rows] == [7, 8]
    p.capture.assert_called_once()
