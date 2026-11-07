"""A/O/V at the next pre-relocation boundary, reusing the short-window runner."""
from __future__ import annotations

import math
import json
from pathlib import Path
import statistics

from . import run, moment_split
from ..e003_minimal_survival import render_probe

CONTRACT = {**moment_split.CONTRACT, 'profile': 'source_adam_full_cycle_v1',
    'arms': ['A', 'O', 'V'], 'full_cycle': True, 'end': 12200, 'arm_updates': 100,
    'intervention': 'source_A_opacity_both_V_only', 'total_updates': 300,
    'camera_forwards': 600, 'loss_backwards': 600, 'optimizer_group_steps': 3600,
    'extra_render_forwards': 12, 'endpoint_phase': 'after_Adam_before_next_relocation',
    'native_strategy_steps_per_arm': 99, 'fixed_train_views': 4}


def camera_contract(source):
    dataset = run.read(Path(source['native']['pair']['replay']) / 'dataset.json')
    return render_probe.camera_contract(dataset, source['config']['effective_config'], count=4)


def expected_calls(n):
    return {'restore': 1, 'batch': n, 'render': n, 'backward': n,
        'raster_forward': n, 'raster_backward': n, 'optimizer': 6 * n,
        'strategy': n - 1, 'relocate': 0, 'sample': 0, 'add': 0,
        'noise': n - 1, 'terminal': 1, 'checkpoint': 0}


def evaluate_endpoint(probe):
    """All updates are over; render four frozen Train images without backward."""
    import torch
    from scripts.mcmc_lifecycle_observer import unchanged_rng
    from image3d_scenegraph.gaussian.training_math import l1_ssim_loss

    model, views, renderer = probe.cycle_render_context
    ids = [str(v.image_id) for v in views.cameras] if hasattr(views, 'cameras') else [
        str(v.camera.image_id) for v in views]
    assigned = probe.cycle_cameras[probe.rank::2]
    output = []
    probe.evaluating_cycle = True
    try:
        with unchanged_rng(), torch.no_grad():
            model.validate(max_count=probe.owner_size)
            for identity in assigned:
                prepared = views[ids.index(identity)].to(model.means.device)
                view, rendered = renderer(model, [prepared], 0, 3, distributed=True)
                prediction = rendered.image.clamp(0, 1) if probe.config['loss']['clamp_render'] else rendered.image
                loss, terms = l1_ssim_loss(prediction, view.image,
                    l1_weight=probe.config['loss']['l1_weight'], ssim_weight=probe.config['loss']['ssim_weight'])
                mse = float((prediction - view.image).square().mean())
                values = {'reconstruction_loss': float(loss), 'l1': terms['l1'],
                    'ssim': terms['ssim'], 'mse': mse,
                    'psnr_db': -10 * math.log10(mse) if mse > 0 else None}
                if not all(math.isfinite(v) for v in values.values() if v is not None):
                    raise ValueError('nonfinite cycle endpoint metric')
                output.append({'image_id': identity, **values})
    finally:
        probe.evaluating_cycle = False
    return output


def finish_boundary(probe):
    probe.consensus('cycle-fixed-views', lambda: run.write_new(
        probe.root / f'fixed-views-rank-{probe.rank}.json',
        {'phase': CONTRACT['endpoint_phase'], 'iteration': probe.end,
         'views': evaluate_endpoint(probe)}))
    def save():
        path = probe.training / 'attempts' / probe.case / 'artifacts/progress.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Native 12200 summary is after strategy; do not manufacture that row.
        rows = native_rows(rows, probe.start, probe.end, probe.owner_size)
        run.write_new(probe.root / f'native-tail-rank-{probe.rank}.json', rows)
        run.write_new(probe.trace / 'index.json', {'owner': probe.rank, 'event': probe.event,
            'end': probe.end, 'initial': probe.initial_masks, 'rows': probe.rows})
        probe.calls['terminal'] += 1
        probe.completed = True
    probe.consensus('cycle-endpoint-save', save)


def native_rows(rows, start, end, owner_size):
    selected = [r for r in rows if type(r.get('iteration')) is int and r['iteration'] > start]
    if [r['iteration'] for r in selected] != list(range(start + 1, end)):
        raise ValueError('incomplete pre-endpoint native history')
    for r in selected:
        if (r.get('event') == 'validation' or r.get('world_size') != 2
                or r.get('gaussian_count') != 2 * owner_size
                or r.get('nominal_iterations') != 30000 or r.get('requested_stop_iteration') != end
                or not r.get('mcmc_noise_applied') or len(r.get('batch_view_ids', [])) != 2):
            raise ValueError('cycle native lifecycle differs')
        for key in ('loss', 'reconstruction_loss', 'l1', 'ssim', 'opacity_regularization', 'scale_regularization'):
            run.base.previous.admission.finite(r[key])
    return selected


def verify_schedule(rows, worker, rank, p, dataset):
    schedule, noise = worker['schedule'], worker['noise_schedule']
    start, end = p['event'], p['end']
    config = p['source_identity']['config']['effective_config']
    train = set(map(str, dataset['splits']['train']))
    scale = run.analyze.scene_scale(dataset)
    if ([e['iteration'] for e in schedule] != list(range(start + 1, end + 1))
            or [e['iteration'] for e in noise] != list(range(start + 1, end))
            or [e['iteration'] for e in rows] != list(range(start + 1, end))):
        raise ValueError('incomplete cycle schedule')
    for e in schedule:
        i = e['iteration']
        sh = min(config['sh_schedule']['max_degree'], config['sh_schedule']['initial_degree']
                 + i // config['sh_schedule']['increase_every_iterations'])
        if (e['learning_rates'] != run.analyze.learning_rates(config, scale, i)
                or e['sh_degree'] != sh or e['image_id'] not in train):
            raise ValueError('cycle LR/SH/Train schedule differs')
    for row, e in zip(rows, schedule):
        if e['image_id'] != str(row['batch_view_ids'][rank]) or not set(map(str, row['batch_view_ids'])) <= train:
            raise ValueError('cycle native camera pairing differs')
    if worker['evaluation_forwards'] != p['fixed_train_views'] // 2:
        raise ValueError('cycle fixed-view budget differs')


def source_pool(alpha, threshold=.005):
    """Native relocation samples living points with replacement, weighted by alpha."""
    import numpy as np
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if not np.isfinite(alpha).all() or ((alpha < 0) | (alpha > 1)).any():
        raise ValueError('invalid source-pool opacity')
    eligible = alpha > threshold
    weights = alpha[eligible]
    draws = int((~eligible).sum())
    if not len(weights):
        return {'eligible_count': 0, 'dead_target_count': draws,
            'opacity_weight_sum': 0., 'effective_sample_size': 0.,
            'expected_unique_sources': 0., 'max_sampling_probability': None}
    probabilities = weights / weights.sum()
    with np.errstate(divide='ignore'):
        unique = float((-np.expm1(draws * np.log1p(-probabilities))).sum()) if draws else 0.
    return {'eligible_count': int(eligible.sum()), 'dead_target_count': draws,
        'opacity_weight_sum': float(weights.sum()),
        'effective_sample_size': float(1 / np.square(probabilities).sum()),
        'expected_unique_sources': unique, 'max_sampling_probability': float(probabilities.max())}


def compare(root, completed):
    arms = {case: {rank: w['analysis'] for rank, w in workers.items()} for case, workers in completed.items()}
    losses = {case: run.read(root / case / 'native-tail-rank-0.json') for case in completed}
    fixed, pools = {}, {}
    for case, workers in completed.items():
        fixed[case], pools[case] = [], {}
        for rank in ('0', '1'):
            stage = root / case
            row = run.read(stage / f'fixed-views-rank-{rank}.json')
            if row['phase'] != CONTRACT['endpoint_phase'] or row['iteration'] != CONTRACT['end']:
                raise ValueError('wrong fixed-view phase')
            fixed[case].extend(row['views'])
            alpha = run.analyze.state(stage / f'final-opacity-rank-{rank}')['alpha']
            pools[case][rank] = source_pool(alpha)
        fixed[case].sort(key=lambda row: row['image_id'])
    fields = ('loss', 'reconstruction_loss', 'l1', 'ssim', 'opacity_regularization', 'scale_regularization')
    means, paired = {}, {}
    for case, rows in losses.items():
        if len(rows) != 99:
            raise ValueError('cycle native summary window differs')
        for a, b in zip(rows, losses['A']):
            if a['iteration'] != b['iteration'] or a['batch_view_ids'] != b['batch_view_ids']:
                raise ValueError('unpaired cycle loss')
            if not all(math.isfinite(a[k]) for k in fields):
                raise ValueError('nonfinite cycle loss')
        means[case] = {k: statistics.mean(r[k] for r in rows[1:]) for k in fields}
        if [r['image_id'] for r in fixed[case]] != [r['image_id'] for r in fixed['A']]:
            raise ValueError('unpaired endpoint views')
        paired[case] = [{'image_id': a['image_id'],
            **{k: a[k] - b[k] if a[k] is not None and b[k] is not None else None
               for k in ('reconstruction_loss', 'l1', 'ssim', 'psnr_db')}}
            for a, b in zip(fixed[case], fixed['A'])]
    return {'status': 'full_cycle_measured', 'endpoint_phase': CONTRACT['endpoint_phase'],
        'new_global_updates': 300, 'extra_render_forwards': 12,
        'arms': arms, 'source_pools': pools, 'fixed_train_views': fixed,
        'fixed_view_changes_vs_A': paired, 'native_loss_before_each_update': losses,
        'mean_loss_12102_through_12199': means,
        'native_12200_loss_summary_available': False,
        'quality_scope': 'four_fixed_Train_views_not_held_out_generalization',
        'statistical_significance_proven': False, 'long_term_root_cause_proven': False}
