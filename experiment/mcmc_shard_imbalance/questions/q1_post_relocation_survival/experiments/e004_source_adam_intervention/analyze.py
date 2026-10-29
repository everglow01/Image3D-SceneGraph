"""CPU-only Gate 0 and paired short-window analysis; no automatic scientific gate."""
from __future__ import annotations

import math

from ..e003_minimal_survival import analyze as previous
from ..e003_minimal_survival.run import read, snapshots

state = previous.state
MOMENTS = ('exp_avg', 'exp_avg_sq')
HORIZONS = (1, 5, 10, 20)


def optimizer_options(group):
    if (len(group.get('betas', ())) != 2 or any(not 0 <= b < 1 for b in group['betas'])
            or not math.isfinite(group['lr']) or group['lr'] <= 0
            or not math.isfinite(group['eps']) or group['eps'] <= 0
            or group.get('weight_decay', 0) != 0
            or any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'fused'))
            or group.get('foreach') not in (None, False, True)):
        raise ValueError('unsupported Adam options')
    return group


def exact(left, right):
    if hasattr(left, 'dtype'):
        return (hasattr(right, 'dtype') and left.dtype == right.dtype and left.shape == right.shape
                and left.tobytes() == right.tobytes())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def distribution(values):
    import numpy as np
    a = np.asarray(values).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite analysis input')
    return {'count': int(a.size), 'nonzero': int(np.count_nonzero(a)),
            'positive': int((a > 0).sum()), 'negative': int((a < 0).sum()),
            'quantiles': np.quantile(a, [0, .1, .5, .9, 1]).tolist() if a.size else None,
            'absolute_quantiles': np.quantile(np.abs(a), [.5, .9, 1]).tolist() if a.size else None,
            'mean': float(a.astype(float).mean()) if a.size else None}


def gate0(stage, rank):
    import numpy as np
    before = state(stage / f'pre-relocation-rank-{rank}')
    after = state(stage / f'post-relocation-rank-{rank}')
    cohort = state(stage / f'cohort-rank-{rank}')
    event, count = cohort['event'], len(cohort['labels'])
    if (before['phase'] != 'after_Adam_before_relocation' or after['phase'] != 'after_relocation_before_noise'
            or before['iteration'] != event or after['iteration'] != event or cohort['owner'] != rank):
        raise ValueError('relocation snapshot phase differs')
    t, s = cohort['targets'], cohort['sources']
    if (t.dtype != np.int64 or s.dtype != np.int64 or t.ndim != 1 or t.shape != s.shape
            or np.unique(t).size != t.size or ((t < 0) | (t >= count) | (s < 0) | (s >= count)).any()):
        raise ValueError('invalid true source mapping')
    ids, k = np.unique(s, return_counts=True)
    if np.intersect1d(ids, t).size:
        raise ValueError('overlapping source and target')
    labels = np.full(count, 2, dtype=np.int8)
    labels[ids], labels[t] = 1, 0
    if not exact(labels, cohort['labels']) or not np.array_equal(np.flatnonzero(cohort['before_alpha'] <= .005), t):
        raise ValueError('cohort partition or original alive set differs')
    if not (cohort['before_alpha'][s] > .005).all() or not exact(cohort['initial_alpha'][t], cohort['initial_alpha'][s]):
        raise ValueError('source validity or initial pair alpha differs')
    non_source, uninvolved = labels != 1, labels == 2
    rows, steps, signal = {}, [], 0
    if any(x.get('optimizer_classes') != dict.fromkeys(snapshots.PARAMETERS, 'torch.optim.adam.Adam') for x in (before, after)):
        raise ValueError('native optimizer class evidence differs')
    if any(set(x[key]) != set(snapshots.PARAMETERS) for x in (before, after) for key in ('parameters', 'adam')):
        raise ValueError('six parameter/optimizer groups required')
    for name in snapshots.PARAMETERS:
        a, b = before['parameters'][name], after['parameters'][name]
        if (a.dtype != np.float32 or b.dtype != a.dtype or a.shape != b.shape or a.shape[0] != count
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or not exact(b[t], b[s]) or not exact(a[uninvolved], b[uninvolved])):
            raise ValueError('parameter copy/unchanged U gate failed: ' + name)
        if name not in ('opacities', 'scales') and not exact(a[ids], b[ids]):
            raise ValueError('unexpected source parameter change')
        aa, bb = before['adam'][name], after['adam'][name]
        if (len(aa['param_groups']) != 1 or not exact(aa['param_groups'], bb['param_groups'])
                or set(aa['state']) != {*MOMENTS, 'step'} or set(bb['state']) != {*MOMENTS, 'step'}
                or not exact(aa['state']['step'], bb['state']['step'])):
            raise ValueError('Adam scalar/options/state fields changed')
        group = optimizer_options(aa['param_groups'][0])
        step = float(aa['state']['step'])
        if not math.isfinite(step) or step <= 0 or step != int(step):
            raise ValueError('invalid restored Adam step')
        steps.append(step)
        moments = {}
        for key in MOMENTS:
            old, new = aa['state'][key], bb['state'][key]
            if (old.dtype != a.dtype or new.dtype != a.dtype or old.shape != a.shape or new.shape != a.shape
                    or not np.isfinite(old).all() or not np.isfinite(new).all()
                    or not exact(new[ids], np.zeros_like(new[ids])) or not exact(old[non_source], new[non_source])
                    or (key == 'exp_avg_sq' and ((old < 0).any() or (new < 0).any()))):
                raise ValueError('source reset or preserved target/U state failed: ' + name + '/' + key)
            source_stats = distribution(old[ids])
            signal += source_stats['nonzero']
            moments[key] = {'source_before': source_stats, 'source_after': distribution(new[ids]),
                            'target_after': distribution(new[t]), 'non_source_bitwise_preserved': True}
        normal = aa['state']['exp_avg'][ids] / (np.sqrt(aa['state']['exp_avg_sq'][ids]) + group['eps'])
        rows[name] = {'step': int(step), 'options': group, 'moments': moments,
                      'normalized_history_not_actual_update': distribution(normal), 'pair_parameters_bitwise_equal': True}
    if len(set(steps)) != 1:
        raise ValueError('Adam steps differ across parameter groups; do not normalize them')
    return {'status': 'structural_verified', 'rank': rank, 'event': event, 'groups': rows,
            'family_count': len(ids), 'target_count': len(t), 'maximum_k': int(k.max()) if k.size else 0,
            'source_moment_nonzero_elements': signal, 'effective_intervention_available': signal > 0,
            'moment_magnitude_automatically_admitted': False, 'arms_authorized': False}


def adam_prediction(logits, gradient, first, second, step, group):
    """Offline algebra only; the native optimizer remains the sole training updater."""
    import numpy as np
    optimizer_options(group)
    if not math.isfinite(float(step)) or float(step) < 0 or int(step) != step:
        raise ValueError('invalid Adam step')
    arrays = (logits, gradient, first, second)
    if any(a.shape != logits.shape or a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays) or (second < 0).any():
        raise ValueError('invalid Adam algebra arrays')
    b1, b2 = group['betas']
    g, m, v = gradient.astype(float), first.astype(float), second.astype(float)
    m = (b1 * m + (1 - b1) * g).astype(np.float32)
    v = (b2 * v + (1 - b2) * g * g).astype(np.float32)
    denom = np.sqrt(v.astype(float)) / math.sqrt(1 - b2 ** (step + 1)) + group['eps']
    change = -(group['lr'] / (1 - b1 ** (step + 1))) * m.astype(float) / denom
    return {'logits': (logits.astype(float) + change).astype(np.float32), 'exp_avg': m, 'exp_avg_sq': v}


def first_step(stage, rank):
    import numpy as np
    before = state(stage / f'opacity-before-1-rank-{rank}')
    after = state(stage / f'opacity-after-1-rank-{rank}')
    predicted = adam_prediction(before['logits'], before['gradient'], before['exp_avg'], before['exp_avg_sq'],
                                float(before['step']), before['options'])
    if float(after['step']) != float(before['step']) + 1:
        raise ValueError('first opacity Adam step increment differs')
    comparisons = {key: snapshots.compare_arrays(value, after[key]) for key, value in predicted.items()}
    comparisons['actual_update'] = snapshots.compare_arrays(predicted['logits'] - before['logits'], after['logits'] - before['logits'])
    usable = all(x['status'] in ('bitwise_equal', 'numerically_equivalent') for x in comparisons.values())
    if any(x['status'] == 'structure_mismatch' or x['num_nonfinite'] for x in comparisons.values()):
        raise ValueError('invalid first-step algebra evidence')
    return {'comparisons': comparisons, 'interpretation_usable': usable,
            'gradient_kind': 'total_parameter_gradient_reconstruction_plus_regularizer',
            'gradient': distribution(before['gradient']), 'actual_update': distribution(after['logits'] - before['logits']),
            'unresolved_update_abs_le_atol': int((np.abs(after['logits'] - before['logits']) <= snapshots.ATOL).sum())}


def trajectory(prefix, stage, rank, event, end, count):
    import numpy as np
    cohort = state(prefix / f'cohort-rank-{rank}')
    labels, targets, sources = cohort['labels'], cohort['targets'], cohort['sources']
    if labels.shape != (count,) or cohort['event'] != event or cohort['owner'] != rank:
        raise ValueError('trajectory source identity differs')
    trace = stage / f'trajectory-rank-{rank}'
    index = read(trace / 'index.json')
    previous.validate_rows(index, rank, event, end, count)
    initial = previous.bits(trace, index['initial']['alive'])
    if not np.array_equal(initial, cohort['initial_alpha'] > .005):
        raise ValueError('initial intervention cohort differs')
    if not np.array_equal(previous.bits(trace, index['initial']['below_cutoff']), cohort['initial_alpha'] < 1 / 255):
        raise ValueError('initial cutoff differs')
    first = np.where(initial, end + 1, event).astype(np.int32)
    ids, inverse, k = np.unique(sources, return_inverse=True, return_counts=True)
    output = {}
    for row in index['rows']:
        alive = previous.bits(trace, row['alive'])
        cutoff = previous.bits(trace, row['below_cutoff'])
        if (alive & cutoff).any():
            raise ValueError('incompatible threshold masks')
        first[(first == end + 1) & ~alive] = row['iteration']
        h = row['iteration'] - event
        if h not in HORIZONS and row['iteration'] != end:
            continue
        groups = {}
        for label, name in enumerate(('T', 'S', 'U')):
            selected = labels == label
            risk = selected & initial
            dead = risk & (first <= row['iteration'])
            groups[name] = {'count': int(selected.sum()), 'initial_not_alive': int((selected & ~initial).sum()),
                            'at_risk': int(risk.sum()), 'first_deaths': int(dead.sum()),
                            'risk': float(dead.sum() / risk.sum()) if risk.any() else None,
                            'currently_dead': int((selected & ~alive).sum())}
        tdead = initial[targets] & (first[targets] <= row['iteration'])
        died = np.bincount(inverse, weights=tdead, minlength=len(ids))
        valid = initial[ids]
        if not np.array_equal(initial[sources], initial[targets]):
            raise ValueError('mixed initial family state')
        family_t = died / k
        groups['T']['family_equal_risk'] = float(family_t[valid].mean()) if valid.any() else None
        groups['T']['at_risk_families'] = int(valid.sum())
        family = {'none': int((valid & (died == 0)).sum()), 'all': int((valid & (died == k)).sum()),
                  'mixed': int((valid & (died > 0) & (died < k)).sum())}
        source_dead = first[ids] <= row['iteration']
        joint = {sname + '/' + tname: int((valid & smask & tmask).sum())
            for sname, smask in (('source_died', source_dead), ('source_no_death', ~source_dead))
            for tname, tmask in (('none', died == 0), ('all', died == k), ('mixed', (died > 0) & (died < k)))}
        exact_k = [{'k': int(v), 'at_risk_families': int((valid & (k == v)).sum()),
                    'target_deaths_histogram': {str(j): int((valid & (k == v) & (died == j)).sum()) for j in range(int(v) + 1)}}
                   for v in np.unique(k)]
        output[str(h)] = {'groups': groups, 'target_family_outcomes': family, 'joint_family_outcomes': joint,
                          'target_member_outcome_distribution': exact_k}
    final = state(stage / f'final-opacity-rank-{rank}')
    if not np.array_equal(final['alpha'] > .005, alive) or not np.array_equal(final['alpha'] < 1 / 255, cutoff):
        raise ValueError('terminal alpha disagrees with bitmaps')
    ds, dt = first[ids], first[targets]
    valid = initial[ids]
    w = 1 / k[inverse] if len(k) else np.empty(0)
    pair_valid = valid[inverse]
    so, to = ds[inverse] <= end, dt <= end
    flags = {'both_source_first': so & to & (ds[inverse] < dt), 'both_target_first': so & to & (dt < ds[inverse]),
             'same_observed_step': so & to & (ds[inverse] == dt), 'source_only': so & ~to,
             'target_only': ~so & to, 'both_right_censored': ~so & ~to}
    timing = {key: float(w[pair_valid & value].sum() / valid.sum()) if valid.any() else None for key, value in flags.items()}
    updates, fate = {}, {}
    for h in output:
        before = state(stage / f'opacity-before-{h}-rank-{rank}')
        after = state(stage / f'opacity-after-{h}-rank-{rank}')
        delta = after['logits'] - before['logits']
        updates[h] = {name: {'actual_update': distribution(delta[labels == i]),
            'exp_avg_before': distribution(before['exp_avg'][labels == i]),
            'exp_avg_after': distribution(after['exp_avg'][labels == i]),
            'exp_avg_sq_before': distribution(before['exp_avg_sq'][labels == i]),
            'exp_avg_sq_after': distribution(after['exp_avg_sq'][labels == i])} for i, name in enumerate(('T', 'S', 'U'))}
        if h == '1':
            for i, name in enumerate(('T', 'S', 'U')):
                fate[name] = {}
                for label, condition in (('initial_not_alive', ~initial), ('initially_alive_later_died', initial & (first <= end)),
                                         ('initially_alive_right_censored', initial & (first > end))):
                    mask = (labels == i) & condition
                    fate[name][label] = {'first_actual_update': distribution(delta[mask]),
                        'first_total_gradient': distribution(before['gradient'][mask]),
                        'first_exp_avg': distribution(before['exp_avg'][mask]), 'first_exp_avg_sq': distribution(before['exp_avg_sq'][mask])}
    return {'horizons': output, 'individual_target_timing_family_equal': timing,
            'opacity_update_windows': updates, 'first_update_by_later_fate_descriptive_only': fate,
            'opacity_first_death_step_histogram': {name: {str(t): int(((labels == i) & (first == t)).sum())
                for t in (event, *range(event + 1, end + 1), end + 1)} for i, name in enumerate(('T', 'S', 'U'))}}


def compare(root, event=12100, end=12120, count=1500000):
    import numpy as np
    arms, mechanism = {}, {}
    for case in ('A', 'B'):
        stage = root / case
        arms[case] = {str(rank): {'survival': trajectory(root / 'prefix', stage, rank, event, end, count),
                                  'first_step': first_step(stage, rank)} for rank in (0, 1)}
    for rank in (0, 1):
        workers = [read(root / case / f'worker-rank-{rank}.json') for case in ('A', 'B')]
        if workers[0]['schedule'] != workers[1]['schedule'] or workers[0]['restored_digest'] != workers[1]['restored_digest']:
            raise ValueError('arms did not share restored state and camera/LR schedule')
        cohort = state(root / 'prefix' / f'cohort-rank-{rank}')
        ids = np.unique(cohort['sources'])
        initial = state(root / 'A' / f'opacity-before-1-rank-{rank}')
        other = state(root / 'B' / f'opacity-before-1-rank-{rank}')
        if not exact(initial['logits'], other['logits']) or not exact(initial['step'], other['step']) or not exact(initial['options'], other['options']):
            raise ValueError('first opacity update did not start from shared parameters/step/options')
        original = state(root / 'prefix' / f'pre-relocation-rank-{rank}')['adam']['opacities']['state']
        saved = [original[key][ids] for key in MOMENTS]
        x, g = initial['logits'][ids], initial['gradient'][ids]
        zero = np.zeros_like(x)
        native = adam_prediction(x, g, zero, zero, float(initial['step']), initial['options'])['logits']
        retained = adam_prediction(x, g, *saved, float(initial['step']), initial['options'])['logits']
        mechanism[str(rank)] = {'fixed_gradient_reference': 'A_total_opacity_gradient',
            'retained_minus_zero_logit_update': distribution(retained - native),
            'direction_flip_count': int(((retained - x) * (native - x) < 0).sum()),
            'gradient_difference_B_minus_A': distribution(state(root / 'B' / f'opacity-before-1-rank-{rank}')['gradient'] - initial['gradient']),
            'interpretation_usable': all(arms[c][str(rank)]['first_step']['interpretation_usable'] for c in ('A', 'B'))}
    effects = {}
    for h in arms['A']['0']['survival']['horizons']:
        effects[h] = {}
        for group in ('S', 'T', 'U'):
            risks = {c: [arms[c][str(r)]['survival']['horizons'][h]['groups'][group]['risk'] for r in (0, 1)] for c in ('A', 'B')}
            def difference(a, b):
                return a - b if a is not None and b is not None else None
            effects[h][group] = {'B_minus_A_by_owner': [difference(b, a) for a, b in zip(risks['A'], risks['B'])],
                'owner1_minus_owner0': {c: difference(v[1], v[0]) for c, v in risks.items()}}
            gaps = effects[h][group]['owner1_minus_owner0']
            effects[h][group]['owner_gap_change_B_minus_A'] = difference(gaps['B'], gaps['A'])
            if group == 'T':
                effects[h][group]['family_equal_B_minus_A_by_owner'] = [difference(
                    arms['B'][str(r)]['survival']['horizons'][h]['groups']['T']['family_equal_risk'],
                    arms['A'][str(r)]['survival']['horizons'][h]['groups']['T']['family_equal_risk']) for r in (0, 1)]
    losses = {c: read(root / c / 'native-tail-rank-0.json') for c in ('A', 'B')}
    loss_changes = []
    if len(losses['A']) != end - event or len(losses['B']) != end - event:
        raise ValueError('incomplete native loss windows')
    for index, (a, b) in enumerate(zip(losses['A'], losses['B']), event + 1):
        if a['iteration'] != b['iteration'] or a['iteration'] != index or a['batch_view_ids'] != b['batch_view_ids']:
            raise ValueError('loss comparison not paired by step/cameras')
        fields = ('loss', 'reconstruction_loss', 'l1', 'ssim', 'opacity_regularization', 'scale_regularization')
        if not all(math.isfinite(row[k]) for row in (a, b) for k in fields):
            raise ValueError('nonfinite native loss metric')
        loss_changes.append({'iteration': index, 'batch_view_ids': a['batch_view_ids'],
            'phase': 'before_current_Adam', 'B_minus_A': {k: b[k] - a[k] for k in fields}})
    return {'status': 'paired_short_window_measured', 'arms': arms, 'mechanism': mechanism, 'effects': effects,
            'native_loss_before_each_update': losses, 'paired_loss_changes': loss_changes, 'first_loss_is_pre_intervention_update': True,
            'endpoint_post_update_quality_evaluated': False, 'statistical_significance_proven': False,
            'long_term_root_cause_proven': False, 'algorithm_improvement_automatically_declared': False}
