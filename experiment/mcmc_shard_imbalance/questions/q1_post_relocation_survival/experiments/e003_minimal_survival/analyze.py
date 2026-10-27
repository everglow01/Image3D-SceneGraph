"""Within-run T/S/U descriptions. No statistical independence or nonintervention claim."""
from __future__ import annotations

import gzip
import json

from scripts.mcmc_lifecycle_observer import GROUPS, reuse_record, survival_summary
from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0.tensor_analysis import decode
from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission import masks
from .observer import snapshots, sha256

REUSE_BINS = ((1, 1, '1'), (2, 3, '2-3'), (4, None, '4+'))
ALPHA_BINS = ((None, .005, 'not_alive'), (.005, .01, 'near_threshold'), (.01, .05, 'moderate'), (.05, None, 'higher'))


def validate_rows(index, rank, event, end, count):
    if ((index['owner'], index['event'], index['end']) != (rank, event, end)
            or [r['iteration'] for r in index['rows']] != list(range(event + 1, end + 1))):
        raise ValueError('trajectory identity or interval differs')
    for row in [index['initial'], *index['rows']]:
        if row is not index['initial'] and row['phase'] != 'after_Adam_before_strategy':
            raise ValueError('trajectory phase differs')
        if any(row[k]['slot_count'] != count for k in ('alive', 'below_cutoff')):
            raise ValueError('trajectory topology differs')


def state(folder):
    return decode(snapshots.load_snapshot(folder)['tree'], folder)


def bits(folder, row):
    import numpy as np
    masks.verify_bitset(folder, row)
    raw = gzip.decompress(snapshots.safe_file(folder, row['path']).read_bytes())
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')[:row['slot_count']].astype(bool)


def cohort(stage, rank, *, event=12100, end=12200, count=1500000):
    import numpy as np
    initial = state(stage / f'cohort-rank-{rank}')
    final = state(stage / f'cohort-final-rank-{rank}')
    for value in (initial, final):
        if (value['owner'], value['event']) != (rank, event):
            raise ValueError('snapshot owner or generation differs')
    if final['opacities'].shape != (count,) or not np.isfinite(final['opacities']).all():
        raise ValueError('invalid terminal opacity logits')
    labels = initial['labels']
    if (labels.shape != (count,) or not np.isin(labels, (0, 1, 2)).all()
            or not np.array_equal(labels, final['labels'])):
        raise ValueError('cohort labels differ')
    targets, sources = initial['targets'], initial['sources']
    if targets.dtype != np.int64 or sources.dtype != np.int64 or targets.ndim != 1 or sources.ndim != 1:
        raise ValueError('invalid source map dtype/shape')
    reuse = reuse_record(count, targets.tolist(), sources.tolist())
    expected = np.full(count, 2, dtype=np.int8)
    expected[sources], expected[targets] = 1, 0
    if not np.array_equal(labels, expected):
        raise ValueError('source map does not partition T/S/U')
    for key, value in (('before_alpha', initial), ('initial_alpha', initial), ('alpha', final)):
        a = value[key]
        if a.shape != (count,) or a.dtype != np.float32 or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
            raise ValueError('invalid native alpha')
    if (not np.array_equal(np.flatnonzero(initial['before_alpha'] <= .005), targets)
            or not (initial['before_alpha'][sources] > .005).all()):
        raise ValueError('mapping is inconsistent with original alive set')
    trace = stage / f'trajectory-rank-{rank}'
    index = json.loads((trace / 'index.json').read_text())
    validate_rows(index, rank, event, end, count)
    alive0 = bits(trace, index['initial']['alive'])
    if (not np.array_equal(alive0, initial['initial_alpha'] > .005)
            or not np.array_equal(bits(trace, index['initial']['below_cutoff']), initial['initial_alpha'] < 1 / 255)):
        raise ValueError('initial masks disagree with native alpha')
    death = np.where(alive0, -1, event).astype(np.int32)
    recovery = np.full(count, -1, dtype=np.int32)
    alive_steps, spell_ended = np.zeros(count, dtype=np.int32), np.zeros(count, dtype=bool)
    for row in index['rows']:
        alive = bits(trace, row['alive'])
        cutoff = bits(trace, row['below_cutoff'])
        if (alive & cutoff).any():
            raise ValueError('cutoff mask contradicts alive mask')
        death[(death < 0) & ~alive] = row['iteration']
        recovery[(recovery < 0) & (death >= 0) & alive] = row['iteration']
        begun = recovery >= 0
        alive_steps += (begun & ~spell_ended & alive).astype(np.int32)
        spell_ended |= begun & ~alive
    if (not np.array_equal(alive, final['alpha'] > .005)
            or not np.array_equal(cutoff, final['alpha'] < 1 / 255)):
        raise ValueError('terminal trajectory differs from saved alpha')
    source_reuse = np.bincount(sources, minlength=count)
    associated = source_reuse.copy()
    associated[targets] = source_reuse[sources]
    family = np.full(count, -1, dtype=np.int64)
    family[np.unique(sources)] = np.unique(sources)
    family[targets] = sources
    redied = alive0 & (death >= 0)
    groups = {}
    for i, name in enumerate(GROUPS):
        selected = labels == i
        summary = survival_summary(alive0[selected].tolist(), death[selected].tolist(),
                                   alive[selected].tolist(), recovery[selected].tolist())
        restored = selected & (recovery >= 0)
        summary['first_recovery_alive_steps_sum'] = int(alive_steps[restored].sum())
        summary['first_recovery_alive_steps_mean'] = float(alive_steps[restored].mean()) if restored.any() else None
        summary['first_recovery_spell_right_censored'] = int((restored & ~spell_ended).sum())
        summary['fraction_of_all_final_dead'] = int((selected & ~alive).sum()) / int((~alive).sum()) if (~alive).any() else None
        strata = []
        if name != 'uninvolved':
            for lo, hi, reuse_name in REUSE_BINS:
                for alo, ahi, alpha_name in ALPHA_BINS:
                    mask = selected & (associated >= lo)
                    if hi is not None:
                        mask &= associated <= hi
                    if alo is not None:
                        mask &= initial['initial_alpha'] > alo
                    if ahi is not None:
                        mask &= initial['initial_alpha'] <= ahi
                    at_risk = mask & alive0
                    ids, inverse = np.unique(family[at_risk], return_inverse=True)
                    denominator = np.bincount(inverse)
                    numerator = np.bincount(inverse, weights=redied[at_risk].astype(float), minlength=len(ids))
                    strata.append({'reuse': reuse_name, 'initial_alpha': alpha_name, 'count': int(mask.sum()),
                        'initially_alive': int(at_risk.sum()), 'redied': int((mask & redied).sum()),
                        'redeath_fraction': float(redied[at_risk].mean()) if at_risk.any() else None,
                        'source_families_at_risk': len(ids),
                        'family_equal_weight_redeath_fraction': float((numerator / denominator).mean()) if len(ids) else None})
        groups[name] = {**summary, 'strata': strata}
    if sum(g['count'] for g in groups.values()) != count:
        raise ValueError('group totals do not cover owner')
    return {'owner': rank, 'event': event, 'end_before_second_relocation': end, 'groups': groups,
            'source_reuse': reuse, 'trajectory_sha256': sha256(trace / 'index.json'),
            'initial_dead_count': int((~alive0).sum()), 'final_dead_count': int((~alive).sum()),
            'initial_below_cutoff': int((initial['initial_alpha'] < 1 / 255).sum()), 'final_below_cutoff': int(cutoff.sum()),
            'cross_run_slot_pairing': False, 'independent_Gaussian_samples_assumed': False,
            'interpretation': 'descriptive_observed_trajectory_not_causal_or_full_nonintervention_proof'}


def direction(values):
    if any(v is None for v in values):
        return 'insufficient_denominator'
    if all(v > 0 for v in values):
        return 'same_positive_direction'
    if all(v < 0 for v in values):
        return 'same_negative_direction'
    if all(v == 0 for v in values):
        return 'zero_in_both'
    return 'direction_not_consistent'


def summarize(trajectories):
    if set(trajectories) != {'T0', 'T1'}:
        raise ValueError('both fixed trajectories required for comparison')
    result = {}
    for group in GROUPS:
        for metric in ('initial_not_alive_fraction', 'redeath_fraction_of_initially_alive', 'fraction_of_all_final_dead'):
            differences = {}
            for case in ('T0', 'T1'):
                values = []
                for rank in ('0', '1'):
                    row = trajectories[case]['owners'][rank]['groups'][group]
                    values.append((row['not_alive_after_relocation'] / row['count'] if row['count'] else None)
                                  if metric == 'initial_not_alive_fraction' else row[metric])
                differences[case] = values[1] - values[0] if all(v is not None for v in values) else None
            result[f'{group}/{metric}'] = {'owner1_minus_owner0': differences, 'direction': direction(list(differences.values()))}
    return {'owner_directions': result, 'meaning': 'two_descriptive_directions_not_statistical_replication',
            'cross_run_slot_pairing': False, 'causal_candidate_selected_automatically': False}
