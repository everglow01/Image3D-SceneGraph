"""Remote CPU analysis of accepted within-run identities; never train or pool slots across runs."""
from __future__ import annotations

import gzip
import json

from scripts.mcmc_lifecycle_observer import GROUPS, reuse_record, survival_summary
from ..e004_fixed_b0.tensor_analysis import decode
from . import admission, masks
from .run import snapshots, sha256


def state(folder):
    record = snapshots.load_snapshot(folder)
    return decode(record['tree'], folder)


def array_summary(value):
    import numpy as np
    if not np.isfinite(value).all():
        raise ValueError('nonfinite cohort diagnostic')
    if value.size == 0:
        return {'count': 0, 'mean': None, 'quantiles': None}
    return {'count': value.size, 'mean': float(value.astype(np.float64).mean()),
            'quantiles': np.quantile(value, [0, .1, .5, .9, 1]).tolist()}


def cohort(stage, rank):
    import numpy as np
    initial = state(stage / f'cohort-initial-rank-{rank}')
    final = state(stage / f'cohort-final-rank-{rank}')
    before = state(stage / f'cohort-before-rank-{rank}')
    n = admission.OWNER_SIZE
    labels, alive0 = initial['labels'], initial['initial_alive']
    if labels.shape != (n,) or alive0.shape != (n,) or alive0.dtype != np.bool_ or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('cohort shape or label differs')
    if not np.array_equal(labels, final['labels']) or not np.array_equal(alive0, final['initial_alive']):
        raise ValueError('cohort identity changed before second relocation')
    with gzip.open(stage / f'cohort-rank-{rank}.json.gz', 'rt') as stream:
        identities = json.load(stream)
    if (identities['owner_rank'], identities['event']) != (rank, admission.EVENT):
        raise ValueError('cohort owner or generation differs')
    targets, sources = identities['target_slots'], identities['sampled_source_slots']
    reuse = reuse_record(n, targets, sources)
    expected_labels = np.full(n, 2, dtype=np.int64)
    expected_labels[sources] = 1
    expected_labels[targets] = 0
    if not np.array_equal(labels, expected_labels):
        raise ValueError('sample map and cohort labels differ')
    source_reuse = np.bincount(sources, minlength=n)
    associated_reuse = source_reuse.copy()
    associated_reuse[targets] = source_reuse[sources]
    index_path = stage / f'trajectory-rank-{rank}/index.json'
    index = json.loads(index_path.read_text())
    if ((index['owner'], index['event'], index['end_before_second_relocation']) != (rank, admission.EVENT, admission.END)
            or [r['iteration'] for r in index['rows']] != list(range(admission.EVENT + 1, admission.END + 1))):
        raise ValueError('trajectory interval differs')
    death = np.where(alive0, -1, admission.EVENT).astype(np.int64)
    recovery = np.full(n, -1, dtype=np.int64)
    first_recovery_alive_steps = np.zeros(n, dtype=np.int32)
    first_spell_ended = np.zeros(n, dtype=np.bool_)
    alive = None
    for row in index['rows']:
        if row['slot_count'] != n or row['phase'] != 'after_Adam_before_strategy':
            raise ValueError('trajectory phase or topology differs')
        masks.verify_bitset(index_path.parent, {**row, 'xor_count': row['alive_count']})
        path = snapshots.safe_file(index_path.parent, row['path'])
        raw = gzip.decompress(path.read_bytes())
        alive = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')[:n].astype(np.bool_)
        death[(death < 0) & ~alive] = row['iteration']
        recovery[(recovery < 0) & (death >= 0) & alive] = row['iteration']
        begun = recovery >= 0
        first_recovery_alive_steps += (begun & ~first_spell_ended & alive).astype(np.int32)
        first_spell_ended |= begun & ~alive
    if (not np.array_equal(death, final['first_death']) or not np.array_equal(recovery, final['first_recovery'])
            or not np.array_equal(alive, final['alive'])):
        raise ValueError('saved first events disagree with full trajectory')
    for payload in (before, initial, final):
        if payload['opacities'].shape != (n,) or not np.isfinite(payload['opacities']).all():
            raise ValueError('invalid saved opacity logits')
    groups = {}
    for i, name in enumerate(GROUPS):
        selected = labels == i
        groups[name] = survival_summary(alive0[selected].tolist(), death[selected].tolist(),
                                       alive[selected].tolist(), recovery[selected].tolist())
        groups[name]['first_recovery_alive_steps'] = array_summary(first_recovery_alive_steps[selected & (recovery >= 0)])
        groups[name]['first_recovery_spell_censored_at_window_end'] = int((selected & (recovery >= 0) & ~first_spell_ended).sum())
        groups[name]['by_fate'] = {}
        for category, fate in (('initially_alive_redied', alive0 & (death >= 0)),
                               ('initially_alive_never_died', alive0 & (death < 0)),
                               ('not_alive_after_relocation', ~alive0)):
            subset = selected & fate
            values = {'associated_source_reuse': array_summary(associated_reuse[subset]),
                      'initial_opacity_logit': array_summary(initial['opacities'][subset]),
                      'final_opacity_logit': array_summary(final['opacities'][subset])}
            for key, value in final['exposure'].items():
                if value.shape != (n,):
                    raise ValueError('exposure topology differs')
                values[key + '_per_step'] = array_summary(value[subset] / (admission.END - admission.EVENT))
            groups[name]['by_fate'][category] = values
    diagnostics = admission.jsonl(stage / f'diagnostic-rank-{rank}.jsonl')
    contributions = [r for r in diagnostics if r.get('event') == 'contribution']
    if [r['iteration'] for r in contributions] != list(range(admission.START + 1, admission.END + 1)):
        raise ValueError('missing group contribution sequence')
    return {'owner': rank, 'event': admission.EVENT, 'end_before_second_relocation': admission.END,
            'groups': groups, 'source_reuse': reuse, 'trajectory_index_sha256': sha256(index_path),
            'group_contributions': contributions,
            'interpretation': 'within_run_associations_only; group_T_alpha_not_per_point_visibility_or_loss_improvement',
            'image_gradient_method': 'total_minus_analytic_regularizer_float_residual',
            'cross_run_slot_pairing': False}


def compare_endpoints(root):
    result = {}
    for name in admission.CASES[1:]:
        result[name + '__C0'] = {str(rank): snapshots.compare_snapshots(
            root / name / f'endpoint-rank-{rank}', root / 'C0' / f'endpoint-rank-{rank}') for rank in (0, 1)}
    return {'reference': 'C0', 'pairs': result, 'atol': snapshots.ATOL, 'rtol': snapshots.RTOL,
            'used_as_admission_gate': False, 'slot_identity_across_runs': 'not_assumed_after_relocation'}
