"""Three fixed short-window interventions, reusing the completed A/B reference."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

from . import run, prefix_recovery
from .probe import SPLIT_CASES, retains_moment

PROFILE = 'source_adam_moment_split_v1'
REFERENCE = run.RUNS / 'adam-intervention-recovery-20261029-v1'
PREFIX = prefix_recovery.SOURCE / 'prefix'
REFERENCE_PROTOCOL_SHA = '15b4a15b2f585fd4d730de9c19f063fdd9fbb3599ed02dfceb4086ce89cf1f98'
REFERENCE_COMPARISON_SHA = 'ec51dc044fc5a5fc7226e3346a5d736692dec65041d001d9fbd75aaeacde55ac'
CONTRACT = {**run.CONTRACT, 'profile': PROFILE, 'arms': list(SPLIT_CASES),
    'intervention': 'source_v_only_m_only_opacity_both', 'prefix_updates': 0,
    'total_updates': 60, 'camera_forwards': 120, 'loss_backwards': 120,
    'optimizer_group_steps': 720, 'global_relocations': 0, 'requires_separate_gate0_review': False}


def contract(p):
    if any(p.get(k) != v for k, v in CONTRACT.items()) or 'prefix_recovery' in p:
        raise ValueError('invalid moment split contract')
    s = p.get('moment_split', {})
    if (s.get('reference_root') != str(REFERENCE) or s.get('reference_protocol_sha256') != REFERENCE_PROTOCOL_SHA
            or s.get('reference_comparison_sha256') != REFERENCE_COMPARISON_SHA
            or s.get('checkpoint_hash') != prefix_recovery.CHECKPOINT_HASH):
        raise ValueError('moment split reference differs')
    return s


def reference_inputs():
    """Validate only files consumed by this experiment, once per entrypoint."""
    files = {}
    def read(name, expected=None):
        file = run.snapshots.safe_file(REFERENCE, name)
        digest = run.sha256(file)
        if expected is not None and digest != expected:
            raise ValueError('reference changed: ' + name)
        files[str(file.relative_to(run.PROJECT))] = digest
        return run.read(file)
    p = read('protocol.json', REFERENCE_PROTOCOL_SHA)
    comparison = read('comparison.json', REFERENCE_COMPARISON_SHA)
    recovery = read('recovery-complete.json')
    recovery_end = read('recovery-execution.json')
    end = read('arms-execution.json')
    for terminal, name in ((recovery_end, 'recovery-complete.json'), (end, 'comparison.json')):
        if (terminal['status'] != 'completed' or terminal['exit_code'] != 0
                or terminal['protocol_sha256'] != REFERENCE_PROTOCOL_SHA or terminal['report'] != name
                or terminal['report_sha256'] != files[str((REFERENCE / name).relative_to(run.PROJECT))]):
            raise ValueError('reference execution did not complete')
    if (recovery['checkpoint_hash'] != prefix_recovery.CHECKPOINT_HASH
            or recovery['full_state_sha256'] != prefix_recovery.STATE_HASHES
            or comparison['status'] != 'paired_short_window_measured'):
        raise ValueError('reference checkpoint/measurement differs')
    workers = {}
    for case in ('A', 'B'):
        complete = read(case + '/complete.json')
        if complete['case'] != case or complete['protocol_sha256'] != REFERENCE_PROTOCOL_SHA:
            raise ValueError('reference completion differs')
        workers[case] = {str(rank): read(f'{case}/worker-rank-{rank}.json', complete['files'][f'{case}/worker-rank-{rank}.json'])
                         for rank in (0, 1)}
        for rank in (0, 1):
            stem = f'{case}/opacity-before-1-rank-{rank}/'
            if stem + 'manifest.json' not in complete['files']:
                raise ValueError('missing reference first-step snapshot')
            for name, digest in complete['files'].items():
                if name.startswith(stem):
                    file = run.snapshots.safe_file(REFERENCE, name)
                    if run.sha256(file) != digest:
                        raise ValueError('reference first-step evidence changed')
                    files[str(file.relative_to(run.PROJECT))] = digest
        for rank, worker in workers[case].items():
            if (worker['restored_digest'] != prefix_recovery.STATE_HASHES[rank]
                    or worker['new_updates'] != 20 or worker['status'] != 'measured'
                    or worker['case'] != case or worker['rank'] != int(rank)
                    or worker['protocol_sha256'] != REFERENCE_PROTOCOL_SHA):
                raise ValueError('reference worker origin differs')
    verify_schedule(workers['B'], workers['A'])
    old = p['prefix_recovery']['source']
    stems = [str((PREFIX / f'{name}-rank-{rank}').relative_to(run.PROJECT)) + '/'
             for name in ('pre-relocation', 'cohort') for rank in (0, 1)]
    states = {str((PREFIX / f'prefix-state-rank-{rank}.json').relative_to(run.PROJECT)) for rank in (0, 1)}
    inputs = {name: digest for name, digest in old['bound_files'].items()
              if name in states or any(name.startswith(stem) for stem in stems)}
    if not states.union(stem + 'manifest.json' for stem in stems) <= inputs.keys():
        raise ValueError('missing intervention snapshots')
    native = p['source_identity']['native']
    replay = Path(native['pair']['replay'])
    # The replay metadata/initialization and native code are inputs; old run logs are not.
    for name, digest in native['bound_files'].items():
        path = Path(name)
        if path.is_relative_to(replay) or path.is_relative_to(run.PROJECT / 'src') or path.is_relative_to(run.PROJECT / 'scripts'):
            inputs[str(path.relative_to(run.PROJECT))] = digest
    package = Path(native['pair']['source_package'])
    for name, digest in native['pair']['source_hashes'].items():
        inputs[str((package / name).relative_to(run.PROJECT))] = digest
    for name, digest in inputs.items():
        if run.sha256(run.snapshots.safe_file(run.PROJECT, name)) != digest:
            raise ValueError('moment split input changed: ' + name)
    files.update(inputs)
    copies = {name: {**row, 'source': str((REFERENCE / name).relative_to(run.PROJECT))}
              for name, row in old['copy_files'].items()}
    binding = {'reference_root': str(REFERENCE), 'reference_protocol_sha256': REFERENCE_PROTOCOL_SHA,
        'reference_comparison_sha256': REFERENCE_COMPARISON_SHA, 'checkpoint_hash': prefix_recovery.CHECKPOINT_HASH,
        'input_sha256': files, 'checkpoint_files': copies}
    return {'protocol': p, 'comparison': comparison, 'workers': workers, 'binding': binding}


def prepare(root):
    if not root.name.startswith('adam-intervention-moment-split-'):
        raise ValueError('new adam-intervention-moment-split-* root required')
    commit = run.execution_commit()
    reference = reference_inputs()
    copies = reference['binding']['checkpoint_files']
    if (sum(row['bytes'] for row in copies.values()) + 1024**3 > CONTRACT['max_output_bytes']
            or shutil.disk_usage(run.PROJECT).free < CONTRACT['disk_admission_bytes']):
        raise ValueError('moment split disk admission')
    root.mkdir(exist_ok=False)
    prefix_recovery.copy_lineage(root, copies)
    run.write_new(root / 'protocol.json', {**CONTRACT, 'implementation_commit': commit,
        'code_sha256': {n: run.sha256(run.PROJECT / n) for n in run.DEPENDENCIES},
        'source_identity': reference['protocol']['source_identity'], 'moment_split': reference['binding'],
        'status': 'moment_split_prepared_not_executed'})


def verify_inputs(p):
    binding = contract(p)
    run.execution_commit()
    reference = reference_inputs()
    if binding != reference['binding'] or p['source_identity'] != reference['protocol']['source_identity']:
        raise ValueError('split input bindings differ')
    return reference


def verify_checkpoint_copies(root, p):
    prefix_recovery.verify_copies(root, {'copy_files': contract(p)['checkpoint_files']})


def verify_schedule(workers, reference):
    for rank in ('0', '1'):
        for key in ('restored_digest', 'schedule', 'noise_schedule'):
            if workers[rank][key] != reference[rank][key]:
                raise ValueError('split/reference ' + key + ' differs')


def compare(root, completed, reference):
    """Reuse per-arm analysis and the frozen A/B report; no second trajectory pass."""
    old = reference['comparison']
    arms = {**old['arms'], **{case: {rank: w['analysis'] for rank, w in workers.items()}
                              for case, workers in completed.items()}}
    losses = {**old['native_loss_before_each_update'],
              **{case: run.read(root / case / 'native-tail-rank-0.json') for case in SPLIT_CASES}}
    effects, loss_changes = {}, {}
    fields = ('loss', 'reconstruction_loss', 'l1', 'ssim', 'opacity_regularization', 'scale_regularization')
    def difference(a, b):
        return a - b if a is not None and b is not None else None
    for case in SPLIT_CASES:
        effects[case] = {}
        for h in ('1', '5', '10', '20'):
            effects[case][h] = {}
            for group in ('S', 'T', 'U'):
                values = {c: [arms[c][r]['survival']['horizons'][h]['groups'][group] for r in ('0', '1')]
                          for c in ('A', 'B', case)}
                row = {f'{case}_minus_{ref}_by_owner': [difference(x['risk'], y['risk']) for x, y in zip(values[case], values[ref])]
                       for ref in ('A', 'B')}
                row['owner1_minus_owner0'] = {c: difference(v[1]['risk'], v[0]['risk']) for c, v in values.items()}
                if group == 'T':
                    row['family_equal_vs_reference'] = {ref: [difference(x['family_equal_risk'], y['family_equal_risk'])
                        for x, y in zip(values[case], values[ref])] for ref in ('A', 'B')}
                effects[case][h][group] = row
        rows = losses[case]
        if len(rows) != 20:
            raise ValueError('incomplete split loss window')
        loss_changes[case] = {}
        for ref in ('A', 'B'):
            if len(losses[ref]) != 20:
                raise ValueError('incomplete reference loss window')
            paired = []
            for iteration, (a, b) in enumerate(zip(rows, losses[ref]), 12101):
                if a['iteration'] != iteration or b['iteration'] != iteration or a['batch_view_ids'] != b['batch_view_ids']:
                    raise ValueError('unpaired split loss')
                if not all(math.isfinite(row[k]) for row in (a, b) for k in fields):
                    raise ValueError('nonfinite split loss')
                paired.append({'iteration': iteration, 'difference': {k: a[k] - b[k] for k in fields}})
            loss_changes[case][ref] = {'per_step': paired,
                'mean_12102_through_12120': {k: sum(row['difference'][k] for row in paired[1:]) / 19 for k in fields}}
    mechanism = first_step_mechanism(root)
    for rank, rows in mechanism.items():
        for case, row in rows.items():
            row['interpretation_usable'] = all(arms[c][rank]['first_step']['interpretation_usable'] for c in ('A', case))
    return {'status': 'moment_split_short_window_measured', 'new_global_updates': 60,
        'reference_root': str(REFERENCE), 'reference_comparison_sha256': REFERENCE_COMPARISON_SHA,
        'reference_implementation_commit': reference['protocol']['implementation_commit'],
        'arms': arms, 'effects': effects, 'mechanism': mechanism,
        'native_loss_before_each_update': losses, 'paired_loss_changes': loss_changes,
        'first_loss_is_pre_intervention_update': True, 'reference_arms_reexecuted': False,
        'endpoint_post_update_quality_evaluated': False, 'long_term_root_cause_proven': False,
        'statistical_significance_proven': False}


def first_step_mechanism(root):
    import numpy as np
    output = {}
    for rank in (0, 1):
        cohort = run.analyze.state(PREFIX / f'cohort-rank-{rank}')
        ids = np.unique(cohort['sources'])
        baseline = run.analyze.state(REFERENCE / 'A' / f'opacity-before-1-rank-{rank}')
        retained = run.analyze.state(REFERENCE / 'B' / f'opacity-before-1-rank-{rank}')
        x, g = baseline['logits'][ids], baseline['gradient'][ids]
        zero = np.zeros_like(x)
        step, options = float(baseline['step']), baseline['options']
        native = run.analyze.adam_prediction(x, g, zero, zero, step, options)['logits']
        output[str(rank)] = {}
        for case in SPLIT_CASES:
            current = run.analyze.state(root / case / f'opacity-before-1-rank-{rank}')
            if (not run.analyze.exact(current['logits'], baseline['logits'])
                    or not run.analyze.exact(current['step'], baseline['step'])
                    or not run.analyze.exact(current['options'], baseline['options'])):
                raise ValueError('split first update origin differs')
            moments = [retained[key][ids] if retains_moment(case, 'opacities', key) else zero
                       for key in run.analyze.MOMENTS]
            prediction = run.analyze.adam_prediction(x, g, *moments, step, options)['logits']
            output[str(rank)][case] = {'fixed_gradient_reference': 'historical_A_total_opacity_gradient',
                'logit_update_minus_zero': run.analyze.distribution(prediction - native),
                'direction_flip_count': int(((prediction - x) * (native - x) < 0).sum()),
                'actual_gradient_minus_reference': run.analyze.distribution(current['gradient'][ids] - g)}
    return output
