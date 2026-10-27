"""Predeclared phase-B metrics and independent holdout admission; standard library."""
from __future__ import annotations

import json
import math
from statistics import median

START, EVENT, END = 12000, 12100, 12200
OWNER_SIZE = 1500000
CASES = ('C0', 'C1', 'C2', 'U0', 'O0', 'O1', 'U1', 'U2', 'O2')
CALIBRATION = ('C0', 'C1', 'C2')
HEALTH_PHASES = ('before_strategy_after_optimizer', 'after_strategy_before_validation')


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('nonfinite or nonnumeric metric')
    return value


def window_rows(rows, *, start=START, end=END, owner_size=OWNER_SIZE):
    selected = [r for r in rows if type(r.get('iteration')) is int and r['iteration'] > start]
    if [r['iteration'] for r in selected] != list(range(start + 1, end + 1)):
        raise ValueError('missing, duplicated, reordered or extra training step')
    for r in selected:
        if (r.get('event') == 'validation' or r.get('world_size') != 2
                or r.get('gaussian_count') != 2 * owner_size
                or r.get('nominal_iterations') != 30000 or r.get('requested_stop_iteration') != end
                or not r.get('mcmc_noise_applied') or len(r.get('batch_view_ids', [])) != 2):
            raise ValueError('native training lifecycle differs')
        for k in ('loss', 'reconstruction_loss', 'opacity_regularization', 'scale_regularization'):
            finite(r[k])
    return selected


def extract(rows, health, train_ids):
    rows = window_rows(rows)
    ids = set(map(str, train_ids))
    sequence = [list(map(str, r['batch_view_ids'])) for r in rows]
    if any(not set(batch) <= ids for batch in sequence):
        raise ValueError('non-Train camera in native progress')
    if [r['iteration'] for r in rows if r.get('mcmc_refinement')] != [EVENT, END]:
        raise ValueError('unexpected relocation cadence')
    metrics = {}
    for rank in (0, 1):
        events = health[str(rank)]
        if [(r['iteration'], r['phase']) for r in events] != [(i, p) for i in (EVENT, END) for p in HEALTH_PHASES]:
            raise ValueError('missing or extra owner health events')
        for r in events:
            h = r['health']
            if (r['rank'] != rank or h['gaussian_count'] != OWNER_SIZE or h['min_opacity'] != .005
                    or h['alive_count'] + h['dead_count'] != OWNER_SIZE
                    or any(h[k] != 0 for k in ('opacity_nonfinite_count', 'log_scale_nan_count',
                        'log_scale_positive_inf_count', 'log_scale_negative_inf_count',
                        'activated_scale_nonfinite_count', 'origin_distance_nonfinite_count'))):
                raise ValueError('owner topology, threshold or finiteness differs')
            count = h['dead_count']
            if type(count) is not int or not 0 <= count <= OWNER_SIZE or h['dead_fraction'] != count / OWNER_SIZE:
                raise ValueError('invalid death count or fraction')
            key = f'owner{rank}/{r["iteration"]}/{r["phase"]}/dead_fraction'
            metrics[key] = count / OWNER_SIZE
        for iteration in (EVENT, END):
            before = next(r['health']['dead_count'] for r in events
                          if r['iteration'] == iteration and r['phase'] == HEALTH_PHASES[0])
            metrics[f'owner{rank}/{iteration}/relocation_fraction'] = before / OWNER_SIZE
    for iteration in (EVENT, END):
        row = next(r for r in rows if r['iteration'] == iteration)
        expected = sum(next(r['health']['dead_count'] for r in health[str(rank)]
                            if r['iteration'] == iteration and r['phase'] == HEALTH_PHASES[0]) for rank in (0, 1))
        if row['mcmc_relocated_count'] != expected or row['mcmc_added_count'] != 0:
            raise ValueError('original all-dead relocation accounting differs')
        loss_rows = [r['reconstruction_loss'] for r in rows if iteration - 19 <= r['iteration'] <= iteration]
        if len(loss_rows) != 20:
            raise ValueError('incomplete twenty-step loss window')
        metrics[f'loss/{iteration}'] = math.fsum(loss_rows) / 20
    return {'metrics': metrics, 'camera_sequence': sequence,
            'relocation_count_method': 'per_owner_native_pre_dead_count_global_actual_count_crosscheck'}


def metric_names():
    return {*(f'owner{r}/{i}/{p}/dead_fraction' for r in (0, 1) for i in (EVENT, END) for p in HEALTH_PHASES),
            *(f'owner{r}/{i}/relocation_fraction' for r in (0, 1) for i in (EVENT, END)),
            *(f'loss/{i}' for i in (EVENT, END))}


def validate_metrics(row):
    if set(row['metrics']) != metric_names():
        raise ValueError('metric coverage differs')
    for name, value in row['metrics'].items():
        finite(value)
        if name.startswith('owner') and not 0 <= value <= 1:
            raise ValueError('fraction out of range')


def calibrate(cases):
    if tuple(cases) != CALIBRATION:
        raise ValueError('calibration must contain only C0/C1/C2 in fixed order')
    for row in cases.values():
        validate_metrics(row)
    sequence = cases['C0']['camera_sequence']
    if any(r['camera_sequence'] != sequence for r in cases.values()):
        raise ValueError('calibration cameras differ')
    limits, exceeded = {}, []
    for name in sorted(metric_names()):
        values = [cases[k]['metrics'][name] for k in CALIBRATION]
        centre = median(values)
        floor = 1 / OWNER_SIZE if name.startswith('owner') else 2e-6 + 2e-4 * abs(centre)
        ceiling = .001 if name.startswith('owner') else 2e-6 + 1e-3 * abs(centre)
        width = max(max(values) - min(values), floor)
        limits[name] = {'centre': centre, 'half_width': width, 'floor': floor, 'ceiling': ceiling}
        if width > ceiling:
            exceeded.append(name)
    return {'status': 'calibration_blocked' if exceeded else 'calibrated', 'exceeded': exceeded,
            'calibration_cases': list(CALIBRATION), 'limits': limits, 'camera_sequence': sequence,
            'scope': 'finite_descriptive_envelope_not_statistical_equivalence'}


def check_case(name, row, calibration):
    if name not in CASES[3:] or calibration['status'] != 'calibrated':
        raise ValueError('holdout requires a frozen successful calibration')
    validate_metrics(row)
    if set(calibration['limits']) != metric_names():
        raise ValueError('calibration metric coverage differs')
    if row['camera_sequence'] != calibration['camera_sequence']:
        raise ValueError('holdout cameras differ')
    differences = {k: abs(v - calibration['limits'][k]['centre']) for k, v in row['metrics'].items()}
    exceeded = [k for k, v in differences.items() if v > calibration['limits'][k]['half_width']]
    return {'case': name, 'passed': not exceeded, 'exceeded': exceeded, 'absolute_differences': differences}


def classify_stop(name, accepted):
    if name.startswith('C') or name.startswith('U'):
        return 'baseline_unstable_or_insufficient'
    return ('observer_suspect' if all(k in accepted for k in ('U0', 'U1', 'U2'))
            else 'observed_outside_envelope_holdout_coverage_incomplete')


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
