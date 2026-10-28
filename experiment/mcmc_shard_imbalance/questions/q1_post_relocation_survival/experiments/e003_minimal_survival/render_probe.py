"""Frozen Train views, event-before/event-after states; runs only after training exits."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys

from .analyze import state
from .observer import snapshots
from scripts.mcmc_lifecycle_observer import unchanged_rng
from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0.run import initialize_backend


def rng_components():
    return {key: snapshots.digest_state(value) for key, value in snapshots.rng_state().items()}


def initialize_render(audit_root, rank):
    cold = 'gsplat.cuda._backend' not in sys.modules
    record = initialize_backend()
    record.update(rank=rank, backend_loaded_before=not cold, initialization_outside_measurement=True,
                  rng_restored=False)
    snapshots.write_new(snapshots.safe_file(audit_root, f'initialization-rank-{rank}.json'), record)
    return record


@contextmanager
def audited_rng(audit_root, rank, label):
    before, error = rng_components(), None
    try:
        yield
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        after = rng_components()
        changed = [k for k in before if before[k] != after[k]]
        snapshots.write_new(snapshots.safe_file(audit_root, f'rng-{label}-rank-{rank}.json'),
            {'rank': rank, 'phase': label, 'before': before, 'after': after,
             'changed_rng_components': changed, 'error': error, 'rng_restored': False})
        if changed and error is None:
            raise ValueError(f'{label}: observer consumed RNG ({", ".join(changed)}); state was not restored or hidden')


def select_cameras(ids, count=8):
    ids = list(map(str, ids))
    if count <= 0 or count % 2 or len(ids) != len(set(ids)) or len(ids) < count:
        raise ValueError('unique Train IDs and an even frozen camera count required')
    return sorted(ids, key=lambda s: (hashlib.sha256(s.encode()).hexdigest(), s))[:count]


def camera_contract(dataset, config, count=8):
    ids = select_cameras(dataset['splits']['train'], count)
    images = {str(r['image_id']): r for r in dataset['images']}
    if not set(ids) <= images.keys():
        raise ValueError('selected Train metadata missing')
    return {'image_ids': ids, 'images': [images[i] for i in ids], 'normalization': dataset['normalization'],
            'resolution': config['resolution'], 'sh_degree': 3, 'loss': config['loss'],
            'selection': 'sha256_utf8_image_id_then_id_before_execution'}


def model_from_snapshot(folder, device):
    import numpy as np
    import torch
    from image3d_scenegraph.gaussian.model import GaussianModel
    value = state(folder)
    if set(value['parameters']) != set(snapshots.PARAMETERS):
        raise ValueError('incomplete event parameters')
    arrays = value['parameters']
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('invalid render parameter dtype/finiteness')
    p = {n: torch.from_numpy(np.array(a, copy=True)).to(device) for n, a in arrays.items()}
    degree = math.isqrt(1 + p['shN'].shape[1]) - 1
    if (degree + 1)**2 != 1 + p['shN'].shape[1]:
        raise ValueError('invalid spherical harmonics shape')
    return GaussianModel(means=p['means'], log_scales=p['scales'], quats=p['quats'],
                         opacity_logits=p['opacities'], sh_coeffs=torch.cat((p['sh0'], p['shN']), dim=1), max_sh_degree=degree)


def comparison(left, right):
    rows = {k: snapshots.compare_arrays(left[k], right[k]) for k in ('rgb', 'alpha')}
    if any(r['status'] == 'structure_mismatch' or r['num_nonfinite'] for r in rows.values()):
        raise ValueError('render arrays have incompatible structure or nonfinite values')
    return rows


def repeat_usable(rows):
    return set(rows) == {'rgb', 'alpha'} and all(r['status'] in ('bitwise_equal', 'numerically_equivalent') for r in rows.values())


def probe(stage, rank, views, camera_ids, config, *, audit_root, sh_degree=3):
    import numpy as np
    import torch
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.training_math import l1_ssim_loss
    audit_root = Path(audit_root)
    if audit_root.resolve().is_relative_to(stage.resolve()):
        raise ValueError('render audit must not write inside source snapshots')
    audit_root.mkdir(parents=True, exist_ok=True)
    initialization = initialize_render(audit_root, rank)
    ids = [str(v.image_id) for v in (views.cameras if hasattr(views, 'cameras') else [v.camera for v in views])]
    if (len(camera_ids) % 2 or not camera_ids or len(set(camera_ids)) != len(camera_ids)
            or any(ids.count(i) != 1 for i in camera_ids)):
        raise ValueError('frozen render camera set differs')
    assigned = camera_ids[rank::2]
    before, repeat_checks, results = {}, {}, {}
    calls, audits = 0, []
    for phase in ('before', 'after'):
        model = model_from_snapshot(stage / f'event-{phase}-rank-{rank}', torch.device(f'cuda:{rank}'))
        first = {}
        for repetition in range(2):
            for position, image_id in enumerate(assigned):
                prefix = f'{phase}-{repetition}-{position}'
                with unchanged_rng(), torch.no_grad():
                    with audited_rng(audit_root, rank, prefix + '-load'):
                        prepared = views[ids.index(image_id)].to(model.means.device)
                    with audited_rng(audit_root, rank, prefix + '-render'):
                        view, rendered = trainer._render_visible_training_view(model, [prepared], 0, sh_degree, distributed=True)
                        calls += 1
                    with audited_rng(audit_root, rank, prefix + '-loss'):
                        prediction = rendered.image.clamp(0, 1) if config['loss']['clamp_render'] else rendered.image
                        loss, terms = l1_ssim_loss(prediction, view.image, l1_weight=config['loss']['l1_weight'],
                                                  ssim_weight=config['loss']['ssim_weight'])
                    with audited_rng(audit_root, rank, prefix + '-export'):
                        values = {'rgb': rendered.image.detach().cpu().numpy().copy(),
                                  'alpha': rendered.alpha.detach().cpu().numpy().copy(),
                                  'loss': float(loss), **terms}
                        if not all(np.isfinite(values[k]).all() for k in ('rgb', 'alpha')) or not math.isfinite(values['loss']):
                            raise ValueError('nonfinite isolated render')
                audits.extend(f'rng-{prefix}-{part}-rank-{rank}.json' for part in ('load', 'render', 'loss', 'export'))
                if repetition == 0:
                    first[image_id] = values
                    if phase == 'after':
                        results[image_id] = {'pixel_comparison': comparison(values, before[image_id]),
                            'before': {k: before[image_id][k] for k in ('loss', 'l1', 'ssim')},
                            'after': {k: values[k] for k in ('loss', 'l1', 'ssim')},
                            'loss_change': values['loss'] - before[image_id]['loss']}
                else:
                    check = comparison(values, first[image_id])
                    repeat_checks[f'{phase}/{image_id}'] = {'pixels': check,
                        'loss': snapshots.numeric_pair(values['loss'], first[image_id]['loss']),
                        'usable': repeat_usable(check) and not snapshots.numeric_pair(values['loss'], first[image_id]['loss'])['mismatch']}
        if phase == 'before':
            before = first
        del model
    usable = all(r['usable'] for r in repeat_checks.values())
    return {'status': 'measured', 'rank': rank, 'initialization': initialization,
            'rng_audit_sha256': {n: snapshots.sha256_file(audit_root / n) for n in audits}, 'image_ids': assigned, 'camera_forwards': calls,
            'backwards': 0, 'optimizer_steps': 0, 'repeat_checks': repeat_checks, 'effects': results,
            'interpretation_usable': usable, 'atol': snapshots.ATOL, 'rtol': snapshots.RTOL,
            'interpretation': 'immediate_render_effect_only' if usable else 'same_state_repeat_outside_original_tolerance',
            'discrete_selection_causality_proven': False, 'long_term_quality_effect_proven': False}
