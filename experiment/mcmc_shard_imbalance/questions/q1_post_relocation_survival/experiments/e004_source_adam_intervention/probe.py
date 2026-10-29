"""Research-only prefix capture and source-moment intervention around native calls."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.mcmc_lifecycle_observer import unchanged_rng, digest_state, distribution, regularizer_gradient
from ..e003_minimal_survival.observer import MinimalObserver, Endpoint, tensor_bytes
from ..e003_minimal_survival.run import snapshots, read, write_new
from . import analyze


def moment_digest(value):
    # Empty tensors have no data bytes; keep the frozen tensor dtype/shape encoding.
    if value.numel() == 0:
        return digest_state(('tensor', str(value.dtype), list(value.shape)))
    return digest_state(value)


class InterventionProbe:
    consensus = Endpoint.consensus
    full_state = snapshots.FirstDivergenceProbe.full_state
    record_masks = MinimalObserver.record_masks

    def __init__(self, root, rank, case, train_ids, reserve, *, prefix=None, start=12000, event=12100,
                 end=12120, count=1500000, training=None, provenance=None):
        if case not in ('prefix', 'A', 'B') or not start < event < end:
            raise ValueError('invalid intervention window/case')
        self.root, self.rank, self.case = Path(root), rank, case
        self.prefix = Path(prefix) if prefix is not None else self.root
        self.start, self.event, self.end = (start, event, event) if case == 'prefix' else (event, event, end)
        self.expected_count = self.owner_size = count
        self.train_ids, self.reserve = set(map(str, train_ids)), reserve
        self.training, self.provenance = training, provenance
        self.params = self.optimizers = self.state = None
        self.camera, self.schedule, self.noise_schedule = {}, [], []
        self.manifests, self.rows, self.loaded = {}, [], set()
        self.handles, self.step_names, self.update_summaries = [], [], {}
        self.context = self.labels = self.sources = self.targets = self.ids = None
        self.before_values, self.config, self.writing_checkpoint = {}, None, False
        self.iteration, self.completed, self.restored_digest = self.start, False, None
        self.calls = dict.fromkeys(('restore', 'batch', 'render', 'backward', 'raster_forward', 'raster_backward',
                                   'optimizer', 'strategy', 'relocate', 'sample', 'add', 'noise', 'terminal', 'checkpoint'), 0)
        self.trace = self.root / f'trajectory-rank-{rank}'
        self.trace.mkdir(exist_ok=False)

    def capture(self, name, payload):
        with unchanged_rng():
            self.reserve(tensor_bytes(payload) + 1024**2)
            self.manifests[name] = snapshots.save_snapshot(self.root / f'{name}-rank-{self.rank}', payload)

    @unchanged_rng()
    def validate_live(self):
        import torch
        value = self.full_state()
        for name, opt in self.optimizers.items():
            if type(opt) is not torch.optim.Adam:
                raise ValueError('only native torch.optim.Adam supported')
            analyze.optimizer_options(value['adam'][name]['param_groups'][0])
            if set(value['adam'][name]['state']) != {*analyze.MOMENTS, 'step'}:
                raise ValueError('unexpected Adam state fields')
        value['optimizer_classes'] = {n: type(o).__module__ + '.' + type(o).__name__ for n, o in self.optimizers.items()}
        return value

    @unchanged_rng()
    def protected_digest(self):
        keep = self.labels != 1
        value = self.full_state()
        adam = {name: {'param_groups': row['param_groups'], 'step': row['state']['step'],
                      **{key: row['state'][key][keep] for key in analyze.MOMENTS}} for name, row in value['adam'].items()}
        return digest_state({**value, 'adam': adam})

    @unchanged_rng()
    def restore_source_moments(self):
        import numpy as np
        import torch
        before = analyze.state(self.prefix / f'pre-relocation-rank-{self.rank}')
        ids = self.ids.cpu().numpy()
        protected = self.protected_digest()
        changed = {}
        with unchanged_rng(), torch.no_grad():
            for name, opt in self.optimizers.items():
                active = opt.state[self.params[name]]
                changed[name] = {}
                for key in analyze.MOMENTS:
                    current = active[key][self.ids]
                    if bool(current.count_nonzero()):
                        raise ValueError('source moments were not zero in common checkpoint')
                    expected = torch.from_numpy(np.array(before['adam'][name]['state'][key][ids], copy=True)).to(active[key].device)
                    changed[name][key] = {'before': moment_digest(current), 'expected_retained': moment_digest(expected)}
                    if self.case == 'B':
                        active[key][self.ids] = expected
                    actual = active[key][self.ids]
                    reference = expected if self.case == 'B' else current
                    if moment_digest(actual) != moment_digest(reference):
                        raise ValueError('source moment intervention did not match whitelist')
                    changed[name][key]['after'] = moment_digest(actual)
            if self.protected_digest() != protected:
                raise ValueError('intervention changed parameters, T/U, step, options, strategy, camera or RNG')
        write_new(self.root / f'intervention-rank-{self.rank}.json',
            {'case': self.case, 'rank': self.rank, 'protected_before': protected, 'protected_after': protected,
             'only_source_moments_changed': True, 'sources': int(self.ids.numel()), 'moments': changed})

    @unchanged_rng()
    def initialize_arm(self):
        import numpy as np
        import torch
        self.restored_digest = digest_state(self.validate_live())
        reference = read(self.prefix / f'prefix-state-rank-{self.rank}.json')
        if self.restored_digest != reference['full_state_sha256']:
            raise ValueError('native checkpoint did not restore full common prefix')
        cohort = analyze.state(self.prefix / f'cohort-rank-{self.rank}')
        device = self.params['means'].device
        self.labels = torch.from_numpy(np.array(cohort['labels'], copy=True)).to(device)
        self.sources = torch.from_numpy(np.array(cohort['sources'], copy=True)).to(device)
        self.targets = torch.from_numpy(np.array(cohort['targets'], copy=True)).to(device)
        self.ids = self.sources.unique()
        self.restore_source_moments()
        self.initial_masks = self.record_masks(self.params['opacities'].detach().sigmoid(), 'initial')

    def before_relocation(self):
        import torch
        if self.case != 'prefix' or self.iteration != self.event or self.calls['relocate']:
            raise ValueError('unexpected relocation')
        self.calls['relocate'] += 1
        with unchanged_rng(), torch.no_grad():
            alpha = self.params['opacities'].detach().sigmoid()
            self.context = {'alpha': alpha, 'targets': (alpha <= .005).nonzero().flatten(),
                            'alive': (alpha > .005).nonzero().flatten(), 'sources': None, 'samples': 0}
            self.capture('pre-relocation', {**self.validate_live(), 'iteration': self.iteration,
                                           'phase': 'after_Adam_before_relocation'})

    def record_sample(self, indices, n):
        import torch
        c = self.context
        if (c is None or c['samples'] or indices.dtype != torch.int64 or indices.ndim != 1
                or len(indices) != n or len(c['targets']) != n
                or bool(((indices < 0) | (indices >= len(c['alive']))).any())):
            raise ValueError('unexpected source sample')
        with unchanged_rng():
            c['sources'] = c['alive'][indices].detach().clone()
            c['samples'] += 1

    def after_relocation(self, count):
        import torch
        c = self.context
        if count != len(c['targets']) or c['samples'] != int(count > 0):
            raise ValueError('relocation/sample count differs')
        with unchanged_rng(), torch.no_grad():
            self.targets = c['targets']
            self.sources = c['sources'] if count else torch.empty_like(self.targets)
            self.ids = self.sources.unique()
            self.labels = torch.full((self.owner_size,), 2, dtype=torch.int8, device=self.params['means'].device)
            self.labels[self.ids], self.labels[self.targets] = 1, 0
            self.capture('post-relocation', {**self.validate_live(), 'iteration': self.iteration,
                                            'phase': 'after_relocation_before_noise'})
            self.capture('cohort', {'owner': self.rank, 'event': self.event, 'labels': self.labels,
                'sources': self.sources, 'targets': self.targets, 'before_alpha': c['alpha'],
                'initial_alpha': self.params['opacities'].detach().sigmoid()})
            self.context = None

    def optimizer_before(self, name):
        import torch
        if name != snapshots.PARAMETERS[len(self.step_names)]:
            raise ValueError('optimizer order differs')
        h = self.iteration - self.event
        if self.case == 'prefix' or h not in (*analyze.HORIZONS, self.end - self.event):
            return
        with unchanged_rng(), torch.no_grad():
            p = self.params[name]
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise ValueError('missing/nonfinite native gradient')
            if name == 'opacities' or h == 1:
                self.before_values[name] = p.detach().clone()
            if h == 1:
                norms = p.grad.detach().reshape(self.owner_size, -1).norm(dim=1)
                self.update_summaries[name] = {'gradient_norm': {g: distribution(norms[self.labels == i]) for i, g in enumerate(('T', 'S', 'U'))}}
            if name == 'opacities':
                opt = self.optimizers[name]
                moment = opt.state[p]
                self.capture(f'opacity-before-{h}', {'logits': p.detach(), 'gradient': p.grad.detach(),
                    'regularizer_gradient': regularizer_gradient(name, p, self.config, 2),
                    **{key: moment[key] for key in (*analyze.MOMENTS, 'step')},
                    'options': {k: v for k, v in opt.param_groups[0].items() if k != 'params'}})

    def optimizer_after(self, name):
        import torch
        self.calls['optimizer'] += 1
        self.step_names.append(name)
        h = self.iteration - self.event
        if self.case == 'prefix':
            return
        with unchanged_rng(), torch.no_grad():
            if name in self.before_values:
                delta = self.params[name].detach() - self.before_values.pop(name)
                values = delta if name == 'opacities' else delta.reshape(self.owner_size, -1).norm(dim=1)
                row = {g: distribution(values[self.labels == i]) for i, g in enumerate(('T', 'S', 'U'))}
                if h == 1:
                    self.update_summaries[name]['actual_update'] = row
                if name == 'opacities':
                    self.capture(f'opacity-after-{h}', {'logits': self.params[name].detach(),
                        **{key: self.optimizers[name].state[self.params[name]][key] for key in (*analyze.MOMENTS, 'step')}})
                    write_new(self.root / f'opacity-summary-{h}-rank-{self.rank}.json', {'iteration': self.iteration, 'groups': row})
            if h == 1 and len(self.step_names) == 6:
                write_new(self.root / f'first-update-summary-rank-{self.rank}.json', self.update_summaries)

    def terminate(self, model, views, config, *, progress_events, **kwargs):
        @unchanged_rng()
        def validate():
            self.calls['terminal'] += 1
            if self.calls['terminal'] != 1 or self.iteration != self.end or self.calls['strategy'] != self.end - self.start:
                raise ValueError('unexpected evaluation boundary')
            from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission.admission import window_rows
            rows = window_rows(progress_events, start=self.start, end=self.end, owner_size=self.owner_size)
            if not self.loaded or not self.loaded <= self.train_ids or model.count != self.owner_size:
                raise ValueError('terminal decoded Train set/topology differs')
            if any(not set(map(str, r['batch_view_ids'])) <= self.train_ids for r in rows):
                raise ValueError('non-Train native history')
            write_new(self.root / f'native-tail-rank-{self.rank}.json', rows)
            return rows
        self.consensus('terminal', validate)
        if self.case == 'prefix':
            @unchanged_rng()
            def boundary():
                value = self.validate_live()
                write_new(self.root / f'prefix-state-rank-{self.rank}.json',
                          {'full_state_sha256': digest_state(value), 'iteration': self.end,
                           'phase': 'after_original_strategy_and_noise_before_evaluation'})
            self.consensus('prefix-state', boundary)
            # Native streaming writer performs its own coordinated atomic publication.
            self.consensus('checkpoint-reserve', lambda: self.reserve(3 * tensor_bytes(self.full_state()) + 1024**2))
            self.writing_checkpoint = True
            try:
                with unchanged_rng():
                    self.writer(self.training, attempt_id='prefix', iteration=self.end, provenance=self.provenance,
                        model=model, optimizers=self.optimizers, strategy_state=self.state,
                        camera_order=self.camera['order'], camera_cursor=self.camera['cursor'], history=progress_events,
                        world_rank=self.rank, world_size=2,
                        execution_budget={'nominal_iterations': 30000, 'requested_stop_iteration': self.end})
            finally:
                self.writing_checkpoint = False
            self.calls['checkpoint'] += 1
        else:
            self.consensus('trajectory-index', lambda: write_new(self.trace / 'index.json',
                {'owner': self.rank, 'event': self.event, 'end': self.end, 'initial': self.initial_masks, 'rows': self.rows}))
        self.completed = True
        raise snapshots.ProbeComplete()

    @contextmanager
    def installed(self):
        import torch
        from gsplat.strategy import ops
        from gsplat.strategy import mcmc
        from gsplat.cuda._wrapper import _RasterizeToPixels
        from image3d_scenegraph.gaussian import trainer, runtime
        originals = {k: getattr(trainer, k) for k in ('_build_mcmc_strategy', '_torch_load', '_restore_rng',
                     '_next_camera_batch', '_render_visible_training_view', '_write_streaming_checkpoint')}
        self.writer = originals['_write_streaming_checkpoint']
        sample, noise = ops._multinomial_sample, mcmc.inject_noise_to_position
        forward, backward, tensor_backward = _RasterizeToPixels.forward, _RasterizeToPixels.backward, torch.Tensor.backward
        load, save_model = runtime._ViewSource.load, trainer.save_model_snapshot

        def loaded(*a, **kw):
            result = originals['_torch_load'](*a, **kw)
            if isinstance(result, dict) and {'strategy_state', 'camera_order', 'camera_cursor'} <= result.keys():
                if self.state is not None:
                    raise ValueError('duplicate strategy restore')
                self.state = result['strategy_state']
                self.camera = {'order': list(result['camera_order']), 'cursor': int(result['camera_cursor'])}
            return result

        def restored(*a, **kw):
            result = originals['_restore_rng'](*a, **kw)
            self.calls['restore'] += 1
            return result

        def build(*a, **kw):
            strategy = originals['_build_mcmc_strategy'](*a, **kw)
            self.config = a[1] if len(a) > 1 else kw['config']
            sanity, post, relocate, add = strategy.check_sanity, strategy.step_post_backward, strategy._relocate_gs, strategy._add_new_gs
            def checked(params, optimizers):
                result = sanity(params, optimizers)
                self.params, self.optimizers = params, optimizers
                return result
            def relocated(*args, **kwargs):
                self.before_relocation()
                result = relocate(*args, **kwargs)
                self.after_relocation(result)
                return result
            def added(*args, **kwargs):
                self.calls['add'] += 1
                if self.case != 'prefix' or self.iteration != self.event or add(*args, **kwargs) != 0:
                    raise ValueError('unexpected topology addition')
                return 0
            def posted(*args, **kwargs):
                def action():
                    if len(self.step_names) != 6 or kwargs['step'] != self.iteration or kwargs['state'] is not self.state:
                        raise ValueError('strategy before six ordered Adam updates')
                    if self.case != 'prefix':
                        with unchanged_rng(), torch.no_grad():
                            alpha = self.params['opacities'].detach().sigmoid()
                            self.rows.append({'iteration': self.iteration, 'phase': 'after_Adam_before_strategy',
                                              **self.record_masks(alpha, str(self.iteration))})
                            if self.iteration == self.end:
                                self.capture('final-opacity', {'alpha': alpha, 'logits': self.params['opacities'].detach()})
                    result = post(*args, **kwargs)
                    if len(self.params['means']) != self.owner_size:
                        raise ValueError('owner topology changed')
                    self.calls['strategy'] += 1
                    if self.case == 'prefix' and self.iteration == self.event:
                        with unchanged_rng():
                            proof = analyze.gate0(self.root, self.rank)
                            write_new(self.root / f'gate0-rank-{self.rank}.json', proof)
                    return result
                return self.consensus(f'post-{self.iteration}', action)
            strategy.check_sanity, strategy.step_post_backward = checked, posted
            strategy._relocate_gs, strategy._add_new_gs = relocated, added
            return strategy

        def batch(count, order, cursor, size):
            self.iteration = self.start + self.calls['batch'] + 1
            if self.iteration > self.end or self.calls['restore'] != 1 or self.state is None:
                raise ValueError('batch outside complete native resume window')
            self.camera = {'order': list(order), 'cursor': cursor}
            if not self.calls['batch']:
                if self.case != 'prefix':
                    self.consensus('arm-initialization', self.initialize_arm)
                else:
                    self.consensus('prefix-initialization', self.validate_live)
                for name, opt in self.optimizers.items():
                    self.handles.append(opt.register_step_pre_hook(lambda o, a, kw, n=name: self.consensus('before-' + n, lambda: self.optimizer_before(n))))
                    self.handles.append(opt.register_step_post_hook(lambda o, a, kw, n=name: self.consensus('after-' + n, lambda: self.optimizer_after(n))))
            self.step_names = []
            result = originals['_next_camera_batch'](count, order, cursor, size)
            self.camera = {'order': list(result[0]), 'cursor': result[1]}
            self.batch_indices = list(result[2])
            self.calls['batch'] += 1
            return result

        def render(*a, **kw):
            if self.calls['render'] != self.calls['batch'] - 1:
                raise ValueError('extra training render')
            result = originals['_render_visible_training_view'](*a, **kw)
            self.calls['render'] += 1
            self.schedule.append({'iteration': self.iteration, 'indices': self.batch_indices,
                'image_id': str(result[0].camera.image_id), 'sh_degree': a[3],
                'learning_rates': {n: o.param_groups[0]['lr'] for n, o in self.optimizers.items()}})
            return result

        def sampled(weights, n, **kw):
            result = sample(weights, n, **kw)
            self.calls['sample'] += 1
            self.record_sample(result, n)
            return result

        def noisy(*a, **kw):
            selected = self.labels is not None and (self.case == 'prefix' or self.iteration - self.event in (*analyze.HORIZONS, self.end - self.event))
            with unchanged_rng(), torch.no_grad():
                rng_before = digest_state(snapshots.rng_state())
                slots = (self.labels != 2).nonzero().flatten() if selected else None
                before = self.params['means'][slots].detach().clone() if selected else None
            result = noise(*a, **kw)
            self.calls['noise'] += 1
            with unchanged_rng():
                self.noise_schedule.append({'iteration': self.iteration, 'rng_before': rng_before, 'rng_after': digest_state(snapshots.rng_state())})
                if selected:
                    self.capture(f'noise-{self.iteration}', {'slots': slots, 'before': before, 'after': self.params['means'][slots].detach(),
                                                         'phase': 'original_position_noise_only'})
            return result

        def counted_forward(*a, **kw):
            self.calls['raster_forward'] += 1
            return forward(*a, **kw)
        def counted_backward(*a, **kw):
            self.calls['raster_backward'] += 1
            return backward(*a, **kw)
        def backward_once(tensor, *a, **kw):
            self.calls['backward'] += 1
            if self.calls['backward'] != self.calls['render']:
                raise ValueError('extra/missing native loss backward')
            return tensor_backward(tensor, *a, **kw)
        def train_only(source):
            identity = str(source.camera.image_id)
            if identity not in self.train_ids:
                raise ValueError('Test/Validation decode forbidden')
            self.loaded.add(identity)
            return load(source)
        def forbidden(*a, **kw):
            raise ValueError('unplanned checkpoint/export forbidden')
        def checkpoint_model(*a, **kw):
            if not self.writing_checkpoint:
                return forbidden(*a, **kw)
            return save_model(*a, **kw)
        try:
            with ExitStack() as stack:
                for name, fn in {'_build_mcmc_strategy': build, '_torch_load': loaded, '_restore_rng': restored,
                                 '_next_camera_batch': batch, '_render_visible_training_view': render,
                                 'evaluate_views': self.terminate, '_write_streaming_checkpoint': forbidden}.items():
                    stack.enter_context(patch.object(trainer, name, fn))
                stack.enter_context(patch.object(trainer, 'save_model_snapshot', checkpoint_model))
                stack.enter_context(patch.object(runtime._ViewSource, 'load', train_only))
                stack.enter_context(patch.object(ops, '_multinomial_sample', sampled))
                stack.enter_context(patch.object(mcmc, 'inject_noise_to_position', noisy))
                stack.enter_context(patch.object(_RasterizeToPixels, 'forward', staticmethod(counted_forward)))
                stack.enter_context(patch.object(_RasterizeToPixels, 'backward', staticmethod(counted_backward)))
                stack.enter_context(patch.object(torch.Tensor, 'backward', backward_once))
                yield self
        finally:
            for handle in self.handles:
                handle.remove()

    def finish(self):
        n = self.end - self.start
        expected = {'restore': 1, 'batch': n, 'render': n, 'backward': n, 'raster_forward': n, 'raster_backward': n,
                    'optimizer': 6 * n, 'strategy': n, 'relocate': int(self.case == 'prefix'),
                    'sample': int(self.case == 'prefix' and len(self.targets) > 0), 'add': int(self.case == 'prefix'),
                    'noise': n, 'terminal': 1, 'checkpoint': int(self.case == 'prefix')}
        if not self.completed or self.calls != expected or len(self.schedule) != n or len(self.noise_schedule) != n:
            raise ValueError('incomplete intervention worker counts')
        return {'status': 'measured', 'case': self.case, 'rank': self.rank, 'calls': self.calls,
                'new_updates': n, 'manifests': self.manifests, 'schedule': self.schedule, 'noise_schedule': self.noise_schedule,
                'restored_digest': self.restored_digest, 'loaded_train_ids': sorted(self.loaded),
                'validation_evaluated': False, 'native_training_complete': False}
