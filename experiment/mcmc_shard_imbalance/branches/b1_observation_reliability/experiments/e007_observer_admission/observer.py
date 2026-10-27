"""Shared terminal boundary and an additive, frozen-lifecycle observer adapter."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import gzip
import json
from unittest.mock import patch

from scripts.mcmc_lifecycle_observer import LifecycleObserver, unchanged_rng
from ..e006_connected_propagation.probe import PropagationProbe
from . import admission
from .run import snapshots, sha256, write_new


def tensor_bytes(value):
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return int(getattr(value, 'nbytes', 0))


class Endpoint:
    consensus = PropagationProbe.consensus

    def __init__(self, root, rank, train_ids, reserve, *, start=admission.START,
                 end=admission.END, owner_size=admission.OWNER_SIZE):
        self.root, self.rank, self.train_ids = root, rank, set(map(str, train_ids))
        self.reserve, self.start, self.end, self.owner_size = reserve, start, end, owner_size
        self.loaded, self.calls, self.completed = set(), 0, False
        self.manifest = None

    def terminate(self, model, views, config, *, progress_events, **kwargs):
        def validate():
            if self.calls:
                raise ValueError('duplicate endpoint')
            self.calls += 1
            rows = admission.window_rows(progress_events, start=self.start, end=self.end, owner_size=self.owner_size)
            if model.count != self.owner_size or not self.loaded or not self.loaded <= self.train_ids:
                raise ValueError('endpoint topology or decoded image identity differs')
            if any(not set(map(str, row['batch_view_ids'])) <= self.train_ids for row in rows):
                raise ValueError('non-Train camera in endpoint history')
            import torch
            if set(model.params) != set(snapshots.PARAMETERS) or not all(bool(torch.isfinite(p).all()) for p in model.params.values()):
                raise ValueError('invalid endpoint model parameters')
            return rows
        rows = self.consensus('terminal-validation', validate)

        def save():
            with unchanged_rng():
                payload = {'parameters': dict(model.params.items()), 'rng': snapshots.rng_state()}
                self.reserve(tensor_bytes(payload) + 1024**2)
                self.manifest = snapshots.save_snapshot(self.root / f'endpoint-rank-{self.rank}', payload)
                self.reserve(len(json.dumps(rows).encode()) + 4096)
                write_new(self.root / f'native-tail-rank-{self.rank}.json', rows)
                self.completed = True
        self.consensus('terminal-save', save)
        raise snapshots.ProbeComplete()

    @contextmanager
    def installed(self):
        from image3d_scenegraph.gaussian import trainer, runtime
        load = runtime._ViewSource.load

        def train_only(source):
            image_id = str(source.camera.image_id)
            if image_id not in self.train_ids:
                raise ValueError('Test/Validation image decode forbidden')
            self.loaded.add(image_id)
            return load(source)

        def forbidden(*args, **kwargs):
            raise ValueError('unexpected model export/checkpoint beyond terminal boundary')

        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime._ViewSource, 'load', train_only))
            stack.enter_context(patch.object(trainer, 'evaluate_views', self.terminate))
            for name in ('_write_streaming_checkpoint', 'save_model_snapshot'):
                stack.enter_context(patch.object(trainer, name, forbidden))
            yield self


class CohortObserver(LifecycleObserver):
    """Keep the original algorithms; persist the state they already compute."""
    consensus = PropagationProbe.consensus

    def __init__(self, root, rank, config, reserve, **kwargs):
        super().__init__(root, rank, config, True, **kwargs)
        self.reserve = reserve
        self.saved, self.trajectory = {}, []
        self.contributions = 0
        self.trace_root = root / f'trajectory-rank-{rank}'
        self.trace_root.mkdir(exist_ok=False)

    def capture(self, stage, payload):
        with unchanged_rng():
            self.reserve(tensor_bytes(payload) + 1024**2)
            self.saved[stage] = snapshots.save_snapshot(self.root / f'{stage}-rank-{self.rank}', payload)

    def emit(self, name, record):
        self.reserve(len(json.dumps(record).encode()) + 4096)
        super().emit(name, record)
        if record.get('event') == 'relocation_before':
            self.capture('cohort-before', {'opacities': self.params['opacities'].detach()})
        elif record.get('event') == 'relocation_after':
            self.capture('cohort-initial', {'opacities': self.params['opacities'].detach(),
                'labels': self.labels, 'initial_alive': self.initial_alive})

    def contribution(self, model, view, rendered):
        result = self.consensus(f'contribution-{self.iteration}',
            lambda: super(CohortObserver, self).contribution(model, view, rendered))
        self.contributions += 1
        return result

    def observe_gradients(self, name, optimizer):
        return self.consensus(f'gradient-{name}-{self.iteration}',
            lambda: super(CohortObserver, self).observe_gradients(name, optimizer))

    def observe_update(self, optimizer, args, kwargs):
        return self.consensus(f'opacity-update-{self.iteration}', lambda: self._observe_update(optimizer, args, kwargs))

    def _observe_update(self, optimizer, args, kwargs):
        recording = self.pre_opacity is not None
        super().observe_update(optimizer, args, kwargs)
        if not recording:
            return
        with unchanged_rng():
            import numpy as np
            alive = (self.params['opacities'].detach().sigmoid().flatten() > .005).cpu().numpy()
            packed = gzip.compress(np.packbits(alive, bitorder='little').tobytes(), mtime=0)
            self.reserve(len(packed) + 4096)
            path = self.trace_root / f'{self.iteration}.bits.gz'
            with path.open('xb') as handle:
                handle.write(packed)
            self.trajectory.append({'iteration': self.iteration, 'path': path.name, 'slot_count': alive.size,
                'alive_count': int(alive.sum()), 'bytes': len(packed), 'sha256': sha256(path),
                'encoding': 'gzip_numpy_packbits_little_zero_based_slots', 'phase': 'after_Adam_before_strategy'})

    def finish_cohort(self):
        super().finish_cohort()
        if [r['iteration'] for r in self.trajectory] != list(range(self.event + 1, self.end + 1)):
            raise ValueError('incomplete or reordered cohort trajectory')
        self.capture('cohort-final', {'opacities': self.params['opacities'].detach(), 'labels': self.labels,
            'initial_alive': self.initial_alive, 'alive': self.params['opacities'].detach().sigmoid().flatten() > .005,
            'first_death': self.first_death,
            'first_recovery': self.first_recovery, 'exposure': self.exposure})
        write_new(self.trace_root / 'index.json', {'owner': self.rank, 'event': self.event,
            'end_before_second_relocation': self.end, 'rows': self.trajectory})

    @contextmanager
    def installed(self):
        from image3d_scenegraph.gaussian import trainer
        with super().installed():
            build = trainer._build_mcmc_strategy

            def coordinated_build(*args, **kwargs):
                strategy = build(*args, **kwargs)
                before, after = strategy.step_pre_backward, strategy.step_post_backward
                strategy.step_pre_backward = lambda *a, **k: self.consensus(
                    f'strategy-pre-{self.iteration}', lambda: before(*a, **k))
                strategy.step_post_backward = lambda *a, **k: self.consensus(
                    f'strategy-post-{self.iteration}', lambda: after(*a, **k))
                return strategy

            with patch.object(trainer, '_build_mcmc_strategy', coordinated_build):
                yield self

    def finish(self):
        if (self.emitted != self.end - self.start or not self.cohort_complete
                or self.contributions != self.end - self.start
                or set(self.saved) != {'cohort-before', 'cohort-initial', 'cohort-final'}
                or len(self.trajectory) != self.end - self.event):
            raise ValueError('incomplete observed lifecycle')
        return {'strategy_posts': self.emitted, 'contribution_forwards': self.contributions,
                'manifests': self.saved, 'trajectory_index_sha256': sha256(self.trace_root / 'index.json')}
