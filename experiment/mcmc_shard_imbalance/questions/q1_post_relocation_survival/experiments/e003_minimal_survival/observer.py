"""Minimal event identities and opacity states; no render/backward/optimizer hooks."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import gzip
from unittest.mock import patch

from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission.observer import Endpoint, tensor_bytes
from experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission.run import snapshots, sha256, write_new
from scripts.mcmc_lifecycle_observer import unchanged_rng


class MinimalObserver:
    consensus = Endpoint.consensus

    def __init__(self, root, rank, reserve, *, window=(12000, 12100, 12200), owner_size=1500000):
        self.root, self.rank, self.reserve = root, rank, reserve
        self.start, self.event, self.end = window
        self.owner_size = owner_size
        self.iteration, self.posts, self.sample_calls = self.start, 0, 0
        self.params = self.context = self.labels = None
        self.manifests, self.rows = {}, []
        self.relocations = []
        self.trace = root / f'trajectory-rank-{rank}'
        self.trace.mkdir(exist_ok=False)

    def capture(self, name, payload):
        with unchanged_rng():
            self.reserve(tensor_bytes(payload) + 1024**2)
            self.manifests[name] = snapshots.save_snapshot(self.root / f'{name}-rank-{self.rank}', payload)

    def record_masks(self, alpha, name):
        import numpy as np
        result = {}
        for kind, values in (('alive', alpha > .005), ('below_cutoff', alpha < 1 / 255)):
            bits = values.detach().cpu().numpy().reshape(-1)
            payload = gzip.compress(np.packbits(bits, bitorder='little').tobytes(), mtime=0)
            self.reserve(len(payload) + 4096)
            path = self.trace / f'{name}-{kind}.bits.gz'
            with path.open('xb') as handle:
                handle.write(payload)
            result[kind] = {'path': path.name, 'slot_count': bits.size, 'xor_count': int(bits.sum()),
                'bytes': len(payload), 'sha256': sha256(path), 'encoding': 'gzip_numpy_packbits_little_zero_based_slots'}
        return result

    def before_relocation(self):
        import torch
        with unchanged_rng(), torch.no_grad():
            alpha = self.params['opacities'].detach().sigmoid().flatten()
            if not bool(torch.isfinite(alpha).all()):
                raise ValueError('nonfinite event opacity')
            targets, alive = (alpha <= .005).nonzero().flatten(), (alpha > .005).nonzero().flatten()
            self.context = {'targets': targets, 'alive': alive, 'sources': torch.empty_like(targets), 'calls': 0, 'alpha': alpha}
            self.capture('event-before', {'parameters': dict(self.params.items()), 'owner': self.rank,
                                         'event': self.event, 'phase': 'after_Adam_before_relocation'})

    def record_sample(self, indices, n):
        import torch
        with unchanged_rng(), torch.no_grad():
            c = self.context
            if (c is None or c['calls'] or n != c['targets'].numel() or indices.ndim != 1
                    or indices.numel() != n or indices.dtype != torch.int64
                    or bool(((indices < 0) | (indices >= c['alive'].numel())).any())):
                raise ValueError('unexpected original source sample')
            c['sources'] = c['alive'][indices].detach().clone()
            c['calls'] += 1

    def after_relocation(self, count):
        import torch
        with unchanged_rng(), torch.no_grad():
            c = self.context
            n = c['targets'].numel()
            if count != n or c['calls'] != int(n > 0):
                raise ValueError('original relocation count or sampler calls differ')
            self.labels = torch.full((self.owner_size,), 2, dtype=torch.int8, device=self.params['opacities'].device)
            self.labels[c['sources']] = 1
            self.labels[c['targets']] = 0
            alpha = self.params['opacities'].detach().sigmoid().flatten()
            self.capture('event-after', {'parameters': dict(self.params.items()), 'owner': self.rank,
                                        'event': self.event, 'phase': 'after_relocation_before_noise'})
            self.capture('cohort', {'owner': self.rank, 'event': self.event, 'labels': self.labels,
                'targets': c['targets'], 'sources': c['sources'], 'before_alpha': c['alpha'], 'initial_alpha': alpha})
            self.initial_masks = self.record_masks(alpha, 'initial')
            self.context = None

    def before_post(self, params, step):
        import torch
        if step != self.start + self.posts + 1 or len(params['means']) != self.owner_size:
            raise ValueError('unexpected step or topology')
        self.params, self.iteration = params, step
        if self.event < step <= self.end:
            with unchanged_rng(), torch.no_grad():
                if self.labels is None:
                    raise ValueError('missing cohort before trajectory')
                alpha = params['opacities'].detach().sigmoid().flatten()
                if not bool(torch.isfinite(alpha).all()):
                    raise ValueError('nonfinite trajectory opacity')
                self.rows.append({'iteration': step, 'phase': 'after_Adam_before_strategy',
                                  **self.record_masks(alpha, str(step))})
                if step == self.end:
                    self.capture('cohort-final', {'owner': self.rank, 'event': self.event,
                        'opacities': params['opacities'].detach(), 'alpha': alpha, 'labels': self.labels})
                    self.reserve(65536)
                    write_new(self.trace / 'index.json', {'owner': self.rank, 'event': self.event, 'end': self.end,
                        'initial': self.initial_masks, 'rows': self.rows})

    @contextmanager
    def installed(self):
        from gsplat.strategy import ops
        from image3d_scenegraph.gaussian import trainer
        build, sample = trainer._build_mcmc_strategy, ops._multinomial_sample

        def sampled(weights, n, **kwargs):
            indices = sample(weights, n, **kwargs)
            self.sample_calls += 1
            if self.iteration == self.event:
                self.record_sample(indices, n)
            return indices

        def built(*args, **kwargs):
            strategy = build(*args, **kwargs)
            post, relocate = strategy.step_post_backward, strategy._relocate_gs

            def relocated(*a, **k):
                if self.iteration not in (self.event, self.end):
                    raise ValueError('unexpected relocation event')
                self.relocations.append(self.iteration)
                if self.iteration == self.event:
                    self.before_relocation()
                result = relocate(*a, **k)
                if self.iteration == self.event:
                    self.after_relocation(result)
                return result

            def posted(params, optimizers, state, step, info, lr):
                # Inner sampling is rank-local (zero targets may skip it); coordinate only the common post boundary.
                def action():
                    self.before_post(params, step)
                    result = post(params=params, optimizers=optimizers, state=state, step=step, info=info, lr=lr)
                    if len(params['means']) != self.owner_size:
                        raise ValueError('topology changed during cohort')
                    self.posts += 1
                    return result
                return self.consensus(f'post-{step}', action)

            strategy._relocate_gs, strategy.step_post_backward = relocated, posted
            return strategy

        with ExitStack() as stack:
            stack.enter_context(patch.object(trainer, '_build_mcmc_strategy', built))
            stack.enter_context(patch.object(ops, '_multinomial_sample', sampled))
            yield self

    def finish(self):
        if (self.posts != self.end - self.start or self.relocations != [self.event, self.end]
                or [r['iteration'] for r in self.rows] != list(range(self.event + 1, self.end + 1))
                or set(self.manifests) != {'event-before', 'event-after', 'cohort', 'cohort-final'}):
            raise ValueError('incomplete minimal lifecycle')
        return {'posts': self.posts, 'relocations': self.relocations, 'sample_calls': self.sample_calls,
                'manifests': self.manifests, 'index_sha256': sha256(self.trace / 'index.json'),
                'extra_training_forwards': 0, 'extra_backwards': 0, 'full_numerical_noninterference_proven': False}
