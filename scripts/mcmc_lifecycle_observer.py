"""Research-only observation of the frozen MCMC trainer; imports Torch only at runtime."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, ExitStack
import gzip
import hashlib
import json
import random
import sys
from unittest.mock import patch

START, EVENT, END = 12000, 12100, 12200
GROUPS = ("target", "source", "uninvolved")
ATOL, RTOL = 2e-6, 2e-4


def digest_state(value):
    digest = hashlib.sha256()

    def visit(item):
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(item, torch.Tensor):
            visit(("tensor", str(item.dtype), list(item.shape)))
            raw = item.detach().reshape(-1).contiguous().view(torch.uint8)
            for offset in range(0, raw.numel(), 4 * 1024**2):
                digest.update(memoryview(raw[offset:offset + 4 * 1024**2].cpu().numpy()))
        elif hasattr(item, "dtype") and hasattr(item, "tobytes"):
            visit(("array", str(item.dtype), list(item.shape)))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for entry in item:
                visit(entry)
            digest.update(b"]")
        elif item is None or type(item) in (str, bool, int, float):
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode() + b"\n")
        else:
            raise TypeError(f"unsupported state type: {type(item)}")
    visit(value)
    return digest.hexdigest()


def reuse_record(count, targets, sampled):
    if len(targets) != len(set(targets)) or len(targets) != len(sampled):
        raise ValueError("relocation target/sample cardinality differs")
    frequencies = Counter(sampled)
    if set(targets) & frequencies.keys() or any(type(i) is not int or not 0 <= i < count for i in targets + sampled):
        raise ValueError("invalid or overlapping source/target slots")
    return {"group_sizes": {"target": len(targets), "source": len(frequencies),
                            "uninvolved": count - len(targets) - len(frequencies)},
            "unique_source_count": len(frequencies), "max_source_reuse": max(frequencies.values(), default=0),
            "reuse_histogram": dict(sorted(Counter(frequencies.values()).items())),
            "reuse_excludes_original": True, "formula_copies_above_51_sources": sum(n + 1 > 51 for n in frequencies.values())}


def survival_summary(initial_alive, first_death, alive, first_recovery):
    n = len(initial_alive)
    if len(first_death) != n or len(alive) != n or len(first_recovery) != n:
        raise ValueError("cohort size changed")
    restored = sum(initial_alive)
    redied = sum(a and t >= 0 for a, t in zip(initial_alive, first_death))
    return {"count": n, "alive_after_relocation": restored, "not_alive_after_relocation": n - restored,
            "initially_alive_ever_died": redied,
            "redeath_fraction_of_initially_alive": redied / restored if restored else None,
            "dead_at_window_end": sum(not x for x in alive),
            "ever_dead_now_alive": sum(t >= 0 and a for t, a in zip(first_death, alive)),
            "ever_recovered_after_death": sum(t >= 0 for t in first_recovery),
            "recovered_then_dead_at_end": sum(t >= 0 and not a for t, a in zip(first_recovery, alive)),
            "first_recovery_iteration_histogram": dict(sorted(Counter(t for t in first_recovery if t >= 0).items())),
            "first_death_iteration_histogram": dict(sorted(Counter(t for t in first_death if t >= 0).items()))}


def regularizer_gradient(name, parameter, config, world_size):
    if name == "opacities":
        alpha = parameter.detach().sigmoid()
        return alpha * (1 - alpha) * (float(config["loss"]["opacity_regularization"]) / (world_size * alpha.numel()))
    if name == "scales":
        return parameter.detach().exp() * (float(config["loss"]["scale_regularization"]) / (world_size * parameter.numel()))
    raise ValueError("only opacity logits and log scales have these regularizers")


def distribution(tensor):
    import torch
    values = tensor.detach().reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite observer statistic")
    if not values.numel():
        return {"count": 0, "quantiles": None, "mean": None, "negative": 0, "positive": 0, "zero": 0}
    q = torch.tensor([0., .1, .5, .9, .99, 1.], device=values.device)
    return {"count": values.numel(), "quantiles": torch.quantile(values.float(), q).cpu().tolist(),
            "mean": float(values.double().mean()), "negative": int((values < 0).sum()),
            "positive": int((values > 0).sum()), "zero": int((values == 0).sum())}


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


@contextmanager
def unchanged_rng():
    before = digest_state(rng_state())
    yield
    if digest_state(rng_state()) != before:
        raise ValueError("observer consumed RNG; state was not restored or hidden")


def verify_alpha(features, auxiliary_alpha, original_alpha):
    import torch
    expected = original_alpha.detach()
    if not all(bool(torch.isfinite(x).all()) for x in (features, auxiliary_alpha, expected)):
        raise ValueError("nonfinite contribution render")
    mass = features.sum(dim=-1, keepdim=True)
    for actual in (auxiliary_alpha, mass):
        if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
            raise ValueError("contribution alpha conservation failed")
    return {"alpha_max_error": float((auxiliary_alpha - expected).abs().max()),
            "mass_max_error": float((mass - expected).abs().max())}


class LifecycleObserver:
    def __init__(self, root, rank, config, observed, *, window=(START, EVENT, END), expected_count=1500000):
        self.root, self.rank, self.config, self.observed = root, rank, config, observed
        self.start, self.event, self.end = window
        self.expected_count = expected_count
        self.iteration = self.start
        self.camera = None
        self.params = self.optimizers = self.strategy_state = None
        self.labels = self.initial_alive = self.first_death = self.first_recovery = None
        self.samples = []
        self.sample_context = None
        self.handles = []
        self.pre_opacity = None
        self.cohort_complete = False
        self.exposure = {}
        self.emitted = 0

    def emit(self, name, record):
        path = self.root / f"{name}-rank-{self.rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")

    def state(self):
        return {"parameters": digest_state(dict(self.params.items())),
                "adam": digest_state({k: o.state_dict() for k, o in self.optimizers.items()}),
                "strategy": digest_state(self.strategy_state), "camera": digest_state(self.camera),
                "rng": digest_state(rng_state())}

    def group_statistics(self):
        opacity = self.params["opacities"].detach().sigmoid().flatten()
        scales = self.params["scales"].detach().exp()
        minimum, maximum = scales.amin(-1), scales.amax(-1)
        if bool((minimum <= 0).any()):
            raise ValueError("zero/nonpositive axis prevents finite anisotropy")
        return {name: {"count": int((self.labels == i).sum()),
                       "dead": int(((opacity <= .005) & (self.labels == i)).sum()),
                       "below_cutoff": int(((opacity < 1 / 255) & (self.labels == i)).sum()),
                       "origin_distance": distribution(self.params["means"].detach().norm(dim=-1)[self.labels == i]),
                       "opacity": distribution(opacity[self.labels == i]),
                       "min_axis": distribution(minimum[self.labels == i]),
                       "max_axis": distribution(maximum[self.labels == i]),
                       "kappa": distribution((maximum / minimum)[self.labels == i])}
                for i, name in enumerate(GROUPS)}

    def adam_statistics(self):
        output = {}
        for name in ("opacities", "scales"):
            param = self.params[name]
            state = self.optimizers[name].state[param]
            output[name] = {group: {"step": float(state["step"]),
                **{key: distribution(state[key][self.labels == i]) for key in ("exp_avg", "exp_avg_sq")}}
                for i, group in enumerate(GROUPS)}
        return output

    def observe_gradients(self, name, optimizer):
        if not self.observed or self.labels is None or not self.event < self.iteration <= self.end:
            return
        with unchanged_rng():
            param = self.params[name]
            if param.grad is None:
                raise ValueError("missing observed gradient")
            total = param.grad.detach()
            regularizer = regularizer_gradient(name, param, self.config, 2)
            image = total - regularizer
            state = optimizer.state[param]
            groups = {}
            for i, group in enumerate(GROUPS):
                mask = self.labels == i
                groups[group] = {"count": int(mask.sum()), "adam_step": float(state["step"]),
                    **{key: distribution(value[mask]) for key, value in (
                        ("total_gradient", total), ("regularizer_gradient", regularizer),
                        ("image_gradient_residual", image), ("exp_avg", state["exp_avg"]), ("exp_avg_sq", state["exp_avg_sq"]))}}
            self.emit("diagnostic", {"event": "gradient", "iteration": self.iteration, "parameter": name,
                "coordinate": "opacity_logit" if name == "opacities" else "log_scale",
                "image_gradient_method": "total_minus_analytic_regularizer_float_residual", "groups": groups})
            if name == "opacities":
                self.pre_opacity = param.detach().clone()
                for key, value in (("image_gradient_signed", image), ("image_gradient_absolute", image.abs()),
                                   ("regularizer_gradient", regularizer), ("adam_m_before_update", state["exp_avg"])):
                    self.exposure[key].add_(value.flatten())

    def observe_update(self, optimizer, args, kwargs):
        if self.pre_opacity is None:
            return
        import torch
        with unchanged_rng():
            logits = self.params["opacities"].detach()
            alive = logits.sigmoid().flatten() > .005
            first = (self.first_death < 0) & ~alive
            self.first_death[first] = self.iteration
            recovered = alive & (self.first_death >= 0) & (self.first_recovery < 0)
            self.first_recovery[recovered] = self.iteration
            delta = logits - self.pre_opacity
            self.exposure["logit_update"].add_(delta.flatten())
            groups = {group: {"update": distribution(delta[self.labels == i]),
                "dead": int((~alive & (self.labels == i)).sum()),
                "newly_dead": int((first & (self.labels == i)).sum()),
                "first_recovered": int((recovered & (self.labels == i)).sum()),
                "ever_dead_now_alive": int(((self.first_death >= 0) & alive & (self.labels == i)).sum())}
                for i, group in enumerate(GROUPS)}
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite opacity after Adam")
            self.emit("diagnostic", {"event": "opacity_update", "iteration": self.iteration, "groups": groups})
        self.pre_opacity = None

    def finish_cohort(self):
        alive = (self.params["opacities"].detach().sigmoid().flatten() > .005).cpu().tolist()
        labels = self.labels.cpu().tolist()
        initial, first = self.initial_alive.cpu().tolist(), self.first_death.cpu().tolist()
        recovered = self.first_recovery.cpu().tolist()
        groups = {}
        for i, name in enumerate(GROUPS):
            ids = [j for j, label in enumerate(labels) if label == i]
            groups[name] = survival_summary([initial[j] for j in ids], [first[j] for j in ids], [alive[j] for j in ids], [recovered[j] for j in ids])
        exposure = {}
        for i, name in enumerate(GROUPS):
            mask = self.labels == i
            exposure[name] = {}
            for category, risk in (("initially_alive_redied", self.initial_alive & (self.first_death >= 0)),
                                   ("initially_alive_never_died", self.initial_alive & (self.first_death < 0)),
                                   ("not_alive_after_relocation", ~self.initial_alive)):
                exposure[name][category] = {key: distribution(value[mask & risk] / (self.end - self.event))
                                            for key, value in self.exposure.items()}
        self.emit("diagnostic", {"event": "cohort_end_before_relocation", "iteration": self.end,
                                 "groups": groups, "state": self.group_statistics(), "adam": self.adam_statistics(),
                                 "mean_exposure_by_survival_class": exposure})
        self.cohort_complete = True

    def contribution(self, model, view, rendered):
        import torch
        from gsplat.rendering import rasterization
        from image3d_scenegraph.gaussian.render import NEAR_PLANE_NORMALIZED, FAR_PLANE_NORMALIZED

        with unchanged_rng(), torch.no_grad():
            n = model.count
            labels = self.labels if self.labels is not None else torch.full((n,), 2, device=model.means.device, dtype=torch.long)
            colors = torch.nn.functional.one_hot(labels + 3 * self.rank, num_classes=6).to(model.means.dtype)
            means, quats, scales, opacities, _ = model.activated()
            camera = view.camera
            image, alpha, _ = rasterization(means, quats, scales, opacities, colors,
                camera.camera_from_normalized[None], camera.intrinsic[None], width=camera.width, height=camera.height,
                sh_degree=None, packed=False, backgrounds=None, render_mode="RGB", absgrad=False, distributed=True,
                near_plane=NEAR_PLANE_NORMALIZED, far_plane=FAR_PLANE_NORMALIZED)
            checks = verify_alpha(image[0], alpha[0], rendered.alpha)
            self.emit("diagnostic", {"event": "contribution", "iteration": self.iteration,
                "camera_rank": self.rank, "image_id": str(camera.image_id), "cohort_defined": self.labels is not None,
                "channel_order": [f"owner_{rank}/{group}" for rank in (0, 1) for group in GROUPS],
                "weight_sum": image[0].double().sum((0, 1)).cpu().tolist(),
                "alpha_sum": float(rendered.alpha.double().sum()), **checks,
                "meaning": "composited_T_alpha_not_loss_improvement"})

    @contextmanager
    def installed(self):
        import torch
        from gsplat.strategy import ops
        from image3d_scenegraph.gaussian import trainer

        original_build = trainer._build_mcmc_strategy
        original_batch, original_render = trainer._next_camera_batch, trainer._render_visible_training_view
        original_sample = ops._multinomial_sample

        def batch(*args, **kwargs):
            result = original_batch(*args, **kwargs)
            self.camera = {"order": list(result[0]), "cursor": result[1], "indices": list(result[2])}
            self.iteration += 1
            return result

        def render(*args, **kwargs):
            result = original_render(*args, **kwargs)
            self.camera["image_id"] = str(result[0].camera.image_id)
            if self.observed:
                self.contribution(args[0], result[0], result[1])
            return result

        def sample(weights, n, **kwargs):
            indices = original_sample(weights, n, **kwargs)
            if self.sample_context is None:
                raise ValueError("unexpected sampling outside fixed-cap relocation")
            with unchanged_rng():
                targets, alive = self.sample_context
                absolute = alive[indices]
                self.samples.append({"targets": digest_state(targets), "sources": digest_state(absolute), "count": n})
                if self.observed and self.iteration == self.event:
                    self.labels = torch.full((self.expected_count,), 2, dtype=torch.long, device=indices.device)
                    self.labels[absolute] = 1
                    self.labels[targets] = 0
                    target_ids, source_ids = targets.cpu().tolist(), absolute.cpu().tolist()
                    record = reuse_record(self.expected_count, target_ids, source_ids)
                    identity = self.root / f"cohort-rank-{self.rank}.json.gz"
                    with identity.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                        handle.write(json.dumps({"owner_rank": self.rank, "event": self.event,
                            "target_slots": target_ids, "sampled_source_slots": source_ids}).encode())
                    self.emit("diagnostic", {"event": "relocation_before", "iteration": self.iteration,
                        **record, "groups": self.group_statistics(), "adam": self.adam_statistics()})
            return indices

        def build(*args, **kwargs):
            strategy = original_build(*args, **kwargs)
            pre, post, relocate = strategy.step_pre_backward, strategy.step_post_backward, strategy._relocate_gs

            def before(params, optimizers, state, step, info):
                self.params, self.optimizers, self.strategy_state = params, optimizers, state
                if step != self.iteration or len(params["means"]) != self.expected_count:
                    raise ValueError("observer iteration or fixed capacity differs")
                if not self.handles:
                    for name in ("opacities", "scales"):
                        self.handles.append(optimizers[name].register_step_pre_hook(
                            lambda opt, a, kw, name=name: self.observe_gradients(name, opt)))
                    self.handles.append(optimizers["opacities"].register_step_post_hook(self.observe_update))
                    self.emit("audit", {"event": "first_render", "iteration": self.start + 1, "state": self.state()})
                return pre(params, optimizers, state, step, info)

            def relocation(params, optimizers, binoms):
                if self.iteration not in (self.event, self.end):
                    raise ValueError("unexpected relocation step")
                if self.observed and self.iteration == self.end:
                    with unchanged_rng():
                        self.finish_cohort()
                with unchanged_rng():
                    alpha = params["opacities"].detach().sigmoid().flatten()
                    self.sample_context = ((alpha <= .005).nonzero().flatten(), (alpha > .005).nonzero().flatten())
                snapshot = None
                if self.observed and self.iteration == self.event:
                    with unchanged_rng():
                        snapshot = {key: params[key].detach().clone() for key in ("means", "opacities")}
                try:
                    count = relocate(params, optimizers, binoms)
                finally:
                    self.sample_context = None
                if self.observed and self.iteration == self.event:
                    if self.labels is None:
                        raise ValueError("no relocation cohort was formed")
                    with unchanged_rng():
                        self.initial_alive = params["opacities"].detach().sigmoid().flatten() > .005
                        self.first_death = torch.full_like(self.labels, -1)
                        self.first_recovery = torch.full_like(self.labels, -1)
                        self.first_death[~self.initial_alive] = self.event
                        self.exposure = {key: torch.zeros_like(params["opacities"].detach()).flatten() for key in (
                            "image_gradient_signed", "image_gradient_absolute", "regularizer_gradient", "adam_m_before_update", "logit_update")}
                        delta = {group: {
                            "position_distance": distribution((params["means"].detach() - snapshot["means"]).norm(dim=-1)[self.labels == i]),
                            "opacity_logit": distribution((params["opacities"].detach() - snapshot["opacities"])[self.labels == i])}
                            for i, group in enumerate(GROUPS)}
                        self.emit("diagnostic", {"event": "relocation_after", "iteration": self.iteration,
                                                 "groups": self.group_statistics(), "adam": self.adam_statistics(), "parameter_delta": delta})
                return count

            def after(*args, **kwargs):
                result = post(*args, **kwargs)
                if len(self.params["means"]) != self.expected_count:
                    raise ValueError("topology growth invalidated cohort slots")
                with unchanged_rng():
                    self.emit("audit", {"event": "step", "iteration": self.iteration,
                                        "state": self.state(), "sampling": self.samples})
                self.samples = []
                self.emitted += 1
                return result

            strategy.step_pre_backward, strategy.step_post_backward, strategy._relocate_gs = before, after, relocation
            return strategy

        try:
            with ExitStack() as stack:
                for name, replacement in (("_build_mcmc_strategy", build), ("_next_camera_batch", batch),
                                          ("_render_visible_training_view", render)):
                    stack.enter_context(patch.object(trainer, name, replacement))
                stack.enter_context(patch.object(ops, "_multinomial_sample", sample))
                yield self
            if self.emitted != self.end - self.start or (self.observed and not self.cohort_complete):
                raise ValueError("incomplete observation window")
        finally:
            for handle in self.handles:
                handle.remove()
