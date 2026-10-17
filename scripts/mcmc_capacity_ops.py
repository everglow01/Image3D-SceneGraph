"""Research-only donor-capacity admission for the pinned gsplat MCMC overlay."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def copy_budget(opacity, minimum, n_max):
    if not 0 < minimum < 1 or n_max < 1:
        raise ValueError("invalid relocation admission bounds")
    threshold = max(minimum, 1. / 255.)
    eligible = torch.isfinite(opacity) & (opacity > threshold) & (opacity < 1)
    safe = torch.where(eligible, opacity, 0.).double()
    # N total copies must satisfy 1-(1-alpha)**(1/N) > threshold, strictly.
    budget = (torch.ceil(torch.log1p(-safe) / math.log1p(-threshold)) - 2)
    return budget.clamp(0, n_max - 1).long()


@torch.no_grad()
def _resample(params, optimizers, state, n, binoms, min_opacity, targets=None):
    from gsplat.strategy import ops

    if n == 0:
        return 0
    opacity = params["opacities"].sigmoid().flatten()
    budget = copy_budget(opacity, min_opacity, binoms.shape[0])
    if targets is not None:
        budget[targets] = 0
    eligible = (budget > 0).nonzero(as_tuple=True)[0]
    if not eligible.numel():
        return 0
    proposals = ops._multinomial_sample(opacity[eligible], n, replacement=True)
    proposed_counts = torch.bincount(proposals, minlength=eligible.numel())
    order = torch.argsort(proposals, stable=True)
    starts = proposed_counts.cumsum(0) - proposed_counts
    ordinal = torch.arange(n, device=proposals.device) - starts[proposals[order]]
    keep = torch.zeros(n, dtype=torch.bool, device=proposals.device)
    keep[order] = ordinal < budget[eligible[proposals[order]]]
    slots = keep.nonzero(as_tuple=True)[0]
    counts = torch.bincount(proposals[slots], minlength=eligible.numel())
    selected = counts > 0
    donors, counts = eligible[selected], counts[selected]
    new_opacity, new_scales = ops.compute_relocation(
        opacities=opacity[donors], scales=params["scales"][donors].exp(),
        ratios=counts + 1, binoms=binoms,
    )
    logits, log_scales = torch.logit(new_opacity), new_scales.log()
    roundtrip_scales = log_scales.exp()
    valid = (torch.isfinite(logits) & (logits.sigmoid() > max(min_opacity, 1. / 255.))
             & torch.isfinite(log_scales).all(dim=-1)
             & torch.isfinite(roundtrip_scales).all(dim=-1)
             & (roundtrip_scales >= torch.finfo(new_scales.dtype).tiny).all(dim=-1))
    valid_sources = torch.zeros_like(selected)
    valid_sources[selected] = valid
    slots = slots[valid_sources[proposals[slots]]]
    donors = donors[valid]
    logits, log_scales = logits[valid], log_scales[valid]
    if not donors.numel():
        return 0
    # Keep proposal/target order; memory is O(points + requests), not points * capacity.
    copies = eligible[proposals[slots]]
    accepted = copies.numel()
    if targets is not None:
        targets = targets[slots]

    def param_fn(name, p):
        if name == "opacities":
            p[donors] = logits.reshape_as(p[donors])
        elif name == "scales":
            p[donors] = log_scales
        if targets is None:
            return torch.nn.Parameter(torch.cat([p, p[copies]]), requires_grad=p.requires_grad)
        p[targets] = p[copies]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key, value):
        if targets is None:
            return torch.cat([value, value.new_zeros((accepted, *value.shape[1:]))])
        value[donors] = 0
        return value

    ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            if targets is None:
                state[key] = torch.cat([value, value.new_zeros((accepted, *value.shape[1:]))])
            else:
                value[donors] = 0
    return accepted


@torch.no_grad()
def relocate(params, optimizers, state, mask, binoms, min_opacity=0.005):
    targets = mask.nonzero(as_tuple=True)[0]
    return _resample(params, optimizers, state, len(targets), binoms, min_opacity, targets)


@torch.no_grad()
def sample_add(params, optimizers, state, n, binoms, min_opacity=0.005):
    return _resample(params, optimizers, state, n, binoms, min_opacity)
