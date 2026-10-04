#
# Copied and modified from the original Adafactor implementation in transformers (https://github.com/huggingface/transformers)
#
# Implements stochastic rounding from "Revisiting BFloat16 Training" (https://arxiv.org/abs/2010.06192)
#

import math
from collections import defaultdict

from modules.util.bf16_stochastic_rounding import copy_stochastic_, copy_stochastic_flat_

import torch

from transformers import Adafactor


@torch.no_grad()
def step_adafactor_parameter(self, p, group, i):
    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state, factored, use_first_moment = _init_adafactor_state(self, p, grad, group)

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state["step"] += 1
    state["RMS"] = self._rms(p_data_fp32)
    lr = self._get_lr(group, state)

    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
    update = (grad ** 2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        # Approximation of exponential moving average of square of gradient
        update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)

    if use_first_moment:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
        update = exp_avg

    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        copy_stochastic_(p, p_data_fp32)
    elif p.dtype in {torch.float16, torch.bfloat16}:
        p.copy_(p_data_fp32)
    else:
        assert p_data_fp32 is p


@torch.no_grad()
def step_adafactor(self, closure=None):
    """
    Performs a single optimization step

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for i, p in enumerate(group["params"]):
            step_adafactor_parameter(self, p, group, i)

    return loss


# batching limit of the foreach step, bounding the extra VRAM of a chunk's fp32 temporaries. Not a hard cap: an
# oversized parameter forms a chunk of its own, like in the per-parameter step.
FOREACH_CHUNK_NUMEL = 1 << 24


def _to_fp32(tensors):
    # fp32 copies, with 16 bit tensors packed into one flat buffer per (device, dtype). Returns the tensors and
    # {(device, dtype): (flat buffer, indices of the tensors in it)}.
    indices = defaultdict(list)
    for i, t in enumerate(tensors):
        if t.dtype in {torch.float16, torch.bfloat16}:
            indices[(t.device, t.dtype)].append(i)

    out = list(tensors)
    buffers = {}
    for (device, dtype), idx in indices.items():
        numels = [tensors[i].numel() for i in idx]
        flat = torch.empty(sum(numels), dtype=torch.float32, device=device)
        views = [v.view(tensors[i].shape) for v, i in zip(flat.split(numels), idx, strict=True)]
        torch._foreach_copy_(views, [tensors[i] for i in idx])
        for i, v in zip(idx, views, strict=True):
            out[i] = v
        buffers[(device, dtype)] = (flat, idx)
    return out, buffers


def _init_adafactor_state(self, p, grad, group):
    state = self.state[p]
    factored, use_first_moment = self._get_options(group, grad.shape)
    if len(state) == 0:
        state["step"] = 0
        if use_first_moment:
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["RMS"] = 0
    else:
        if use_first_moment:
            state["exp_avg"] = state["exp_avg"].to(grad)
        if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)
    return state, factored, use_first_moment


@torch.no_grad()
def _step_adafactor_chunk(self, group, params):
    # Same math as step_adafactor_parameter, batched over parameters that share the same step count.
    grads, _ = _to_fp32([p.grad for p in params])
    states = []
    factored = []
    for p, grad in zip(params, grads, strict=True):
        state, is_factored, _ = _init_adafactor_state(self, p, grad, group)
        states.append(state)
        factored.append(is_factored)

    p_fp32, p_buffers = _to_fp32(params)

    for state in states:
        state["step"] += 1
    rms = torch._foreach_norm(p_fp32)
    torch._foreach_div_(rms, [p.numel() ** 0.5 for p in p_fp32])
    for state, r in zip(states, rms, strict=True):
        state["RMS"] = r
    lrs = [self._get_lr(group, state) for state in states]
    lrs = [float(lr) if isinstance(lr, torch.Tensor) else lr for lr in lrs]

    beta2t = 1.0 - math.pow(states[0]["step"], group["decay_rate"])
    updates = [None] * len(params)

    # factored params are grouped by shape, so their reductions and outer products run once per shape
    f_groups = defaultdict(list)
    u_idx = []
    for i, (is_factored, grad) in enumerate(zip(factored, grads, strict=True)):
        if is_factored:
            f_groups[(grad.device, grad.dtype, grad.shape)].append(i)
        else:
            u_idx.append(i)

    for idx in f_groups.values():
        sq = torch.stack([grads[i] for i in idx]).square_().add_(group["eps"][0])
        rows = [states[i]["exp_avg_sq_row"] for i in idx]
        cols = [states[i]["exp_avg_sq_col"] for i in idx]
        torch._foreach_mul_(rows, beta2t)
        torch._foreach_add_(rows, sq.mean(dim=-1).unbind(), alpha=(1.0 - beta2t))
        torch._foreach_mul_(cols, beta2t)
        torch._foreach_add_(cols, sq.mean(dim=-2).unbind(), alpha=(1.0 - beta2t))
        del sq

        # _approx_sq_grad, batched over the shape group
        r_factor = torch.stack(rows)
        r_factor = r_factor.div_(r_factor.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = torch.stack(cols).rsqrt_()
        f_update = r_factor.unsqueeze(-1) * c_factor.unsqueeze(-2)
        del r_factor, c_factor
        f_updates = f_update.unbind()
        torch._foreach_mul_(f_updates, [grads[i] for i in idx])

        # clipping and lr as one lr / clip multiply, not a divide plus a multiply as in step_adafactor_parameter
        dims = tuple(range(1, f_update.dim()))
        clip = torch.linalg.vector_norm(f_update, dim=dims).div_(f_updates[0].numel() ** 0.5)
        clip = clip.div_(group["clip_threshold"]).clamp_(min=1.0)
        group_lrs = [lrs[i] for i in idx]
        if all(lr == group_lrs[0] for lr in group_lrs):
            scale = clip.reciprocal_().mul_(group_lrs[0])
        else:
            scale = torch.tensor(group_lrs, dtype=clip.dtype, device=clip.device).div_(clip)
        f_update.mul_(scale.view(-1, *(1 for _ in dims)))
        del clip, scale
        for i, u in zip(idx, f_updates, strict=True):
            updates[i] = u
        del f_update, f_updates

    if u_idx:
        sqs = [states[i]["exp_avg_sq"] for i in u_idx]
        u_sq = torch._foreach_pow([grads[i] for i in u_idx], 2)
        torch._foreach_add_(u_sq, group["eps"][0])
        torch._foreach_mul_(sqs, beta2t)
        torch._foreach_add_(sqs, u_sq, alpha=(1.0 - beta2t))
        del u_sq
        u_updates = torch._foreach_rsqrt(sqs)
        torch._foreach_mul_(u_updates, [grads[i] for i in u_idx])

        clip = torch._foreach_norm(u_updates)
        torch._foreach_div_(clip, [u.numel() ** 0.5 for u in u_updates])
        torch._foreach_div_(clip, group["clip_threshold"])
        torch._foreach_clamp_min_(clip, 1.0)
        torch._foreach_div_(u_updates, clip)
        del clip
        torch._foreach_mul_(u_updates, [lrs[i] for i in u_idx])
        for i, u in zip(u_idx, u_updates, strict=True):
            updates[i] = u
        del u_updates

    if group["beta1"] is not None:
        exp_avgs = [state["exp_avg"] for state in states]
        torch._foreach_mul_(exp_avgs, group["beta1"])
        torch._foreach_add_(exp_avgs, updates, alpha=(1 - group["beta1"]))
        updates = exp_avgs

    if group["weight_decay"] != 0:
        if all(lr == lrs[0] for lr in lrs):
            torch._foreach_add_(p_fp32, p_fp32, alpha=(-group["weight_decay"] * lrs[0]))
        else:
            for p, lr in zip(p_fp32, lrs, strict=True):
                p.add_(p, alpha=(-group["weight_decay"] * lr))

    torch._foreach_sub_(p_fp32, updates)

    # fp32 params were updated in place, 16 bit params are written back from their flat buffers, in parameter order so
    # stochastic rounding draws the same numbers as the per-parameter path
    for (_, dtype), (flat, idx) in p_buffers.items():
        targets = [params[i] for i in idx]
        if dtype == torch.bfloat16 and self.stochastic_rounding:
            copy_stochastic_flat_(targets, flat)
        else:
            torch._foreach_copy_(targets, [p_fp32[i] for i in idx])


@torch.no_grad()
def step_adafactor_foreach(self, closure=None):
    """
    Performs a single optimization step, batching parameters with torch._foreach_* ops. Uses the same optimizer state
    and mathematically equivalent updates as step_adafactor; batched reductions and fused clip/LR scaling can
    introduce small rounding differences.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        params, numel, step = [], 0, None
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients.")
            p_step = self.state[p].get("step", 0)
            # chunks are contiguous in parameter order and share one step count (and therefore one beta2t)
            if params and (p_step != step or numel + p.numel() > FOREACH_CHUNK_NUMEL):
                _step_adafactor_chunk(self, group, params)
                params, numel = [], 0
            params.append(p)
            numel += p.numel()
            step = p_step
        if params:
            _step_adafactor_chunk(self, group, params)

    return loss


def patch_adafactor(optimizer: Adafactor, stochastic_rounding: bool, foreach: bool = False):
    optimizer.stochastic_rounding = stochastic_rounding
    optimizer.step = (step_adafactor_foreach if foreach else step_adafactor).__get__(optimizer, Adafactor)
    optimizer.step_parameter = step_adafactor_parameter.__get__(optimizer, Adafactor)
