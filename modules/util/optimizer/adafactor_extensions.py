#
# Copied and modified from the original Adafactor implementation in transformers (https://github.com/huggingface/transformers)
#
# Implements stochastic rounding from "Revisiting BFloat16 Training" (https://arxiv.org/abs/2010.06192)
#

import math

import torch
from transformers import Adafactor

from modules.util.bf16_stochastic_rounding import copy_stochastic_


@torch.no_grad()
def step_adafactor_parameter(self, p, group, i):
    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored, use_first_moment = self._get_options(group, grad_shape)
    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        if use_first_moment:
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
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
    if p.dtype in {torch.float16, torch.bfloat16}:
        p.copy_(p_data_fp32)


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


def patch_adafactor(optimizer: Adafactor, stochastic_rounding: bool):
    optimizer.stochastic_rounding = stochastic_rounding
    optimizer.step = step_adafactor.__get__(optimizer, Adafactor)
    optimizer.step_parameter = step_adafactor_parameter.__get__(optimizer, Adafactor)
