#
# Copied and modified from the original CAME implementation (https://github.com/yangluo7/CAME)
#
# Implements stochastic rounding from "Revisiting BFloat16 Training" (https://arxiv.org/abs/2010.06192)
#

import torch
from came_pytorch import CAME

from modules.util.bf16_stochastic_rounding import add_stochastic_

@torch.no_grad()
def step_came_parameter(self, p, group, i):
    if p.grad is None:
        return
    grad = p.grad.data
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("CAME does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored = self._get_options(grad_shape)
    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
            state["exp_avg_sq_col"] = torch.zeros(
                grad_shape[:-2] + grad_shape[-1:]
            ).type_as(grad)

            state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
            state["exp_avg_res_col"] = torch.zeros(
                grad_shape[:-2] + grad_shape[-1:]
            ).type_as(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["RMS"] = 0

    state["step"] += 1
    state["RMS"] = self._rms(p.data)

    update = (grad**2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(group["betas"][1]).add_(
            update.mean(dim=-1), alpha=1.0 - group["betas"][1]
        )
        exp_avg_sq_col.mul_(group["betas"][1]).add_(
            update.mean(dim=-2), alpha=1.0 - group["betas"][1]
        )

        # Approximation of exponential moving average of square of gradient
        update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_(
        (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
    )

    exp_avg = state["exp_avg"]
    exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

    # Confidence-guided strategy
    # Calculation of instability
    res = (update - exp_avg)**2 + group["eps"][1]

    if factored:
        exp_avg_res_row = state["exp_avg_res_row"]
        exp_avg_res_col = state["exp_avg_res_col"]

        exp_avg_res_row.mul_(group["betas"][2]).add_(
            res.mean(dim=-1), alpha=1.0 - group["betas"][2]
        )
        exp_avg_res_col.mul_(group["betas"][2]).add_(
            res.mean(dim=-2), alpha=1.0 - group["betas"][2]
        )

        # Approximation of exponential moving average of instability
        res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
        update = res_approx.mul_(exp_avg)
    else:
        update = exp_avg.clone()

    if group["weight_decay"] != 0:
        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, p.data,
                            alpha=-group["weight_decay"] * group["lr"])
        else:
            p.data.add_(
                p.data, alpha=-group["weight_decay"] * group["lr"]
            )

    update.mul_(group["lr"])
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        add_stochastic_(p.data, -update)
    else:
        p.data.add_(-update)


@torch.no_grad()
def step_came(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                step_came_parameter(self, p, group, i)

        return loss


def patch_came(optimizer: CAME, stochastic_rounding: bool):
    optimizer.stochastic_rounding = stochastic_rounding
    optimizer.step = step_came.__get__(optimizer, CAME)
    optimizer.step_parameter = step_came_parameter.__get__(optimizer, CAME)
