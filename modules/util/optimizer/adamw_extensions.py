#
# Copied and modified from the original AdamW implementation in PyTorchm (https://github.com/pytorch/pytorch/)
#
# Implements stochastic rounding from "Revisiting BFloat16 Training" (https://arxiv.org/abs/2010.06192)
#

import math
from typing import Optional

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.optimizer import _use_grad_for_differentiable

from modules.util.bf16_stochastic_rounding import addcdiv_stochastic_


@torch.no_grad()
def step_adamw_parameter(self, p, group, i):
    if p.grad is None:
        return
    grad = p.grad
    if p.grad.is_sparse:
        raise RuntimeError("AdamW does not support sparse gradients")

    state = self.state[p]

    # State initialization
    if len(state) == 0:
        # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
        # This is because kernel launches are costly on CUDA and XLA.
        state["step"] = (
            torch.zeros((), dtype=_get_scalar_dtype(is_fused=group["fused"]), device=p.device)
            if group["capturable"] or group["fused"]
            else torch.tensor(0.0, dtype=_get_scalar_dtype())
        )
        # Exponential moving average of gradient values
        state["exp_avg"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )
        # Exponential moving average of squared gradient values
        state["exp_avg_sq"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )
        if group['amsgrad']:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state["max_exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

    if group['differentiable'] and state['step'].requires_grad:
        raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

    # Foreach without capturable does not support a tensor lr
    if group['foreach'] and isinstance(group['lr'], Tensor) and not group['capturable']:
        raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

    if group["maximize"]:
        grad = -grad

    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    step_t = state["step"]
    beta1, beta2 = group["betas"]

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and group["capturable"]:
        assert (
                (p.is_cuda and step_t.is_cuda) or (p.is_xla and step_t.is_xla)
        ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

    if torch.is_complex(p):
        grad = torch.view_as_real(grad)
        exp_avg = torch.view_as_real(exp_avg)
        exp_avg_sq = torch.view_as_real(exp_avg_sq)
        if group['amsgrad']:
            state["max_exp_avg_sq"] = torch.view_as_real(state["max_exp_avg_sq"])
        p = torch.view_as_real(p)

    # update step
    step_t += 1

    # Perform stepweight decay
    p.mul_(1 - group["lr"] * group["weight_decay"])

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    if group["capturable"] or group["differentiable"]:
        step = step_t

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = group["lr"] / bias_correction1
        step_size_neg = step_size.neg()

        bias_correction2_sqrt = bias_correction2.sqrt()

        if group['amsgrad']:
            # Maintains the maximum of all 2nd moment running avg. till now
            if group["differentiable"]:
                max_exp_avg_sq = state["max_exp_avg_sq"].clone()
            else:
                max_exp_avg_sq = state["max_exp_avg_sq"]

            state["max_exp_avg_sq"].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

            # Uses the max. for normalizing running avg. of gradient
            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
            denom = (
                    state["max_exp_avg_sq"].sqrt() / (bias_correction2_sqrt * step_size_neg)
            ).add_(group["eps"] / step_size_neg)
        else:
            denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
            ).add_(group["eps"] / step_size_neg)

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            addcdiv_stochastic_(p, exp_avg, denom)
        else:
            p.addcdiv_(exp_avg, denom)
    else:
        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = group["lr"] / bias_correction1

        if isinstance(bias_correction2, torch.Tensor):
            bias_correction2_sqrt = bias_correction2.sqrt()
        else:
            bias_correction2_sqrt = math.sqrt(bias_correction2)

        if group['amsgrad']:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])

            # Use the max. for normalizing running avg. of gradient
            denom = (state["max_exp_avg_sq"].sqrt() / bias_correction2_sqrt).add_(group["eps"])
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            addcdiv_stochastic_(p, exp_avg, denom, value=-step_size)
        else:
            p.addcdiv_(exp_avg, denom, value=-step_size)

    # Lastly, switch back to complex view
    if group['amsgrad'] and torch.is_complex(p):
        state["max_exp_avg_sq"] = torch.view_as_complex(state["max_exp_avg_sq"])

def _get_scalar_dtype(is_fused=None):
    if is_fused:
        return torch.float32
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _single_tensor_adamw(
        self,
        group,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(group["lr"], float)

    for i, p in enumerate(group["params"]):
        step_adamw_parameter(self, p, group, i)


@_use_grad_for_differentiable
def step_adamw(self, closure=None):
    """Performs a single optimization step.

    Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        _single_tensor_adamw(
            self,
            group=group,
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
        )

    return loss

def patch_adamw(optimizer: AdamW, stochastic_rounding: bool):
    optimizer.stochastic_rounding = stochastic_rounding
    optimizer.step = step_adamw.__get__(optimizer, AdamW)
    optimizer.step_parameter = step_adamw_parameter.__get__(optimizer, AdamW)
