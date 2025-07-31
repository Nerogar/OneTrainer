from typing import Tuple, Optional, Any

import math

import torch
from torch.optim.optimizer import Optimizer

from modules.util.bf16_stochastic_rounding import add_stochastic_

class LION(Optimizer):
    """
    Implements the LION (Evolved Sign Momentum) optimizer, refactored for better integration and readability.
    This implementation is based on:
    `Symbolic Discovery of Optimization Algorithms` (https://arxiv.org/abs/2302.06675)
    and has been refactored to support fused backward passes and other features.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients for computing running averages of the update (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        use_bias_correction (bool, optional): whether to use bias correction on the first moment, similar to Adam. (default: True)        
        stochastic_rounding (bool, optional): whether to use stochastic rounding for BF16 (default: True)
        use_cautious (bool, optional): use cautious masking (default: False).
        use_arctan (bool, optional): Use the Refined LION variant, replacing the discontinuous sign with continuous arctan (default: False).
        alpha (float, optional): The scaling factor for the arctan function. A larger alpha makes the function behave more like `sign` (default: 50.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_bias_correction: bool = False,
        stochastic_rounding: bool = True,       
        use_cautious: bool = False,
        use_orthograd: bool = False,
        use_arctan: bool = False,
        alpha: float = 50.0,          
    ):
        if not lr > 0.0:
            raise ValueError(f"Learning rate must be > 0.0, but got {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Betas should be in [0.0, 1.0], but got {betas}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")

        self._init_lr = lr

        defaults = {
            "lr":lr,
            "betas":betas,
            "weight_decay":weight_decay,
            "alpha":alpha,
        }
        self.use_bias_correction = use_bias_correction
        self.stochastic_rounding = stochastic_rounding
        self.use_cautious = use_cautious
        self.use_orthograd = use_orthograd
        self.use_arctan = use_arctan
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Projects the gradient `grad` to be orthogonal to the parameter `p`,
        and then re-scales the result to have the same norm as the original gradient.
        This method is based on the `OrthoGrad` optimizer from the paper
        "Grokking at the Edge of Numerical Stability" (Prieto et al., 2025). It is
        intended to prevent Naïve Loss Minimization (NLM) by removing the component
        of the gradient that is parallel to the weight vector.
        """
        if grad.is_sparse:
            raise RuntimeError("OrthoGrad logic does not support sparse gradients.")
        original_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        # Project g onto w: proj_w(g) = (w·g / w·w) * w
        # The small epsilon is for numerical stability.
        w_norm_sq = torch.dot(w, w).add_(1e-30)
        proj = torch.dot(w, g) / w_norm_sq
        # The orthogonal component: g_orth = g - proj_w(g)
        g_orth = g.sub(w, alpha=proj)
        # Rescale g_orth to have the same L2 norm as the original gradient g.
        g_norm = g.norm(2)
        g_orth_norm = g_orth.norm(2).add_(1e-30)
        g_orth_scaled = g_orth * (g_norm / g_orth_norm)
        return g_orth_scaled.view(original_shape)

    @property
    def supports_fused_back_pass(self) -> bool:
        return True

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: Optional[int] = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad
        if self.use_orthograd:
            grad = self._orthogonalize_gradient(p, grad)
        state = self.state[p]

        # State Initialization
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)

        # Get hyperparameters
        state["step"] += 1
        exp_avg = state["exp_avg"]
        beta1, beta2 = group["betas"]

        # --- LION Update Rule ---
        update = torch.lerp(grad, exp_avg, beta1)

        if  self.use_bias_correction:
            bias_correction1 = 1.0 - beta1 ** state["step"]
            update = update.div(bias_correction1)  

        if not self.use_arctan:
            update.sign_()
        else:
            update = torch.arctan(update.mul_(group["alpha"]))
            update.mul_(2.0 / math.pi)  

        if self.use_cautious:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)     

        if group["weight_decay"] != 0:
            update.add_(p.data, alpha=group["weight_decay"])

        update.mul_(group["lr"])
        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)
        
        exp_avg.lerp_(grad, 1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        return loss