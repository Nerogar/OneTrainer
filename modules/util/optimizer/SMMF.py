#
# SMMF optimizer, refactored for better integration and readability.
# Original paper: "SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization" (https://arxiv.org/abs/2412.08894)
#
# This version incorporates:
# - A structure that supports fused backward passes.
# - Stochastic rounding from "Revisiting BFloat16 Training" (https://arxiv.org/abs/2010.06192) for the final parameter update.
# - Decoupling the direction and magnitude of parameter updates, using current gradients for the update direction and momentum solely for adaptive magnitude scaling, as described in "Grams: Gradient Descent with Adaptive Momentum Scaling" (https://arxiv.org/abs/2412.17107).
# - A toggleable 'factored_sign' mode for maximum memory efficiency.
#


from modules.util.bf16_stochastic_rounding import add_stochastic_

import torch
import torch.optim


class SMMF(torch.optim.Optimizer):
    """
    Implements SMMF algorithm.
    This implementation is based on:
    `SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (Optional[float], optional): coefficient for computing running average of gradient (default: 0.9)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decay_rate (float, optional): decay-rate for 2nd moment ('gamma' in paper) (default: -0.8)
        beta1_growth_rate (Optional[float], optional): growth-rate for 1st moment ('lambda' in paper) (default: 0.999)
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D matrices (default: True)
        stochastic_rounding (bool, optional): whether to use stochastic rounding for BF16 (default: True)
        factored_sign (bool, optional): whether to use the memory-efficient pos/neg factorization instead of a sign matrix (default: False)
        use_grams (bool, optional): whether to use the grams method, Decoupling the direction and magnitude of parameter updates (default: False)
        use_orthograd (bool): whether to use OrthoGrad.  (default: False)
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float | None = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decay_rate: float = -0.8,
        beta1_growth_rate: float | None = 0.999,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        factored_sign: bool = False,
        use_grams: bool = False,
        use_orthograd: bool = False,
        use_atan2: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if beta1 is not None and not (0.0 <= beta1 <= 1.0):
            raise ValueError(f"beta1 should be in [0.0, 1.0]. Got {beta1}")
        if not (eps >= 0.0):
            raise ValueError(f"Epsilon should be >= 0.0. Got {eps}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (-1.0 <= decay_rate <= 0.0):
            raise ValueError(f"Decay-rate should be in [-1.0, 0.0]. Got {decay_rate}")
        if beta1_growth_rate is not None and not (0.0 <= beta1_growth_rate <= 1.0):
            raise ValueError(f"Growth-rate should be in [0.0, 1.0]. Got {beta1_growth_rate}")
        if use_grams and factored_sign:
            raise ValueError("Cannot use both 'use_grams' and 'factored_sign' modes simultaneously as they are mutually exclusive.")
        
        defaults = {
            "lr": lr, "beta1": beta1, "eps": eps, "weight_decay": weight_decay,
            "decay_rate": decay_rate, "beta1_growth_rate": beta1_growth_rate,
            "vector_reshape": vector_reshape,
            "factored_sign": factored_sign,
        }
        self.stochastic_rounding = stochastic_rounding
        self.factored_sign = factored_sign
        self.use_grams = use_grams
        self.use_orthograd = use_orthograd
        self.use_atan2 = use_atan2
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
        w = p.view(-1).float()
        g = grad.view(-1).float()
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
    def supports_fused_back_pass(self):
        return True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_effective_shape(self, numel: int) -> tuple:
        """Finds two factors of numel that are closest to its square root."""
        sqrt_num_sq = int(numel ** 0.5) ** 2
        if numel == sqrt_num_sq:
            sqrt_num = int(numel ** 0.5)
            return (sqrt_num, sqrt_num)

        for i in reversed(range(1, int(numel ** 0.5) + 1)):
            if numel % i == 0:
                return (numel // i, i)
        return (numel, 1)

    def _unnmf(self, row_col: tuple) -> torch.Tensor:
        """Reconstructs a matrix from its rank-1 factors (outer product)."""
        return torch.outer(row_col[0], row_col[1])

    def _nnmf(self, matrix: torch.Tensor, out: tuple):
        """Performs a rank-1 non-negative matrix factorization."""
        shape = matrix.shape
        torch.sum(matrix, dim=1, out=out[0])
        torch.sum(matrix, dim=0, out=out[1])

        # Normalize one of the factors for stability
        if shape[0] < shape[1]:
            scale = out[0].sum()
            if scale != 0:
                out[0].div_(scale)
        else:
            scale = out[1].sum()
            if scale != 0:
                out[1].div_(scale)

    def _decompress_momentum(self, state, momentum_name: str, group: dict) -> torch.Tensor:
        """Decompresses a momentum tensor from its factorized form."""
        if momentum_name != 'momentum_m':
            return self._unnmf(state[momentum_name])
        if not self.factored_sign:
            # Original method (with patch for resume-spike)
            update = self._unnmf(state['momentum_m'])
            if not self.use_grams:
                if state['sign'].dtype != torch.bool:
                    state['sign'] = state['sign'].to(torch.bool)
                torch.where(state['sign'], update, -update, out=update)
            return update
        else:
            m_pos = self._unnmf(state['momentum_m_pos'])
            m_neg = self._unnmf(state['momentum_m_neg'])
            return torch.sub(m_pos, m_neg, out=m_pos)

    def _compress_momentum(self, matrix: torch.Tensor, state, momentum_name: str, group: dict):
        """Compresses a momentum tensor into its factorized form."""
        # The second moment 'v' is always non-negative. Handle its compression simply.
        if momentum_name == 'momentum_v':
            # We take abs() to be safe against any minor numerical noise, though v should be >= 0.
            self._nnmf(torch.abs(matrix), out=state['momentum_v'])
            return

        # The logic depends on the optimizer mode (use_grams, factored_sign, or default).
        if self.use_grams:
            # Grams mode: store magnitude only.
            self._nnmf(torch.abs(matrix), out=state['momentum_m'])
            
        elif self.factored_sign:
            # Factored sign mode: store positive and negative parts in separate factors.
            m_pos = matrix.relu()
            m_neg = matrix.neg().relu_()
            self._nnmf(m_pos, out=state['momentum_m_pos'])
            self._nnmf(m_neg, out=state['momentum_m_neg'])
            
        else:
            # Default: store magnitude and a separate boolean sign matrix.
            state['sign'] = matrix > 0
            self._nnmf(torch.abs(matrix), out=state['momentum_m'])

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad

        if grad.dtype != torch.float32:
            grad = grad.float()

        if self.use_orthograd:
            grad = self._orthogonalize_gradient(p, grad)

        state = self.state[p]

        # State Initialization
        if len(state) == 0:
            state['step'] = 0

            dimension = len(grad.squeeze().shape)
            state['factored'] = not (dimension == 1 and not group['vector_reshape'])

            if state['factored']:
                state['effective_shape'] = self._get_effective_shape(p.numel())
                device = p.device
                shape0, shape1 = state['effective_shape']

                if group['beta1'] is not None:
                    if self.factored_sign: 
                        state['momentum_m_pos'] = (torch.zeros(shape0, device=device), torch.zeros(shape1, device=device)) 
                        state['momentum_m_neg'] = (torch.zeros(shape0, device=device), torch.zeros(shape1, device=device)) 
                    elif not self.use_grams: # Only initialize sign if not using grams
                        state['momentum_m'] = (torch.zeros(shape0, device=device), torch.zeros(shape1, device=device)) 
                        state['sign'] = torch.zeros(state['effective_shape'], dtype=torch.bool, device=device) 
                    else: 
                        state['momentum_m'] = (torch.zeros(shape0, device=device), torch.zeros(shape1, device=device)) 

                state['momentum_v'] = (torch.zeros(shape0, device=device), torch.zeros(shape1, device=device))
            else:  # not factored
                if group['beta1'] is not None:
                    state['momentum_m'] = torch.zeros_like(p, dtype=torch.float32)
                state['momentum_v'] = torch.zeros_like(p, dtype=torch.float32)

        state['step'] += 1

        beta1 = group['beta1']
        eps = group['eps']
        decay_rate = group['decay_rate']
        beta1_growth_rate = group['beta1_growth_rate']

        if state['factored']:
            original_shape = p.shape
            if not grad.is_contiguous():
                grad = grad.contiguous()
            grad_reshaped = grad.view(state['effective_shape'])

            # Decompress, update, and compress momentums
            # 1. Decompress momentums ONCE into full local matrices
            update_m = None
            if beta1 is not None:
                update_m = self._decompress_momentum(state, 'momentum_m', group)
            update_v = self._decompress_momentum(state, 'momentum_v', group)

            # 2. Update the full local matrices
            if beta1 is not None:
                beta_m = beta1 * (beta1_growth_rate ** (state['step'] - 1.0))
                update_m.mul_(beta_m).add_(grad_reshaped, alpha=(1.0 - beta_m))
            else:
                # If no beta1, update_m is just the gradient for the division step
                update_m = grad_reshaped

            beta_v = 1.0 - (state['step'] ** decay_rate)
            update_v.mul_(beta_v).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta_v)

            # 3. Compute the final update using the full local matrices
            if self.use_atan2:
                a = 1.2732395
                update = torch.atan2(update_m, update_v.sqrt()).mul_(a)
            else:
                update = update_m / (update_v.sqrt() + eps)

            if self.use_grams:
                update = grad_reshaped.sign() * update.abs()

            update = update.contiguous().view(original_shape)

            # 4. Compress the updated full local matrices back into state ONCE
            if beta1 is not None:
                self._compress_momentum(update_m, state, 'momentum_m', group)
                del update_m
            self._compress_momentum(update_v, state, 'momentum_v', group)
            del update_v

        else:  # Non-factorized path
            if beta1 is not None:
                update_m = state['momentum_m']
                beta_m = beta1 * (beta1_growth_rate ** (state['step'] - 1.0))
                update_m.mul_(beta_m).add_(grad, alpha=(1.0 - beta_m))
            else:
                update_m = grad

            update_v = state['momentum_v']
            beta_v = 1.0 - (state['step'] ** decay_rate)
            update_v.mul_(beta_v).addcmul_(grad, grad, value=1.0 - beta_v)
            
            # Compute parameter update
            if self.use_atan2:
                a = 1.2732395
                update = torch.atan2(update_m, update_v.sqrt()).mul_(a)
            else:
                update = update_m / (update_v.sqrt() + eps)
            del update_m, update_v

            if self.use_grams:
                update = grad.sign() * update.abs()

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
        del update



    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (or handles the closure)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.step_parameter(p, group, i)
        return loss
