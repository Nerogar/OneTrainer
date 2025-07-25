from typing import Tuple, Optional, Any

import torch
from torch.optim.optimizer import Optimizer

from modules.util.bf16_stochastic_rounding import add_stochastic_

class MLorc_Lion(Optimizer):
    """
    Implements the MLorc-Lion algorithm.

    This optimizer combines the Lion update rule with the memory-saving low-rank
    compression (MLorc) technique from https://arxiv.org/abs/2506.01897 (see Algorithm 2).
    It stores the momentum state in a compressed format (U, S, Vh) and only
    reconstructs it to full size during the update step, significantly
    reducing memory usage for optimizer states.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-4).
        betas (Tuple[float, float], optional): coefficients for computing
            running averages of the update (default: (0.9, 0.99)).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0).
        rank (int, optional): the rank for the low-rank approximation (default: 4).
        oversampling (int, optional): oversampling parameter for Randomized SVD.
            The paper suggests 0 (default: 0).
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool, optional): whether to use stochastic
            rounding for BF16 parameter updates (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        rank: int = 4,
        oversampling: int = 0,
        vector_reshape: bool = True,
        stochastic_rounding: bool = False,
        use_orthograd: bool = False,
    ):
        if not lr > 0.0:
            raise ValueError(f"Learning rate must be > 0.0, but got {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Betas should be in [0.0, 1.0], but got {betas}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")
        if not rank >= 1:
            raise ValueError(f"Rank must be >= 1, but got {rank}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            rank=rank,
            oversampling=oversampling,
            vector_reshape=vector_reshape,
            use_orthograd=use_orthograd,
        )
        self.stochastic_rounding = stochastic_rounding
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Projects the gradient `grad` to be orthogonal to the parameter `p`."""
        if grad.is_sparse:
            raise RuntimeError("OrthoGrad logic does not support sparse gradients.")
        original_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        w_norm_sq = torch.dot(w, w).add_(1e-30)
        proj = torch.dot(w, g) / w_norm_sq
        g_orth = g.sub(w, alpha=proj)
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

    def _get_effective_shape(self, numel: int) -> tuple[int, int]:
        """Finds two factors of numel that are closest to its square root."""
        # Handle non-positive numel for robustness.
        if numel <= 0:
            return (0, 0)
        for i in reversed(range(1, int(numel ** 0.5) + 1)):
            if numel % i == 0:
                return (numel // i, i)
        return (numel, 1)

    def _rsvd(self, A: torch.Tensor, rank: int, oversampling: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs Randomized SVD."""
        orig_dtype = A.dtype
        device = A.device
        A_float = A.float()

        m, n = A_float.shape
        l = rank + oversampling
        true_rank = min(m, n, rank)

        # Explicitly handle zero-rank case for robustness.
        if true_rank == 0:
            U = torch.zeros(m, rank, dtype=orig_dtype, device=device)
            S = torch.zeros(rank, dtype=orig_dtype, device=device)
            Vh = torch.zeros(rank, n, dtype=orig_dtype, device=device)
            return U, S, Vh

        if l >= min(m, n):  # Fallback to full SVD
            U_full, S_full, Vh_full = torch.linalg.svd(A_float, full_matrices=False)
            U, S, Vh = U_full[:, :true_rank], S_full[:true_rank], Vh_full[:true_rank, :]
        else:  # Standard RSVD path
            Omega = torch.randn(n, l, dtype=A_float.dtype, device=device)
            Y = A_float @ Omega
            Q, _ = torch.linalg.qr(Y)
            B = Q.T @ A_float
            U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
            # Truncate to the true computable rank
            U, S, Vh = (Q @ U_tilde)[:, :true_rank], S[:true_rank], Vh[:true_rank, :]

        # Pad factors with zeros if true_rank < rank
        if true_rank < rank:
            U_padded = torch.zeros(m, rank, dtype=A_float.dtype, device=device)
            S_padded = torch.zeros(rank, dtype=A_float.dtype, device=device)
            Vh_padded = torch.zeros(rank, n, dtype=A_float.dtype, device=device)
            U_padded[:, :true_rank] = U
            S_padded[:true_rank] = S
            Vh_padded[:true_rank, :] = Vh
            U, S, Vh = U_padded, S_padded, Vh_padded

        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: Optional[int] = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad
        if group["use_orthograd"]:
            grad = self._orthogonalize_gradient(p, grad)
        state = self.state[p]

        # State Initialization
        if len(state) == 0:
            state['step'] = 0
            state['factored'] = not (len(p.shape) == 1 and not group['vector_reshape'])

            if state['factored']:
                state['effective_shape'] = self._get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                r = group['rank']
                state['mu'] = torch.zeros(d1, r, device=p.device, dtype=p.dtype)
                state['ms'] = torch.zeros(r, device=p.device, dtype=p.dtype)
                state['mv'] = torch.zeros(r, d2, device=p.device, dtype=p.dtype)
            else: # Fallback to standard Lion
                state['exp_avg'] = torch.zeros_like(p)

        state['step'] += 1
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]

        if state['factored']:
            # --- MLorc-Lion Path ---
            d1, d2 = state['effective_shape']
            rank = group['rank']
            oversampling = group['oversampling']

            # 1. Reconstruct momentum from previous step's factors (m_{t-1})
            exp_avg_prev = state['mu'] @ torch.diag(state['ms']) @ state['mv']
            grad_reshaped = grad.view(d1, d2)

            # 2. Lion's update rule
            # Compute update term c_t = β1*m_{t-1} + (1-β1)*g_t
            update_term_ct = torch.lerp(grad_reshaped, exp_avg_prev, beta1)

            # Compute new momentum m_t = β2*m_{t-1} + (1-β2)*g_t
            exp_avg_new = torch.lerp(grad_reshaped, exp_avg_prev, 1.0 - beta2)

            # Take the sign of c_t for the parameter update
            signed_update = update_term_ct.sign()

            # 3. Apply weight decay to parameter
            if weight_decay != 0:
                p.data.add_(p.data, alpha=-weight_decay * lr)

            # 4. Apply update to parameter: p_t = p_{t-1} - lr * sign(c_t)
            signed_update.mul_(lr)
            update_reshaped = signed_update.view(p.shape)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, -update_reshaped)
            else:
                p.data.add_(-update_reshaped)

            # 5. Compress new momentum (m_t) and store factors for the next step
            mu_new, ms_new, mv_new = self._rsvd(exp_avg_new, rank, oversampling)
            state['mu'].copy_(mu_new)
            state['ms'].copy_(ms_new)
            state['mv'].copy_(mv_new)
        else:
            # --- Fallback to standard Lion logic ---
            exp_avg = state["exp_avg"]
            update = torch.lerp(grad, exp_avg, beta1)
            update.sign_()
            exp_avg.lerp_(grad, 1.0 - beta2)

            if weight_decay != 0:
                p.data.add_(p.data, alpha=-weight_decay * lr)

            update.mul_(lr)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, -update)
            else:
                p.data.add_(-update)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        return loss