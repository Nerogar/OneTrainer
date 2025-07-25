# --- START OF FILE MLorc.py (Fused Backward Pass Compatible, Final Check with slice_p) ---

import torch
import torch.optim
import math
import torch.distributed as dist
from typing import Tuple

# Assumes this utility exists in the specified path
from modules.util.bf16_stochastic_rounding import add_stochastic_


class MLorc_Prodigy(torch.optim.Optimizer):
    """
    Implements MLorc algorithm with Prodigy D-adaptation.
    This implementation is based on:
    `MLorc: Momentum Low-rank Compression for Large Language Model Adaptation`
    (https://arxiv.org/abs/2506.01897)
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1)
        betas (tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        use_bias_correction (boolean): Turn on Adam's bias correction. (default: False)
        rank (int): the rank for the low-rank approximation (default: 4).
        oversampling (int): oversampling parameter for Randomized SVD.
            The paper suggests 0. (default: 0).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: False).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        d0 (float, optional): Initial D estimate for D-adaptation (default 1e-6).
        slice_p (int, optional): Reduce memory usage by calculating LR adaptation statistics
            on only every p-th entry of each tensor. For values greater than 1 this is
            an approximation. (default: 11).
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
        use_grams (bool): whether to use Grams-style updates. (default: False)
        use_orthograd (bool): whether to use OrthoGrad.  (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        beta3: float = None,
        d0: float = 1e-6,
        d_coef: float = 1,
        growth_rate: float = float('inf'),
        safeguard_warmup: bool = False,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_bias_correction: bool = False,
        rank: int = 4,
        oversampling: int = 0,
        slice_p: int = 11,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        use_atan2: bool = False,
        use_grams: bool = False,
        use_orthograd: bool = False,
    ):
        if not (d0 > 0.0):
            raise ValueError(f"Invalid d0 value: {d0}")
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas should be in [0.0, 1.0). Got {betas}")
        if not (eps >= 0.0):
            raise ValueError(f"Epsilon should be >= 0.0. Got {eps}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (rank >= 1):
            raise ValueError(f"Rank should be >= 1. Got {rank}")
        if not (slice_p >= 1):
            raise ValueError(f"slice_p should be >= 1. Got {slice_p}")

        defaults = {
            "lr": lr, "betas": betas, "beta3": beta3, "eps": eps, "weight_decay": weight_decay,
            "rank": rank, "oversampling": oversampling, "slice_p": slice_p, "vector_reshape": vector_reshape,
            "use_bias_correction": use_bias_correction,"use_atan2": use_atan2, "use_grams": use_grams,
            "use_orthograd": use_orthograd, "d": d0, "d0": d0, "d_max": d0, "d_numerator": 0.0, "d_coef": d_coef,
            "growth_rate": growth_rate, "safeguard_warmup": safeguard_warmup, "k": 0,
        }
        self.stochastic_rounding = stochastic_rounding
        super().__init__(params, defaults)
        self.init_step()

    @staticmethod
    def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """ Projects the gradient grad to be orthogonal to the parameter p, and then re-scales the result to have the same norm as the original gradient. This method is based on the OrthoGrad optimizer from the paper "Grokking at the Edge of Numerical Stability" (Prieto et al., 2025). It is intended to prevent Naïve Loss Minimization (NLM) by removing the component of the gradient that is parallel to the weight vector. """
        if grad.is_sparse: raise RuntimeError("OrthoGrad does not support sparse gradients.")
        original_shape = grad.shape
        w, g = p.view(-1).float(), grad.view(-1).float()
        w_norm_sq = torch.dot(w, w).add_(1e-30)
        proj = torch.dot(w, g) / w_norm_sq
        g_orth = g.sub(w, alpha=proj)
        g_orth_scaled = g_orth * (g.norm(2) / g_orth.norm(2).add_(1e-30))
        return g_orth_scaled.view(original_shape).to(p.dtype)

    @property
    def supports_fused_back_pass(self):
        return True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
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

    def _rsvd(self, A: torch.Tensor, rank: int, oversampling: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def init_step(self):
        """Resets accumulators and calculates dlr for the upcoming step."""
        self.d_denom = 0.0
        
        g_group = self.param_groups[0]
        self.beta1, self.beta2 = g_group['betas']
        self.beta3 = g_group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2)
        
        k = g_group['k']
        self.d = g_group['d']
        lr = g_group['lr']
        use_bias_correction = g_group['use_bias_correction']

        if use_bias_correction:
            bias_correction = ((1 -  self.beta2**(k+1))**0.5) / (1 -  self.beta1**(k+1))
        else:
            bias_correction = 1
        self.dlr = self.d * lr * bias_correction

        self.d_numerator = g_group.get('d_numerator', 0.0) * self.beta3

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        grad = p.grad.data
        if group["use_orthograd"]:
            grad = self._orthogonalize_gradient(p, grad)
        
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
            state['factored'] = not (len(p.shape) == 1 and not group['vector_reshape'])
            p_dtype, p_device = p.dtype, p.device
            slice_p = group['slice_p']
            
            if state['factored']:
                d1, d2 = self._get_effective_shape(p.numel())
                r = group['rank']
                state['mu_m'] = torch.zeros(d1, r, device=p_device, dtype=p_dtype)
                state['ms_m'] = torch.zeros(r, device=p_device, dtype=p_dtype)
                state['mv_m'] = torch.zeros(r, d2, device=p_device, dtype=p_dtype)
                state['mu_v'] = torch.zeros(d1, r, device=p_device, dtype=p_dtype)
                state['ms_v'] = torch.zeros(r, device=p_device, dtype=p_dtype)
                state['mv_v'] = torch.zeros(r, d2, device=p_device, dtype=p_dtype)
            else:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            flat_p = p.data.flatten()
            state['s'] = torch.zeros_like(flat_p[::slice_p]).detach()
            state['p0'] = flat_p[::slice_p].detach().clone()
        
        
        # --- Update Adam states and re-compress (MLorc logic) ---
        if state['factored']:
            d1, d2 = self._get_effective_shape(p.numel())
            mt_prev = state['mu_m'] @ torch.diag(state['ms_m']) @ state['mv_m']
            vt_prev = state['mu_v'] @ torch.diag(state['ms_v']) @ state['mv_v']
            neg_mask = vt_prev < 0
            if neg_mask.any():
                adaptive_constant = torch.abs(vt_prev[neg_mask].mean())
                vt_prev_corrected = vt_prev.relu_().add_(neg_mask * adaptive_constant)
            else:
                vt_prev_corrected = vt_prev.relu_()
            grad_reshaped = grad.view(d1, d2)
            mt = mt_prev.mul_(self.beta1).add_(grad_reshaped, alpha=self.d * (1.0 - self.beta1))
            vt = vt_prev_corrected.mul_(self.beta2).addcmul_(grad_reshaped, grad_reshaped, value=self.d * self.d * (1.0 - self.beta2))
            exp_avg, exp_avg_sq = mt.view_as(p), vt.view_as(p)
            mu_m, ms_m, mv_m = self._rsvd(mt, group['rank'], group['oversampling'])
            mu_v, ms_v, mv_v = self._rsvd(vt, group['rank'], group['oversampling'])
            state['mu_m'].copy_(mu_m); state['ms_m'].copy_(ms_m); state['mv_m'].copy_(mv_m)
            state['mu_v'].copy_(mu_v); state['ms_v'].copy_(ms_v); state['mv_v'].copy_(mv_v)
        else:
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.mul_(self.beta1).add_(grad, alpha=self.d * (1.0 - self.beta1))
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=self.d * self.d * (1.0 - self.beta2))

        # --- Accumulate Prodigy stats ---
        d0, safeguard_warmup, slice_p = group['d0'], group['safeguard_warmup'], group['slice_p']
        s, p0 = state['s'], state['p0']
        grad_flat = grad.flatten()
        p_flat = p.data.flatten()
        
        self.d_numerator += (self.d / d0) * self.dlr * torch.dot(grad_flat[::slice_p], p0.data - p_flat[::slice_p]).item()
        
        alpha = ((self.d / d0) * self.d) if safeguard_warmup else ((self.d / d0) * self.dlr)
        s.mul_(self.beta3).add_(grad_flat[::slice_p], alpha=alpha)
        self.d_denom += s.abs().sum().item()

        if group["weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-group["weight_decay"] * self.dlr)
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * self.dlr)

        if group['use_atan2']:
            denom = exp_avg_sq.sqrt()
            update = torch.atan2(exp_avg, denom).mul_(1.2732395)
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'] * self.d)
            update = exp_avg / denom

        if group['use_grams']:
            update.abs_().mul_(grad.sign())

        update.mul_(self.dlr)
        
        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)
            
        state['step'] += 1

    def calculate_d(self):
        """Calculates the new `d` based on the accumulated stats."""
        g_group = self.param_groups[0]
        d_max, d_coef, growth_rate = g_group['d_max'], g_group['d_coef'], g_group['growth_rate']
        
        global_d_numerator = self.d_numerator
        global_d_denom = self.d_denom

        d_hat = self.d
        if global_d_denom > 0:
            d_hat = d_coef * global_d_numerator / global_d_denom
            if self.d == g_group['d0']:
                self.d = max(self.d, d_hat)
            d_max = max(d_max, d_hat)
            self.d = min(d_max, self.d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d'] = self.d
            group['d_max'] = d_max
            group['k'] += 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        self.calculate_d()
        self.init_step()
        return loss