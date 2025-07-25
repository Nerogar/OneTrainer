# FILE: mlorc_dadapt_lion.py

from typing import Tuple, Optional, Any, Callable

import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import logging

from modules.util.bf16_stochastic_rounding import add_stochastic_


class MLorc_DAdapt_Lion(Optimizer):
    """
    Implements the MLorc-Lion algorithm with D-Adaptation.

    This optimizer combines three techniques:
    1. The Lion update rule (https://arxiv.org/abs/2302.06675).
    2. Memory-saving low-rank compression (MLorc) from https://arxiv.org/abs/2506.01897.
    3. Learning-Rate-Free Learning by D-Adaptation (https://arxiv.org/abs/2301.07733).

    It stores the momentum state in a compressed format (U, S, Vh) and only
    reconstructs it during the update, significantly reducing memory. The learning
    rate is dynamically adjusted based on the gradient and update history.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): A global scaling factor for the D-adapted learning
            rate. For D-Adaptation, this is typically left at 1.0 (default: 1.0).
        betas (Tuple[float, float], optional): Coefficients for Lion's momentum
            (default: (0.9, 0.99)).
        weight_decay (float, optional): AdamW-style weight decay (default: 0.0).
        rank (int, optional): The rank for low-rank approximation (default: 4).
        oversampling (int, optional): Oversampling for Randomized SVD (default: 0).
        vector_reshape (bool, optional): Reshape 1D vectors to 2D for compression
            (default: True).
        use_orthograd (bool, optional): whether to use OrthoGrad. (default: False)
        stochastic_rounding (bool, optional): Use stochastic rounding for BF16
            updates (default: True).
        d0 (float, optional): Initial D estimate for D-adaptation (default 1e-6).
        log_every (int, optional): Log D-adaptation statistics every k steps.
            0 means no logging (default: 0).
        fsdp_in_use (bool, optional): Set to True if using FSDP, for correct
            distributed aggregation (default: False).
        slice_p (int, optional): Reduce memory usage by calculating LR adaptation statistics
            on only every p-th entry of each tensor. For values greater than 1 this is
            an approximation. (default: 11).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        rank: int = 4,
        oversampling: int = 0,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        d0: float = 1e-6,
        slice_p: int = 11,
        log_every: int = 0,
        fsdp_in_use: bool = False,
        use_orthograd: bool = False,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Learning rate must be >= 0.0, but got {lr}")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Betas must be in [0.0, 1.0), but got {betas}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")
        if not rank >= 1:
            raise ValueError(f"Rank must be >= 1, but got {rank}")
        if not slice_p >= 1:
            raise ValueError(f"slice_p must be >= 1, but got {slice_p}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            # MLorc settings
            rank=rank,
            oversampling=oversampling,
            vector_reshape=vector_reshape,
            use_orthograd=use_orthograd,
            # D-Adaptation settings
            d=d0,
            slice_p=slice_p,
            k=0,
            log_every=log_every,
            numerator_weighted=0.0,
            fsdp_in_use=fsdp_in_use,
        )
        self.stochastic_rounding = stochastic_rounding
        super().__init__(params, defaults)



        # Global state for accumulating metrics across parameter updates within a single step.
        # This is essential for fused backward pass support, where step_parameter is
        # called from hooks before the main step() is called.
        self.global_state = {"numerator_acum": 0.0, "sk_l1": 0.0}

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

    def _get_effective_shape(self, numel: int) -> tuple[int, int]:
        """Finds two factors of numel that are closest to its square root."""
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

        if true_rank == 0:
            U = torch.zeros(m, rank, dtype=orig_dtype, device=device)
            S = torch.zeros(rank, dtype=orig_dtype, device=device)
            Vh = torch.zeros(rank, n, dtype=orig_dtype, device=device)
            return U, S, Vh

        if l >= min(m, n):
            U_full, S_full, Vh_full = torch.linalg.svd(A_float, full_matrices=False)
            U, S, Vh = U_full[:, :true_rank], S_full[:true_rank], Vh_full[:true_rank, :]
        else:
            Omega = torch.randn(n, l, dtype=A_float.dtype, device=device)
            Y = A_float @ Omega
            Q, _ = torch.linalg.qr(Y)
            B = Q.T @ A_float
            U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U, S, Vh = (Q @ U_tilde)[:, :true_rank], S[:true_rank], Vh[:true_rank, :]

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
        """
        Performs a single optimization step on a parameter and accumulates
        metrics for D-Adaptation.
        """
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
            slice_p = group['slice_p']
            # D-Adaptation state
            state['s'] = torch.zeros_like(p.flatten()[::slice_p], device=p.device) if slice_p > 1 else torch.zeros_like(p)

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
        sqrt_beta2 = beta2 ** 0.5
        dlr = group['d'] * group['lr']

        if group["weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data,
                                alpha=-group["weight_decay"] * dlr)
            else:
                p.data.add_(
                    p.data, alpha=-group["weight_decay"] * dlr
                )

        signed_update = None
        s = state['s']

        if state['factored']:
            # --- MLorc-Lion Path ---
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)
            
            # Reconstruct momentum m_{t-1}
            exp_avg_prev = state['mu'] @ torch.diag(state['ms']) @ state['mv']
            
            # Compute update term c_t = β1*m_{t-1} + (1-β1)*g_t
            update_term_ct = torch.lerp(grad_reshaped, exp_avg_prev, beta1)
            signed_update = update_term_ct.sign()
            
            # Parameter update: p_t = p_{t-1} - dlr * sign(c_t)
            update_reshaped = signed_update.mul(dlr)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, -update_reshaped.view(p.shape))
            else:
                p.data.add_(-update_reshaped.view(p.shape))

            # Update momentum m_t = β2*m_{t-1} + (1-β2)*dlr*g_t
            exp_avg_new = exp_avg_prev.mul(beta2).add_(grad_reshaped, alpha=(1-beta2)*dlr)
            
            # Compress new momentum m_t and store factors
            mu_new, ms_new, mv_new = self._rsvd(exp_avg_new, group['rank'], group['oversampling'])
            state['mu'].copy_(mu_new)
            state['ms'].copy_(ms_new)
            state['mv'].copy_(mv_new)
        else:
            # --- Fallback to standard D-Adapt Lion logic ---
            exp_avg = state["exp_avg"]
            
            # Compute update term and sign for the update
            update = exp_avg.clone().mul_(beta1).add_(grad, alpha=(1-beta1)).sign_()
            signed_update = update
            
            # Parameter update
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, -update.mul(dlr))
            else:
                p.data.add_(update, alpha=-dlr)
            
            # Update momentum with dlr scaling
            exp_avg.mul_(beta2).add_(grad, alpha=(1-beta2)*dlr)

        # --- D-Adaptation Metric Accumulation (for both paths) ---
        # This part populates the global state for the final d-update in step()
        # It uses slicing for memory efficiency if slice_p > 1.
        slice_p = group['slice_p']

        if slice_p > 1:
            sliced_signed_update = signed_update.flatten()[::slice_p]
            # s is already sliced from the previous step
            numerator_contrib = dlr * torch.dot(sliced_signed_update, s)
            s.mul_(sqrt_beta2).add_(sliced_signed_update, alpha=(1 - sqrt_beta2) * dlr)
            sk_l1_contrib = s.abs().sum()
        else: # Original, non-slicing path
            numerator_contrib = dlr * torch.dot(signed_update.view(-1), s.view(-1))
            # Update full-sized s
            s.mul_(sqrt_beta2).add_(signed_update.view_as(s), alpha=(1 - sqrt_beta2) * dlr)
            sk_l1_contrib = s.abs().sum()

        self.global_state["numerator_acum"] += numerator_contrib.item()
        self.global_state["sk_l1"] += sk_l1_contrib.item()


    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Performs a single optimization step.

        This method acts as a controller. In a standard setup, it iterates over
        parameters and calls `step_parameter`. In a fused backward setup, the
        hooks call `step_parameter` first, and this method then only performs
        the final D-Adaptation calculation.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # In non-fused mode, drive the parameter updates ourselves.
        # In fused mode, `step_parameter` has already been called by hooks,
        # so p.grad will be None, and this loop is skipped.
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        # --- D-Adaptation Finalization Step ---
        # This logic runs after all parameter updates for the step are complete.
        g_ref = self.param_groups[0]
        fsdp_in_use = g_ref['fsdp_in_use']
        numerator_weighted = g_ref['numerator_weighted']
        d = g_ref['d']
        lr = g_ref['lr']
        beta2 = g_ref['betas'][1]
        sqrt_beta2 = beta2 ** 0.5
        log_every = g_ref['log_every']
        k = g_ref['k']

        numerator_acum = self.global_state["numerator_acum"]
        sk_l1 = self.global_state["sk_l1"]

        # If no gradients were processed, skip d-adaptation update
        if sk_l1 == 0:
            # Reset state for the next step and return
            self.global_state = {"numerator_acum": 0.0, "sk_l1": 0.0}
            return loss

        if fsdp_in_use:
            # Aggregate numerator and sk_l1 across all ranks
            dist_tensor = torch.tensor([numerator_acum, sk_l1], device=self.param_groups[0]['params'][0].device)
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            global_numerator_acum = dist_tensor[0].item()
            global_sk_l1 = dist_tensor[1].item()
        else:
            global_numerator_acum = numerator_acum
            global_sk_l1 = sk_l1

        # Update weighted numerator
        global_numerator_weighted = sqrt_beta2 * numerator_weighted + (1 - sqrt_beta2) * global_numerator_acum

        d_hat = 0.0
        if global_sk_l1 > 0:
            d_hat = global_numerator_weighted / ((1 - sqrt_beta2) * global_sk_l1)

        if lr > 0.0:
            d = max(d, d_hat)

        if log_every > 0 and k % log_every == 0:
            logging.info(f"lr: {lr} d: {d:.4e} d_hat: {d_hat:.4e} dlr: {d*lr:.4e} sk_l1={global_sk_l1:1.1e}")

        # Update shared state in all param groups
        for group in self.param_groups:
            group['d'] = d
            group['numerator_weighted'] = global_numerator_weighted
            group['k'] += 1

        # Reset global state for the next optimization step
        self.global_state = {"numerator_acum": 0.0, "sk_l1": 0.0}

        return loss