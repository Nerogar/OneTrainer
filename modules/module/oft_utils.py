import math

import torch
import torch.nn as nn


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    This layer randomly replaces the learned rotation blocks with identity matrices during training.
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            D, H, _ = x.shape

            if D == 1:
                return x

            keep_prob = 1.0 - self.p

            # This 'stochastic_mask' has 1s for blocks to keep, and 0s for blocks to replace with Identity.
            stochastic_mask = torch.empty(D, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(p=keep_prob)

            eye_matrix = torch.eye(H, device=x.device, dtype=x.dtype).repeat(D, 1, 1)

            x = stochastic_mask * x + (1 - stochastic_mask) * eye_matrix

        return x


class OFTRotationModule(nn.Module):
    def __init__(
        self,
        r,
        n_elements,
        block_size,
        in_features,
        block_share=False,
        oft_scaled=False,
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
        oft_cans=False,
        dropout_probability=0.0,
        oft_clipped_norm: float | None = 0.95,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(r, n_elements))
        self.block_share = block_share
        if oft_scaled:
            # Register a persistent buffer to indicate this module uses Scaled OFT.
            # This embeds the scaling configuration directly into the state_dict,
            # allowing inference tools to automatically detect scaled oft.
            self.register_buffer("scaled_oft", torch.tensor(True))
        self.oft_scaled = oft_scaled
        self.use_cayley_neumann = use_cayley_neumann and not oft_cans
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.oft_cans = oft_cans
        if oft_cans:
            self.register_buffer("cans_exp", torch.tensor([]))
        # Create indices for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)
        if not self.use_cayley_neumann:
            id_mat = (torch.eye(block_size).unsqueeze(0).expand(r, block_size, block_size))
            self.register_buffer("id_mat", id_mat, persistent=False)
        if oft_clipped_norm == -1:
            if oft_cans:
                # 0.95 * pi (~3.11) avoids the gradient ambiguity/singularity
                # at exactly 180 degrees (pi).
                self.oft_clipped_norm = 0.95 * math.pi
            elif use_cayley_neumann:
                # Neumann series diverges if norm >= 1.0, so 0.95 is the safe max.
                self.oft_clipped_norm = 0.95
        else:
            self.oft_clipped_norm = oft_clipped_norm
        if oft_clipped_norm is not None:
            self.register_buffer("clipped_oft", torch.tensor(self.oft_clipped_norm))
            # Initialize states for Spectral Normalization via Power Iteration
            u = torch.randn(r, block_size)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("u_state", u, persistent=False)
            v = torch.randn(r, block_size)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("v_state", v, persistent=False)

    def _pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

        #the following two lines are equivalent to "matrix[:, self.rows, self.cols] = vec",
        #but they work around a pytorch issue: https://github.com/pytorch/pytorch/issues/169179
        batch_idx = torch.arange(batch_size, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, self.rows, self.cols), vec)

        matrix = matrix - matrix.transpose(-2, -1)
        return matrix

    def _pytorch_skew_symmetric_inv(self, matrix, block_size):
        # Extract the upper triangular elements
        vec = matrix[:, self.rows, self.cols]
        return vec

    def _break_inductor_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Acts as an opaque boundary for TorchInductor. Forces materialization
        of the tensor to sever deeply nested AST trees in torch.compile.
        """
        return torch.bmm(x, torch.ones_like(x))

    @torch.no_grad()
    def _spectral_norm(self, Q_skew):
        u = self.u_state.unsqueeze(-1).to(Q_skew.dtype)
        v = self.v_state.unsqueeze(-1).to(Q_skew.dtype)
        # Update v (Right Singular Vector)
        v_raw = torch.bmm(Q_skew.mT, u)
        v_norm = torch.linalg.vector_norm(v_raw, dim=1, keepdim=True)
        candidate_v = v_raw / v_norm.clamp_min(1e-8)
        next_v = torch.where(v_norm >= 1e-6, candidate_v, v)
        # Update u (Left Singular Vector)
        u_raw = torch.bmm(Q_skew, next_v)
        u_norm = torch.linalg.vector_norm(u_raw, dim=1, keepdim=True)
        candidate_u = u_raw / u_norm.clamp_min(1e-8)
        next_u = torch.where(u_norm >= 1e-6, candidate_u, u)
        if self.training:
            self.v_state.copy_(next_v.squeeze(-1))
            self.u_state.copy_(next_u.squeeze(-1))
        return next_v, next_u

    def _cans_newton_schulz_iteration(
        self,
        G: torch.Tensor,
        steps: int = 3,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Chebyshev-Optimized Newton-Schulz iteration with a dynamically computed Chebyshev lower bound.
        """
        original_dtype = G.dtype
        X = G

        # Max row sum is guaranteed to be >= the maximum singular value of X.
        g_norm = X.abs().sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(eps).detach()
        X = X / g_norm

        # The 4th-order Taylor expansion of exp(Q) has a minimum singular value of exactly 0.5
        # (occurring at ||Q|| = sqrt(6) ~ 2.449). Therefore, the min_singular_value of normalized X
        # is guaranteed to be >= 0.5 / g_norm.
        lower_bound = 0.5 / g_norm
        upper_bound = 1

        for _ in range(steps):
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub
            e_sq = (lb**2 + lb_ub + ub**2) / 3.0
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)
            denom = K + L
            alpha = 6.0 / denom
            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            A = torch.bmm(X, X.mT)
            X = c1 * X + c3 * torch.bmm(A, X)

            # Dynamically update bounds for the next step
            eps_val = (K - L) / denom
            eps_val = self._break_inductor_graph(eps_val)

            lower_bound = 1 - eps_val
            upper_bound = 1 + eps_val

        return X.to(original_dtype)

    def _matrix_exp_cans(self, Q_skew: torch.Tensor) -> torch.Tensor:
        """
        Approximates the Matrix Exponential using a 4th-order Taylor expansion,
        Scaling & Squaring, and Chebyshev-Optimized Newton-Schulz (CANS).
        """
        num_squarings = 2
        id_mat = self.id_mat

        # Scaling step
        Q_scaled = Q_skew / (2 ** num_squarings)
        Q_squared = torch.bmm(Q_scaled, Q_scaled)

        # 4th-order Taylor expansion: exp(Q) ≈ I + Q + Q^2/2 + Q^3/6 + Q^4/24
        # Factored to minimize matrix multiplications: (I + Q) + Q^2 * (0.5*I + 1/6*Q + 1/24*Q^2)
        taylor_higher_order = 0.5 * id_mat + (1.0 / 6.0) * Q_scaled + (1.0 / 24.0) * Q_squared
        G = torch.baddbmm(id_mat + Q_scaled, Q_squared, taylor_higher_order)

        # Orthogonalize the approximation (CANS)
        # Empirically, CANS requires 3 steps to converge
        R = self._cans_newton_schulz_iteration(G=G, steps=3)

        # Squaring step to recover full rotation
        for _ in range(num_squarings):
            R = torch.bmm(R, R)

        # Final standard Newton-Schulz step to correct drift caused by squaring in lower precision
        # R_new = R + 0.5 * R * (I - R^T R)
        residual = torch.baddbmm(id_mat, R.mT, R, beta=1.0, alpha=-1.0)
        R = torch.baddbmm(R, R, residual, beta=1.0, alpha=0.5)

        return R

    def _compute_orthogonal_matrix(
        self, Q: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5, oft_cans: bool = False,
    ) -> torch.Tensor:
        """
        Converts learned weights into a batch of orthogonal matrices.
        """
        b, _ = Q.shape
        previous_dtype = Q.dtype

        Q_skew = self._pytorch_skew_symmetric(Q, block_size)

        # Spectral Normalization / Clipping
        if self.oft_clipped_norm is not None:
            v_norm, u_norm = self._spectral_norm(Q_skew)
            u_raw_grad = torch.bmm(Q_skew, v_norm)
            sigma = torch.sum(u_norm * u_raw_grad, dim=1, keepdim=True)
            max_norm = self.oft_clipped_norm
            Q_skew = Q_skew * (max_norm / torch.clamp(sigma, min=max_norm))

        if oft_cans:
            R = self._matrix_exp_cans(Q_skew)
        elif use_cayley_neumann:
            R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
            if num_neumann_terms > 1:
                R.add_(Q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    Q_squared = torch.bmm(Q_skew, Q_skew)
                    R.add_(Q_squared, alpha=2.0)

                    Q_power = Q_squared
                    for _ in range(3, num_neumann_terms - 1):
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power, alpha=2.0)
                    Q_power = torch.bmm(Q_power, Q_skew)
                    R.add_(Q_power)
        else:
            R = torch.linalg.solve(self.id_mat + Q_skew, self.id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    def forward(self, x):
        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape

        if self.oft_scaled:
            # Cayley has a 2x gradient multiplier (I + 2Q). Exp has a 1x multiplier (I + Q).
            # We drop the 2 for Exp/CANS to maintain consistent effective learning rates.
            is_cayley = self.use_cayley_neumann and not self.oft_cans
            multiplier = 2.0 if is_cayley else 1.0
            scaling_factor = multiplier * math.sqrt(self.block_size - 1)
            effective_weight = self.weight / scaling_factor
        else:
            effective_weight = self.weight

        orth_rotate = self._compute_orthogonal_matrix(
            effective_weight, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms, self.oft_cans
        )
        orth_rotate = self.dropout(orth_rotate)

        rank = self.in_features // self.block_size if self.block_share else self.r
        batch_dims = x.shape[:-1]
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)

        if self.block_share:
            orth_rotate = orth_rotate.repeat(rank, 1, 1)
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
        else:
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)

        x_rotated = x_rotated_reshaped.reshape(*orig_shape)

        return x_rotated.to(required_dtype)
