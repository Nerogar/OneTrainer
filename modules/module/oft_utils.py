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
            self.register_buffer("cans_oft", torch.tensor(True))
        # Create indices for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)


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

    def _cans_newton_schulz_iteration(
        self,
        G: torch.Tensor,
        steps: int = 7,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Chebyshev-Optimized Newton-Schulz iteration with a dynamically computed Chebyshev lower bound.
        Optimized for G = I + Q (where Q is skew-symmetric).
        """
        original_dtype = G.dtype
        X = G.bfloat16()

        # Max row sum is guaranteed to be >= the maximum singular value of X.
        g_norm = torch.linalg.matrix_norm(X, ord=float('inf'), keepdim=True).clamp_min(eps)
        X = X / g_norm

        # Since min_singular_value(I + Q) >= 1, the min_singular_value of normalized X
        # is guaranteed to be >= 1 / ||G||_F.
        # We clamp it to prevent numerical edge cases (e.g. extremely large norms).
        lower_bound = (1.0 / g_norm.detach()).clamp(min=1e-5, max=0.9)
        one = torch.ones_like(lower_bound)
        upper_bound = one

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
            lower_bound = one - eps_val
            upper_bound = one + eps_val

        return X.to(original_dtype)

    def _cayley_batch(
        self, Q: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5, oft_cans: bool = False,
    ) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.
        """
        b, _ = Q.shape
        previous_dtype = Q.dtype

        Q_skew = self._pytorch_skew_symmetric(Q, block_size)

        if use_cayley_neumann:
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
            id_mat = (
                torch.eye(Q_skew.shape[-1], device=Q_skew.device)
                .unsqueeze(0)
                .expand(b, Q_skew.shape[-1], Q_skew.shape[-1])
            )
            if oft_cans:
                G = id_mat + Q_skew
                R = self._cans_newton_schulz_iteration(G=G, steps=5)
            else:
                R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    def forward(self, x):
        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape

        scaling_factor = 2 * math.sqrt(self.block_size - 1) if self.oft_scaled else 1
        effective_weight = self.weight / scaling_factor

        orth_rotate = self._cayley_batch(
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
