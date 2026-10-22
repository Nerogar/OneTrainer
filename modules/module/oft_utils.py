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
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        # Create indices for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)
        # Set via set_merge_rot_source: the LinearQuantRot module whose O^T input rotation is
        # folded into this module's R, so its quantized module can skip applying O^T itself.
        # Held in a list so nn.Module.__setattr__ does not register it as a submodule.
        self._merge_rot_source = None


    def set_merge_rot_source(self, source):
        self._merge_rot_source = [source] if source is not None else None

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

    def _cayley_batch(
        self, Q: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5
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
            effective_weight, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
        )
        orth_rotate = self.dropout(orth_rotate)

        rank = self.in_features // self.block_size
        if self.block_share:
            # Always ensure full rank before merging or contracting
            orth_rotate = orth_rotate.repeat(rank, 1, 1)

        # Ensure rotation matrix matches the input activation dtype
        orth_rotate = orth_rotate.to(x.dtype)

        if self._merge_rot_source is not None:
            source = self._merge_rot_source[0]
            b_o, b_q = self.block_size, source.quant_rot_block_size
            b_c, b_f = max(b_o, b_q), min(b_o, b_q)
            k, n = b_c // b_f, self.in_features // b_c

            # Ensure quant_rot matches device and dtype upfront
            quant_rot = source.quant_rot.to(device=orth_rotate.device, dtype=orth_rotate.dtype)

            if k == 1:
                M = torch.matmul(orth_rotate, quant_rot.transpose(-1, -2))
            elif b_o > b_q:
                M = torch.matmul(
                    orth_rotate.view(n, b_c, k, b_f).permute(0, 2, 1, 3),
                    quant_rot.view(n, k, b_f, b_f).transpose(-1, -2)
                ).permute(0, 2, 1, 3).reshape(n, b_c, b_c)
            else:
                M = torch.matmul(
                    quant_rot.view(n, b_c, k, b_f).permute(0, 2, 1, 3),
                    orth_rotate.view(n, k, b_f, b_f).transpose(-1, -2)
                ).permute(0, 2, 1, 3).reshape(n, b_c, b_c).transpose(-1, -2)

            x_reshaped = x.reshape(-1, n, b_c).permute(1, 0, 2)
            x_rotated_reshaped = torch.bmm(x_reshaped, M).permute(1, 0, 2)
        else:
            x_reshaped = x.reshape(-1, rank, self.block_size).permute(1, 0, 2)
            x_rotated_reshaped = torch.bmm(x_reshaped, orth_rotate).permute(1, 0, 2)

        x_rotated = x_rotated_reshaped.reshape(*orig_shape)

        return x_rotated.to(required_dtype)
