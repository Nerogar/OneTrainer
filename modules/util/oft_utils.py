import torch
import torch.nn as nn


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    This layer randomly replaces a fraction of the learned rotation blocks with identity matrices during training.
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

            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])
            mask = mask[torch.randperm(D)].view(D, 1, 1)
            eye_matrix = torch.eye(H, device=x.device).repeat(D, 1, 1)
            x = (1 - mask) * x + mask * eye_matrix
        return x


class OFTRotationModule(nn.Module):
    def __init__(
        self,
        r,
        n_elements,
        block_size,
        in_features,
        coft=False,
        eps=6e-5,
        block_share=False,
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
        self.coft = coft
        self.eps = eps
        self.block_share = block_share
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        # Create indices for upper triangle (excluding diagonal)
        self.rows, self.cols = torch.triu_indices(block_size, block_size, 1)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)


    def _pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

        matrix[:, self.rows, self.cols] = vec
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
                    for _i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power, alpha=2.0)
        else:
            id_mat = (
                torch.eye(Q_skew.shape[-1], device=Q_skew.device)
                .unsqueeze(0)
                .expand(b, Q_skew.shape[-1], Q_skew.shape[-1])
            )
            R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    def _project_batch(self, Q, eps=1e-5):
        oft_R = self._pytorch_skew_symmetric(Q, self.block_size)
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_R.shape[0]))
        origin_matrix = (
            torch.zeros((oft_R.size(1), oft_R.size(1)), device=oft_R.device, dtype=oft_R.dtype)
            .unsqueeze(0)
            .expand_as(oft_R)
        )
        diff = oft_R - origin_matrix
        norm_diff = torch.norm(oft_R - origin_matrix, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_R, origin_matrix + eps * (diff / norm_diff))

        return self._pytorch_skew_symmetric_inv(out, self.block_size)

    def forward(self, x):
        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape

        if self.coft:
            with torch.no_grad():
                self.weight.copy_(self._project_batch(self.weight, eps=self.eps))

        orth_rotate = self._cayley_batch(
            self.weight, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
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
