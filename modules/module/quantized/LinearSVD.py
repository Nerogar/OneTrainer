import hashlib
from contextlib import suppress

from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin

import torch


class BaseLinearSVD(
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _get_tensor_hash(t: torch.Tensor) -> str:
        tensor = t.detach().cpu().contiguous()
        tensor_bytes = tensor.numpy().tobytes()
        hash_obj = hashlib.sha256(tensor_bytes)
        return hash_obj.hexdigest()


log_obj = None

def make_svd_linear(linear_class):
    class LinearSVD(
        linear_class,
        BaseLinearSVD,
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_buffer("svd_up", None)
            self.register_buffer("svd_down", None)

        def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            if self.svd_up is None:
                return super().unquantized_weight(dtype, device)
            else:
                return (self.svd_up @ self.svd_down).to(dtype) + super().unquantized_weight(dtype, device)

        def quantize(self, rank: int, svd_dtype: torch.dtype, device: torch.device | None = None, cache_dir: str | None = None, max_cache_rank: int = 128):
            if self.svd_up is not None:
                return

            W = super().unquantized_weight(torch.float32, device)
            orig_device = W.device
            if device is not None:
                W = W.to(device=device)

            U = None
            if cache_dir is not None:
                filename = cache_dir + "/" + _get_tensor_hash(W) + ".pt"
                with suppress(FileNotFoundError):
                    U, S, Vh = torch.load(filename, map_location=device)

            if U is None:
                #use full svd - torch.svd_lowrank is not reducing the quant range nearly as much:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                if cache_dir is not None:
                    torch.save((
                        U[:, :max_cache_rank].clone(),
                        S[:max_cache_rank].clone(),
                        Vh[:max_cache_rank, :].clone(),
                    ), filename)

            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]

            svd_down = Vh_r.clone().to(svd_dtype)
            svd_up = (U_r * S_r.unsqueeze(0)).to(svd_dtype)
            self.register_buffer("svd_up", svd_up)
            self.register_buffer("svd_down", svd_down)

            self.weight.data = (W - (svd_up @ svd_down)).to(dtype=self.weight.dtype, device=orig_device)
            super().quantize(device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert not self.svd_down.requires_grad and not self.svd_up.requires_grad
            return ((x @ self.svd_down.T) @ self.svd_up.T).to(x.dtype) + super().forward(x)

    return LinearSVD
