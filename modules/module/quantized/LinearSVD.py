from abc import abstractmethod
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

    @abstractmethod
    def forward_with_lora(self, x: torch.Tensor, lora_down: torch.nn.Linear, lora_up: torch.nn.Linear, dropout: torch.nn.Dropout, alpha: float) -> torch.Tensor:
        pass



def _get_tensor_hash(t: torch.Tensor) -> str:
    t = t.flatten().to(torch.float32)
    vals = torch.stack([
        torch.sum(t),
        torch.sum(t**2),
        torch.sum(torch.sin(t)),
        torch.sum(torch.cos(t))
    ])
    return vals.cpu().numpy().tobytes().hex()

def make_svd_linear(linear_class):
    class LinearSVD(
        linear_class,
        BaseLinearSVD,
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__svd_is_quantized = False

            #use parameters instead of buffer to allow offloading:
            self.svd_up = torch.nn.Parameter(torch.empty(()), requires_grad=False)
            self.svd_down = torch.nn.Parameter(torch.empty(()), requires_grad=False)

        def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            if self.__svd_is_quantized:
                return (self.svd_up @ self.svd_down).to(dtype) + super().unquantized_weight(dtype, device)
            else:
                return super().unquantized_weight(dtype, device)

        @torch.no_grad()
        def quantize(self, rank: int, svd_dtype: torch.dtype, device: torch.device | None = None, cache_dir: str | None = None, max_cache_rank: int = 128, **kwargs):
            if self.__svd_is_quantized:
                return
            self.__svd_is_quantized = True

            W = super().unquantized_weight(torch.float32, device)
            orig_device = W.device
            if device is not None:
                W = W.to(device=device)

            U = None
            if cache_dir is not None:
                filename = cache_dir + "/" + _get_tensor_hash(W) + ".pt"
                with suppress(FileNotFoundError):
                    U, S, Vh = torch.load(filename, map_location=W.device)

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

            svd_down = Vh_r.clone().contiguous().to(svd_dtype)
            svd_up = (U_r * S_r.unsqueeze(0)).clone().contiguous().to(svd_dtype)
            weight = (W - (svd_up @ svd_down)).to(dtype=self.weight.dtype)

            if device is not None:
                weight = weight.to(device=orig_device)
                svd_up = svd_up.to(device=orig_device)
                svd_down = svd_down.to(device=orig_device)

            self.requires_grad_(False)
            self.svd_up = torch.nn.Parameter(svd_up, requires_grad=False)
            self.svd_down = torch.nn.Parameter(svd_down, requires_grad=False)
            self.weight.data = weight
            super().quantize(device=device, **kwargs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert self.__svd_is_quantized
            assert not self.svd_down.requires_grad and not self.svd_up.requires_grad
            x_down = torch.nn.functional.linear(x, self.svd_down)
            x_up = torch.nn.functional.linear(x_down, self.svd_up)
            return x_up + super().forward(x)

        def forward_with_lora(self, x: torch.Tensor, lora_down: torch.nn.Linear, lora_up: torch.nn.Linear, dropout: torch.nn.Dropout, alpha: float) -> torch.Tensor:
            assert self.__svd_is_quantized
            assert not self.svd_down.requires_grad and not self.svd_up.requires_grad
            assert lora_down.bias is None and lora_up.bias is None

            lora_rank = lora_down.weight.shape[0]
            down_merged = torch.cat([lora_down.weight, self.svd_down], dim=0)
            x_down = torch.nn.functional.linear(x, down_merged)
            if dropout.p > 0.0 and self.training:
                x_down[..., :lora_rank] = dropout(x_down[..., :lora_rank])

            lora_up_scaled = lora_up.weight * (alpha / lora_rank)
            up_merged = torch.cat([lora_up_scaled, self.svd_up], dim=1)
            x_up = torch.nn.functional.linear(x_down, up_merged)
            return x_up + super().forward(x)

    return LinearSVD
