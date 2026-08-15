import torch

try:
    from modules.util.triton_mm_8bit import (
        mm_8bit,
        rowcol_scaled_lora_mm_8bit,
        rowcol_scaled_mm_8bit,
        scaled_lora_mm_8bit,
        scaled_mm_8bit,
    )
except ImportError as e:
    print(str(e) + ", continuing without triton")
    def mm_8bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert a.dtype == b.dtype, "Incompatible dtypes"
        assert a.dtype in [torch.int8, torch.float8_e4m3fn]
        if a.dtype == torch.int8:
            return torch._int_mm(a, b)
        else:
            one = torch.ones(1, device=a.device)
            #out_dtype defaults to a's dtype, which would round the accumulator back to fp8
            return torch._scaled_mm(a, b.T.contiguous().T, scale_a=one, scale_b=one, out_dtype=torch.float32)

    def scaled_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return mm_8bit(a, b).float().mul_(scale.reshape(-1, 1)).to(out_dtype)

    def scaled_lora_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, xd: torch.Tensor, up: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return mm_8bit(a, b).float().mul_(scale.reshape(-1, 1)).add_(xd @ up).to(out_dtype)

    def rowcol_scaled_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return mm_8bit(a, b).float().mul_(scale.reshape(-1, 1)).mul_(scale_n.reshape(1, -1)).to(out_dtype)

    def rowcol_scaled_lora_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, xd: torch.Tensor, up: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return mm_8bit(a, b).float().mul_(scale.reshape(-1, 1)).mul_(scale_n.reshape(1, -1)).add_(xd @ up).to(out_dtype)
