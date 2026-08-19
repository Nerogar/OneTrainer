import torch

try:
    from modules.util.triton_mm_8bit import TILE_SIZE, mm_8bit
except ImportError as e:
    print(str(e) + ", continuing without triton")
    #a tile size only exists where a kernel implements it; LinearW8A8 rejects tilewise on None
    TILE_SIZE = None

    def mm_8bit(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype,
                scale_m: torch.Tensor | None = None, scale_n: torch.Tensor | None = None,
                scale_kn: torch.Tensor | None = None,
                lora_xd: torch.Tensor | None = None, lora_up: torch.Tensor | None = None) -> torch.Tensor:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert a.dtype == b.dtype, "Incompatible dtypes"
        assert a.dtype in [torch.int8, torch.float8_e4m3fn]
        assert scale_kn is None, "a tilewise weight scale needs the triton kernel"
        if a.dtype == torch.int8:
            #cublas rejects an n-major b (the backward's weight) outright, so make it k-major
            res = torch._int_mm(a, b.T.contiguous().T)
        else:
            one = torch.ones(1, device=a.device)
            res = torch._scaled_mm(a, b.T.contiguous().T, scale_a=one, scale_b=one, out_dtype=torch.float32)

        if scale_m is not None or scale_n is not None or lora_xd is not None:
            res = res.float()
        if scale_m is not None:
            res = res.mul_(scale_m.reshape(-1, 1))
        if scale_n is not None:
            res = res.mul_(scale_n.reshape(1, -1))
        if lora_xd is not None:
            res = res.add_(lora_xd @ lora_up)
        return res.to(out_dtype)
