try:
    from modules.util.triton_mm_8bit import mm_8bit
except ImportError as e:
    print(str(e) + ", continueing without triton")
    import torch
    def mm_8bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert a.dtype == b.dtype, "Incompatible dtypes"
        assert a.dtype in [torch.int8, torch.float8_e4m3fn]
        if a.dtype == torch.int8:
            return torch._int_mm(a, b)
        else:
            one = torch.ones(1, device=a.device)
            return torch._scaled_mm(a, b.T.contiguous().T, scale_a=one, scale_b=one)
