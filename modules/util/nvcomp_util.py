import torch

try:
    from modules.util.nvcomp_util_lowlevel import compress as compress
    from modules.util.nvcomp_util_lowlevel import decompress_into
    _AVAILABLE = True
    print("nvCOMP: low-level batched-ANS backend loaded")
except ImportError:
    try:
        from modules.util.nvcomp_util_highlevel import compress as compress
        from modules.util.nvcomp_util_highlevel import decompress_into
        _AVAILABLE = True
        print("nvCOMP: high-level Codec backend loaded (low-level unavailable)")
    except ImportError:
        _AVAILABLE = False
        print("nvCOMP: no backend available")


def available() -> bool:
    return _AVAILABLE


@torch.library.custom_op("nvcomp_ot::decompress", mutates_args=())
def _decompress(compressed: torch.Tensor, uncompressed_bytes: int, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    out = torch.empty(uncompressed_bytes, dtype=torch.uint8, device=compressed.device)
    decompress_into(compressed, out)
    return out.view(dtype).view(shape)


@_decompress.register_fake
def _(compressed: torch.Tensor, uncompressed_bytes: int, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    return compressed.new_empty(shape, dtype=dtype)


def decompress(compressed: torch.Tensor, uncompressed_bytes: int, dtype: torch.dtype, shape) -> torch.Tensor:
    return torch.ops.nvcomp_ot.decompress(compressed, uncompressed_bytes, dtype, list(shape))
