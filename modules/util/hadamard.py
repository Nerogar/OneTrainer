import torch
import torch.nn.functional as F
from torch import Tensor

# Normalized Sylvester-Hadamard matrices, cached per (size, device, dtype).
# H is symmetric and involutive (H @ H == I), so the same matrix rotates and un-rotates.
_hadamard_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}

def hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    assert n > 0 and (n & (n - 1)) == 0, f"Hadamard size must be a power of 2, got {n}"
    key = (n, device, dtype)
    h = _hadamard_cache.get(key)
    if h is not None:
        return h

    h = torch.ones((1, 1), device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                        torch.cat([h, -h], dim=1)], dim=0)
    h = (h / (n ** 0.5)).to(dtype)

    _hadamard_cache[key] = h
    return h

def pad_to_block(x: Tensor, block_size: int) -> Tensor:
    # Zero-pads the last dim up to a multiple of block_size. The padded columns are exact
    # zeros; callers must carry them through block_hadamard uncut and only truncate back
    # down after undoing the rotation (see block_hadamard) - never in between the two, or
    # the truncation drops values that real data was mixed into.
    pad = (-x.shape[-1]) % block_size
    if pad:
        x = F.pad(x, (0, pad))
    return x

def block_hadamard(x: Tensor, block_size: int, h: Tensor | None = None) -> Tensor:
    # Rotates the last dim in independent blocks of block_size (block-diagonal Hadamard:
    # O(dim * block_size) cost, not a dense O(dim^2) global rotation). x's last dim must
    # already be a multiple of block_size (pad with pad_to_block first). This function never
    # pads or truncates itself: H only inverts itself (H @ H == I) when applied twice to the
    # same padded width; truncating between the two applications is a lossy projection, not
    # an inverse, whenever the padding is non-zero.
    # Pass a precomputed h (any dtype) from callers inside a torch.compile'd + activation-
    # checkpointed region: the lazy hadamard_matrix() cache write is an in-graph side effect the
    # checkpoint HOP rejects (it would be replayed on backward recompute). Eager callers leave h
    # None and take the cached path.
    assert x.shape[-1] % block_size == 0, "call pad_to_block first"
    if h is None:
        h = hadamard_matrix(block_size, x.device, x.dtype)
    blocks = x.unflatten(-1, (-1, block_size))
    rotated = blocks @ h.to(blocks.dtype)
    return rotated.flatten(-2, -1)
