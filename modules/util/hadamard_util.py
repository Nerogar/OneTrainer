import math

import torch
from torch import Tensor


def block_hadamard(x: Tensor, block_size: int = 128) -> Tensor:
    """Normalized, self-inverse Walsh-Hadamard transform along the last dimension.

    Full blocks are followed by power-of-two tail blocks, so arbitrary widths keep
    their shape and orthogonality. Only shape-dependent Python work is performed;
    no matrix construction, device caches, or RNG enters a compiled forward.
    """
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    if block_size == 1 or x.shape[-1] == 0:
        return x

    parts = []
    offset = 0
    width = x.shape[-1]
    size = block_size
    while size:
        length = ((width - offset) // size) * size
        if length:
            shape = (*x.shape[:-1], length // size, size)
            part = x[..., offset:offset + length].float().reshape(shape)
            stride = 1
            while stride < size:
                pairs = part.reshape(*shape[:-1], size // (2 * stride), 2, stride)
                a, b = pairs.unbind(dim=-2)
                part = torch.stack((a + b, a - b), dim=-2).reshape(shape)
                stride *= 2
            parts.append((part * (1.0 / math.sqrt(size))).reshape(*x.shape[:-1], length))
            offset += length
        size //= 2
    return torch.cat(parts, dim=-1).to(x.dtype)
