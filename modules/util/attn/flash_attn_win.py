"""
Flash Attention support for Windows platforms.

This module provides a dynamic fallback mechanism for Flash Attention on Windows,
patching PyTorch's scaled_dot_product_attention when native support is unavailable.
"""

import sys
from functools import cache

import torch
import torch.nn.functional as F

from diffusers.utils import is_flash_attn_available

ALLOWED_TYPES = {torch.float16, torch.bfloat16}


@cache
def is_supported_hardware(device):
    """
    Check if the given device supports Flash Attention based on its compute capability.
    """
    # FlashAttention-2 only supports Ampere (sm_80) and newer GPUs
    properties = torch.cuda.get_device_properties(device)
    return properties.major >= 8


def can_use_flash_attn(query: torch.Tensor,
                       key: torch.Tensor,
                       value: torch.Tensor,
                       attn_mask: torch.Tensor | None = None,
                       is_causal: bool = False,
                       enable_gqa: bool = False):
    """
    Check if Flash Attention can be used for the given tensors.

    Args:
        query: Query tensor of shape (B, H, L, D)
        key: Key tensor of shape (B, H, L, D)
        value: Value tensor of shape (B, H, L, D)
        attn_mask: Optional attention mask (not supported by flash_attn)
        is_causal: Whether to use causal attention
        enable_gqa: Whether grouped query attention is enabled

    Returns:
        bool: True if Flash Attention can be used, False otherwise
    """
    # Fast grouped early rejects (most common failures first).
    dt = query.dtype
    if (
        attn_mask is not None  # Explicit attention masks are not supported by flash_attn
        or dt not in ALLOWED_TYPES  # flash_attn requires fp16/bf16
        or dt != key.dtype or dt != value.dtype  # Q/K/V must have identical dtypes
        or not (query.is_cuda and key.is_cuda and value.is_cuda)  # flash_attn is CUDA-only
        or query.dim() != 4 or key.dim() != 4 or value.dim() != 4  # Expect rank-4 (B, H, L, D)
        or query.is_nested or key.is_nested or value.is_nested  # Nested tensors unsupported, keep our use-case simple
    ):
        return False

    # Hardware capability check
    if not is_supported_hardware(query.device):
        return False

    # Unpack shapes once.
    (bq, q_heads, q_len, head_dim) = query.shape
    (bk, k_heads, k_len, k_head_dim) = key.shape
    (bv, v_heads, v_len, v_head_dim) = value.shape

    # Batch & head dim validation.
    if bq != bk or bq != bv:
        return False
    if not (0 < head_dim <= 256 and head_dim == k_head_dim == v_head_dim):
        return False

    # Sequence length checks.
    if q_len == 0 or k_len == 0:
        return False
    if is_causal and q_len != k_len:  # causal path requires equal seq lengths
        return False

    # Head count validation (GQA aware).
    if enable_gqa:
        if k_heads != v_heads or k_heads == 0 or (q_heads % k_heads) != 0:
            return False
    else:
        if not (q_heads == k_heads == v_heads):
            return False

    # Stride check (only if dim > 1).
    if head_dim != 1:
        qs = query.stride(-1)
        ks = key.stride(-1)
        vs = value.stride(-1)
        if qs != 1 or ks != 1 or vs != 1:  # All last-dim strides must be 1 (contiguous)
            return False

    return True


def supports_flash_attention_in_sdp():
    """Check if Flash Attention is natively supported in scaled_dot_product."""
    return torch.cuda.is_available() and torch.backends.cuda.is_flash_attention_available()


def register():
    """
    Register Flash Attention fallback on Windows when native support is unavailable.

    Patches F.scaled_dot_product_attention to use flash_attn_func when conditions allow,
    falling back to the original implementation otherwise.
    """
    if sys.platform == "win32" and is_flash_attn_available() and not supports_flash_attention_in_sdp():
        try:
            from flash_attn.flash_attn_interface import flash_attn_func
        except Exception:
            return

        _scaled_dot_product_attention = F.scaled_dot_product_attention

        def _flash_dynamic_scaled_dot_product_attention(query: torch.Tensor,
                                                        key: torch.Tensor,
                                                        value: torch.Tensor,
                                                        attn_mask: torch.Tensor | None = None,
                                                        dropout_p: float = 0.0,
                                                        is_causal: bool = False,
                                                        scale: float | None = None,
                                                        enable_gqa: bool = False):
            if can_use_flash_attn(query, key, value, attn_mask, is_causal, enable_gqa):
                # transpose(1,2) is equivalent to permute(0,2,1,3) for (B,H,L,D) -> (B,L,H,D)
                q = query.transpose(1, 2)
                k = key.transpose(1, 2)
                v = value.transpose(1, 2)
                out = flash_attn_func(
                    q=q, k=k, v=v,
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=is_causal
                )
                return out.transpose(1, 2)

            # Fallback
            return _scaled_dot_product_attention(
                query=query, key=key, value=value,
                attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

        F.scaled_dot_product_attention = _flash_dynamic_scaled_dot_product_attention
