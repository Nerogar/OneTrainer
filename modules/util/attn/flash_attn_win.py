import sys

import torch

from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
from diffusers.utils import is_flash_attn_available


def all_tensors_on_device(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor):
    # Check that all tensors are on the GPU device
    return query.is_cuda and key.is_cuda and value.is_cuda


def check_for_attn_mask(attn_mask: torch.Tensor | None):
    # Flash Attention does not support non-null attn_mask
    return attn_mask is None


def check_tensor_shapes(query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor):
    # All fused kernels requires query, key and value to be 4 dimensional
    query_dim = query.dim()
    return query_dim == key.dim() and query_dim == value.dim() and query_dim == 4


def check_head_dim_size_flash(query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor):
    # All head_dim sizes must be equal and less than 256
    # (ROCm with AOTriton 0.9+ supports up to 512, but we keep 256 for simplicity)
    max_size = 256
    query_size_last = query.size(-1)
    key_size_last = key.size(-1)
    value_size_last = value.size(-1)

    same_head_dim_size = (query_size_last == key_size_last and
                          query_size_last == value_size_last)

    # Check that all head dims are equal, all <= max_size, and query_size_last > 0
    return same_head_dim_size and max_size >= query_size_last > 0


def check_flash_causal_non_square_seqlens(query: torch.Tensor,
                                          key: torch.Tensor,
                                          is_causal: bool):
    # FlashAttention does not support the is_causal flag when seqlen_q != seqlen_k
    # Flash attention layout is (N, S, H, E), so sequence length is at index -3
    return not (is_causal and not query.is_nested and not key.is_nested and query.shape[-3] != key.shape[-3])


def has_for_nested_inputs(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor):
    return (query.is_nested and query.layout == torch.strided) or \
        (key.is_nested and key.layout == torch.strided) or \
        (value.is_nested and value.layout == torch.strided)

def has_only_dense_inputs(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor):
    return (not query.is_nested) and (not key.is_nested) and (not value.is_nested)

def check_grouped_query_attention(query: torch.Tensor,
                                  key: torch.Tensor,
                                  value: torch.Tensor,
                                  requires_same_num_heads: bool = True) -> bool:
    """Check if grouped query attention configuration is valid."""
    # Flash attention layout is (N, S, H, E), so num_heads is at index -2
    q_num_heads = query.size(-2)
    k_num_heads = key.size(-2)
    v_num_heads = value.size(-2)
    same_kv_heads = k_num_heads == v_num_heads

    if requires_same_num_heads and not same_kv_heads:
        return False

    # Check if grouped query attention is supported and validate the number of heads
    return not (q_num_heads % k_num_heads != 0 or (not requires_same_num_heads and q_num_heads % v_num_heads != 0))


def check_batch_size_and_num_heads_dense(query: torch.Tensor,
                                         key: torch.Tensor,
                                         value: torch.Tensor,
                                         enable_gqa: bool = False,
                                         supports_gqa: bool = True,
                                         requires_same_num_heads: bool = True) -> bool:
    """Check batch size and num_heads compatibility for dense tensors.

    This is expected to be called after check_tensor_shapes ensuring that the
    size() calls won't error since the inputs are all 4 dimensional.
    """
    q_batch_size = query.size(0)
    k_batch_size = key.size(0)
    v_batch_size = value.size(0)

    same_batch_size = (q_batch_size == k_batch_size and q_batch_size == v_batch_size)

    # Flash attention layout is (N, S, H, E), so num_heads is at index -2
    q_num_heads = query.size(-2)
    k_num_heads = key.size(-2)
    v_num_heads = value.size(-2)

    same_num_heads = (q_num_heads == k_num_heads and q_num_heads == v_num_heads)

    # For dense inputs, both fused kernels require query, key and value to have the same batch_size
    if not same_batch_size:
        return False

    if enable_gqa and supports_gqa:
        return check_grouped_query_attention(query, key, value, requires_same_num_heads)

    # same num heads condition for non-gqa case
    return same_num_heads


def check_nonzero_sequence_lengths_dense(query: torch.Tensor,
                                         key: torch.Tensor,
                                         value: torch.Tensor) -> bool:
    """Check that sequence lengths are non-zero for dense tensors."""
    # In some cases people will pass in 0 sized tensors, this will
    # cause the fused path to error with unaligned mask
    # Flash attention layout is (N, S, H, E), so sequence length is at index -3
    zero_seq_len_q = query.size(-3) == 0
    zero_seq_len_k = key.size(-3) == 0
    return not (zero_seq_len_q or zero_seq_len_k)


def check_last_dim_stride_equals_1_dense(query: torch.Tensor,
                                         key: torch.Tensor,
                                         value: torch.Tensor,
                                         attn_mask: torch.Tensor | None = None,
                                         ignore_singleton_dim: bool = True) -> bool:
    """Check that the last dimension of inputs has stride 1.

    The stride checking for NestedTensors is done within the kernel
    and .contiguous will be called if needed.

    This function checks that the last dimension of the inputs to
    fused_attention have stride 1.
    """
    qkv_strides_equal_1 = (query.stride(-1) == 1 and
                          key.stride(-1) == 1 and
                          value.stride(-1) == 1)

    # If the head_dim is size 1 the stride won't matter, but we
    # check this condition before padding the head_dim to 1
    if ignore_singleton_dim:
        qkv_strides_equal_1 = qkv_strides_equal_1 or query.size(-1) == 1

    is_cpu = query.is_cpu
    mask_stride_equal_1 = attn_mask.stride(-1) == 1 if attn_mask is not None else True
    mask_stride_valid = True if is_cpu else mask_stride_equal_1

    return qkv_strides_equal_1 and mask_stride_valid


def check_dtypes_low_precision(query: torch.Tensor,
                       key: torch.Tensor,
                       value: torch.Tensor):
    return query.dtype == key.dtype and query.dtype == value.dtype and query.dtype in [torch.float16, torch.bfloat16]


def can_use_flash_attn(query: torch.Tensor,
                       key: torch.Tensor,
                       value: torch.Tensor,
                       attn_mask: torch.Tensor | None = None,
                       is_causal: bool = False,
                       enable_gqa: bool = False):
    # Define gate functions that determine if a flash kernel can be ran
    if not (all_tensors_on_device(query, key, value) and
            check_tensor_shapes(query, key, value) and
            check_for_attn_mask(attn_mask) and
            check_head_dim_size_flash(query, key, value) and
            check_flash_causal_non_square_seqlens(query, key, is_causal) and
            check_dtypes_low_precision(query, key, value)):
        return False

    # While PyTorch's Flash Attention implementation supports nested tensors,
    # we want to keep our use-case simple for now and avoid nested strided tensors as validations
    # require digging into tensor internals.
    if has_for_nested_inputs(query, key, value):
        return False

    if has_only_dense_inputs(query, key, value):
        if not (check_batch_size_and_num_heads_dense(query, key, value, enable_gqa, supports_gqa=True) and
                check_nonzero_sequence_lengths_dense(query, key, value) and
                check_last_dim_stride_equals_1_dense(query, key, value, attn_mask, ignore_singleton_dim=True)):
            return False

    return True


def supports_flash_attention_in_sdp():
    return torch.cuda.is_available() and torch.backends.cuda.is_flash_attention_available()


def register():
    if sys.platform == "win32" and is_flash_attn_available() and not supports_flash_attention_in_sdp():
        from flash_attn.flash_attn_interface import flash_attn_func

        def _native_flash_attention(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                scale: float | None = None,
                enable_gqa: bool = False
        ) -> torch.Tensor:
            # Determine if we can use flash attention
            if can_use_flash_attn(query, key, value, attn_mask, is_causal, enable_gqa):
                return flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=is_causal
                )
            else:
                query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
                out = torch.nn.functional.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
                out = out.permute(0, 2, 1, 3)
                return out

        # Register the dynamic flash attention backend in place of the native one
        _AttentionBackendRegistry.register(AttentionBackendName.NATIVE, [])(_native_flash_attention)
