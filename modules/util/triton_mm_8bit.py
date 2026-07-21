#This is a 8bit matmul kernel adapted from the Triton tutorial here:
#https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

#It is not optimized and about 10% slower than torch._int_mm and torch._scaled_mm
#However, the torch functions don't work on row-major rhs matrices:
#_scaled_mm fails, _int_mm automatically converts to column-major
#
#Converting to column-major is slow, which is significant because the weights matrix
#of a Linear layer is always column-major during the backward pass.
#
#In these cases, this Triton kernel is *much* faster because it can access the
#row-major weight matrix directly, using strided memory access

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  32}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':  64}, num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K':  32}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K':  64}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K':  32}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K':  64}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 128}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K':  32}, num_stages=5,num_warps=2),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  32}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  64}, num_stages=3,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  64}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':  32}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K':  32}, num_stages=5,num_warps=2),


    ],
    key=[
        'QUANTIZED_M', #only tune roughly on M, because M is the transformer sequence length - can vary on data
        'N',
        'K',
        'stride_bk'    #use stride of b as key, to autotune again for a strided rhs matrix (backward pass)
    ],
)

@triton.jit
def _mm_kernel(
        a_ptr, b_ptr, c_ptr,
        scale_a_ptr, scale_b_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        QUANTIZED_M,
        FLOAT: tl.constexpr,
):

    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32 if FLOAT else tl.int32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # offs_am/offs_bn are taken mod M/N, so they are always in range; only the
        # K dimension needs masking here. Out-of-range rows/cols (from the wrap) are
        # discarded by c_mask at the store.
        k_mask = offs_k < K - k*BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32 if FLOAT else tl.int32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if scale_a_ptr is not None:
        # fused dequant epilogue: scale each row by scale_a[row] * scale_b and store
        # directly in c's dtype, avoiding a full-size int32/fp32 intermediate that
        # would otherwise be written, re-read, scaled and converted by a second kernel
        scale = tl.load(scale_a_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32) \
              * tl.load(scale_b_ptr).to(tl.float32)
        result = accumulator.to(tl.float32) * scale[:, None]
        tl.store(c_ptrs, result.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator, mask=c_mask)

def _mm_8bit_launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                    scale_a: torch.Tensor | None, scale_b: torch.Tensor | None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert a.dtype in [torch.int8, torch.float8_e4m3fn]

    M, K = a.shape
    K, N = b.shape

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']) , triton.cdiv(M, META['BLOCK_SIZE_M']), )
    _mm_kernel[grid](
        a, b, c,
        scale_a, scale_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        QUANTIZED_M = M // 64,
        FLOAT = (a.dtype == torch.float8_e4m3fn),
    )
    return c

def mm_8bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    FLOAT = (a.dtype == torch.float8_e4m3fn)
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32 if FLOAT else torch.int32)
    return _mm_8bit_launch(a, b, c, None, None)

def mm_8bit_scaled(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
                   out_dtype: torch.dtype) -> torch.Tensor:
    """(a @ b) * scale_a[:, None] * scale_b, stored directly as out_dtype.

    Fuses the row-wise dequantization into the matmul epilogue, avoiding a
    full-size int32/fp32 intermediate plus a separate scaling kernel.
    scale_a: per-row scales with a.shape[0] elements, scale_b: a single-element scale.
    """
    assert scale_a.numel() == a.shape[0] and scale_a.is_contiguous(), "scale_a must be one contiguous scale per row of A"
    assert scale_b.numel() == 1, "scale_b must be a single-element tensor"
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
    return _mm_8bit_launch(a, b, c, scale_a, scale_b)
