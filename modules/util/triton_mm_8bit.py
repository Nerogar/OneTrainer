#8bit matmul kernels adapted from the Triton tutorial here:
#https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

#All three mm entry points share the _mm_accumulate compute core (grouped launch order
#for L2 reuse, compile-time layout/divisibility specialization, EVEN_K loop selection)
#and differ only in the epilogue: _mm_kernel stores the raw int32/fp32 product,
#_scaled_mm_kernel folds a per-row dequant scale in and casts to the output dtype, and
#_scaled_lora_mm_kernel additionally fuses a low-rank update into the same tile.

from modules.util.tqdm_util import tqdm

import torch

import triton
import triton.language as tl


#Blackwell's block-scaled fp8 mma (mxf8f6f4) runs at the full 8-bit tensor core rate, the legacy
#fp8 mma only at half rate. Pre-Blackwell has no such instruction and triton emulates
#tl.dot_scaled with a bf16 mma, which is slower than plain tl.dot - so pick by compute capability.
#On ROCm the capability is the gfx arch number and RDNA4 reports 12, so require CUDA
def _prefer_mxfp8(device: torch.device) -> bool:
    return torch.version.cuda is not None and torch.cuda.get_device_capability(device)[0] >= 12


def announce_autotuning(kernel, name=None):
    prefix = f"autotuning {name} " if name else "autotuning "
    orig_check_disk_cache = kernel.check_disk_cache
    variants = 0
    def check_disk_cache(tuning_key, configs, bench_fn):
        def announced_bench():
            nonlocal variants
            variants += 1
            tqdm.show_status(f"{prefix}variant #{variants}...")
            bench_fn()
        return orig_check_disk_cache(tuning_key, configs, announced_bench)
    kernel.check_disk_cache = check_disk_cache


#tiled 8-bit transpose, used to rewrite the backward's B matrix to k-major before the mm.
#The int8 tensor-core op needs its B operand k-major and Ada has no 8-bit ldmatrix.trans, so an
#n-major B - a Linear weight in the backward pass - makes the mm emulate the transpose with byte
#shuffles in its inner loop, ~30% slower on every shape. This copy runs at memory bandwidth
#(~570GB/s) once per weight instead, and the mm then takes the fast k-major path.
_TRANSPOSE_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
]

@triton.autotune(configs=_TRANSPOSE_AUTOTUNE_CONFIGS, key=['M', 'N'], cache_results=True)
@triton.jit
def _transpose_kernel(
        src_ptr, dst_ptr,
        M, N,
        stride_sm, stride_dn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tile = tl.load(src_ptr + offs_m[:, None] * stride_sm + offs_n[None, :],
                   mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    tl.store(dst_ptr + offs_n[:, None] * stride_dn + offs_m[None, :],
             tl.trans(tile),
             mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))

def transpose_8bit(src: torch.Tensor) -> torch.Tensor:
    #returns src^T as a new contiguous tensor (any 1-byte dtype)
    assert src.stride(1) == 1, "src must be contiguous along axis 1"
    M, N = src.shape
    dst = torch.empty((N, M), device=src.device, dtype=src.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    _transpose_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), dst.stride(0),
    )
    return dst

announce_autotuning(_transpose_kernel, name="8-bit transpose")

#minimum M for the transpose-to-k-major rewrite in the mm wrappers below: the mm saves ~30%
#(~205 -> ~275 TOPS) but the copy costs 2*K*N bytes of traffic regardless of M, so the rewrite
#only pays above a token count. Breakeven is shape-dependent - measured on Ada (4070 Ti SUPER)
#at ~1550 for the widest layers and below 512 for the attention projections - and the wrappers
#see only M, so this takes the widest layer's breakeven and every shape wins above it
_TRANSPOSE_MIN_M = 1536


_AUTOTUNE_KEY = [
    #M is batch*sequence, so unlike N and K it is data-dependent and unbounded: it moves with
    #resolution, frame count, batch size and (on models that prune prompt padding) the longest
    #caption in the batch. bucketing it per doubling keeps the number of tuning keys logarithmic
    #in M instead of linear. proportional resolution is the right shape because the winning config
    #is decided by block count against SM count, which is linear in M - so equal ratios of M
    #matter equally at every scale, and a fixed stride is too fine at large M and too coarse at small
    'QUANTIZED_M',
    'N',
    'K',
    'stride_bk'    #use stride of b as key, to autotune again for a strided rhs matrix (backward pass)
]

#the LoRA epilogue keys on the rank tiling as well. up's row stride is deliberately not a key even
#though it varies: it is 1 in the forward (up is a transposed view of lora_up) and N in the backward
#(up is lora_down, stored row-major), so keying on it would double the key count on every model.
#the two layouts load differently but the slab is only BLOCK_R x BLOCK_N and is reused by every
#M block in the group, too little traffic next to A and B to move the tile choice
_LORA_AUTOTUNE_KEY = [*_AUTOTUNE_KEY, 'R_TILES', 'BLOCK_R']

#configs for the shared _mm_accumulate core: GROUP_SIZE_M is required by its grouped launch
#order. Shared memory per config is stages*(BLOCK_M+BLOCK_N)*BLOCK_K bytes and must stay
#under the ~99KB per-CTA limit (identical on sm86/sm89/sm120, i.e. consumer/workstation
#Ampere through Blackwell); oversized configs would be skipped by the autotuner.
#GROUP_SIZE_M=8 suits the large L2 of Ada/Blackwell; the 16-variants keep fewer B
#panels in flight per wave, for the small L2 (4-6MB) of Ampere - autotuning picks
_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K':  64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
]


#shared compute core of the 8-bit mm kernels: grouped launch order, divisibility hints and
#the EVEN_K-specialized main loop. Returns the raw accumulator plus the output tile offsets;
#each entry kernel below adds its own epilogue and stores.
@triton.jit
def _mm_accumulate(
        a_ptr, b_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        FLOAT: tl.constexpr, EVEN_K: tl.constexpr, MXFP8_MMA: tl.constexpr,
):

    #grouped launch order: consecutive pids walk down GROUP_SIZE_M M-blocks before advancing
    #to the next N-block, so the concurrent wave covers a rectangle of blocks and each B panel
    #is read from DRAM once and reused by GROUP_SIZE_M CTAs out of L2. A naive row-major grid
    #re-reads all of B per M-row instead, which makes the mm DRAM-bound
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32 if FLOAT else tl.int32)

    #the mma multiplies each group of 32 elements along K by one ue8m0 scale. ue8m0 is a bare
    #exponent with bias 127, so the value 127 means a scale of 1.0 and every element is left
    #unchanged - the result is the same as an unscaled fp8 matmul
    if MXFP8_MMA:
        a_scale = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K // 32), 127, dtype=tl.uint8)
        b_scale = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_K // 32), 127, dtype=tl.uint8)

    if EVEN_K:
        for _k in range(K // BLOCK_SIZE_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

            if MXFP8_MMA:
                accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc=accumulator)
            else:
                accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32 if FLOAT else tl.int32)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k*BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k*BLOCK_SIZE_K, other=0.0)

            if MXFP8_MMA:
                accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc=accumulator)
            else:
                accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32 if FLOAT else tl.int32)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    return accumulator, offs_cm, offs_cn


#shared epilogue tail for every mm kernel: c is allocated row-major in _prepare_mm, so only
#stride_cm is passed and the n stride is 1
@triton.jit
def _store_c(c_ptr, value, offs_cm, offs_cn, M, N, stride_cm):
    tl.assume(stride_cm > 0)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, value, mask=c_mask)


def _prepare_mm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert a.dtype in [torch.int8, torch.float8_e4m3fn]

    FLOAT = (a.dtype == torch.float8_e4m3fn)

    M, K = a.shape
    K, N = b.shape

    #the kernel handles exactly two B layouts: k-major (forward, weight.T) and n-major (backward, weight)
    B_K_MAJOR = (b.stride(0) == 1)
    assert B_K_MAJOR or b.stride(1) == 1, "Matrix B must be contiguous along one axis"
    #n-major B runs the mm ~30% slower; for large M a transpose copy pays for itself (see transpose_8bit)
    if not B_K_MAJOR and M >= _TRANSPOSE_MIN_M:
        b = transpose_8bit(b).t()
        B_K_MAJOR = True
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    return b, c, M, N, K, FLOAT


def _prepare_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype):
    #_prepare_mm plus the per-row scale reshape (the scaled kernels fold it into the epilogue)
    b, c, M, N, K, FLOAT = _prepare_mm(a, b, out_dtype)
    scale = scale.reshape(-1).to(torch.float32).contiguous()
    assert scale.shape[0] == M, "scale must have one entry per row of a"
    return b, c, scale, M, N, K, FLOAT


def _prepare_rowcol_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, out_dtype: torch.dtype):
    b, c, scale, M, N, K, FLOAT = _prepare_scaled_mm(a, b, scale, out_dtype)
    scale_n = scale_n.reshape(-1).to(torch.float32).contiguous()
    assert scale_n.shape[0] == N, "scale_n must have one entry per column of b"
    return b, c, scale, scale_n, M, N, K, FLOAT


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY, cache_results=True)
@triton.jit
def _mm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        QUANTIZED_M, FLOAT: tl.constexpr, EVEN_K: tl.constexpr, MXFP8_MMA: tl.constexpr,
):
    accumulator, offs_cm, offs_cn = _mm_accumulate(
        a_ptr, b_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        FLOAT, EVEN_K, MXFP8_MMA,
    )

    _store_c(c_ptr, accumulator, offs_cm, offs_cn, M, N, stride_cm)

announce_autotuning(_mm_kernel, name="8-bit matmul")

#Opaque custom ops work around pytorch#164124: torch.compile otherwise absorbs the traced
#@triton.autotune kernels and freezes the config benchmarked for the first shape. Kept opaque,
#these bodies run eagerly, so Triton's autotuner selects per key and the JIT sees real sizes
@torch.library.custom_op("ot_quant::mm_8bit", mutates_args=())
def mm_8bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out_dtype = torch.float32 if a.dtype == torch.float8_e4m3fn else torch.int32
    b, c, M, N, K, FLOAT = _prepare_mm(a, b, out_dtype)

    #1D grid: the kernel derives pid_m/pid_n itself in grouped order for L2 reuse
    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    _mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
        QUANTIZED_M = M.bit_length(), FLOAT = FLOAT, EVEN_K = (K % 128 == 0), MXFP8_MMA = FLOAT and _prefer_mxfp8(a.device),
    )
    return c

@mm_8bit.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out_dtype = torch.float32 if a.dtype == torch.float8_e4m3fn else torch.int32
    return a.new_empty((a.shape[0], b.shape[1]), dtype=out_dtype)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY, cache_results=True)
@triton.jit
def _scaled_mm_kernel(
        a_ptr, b_ptr, c_ptr, scale_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        QUANTIZED_M, FLOAT: tl.constexpr, EVEN_K: tl.constexpr, MXFP8_MMA: tl.constexpr,
):
    accumulator, offs_cm, offs_cn = _mm_accumulate(
        a_ptr, b_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        FLOAT, EVEN_K, MXFP8_MMA,
    )

    #per-row scale on axis 0 (M), fold into the epilogue and cast to the output (compute) dtype directly
    scale = tl.load(scale_ptr + offs_cm, mask=offs_cm < M, other=0.0)
    result = accumulator.to(tl.float32) * scale[:, None]
    result = result.to(c_ptr.dtype.element_ty)

    _store_c(c_ptr, result, offs_cm, offs_cn, M, N, stride_cm)

announce_autotuning(_scaled_mm_kernel, name="8-bit scaled matmul")

@torch.library.custom_op("ot_quant::scaled_mm_8bit", mutates_args=())
def scaled_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    b, c, scale, M, N, K, FLOAT = _prepare_scaled_mm(a, b, scale, out_dtype)

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    _scaled_mm_kernel[grid](
        a, b, c, scale,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
        QUANTIZED_M = M.bit_length(), FLOAT = FLOAT, EVEN_K = (K % 128 == 0), MXFP8_MMA = FLOAT and _prefer_mxfp8(a.device),
    )
    return c

@scaled_mm_8bit.register_fake
def _(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]), dtype=out_dtype)


#_scaled_mm_kernel's epilogue with a second dequant scale on axis 1 (N), for callers whose weight
#is quantized axiswise rather than tensorwise (LinearGGUFA8), where the weight scale is one entry
#per output column and so cannot be folded into the per-row scale
@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY, cache_results=True)
@triton.jit
def _rowcol_scaled_mm_kernel(
        a_ptr, b_ptr, c_ptr, scale_ptr, scale_n_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        QUANTIZED_M, FLOAT: tl.constexpr, EVEN_K: tl.constexpr, MXFP8_MMA: tl.constexpr,
):
    accumulator, offs_cm, offs_cn = _mm_accumulate(
        a_ptr, b_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        FLOAT, EVEN_K, MXFP8_MMA,
    )

    scale = tl.load(scale_ptr + offs_cm, mask=offs_cm < M, other=0.0)
    scale_n = tl.load(scale_n_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    result = accumulator.to(tl.float32) * scale[:, None] * scale_n[None, :]
    result = result.to(c_ptr.dtype.element_ty)

    _store_c(c_ptr, result, offs_cm, offs_cn, M, N, stride_cm)

announce_autotuning(_rowcol_scaled_mm_kernel, name="8-bit row/column scaled matmul")

@torch.library.custom_op("ot_quant::rowcol_scaled_mm_8bit", mutates_args=())
def rowcol_scaled_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    b, c, scale, scale_n, M, N, K, FLOAT = _prepare_rowcol_scaled_mm(a, b, scale, scale_n, out_dtype)

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    _rowcol_scaled_mm_kernel[grid](
        a, b, c, scale, scale_n,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
        QUANTIZED_M = M.bit_length(), FLOAT = FLOAT, EVEN_K = (K % 128 == 0), MXFP8_MMA = FLOAT and _prefer_mxfp8(a.device),
    )
    return c

@rowcol_scaled_mm_8bit.register_fake
def _(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]), dtype=out_dtype)


#rank-tile width is min(next_pow2(R), cap): the cap bounds the staged tile, and with it the
#epilogue's shared memory, so occupancy stays rank-independent. 64 keeps
#2*BLOCK_R*(BLOCK_M+BLOCK_N) inside the main mm's pool on Ada; 128 overflowed it (backward OOM)
_LORA_BLOCK_R_CAP = 64


def _prepare_lora(a: torch.Tensor, b: torch.Tensor, xd: torch.Tensor, up: torch.Tensor):
    R = xd.shape[1]
    assert xd.shape[0] == a.shape[0] and up.shape[0] == R and up.shape[1] == b.shape[1], "Incompatible low-rank dimensions"
    assert xd.stride(1) == 1, "xd must be contiguous along the rank axis"
    assert xd.dtype == up.dtype
    return R, min(max(16, triton.next_power_of_2(R)), _LORA_BLOCK_R_CAP)

@triton.jit
def _add_lora(
        result, xd_ptr, up_ptr, offs_cm, offs_cn, M, N, R,
        stride_xdm, stride_upr, stride_upn,
        BLOCK_R: tl.constexpr, R_TILES: tl.constexpr,
):
    for r0 in tl.static_range(R_TILES):
        offs_r = r0 * BLOCK_R + tl.arange(0, BLOCK_R)
        xd_ptrs = xd_ptr + offs_cm[:, None] * stride_xdm + offs_r[None, :]
        up_ptrs = up_ptr + offs_r[:, None] * stride_upr + offs_cn[None, :] * stride_upn
        xd = tl.load(xd_ptrs, mask=(offs_cm[:, None] < M) & (offs_r[None, :] < R), other=0.0)
        up = tl.load(up_ptrs, mask=(offs_r[:, None] < R) & (offs_cn[None, :] < N), other=0.0)
        result += tl.dot(xd, up)
    return result


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_LORA_AUTOTUNE_KEY, cache_results=True)
@triton.jit
def _scaled_lora_mm_kernel(
        a_ptr, b_ptr, c_ptr, scale_ptr, xd_ptr, up_ptr,
        M, N, K, R,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_xdm, stride_upr, stride_upn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        QUANTIZED_M, FLOAT: tl.constexpr, EVEN_K: tl.constexpr, MXFP8_MMA: tl.constexpr,
        BLOCK_R: tl.constexpr, R_TILES: tl.constexpr,
):
    accumulator, offs_cm, offs_cn = _mm_accumulate(
        a_ptr, b_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        FLOAT, EVEN_K, MXFP8_MMA,
    )

    scale = tl.load(scale_ptr + offs_cm, mask=offs_cm < M, other=0.0)
    result = accumulator.to(tl.float32) * scale[:, None]

    result = _add_lora(result, xd_ptr, up_ptr, offs_cm, offs_cn, M, N, R,
                       stride_xdm, stride_upr, stride_upn, BLOCK_R, R_TILES)
    result = result.to(c_ptr.dtype.element_ty)

    _store_c(c_ptr, result, offs_cm, offs_cn, M, N, stride_cm)

announce_autotuning(_scaled_lora_mm_kernel, name="8-bit scaled LoRA matmul")

@torch.library.custom_op("ot_quant::scaled_lora_mm_8bit", mutates_args=())
def scaled_lora_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, xd: torch.Tensor, up: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    #returns (a @ b) * scale + xd @ up in out_dtype (rank-tiled epilogue). xd is (M, r), up is
    #(r, N) (the lora_up weight transposed, alpha folded in)
    R, block_r = _prepare_lora(a, b, xd, up)
    b, c, scale, M, N, K, FLOAT = _prepare_scaled_mm(a, b, scale, out_dtype)

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    _scaled_lora_mm_kernel[grid](
        a, b, c, scale, xd, up,
        M, N, K, R,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), xd.stride(0), up.stride(0), up.stride(1),
        QUANTIZED_M = M.bit_length(), FLOAT = FLOAT, EVEN_K = (K % 128 == 0), MXFP8_MMA = FLOAT and _prefer_mxfp8(a.device),
        BLOCK_R = block_r, R_TILES = triton.cdiv(R, block_r),
    )
    return c

@scaled_lora_mm_8bit.register_fake
def _(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, xd: torch.Tensor, up: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]), dtype=out_dtype)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_LORA_AUTOTUNE_KEY, cache_results=True)
@triton.jit
def _rowcol_scaled_lora_mm_kernel(
        a_ptr, b_ptr, c_ptr, scale_ptr, scale_n_ptr, xd_ptr, up_ptr,
        M, N, K, R,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_xdm, stride_upr, stride_upn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        QUANTIZED_M, FLOAT: tl.constexpr, EVEN_K: tl.constexpr, MXFP8_MMA: tl.constexpr,
        BLOCK_R: tl.constexpr, R_TILES: tl.constexpr,
):
    accumulator, offs_cm, offs_cn = _mm_accumulate(
        a_ptr, b_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        FLOAT, EVEN_K, MXFP8_MMA,
    )

    scale = tl.load(scale_ptr + offs_cm, mask=offs_cm < M, other=0.0)
    scale_n = tl.load(scale_n_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    result = accumulator.to(tl.float32) * scale[:, None] * scale_n[None, :]

    result = _add_lora(result, xd_ptr, up_ptr, offs_cm, offs_cn, M, N, R,
                       stride_xdm, stride_upr, stride_upn, BLOCK_R, R_TILES)
    result = result.to(c_ptr.dtype.element_ty)

    _store_c(c_ptr, result, offs_cm, offs_cn, M, N, stride_cm)

announce_autotuning(_rowcol_scaled_lora_mm_kernel, name="8-bit row/column scaled LoRA matmul")

@torch.library.custom_op("ot_quant::rowcol_scaled_lora_mm_8bit", mutates_args=())
def rowcol_scaled_lora_mm_8bit(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, xd: torch.Tensor, up: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    R, block_r = _prepare_lora(a, b, xd, up)
    b, c, scale, scale_n, M, N, K, FLOAT = _prepare_rowcol_scaled_mm(a, b, scale, scale_n, out_dtype)

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    _rowcol_scaled_lora_mm_kernel[grid](
        a, b, c, scale, scale_n, xd, up,
        M, N, K, R,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), xd.stride(0), up.stride(0), up.stride(1),
        QUANTIZED_M = M.bit_length(), FLOAT = FLOAT, EVEN_K = (K % 128 == 0), MXFP8_MMA = FLOAT and _prefer_mxfp8(a.device),
        BLOCK_R = block_r, R_TILES = triton.cdiv(R, block_r),
    )
    return c

@rowcol_scaled_lora_mm_8bit.register_fake
def _(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, scale_n: torch.Tensor, xd: torch.Tensor, up: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]), dtype=out_dtype)
