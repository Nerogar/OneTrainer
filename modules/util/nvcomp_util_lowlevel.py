# Low-level batched-ANS nvCOMP decode backend (AI-generated, unmaintained). Calls
# nvcompBatchedANSDecompressAsync directly via ctypes to avoid the high-level Codec's per-decode
# Device->Pinned status copy, which splices a copy-engine op into the compute stream and stalls the
# consuming matmul (~40us/decode). Everything decode needs rides inside the blob as a device header,
# so no host<->device transfer touches the stream. The blob format is NOT interchangeable with the
# high-level Codec's, so a run must compress and decompress with the same backend.

import ctypes
import math

import torch

CHUNK = 65536        # 64KiB uncompressed chunk -- NVIDIA's recommended size for decode throughput
ALIGN = 16           # compressed-chunk alignment; ANS requires 8, 16 is a safe multiple


def _round_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class _Opts(ctypes.Structure):
    # nvcompBatchedANS{Compress,Decompress}Opts_t share this layout. All-zero selects rANS / the
    # default decompress backend, CHAR data type, and auto sub-chunk count.
    # {int algo_or_backend; nvcompType_t data_type; uint8 max_sub_chunk_count; char reserved[55]}
    _fields_ = [("algo_or_backend", ctypes.c_int), ("data_type", ctypes.c_int),
                ("max_sub_chunk_count", ctypes.c_uint8), ("reserved", ctypes.c_char * 55)]


def _load_lib():
    # raises ImportError if the package or the shared library is missing, so importing this backend
    # fails and nvcomp_util falls through to the next one
    import glob
    import importlib
    import os

    libnvcomp = importlib.import_module("nvidia.libnvcomp")
    matches = glob.glob(os.path.join(os.path.dirname(libnvcomp.__file__), "lib64", "libnvcomp.so*"))
    if not matches:
        raise ImportError("nvidia.libnvcomp is installed but libnvcomp.so was not found")
    lib = ctypes.CDLL(matches[0])
    P, Z = ctypes.c_void_p, ctypes.c_size_t
    sigs = {
        "nvcompBatchedANSCompressGetMaxOutputChunkSize": [Z, _Opts, ctypes.POINTER(Z)],
        "nvcompBatchedANSCompressGetTempSizeAsync":      [Z, Z, _Opts, ctypes.POINTER(Z), Z],
        "nvcompBatchedANSCompressAsync":                 [P, P, Z, Z, P, Z, P, P, _Opts, P, P],
        "nvcompBatchedANSDecompressGetTempSizeAsync":    [Z, Z, _Opts, ctypes.POINTER(Z), Z],
        "nvcompBatchedANSDecompressAsync":               [P, P, P, P, Z, P, Z, P, _Opts, P, P],
    }
    for name, argt in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argt
        fn.restype = ctypes.c_int
    return lib


_LIB = _load_lib()
_OPTS = _Opts(0, 0, 0, b"\x00" * 55)
_max_out_chunk = None       # worst-case compressed bytes for one CHUNK (constant); filled on first compress
_comp_temp = {}             # num_chunks -> compress temp bytes
_decomp_temp = {}           # num_chunks -> decompress temp bytes
_shape_cache = {}           # (uncompressed_bytes, device) -> (output offsets, output buffer sizes); identical for every weight of a size


def _shape_arrays(total: int, dev: torch.device):
    # output chunk offsets and per-chunk buffer sizes depend only on the uncompressed
    # size, so compute them once per distinct weight size and share read-only.
    key = (total, str(dev))
    v = _shape_cache.get(key)
    if v is None:
        num = math.ceil(total / CHUNK)
        o_off = torch.arange(num, device=dev, dtype=torch.int64) * CHUNK
        v = (o_off, (total - o_off).clamp(max=CHUNK))
        _shape_cache[key] = v
    return v


def _check(rc: int, what: str):
    if rc != 0:
        raise RuntimeError(f"nvcomp {what} failed with nvcompStatus {rc}")


def _stream():
    return torch.cuda.current_stream().cuda_stream


@torch.no_grad()
def compress(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    # weight: contiguous 1-byte (int8/fp8) cuda tensor. blob layout: [per-chunk compressed sizes:
    # num x int64][per-chunk offsets: num x int64][compressed chunks, each ALIGN-padded]. The
    # header rides with the blob so decode needs nothing else.
    assert weight.is_cuda and weight.is_contiguous()
    global _max_out_chunk
    dev = weight.device
    stream = _stream()
    flat = weight.view(torch.uint8).reshape(-1)
    total = flat.numel()
    num = math.ceil(total / CHUNK)

    # uncompressed chunk descriptors (device arrays the API reads on-stream)
    off = torch.arange(num, device=dev, dtype=torch.int64) * CHUNK
    u_bytes = (total - off).clamp_(max=CHUNK)                       # CHUNK for all but the last chunk
    u_ptrs = off + flat.data_ptr()

    if _max_out_chunk is None:
        m = ctypes.c_size_t()
        _check(_LIB.nvcompBatchedANSCompressGetMaxOutputChunkSize(CHUNK, _OPTS, ctypes.byref(m)), "GetMaxOutputChunkSize")
        _max_out_chunk = m.value
    stride = _round_up(_max_out_chunk, ALIGN)
    scratch = torch.empty(num * stride, dtype=torch.uint8, device=dev)
    c_ptrs = torch.arange(num, device=dev, dtype=torch.int64) * stride + scratch.data_ptr()
    c_bytes = torch.empty(num, dtype=torch.int64, device=dev)       # out: actual compressed size per chunk

    if num not in _comp_temp:
        t = ctypes.c_size_t()
        _check(_LIB.nvcompBatchedANSCompressGetTempSizeAsync(num, CHUNK, _OPTS, ctypes.byref(t), total), "CompressGetTempSize")
        _comp_temp[num] = t.value
    temp = torch.empty(max(_comp_temp[num], 1), dtype=torch.uint8, device=dev)
    statuses = torch.empty(num, dtype=torch.int32, device=dev)      # written but unused

    _check(_LIB.nvcompBatchedANSCompressAsync(
        u_ptrs.data_ptr(), u_bytes.data_ptr(), CHUNK, num, temp.data_ptr(), _comp_temp[num],
        c_ptrs.data_ptr(), c_bytes.data_ptr(), _OPTS, statuses.data_ptr(), stream), "CompressAsync")

    # packing needs the actual sizes on the host -- a sync here is fine, compression runs once at load
    torch.cuda.current_stream().synchronize()
    sizes = c_bytes.tolist()
    offsets, cur = [], 0
    for s in sizes:
        offsets.append(cur)
        cur += _round_up(s, ALIGN)                                  # per-chunk offset into the packed data, ALIGN-aligned
    header = num * 16                                               # [c_bytes: num x int64][c_off: num x int64], already 16-aligned
    c_off = torch.tensor(offsets, dtype=torch.int64, device=dev)
    blob = torch.empty(header + cur, dtype=torch.uint8, device=dev)
    blob[:num * 8].copy_(c_bytes.view(torch.uint8))                 # compressed size per chunk (int64 -> raw bytes)
    blob[num * 8:header].copy_(c_off.view(torch.uint8))             # offset of each chunk within the packed data
    data = blob[header:]
    for i, (o, s) in enumerate(zip(offsets, sizes, strict=True)):
        data[o:o + s].copy_(scratch[i * stride:i * stride + s])
    return blob, total


def decompress_into(compressed: torch.Tensor, out: torch.Tensor):
    # Sync-free decode on the current compute stream, writing into the caller's `out` buffer (its
    # size is the uncompressed length). The per-chunk statuses and actual sizes stay on the device
    # (never copied to the host), so there is no status readback and nothing serializes the following
    # matmul. Pointer/size arrays are derived on the device from the blob's header, so no
    # host<->device transfer touches the stream.
    dev = compressed.device
    stream = _stream()
    total = out.numel()
    num = math.ceil(total / CHUNK)
    hb = num * 8

    c_bytes = compressed[:hb].view(torch.int64)                    # compressed size per chunk (device slice, no copy)
    c_off = compressed[hb:2 * hb].view(torch.int64)                # chunk offsets into the packed data (precomputed at compress)
    data = compressed[2 * hb:]
    c_ptrs = c_off + data.data_ptr()                               # only the current base is added on-device (one int64 add)

    o_off, u_bytes = _shape_arrays(total, dev)
    o_ptrs = o_off + out.data_ptr()

    if num not in _decomp_temp:
        t = ctypes.c_size_t()
        _check(_LIB.nvcompBatchedANSDecompressGetTempSizeAsync(num, CHUNK, _OPTS, ctypes.byref(t), total), "DecompressGetTempSize")
        _decomp_temp[num] = t.value
    temp = torch.empty(max(_decomp_temp[num], 1), dtype=torch.uint8, device=dev)
    actual = torch.empty(num, dtype=torch.int64, device=dev)        # out actual sizes -- kept on device, never read
    statuses = torch.empty(num, dtype=torch.int32, device=dev)      # out per-chunk status -- kept on device, never read

    _check(_LIB.nvcompBatchedANSDecompressAsync(
        c_ptrs.data_ptr(), c_bytes.data_ptr(), u_bytes.data_ptr(), actual.data_ptr(), num,
        temp.data_ptr(), _decomp_temp[num], o_ptrs.data_ptr(), _OPTS, statuses.data_ptr(), stream), "DecompressAsync")
