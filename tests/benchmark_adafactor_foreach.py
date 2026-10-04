"""
CUDA benchmark: OneTrainer Adafactor step, per-parameter loop vs foreach.

Variants of a case run interleaved for several rounds, with the order rotated every round, and the median is
reported, because GPU clocks (especially on laptops) drift between back-to-back runs.

Run with:  python tests/benchmark_adafactor_foreach.py [steps] [chunks] [writeback]   (default: all sections)
"""
import gc
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.util import bf16_stochastic_rounding
from modules.util.optimizer import adafactor_extensions
from modules.util.optimizer.adafactor_extensions import patch_adafactor

import torch

from transformers import Adafactor


def lora_shapes(linears, rank):
    shapes = []
    for in_dim, out_dim in linears:
        shapes += [(rank, in_dim), (out_dim, rank)]
    return shapes


def sdxl_unet_linears():
    # SDXL UNet transformer blocks: 10 blocks at 640 and 60 at 1280 channels (attn1, attn2 with 2048 dim context,
    # GEGLU feed forward), plus proj_in/proj_out of the 11 transformer models
    linears = []
    for dim, blocks, models in [(640, 10, 5), (1280, 60, 6)]:
        block = [(dim, dim)] * 4 + [(dim, dim), (2048, dim), (2048, dim), (dim, dim)] + [(dim, 8 * dim), (4 * dim, dim)]
        linears += block * blocks + [(dim, dim)] * 2 * models
    return linears


SETUPS = {
    "sdxl unet LoRA r32": lora_shapes(sdxl_unet_linears(), 32),
    "small full finetune": [(1280, 1280)] * 40 + [(1280,)] * 80 + [(320, 320, 3, 3)] * 20
                           + [(2048, 2048)] * 4 + [(5120, 1280)] * 4,  # 4.2M and 6.6M element tensors
}

DTYPES = [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]
CHUNK_SIZES = [1 << 20, 1 << 22, 1 << 24, 1 << 26]

WARMUP = 10
ITERS = 50
MIN_ROUNDS = 3


def bench(shapes, dtype, foreach, stochastic_rounding, chunk_numel=adafactor_extensions.FOREACH_CHUNK_NUMEL):
    """returns (ms per step, peak allocated MiB above the params and state, peak reserved MiB)"""
    gc.collect()  # the patched optimizer is in a reference cycle (opt.step is bound to opt)
    torch.cuda.empty_cache()
    params = [torch.randn(s, device="cuda", dtype=dtype).mul_(0.01).requires_grad_(True) for s in shapes]
    for p in params:
        p.grad = torch.randn_like(p)
    opt = Adafactor(params, lr=1e-4, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, beta1=None,
                    weight_decay=0.0, scale_parameter=False, relative_step=False, warmup_init=False)
    patch_adafactor(opt, stochastic_rounding, foreach)
    bf16_stochastic_rounding.set_seed(0, torch.device("cuda"))
    default_chunk_numel = adafactor_extensions.FOREACH_CHUNK_NUMEL
    adafactor_extensions.FOREACH_CHUNK_NUMEL = chunk_numel
    try:
        for _ in range(WARMUP):
            opt.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()

        start = time.perf_counter()
        for _ in range(ITERS):
            opt.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / ITERS * 1000
    finally:
        adafactor_extensions.FOREACH_CHUNK_NUMEL = default_chunk_numel

    peak_extra = (torch.cuda.max_memory_allocated() - base_alloc) / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    del opt, params
    return ms, peak_extra, reserved


def interleaved(variants):
    """runs {name: fn} in turn for at least MIN_ROUNDS rounds, starting each round at the next variant so that every
    variant runs equally often in every position. Returns {name: (median ms, max peak extra, max peak reserved)}.
    """
    names = list(variants)
    rounds = len(names) * -(-MIN_ROUNDS // len(names))
    results = {name: [] for name in names}
    for r in range(rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            results[name].append(variants[name]())
    return {name: (statistics.median(r[0] for r in res), max(r[1] for r in res), max(r[2] for r in res))
            for name, res in results.items()}


def print_row(label, ms, extra, reserved):
    print(f"  {label:30s} {ms:8.2f} {extra:15.1f} {reserved:18.1f}")


def print_header(title):
    print(f"-- {title}")
    print(f"  {'':30s} {'step ms':>8s} {'peak extra MiB':>15s} {'peak reserved MiB':>18s}")


def bench_steps():
    for name, shapes in SETUPS.items():
        n = sum(torch.Size(s).numel() for s in shapes)
        for dtype, sr in DTYPES:
            print_header(f"{name}: {len(shapes)} tensors, {n / 1e6:.1f}M params, {str(dtype)[6:]}, SR={sr}")
            res = interleaved({
                "loop": lambda shapes=shapes, dtype=dtype, sr=sr: bench(shapes, dtype, False, sr),
                "foreach": lambda shapes=shapes, dtype=dtype, sr=sr: bench(shapes, dtype, True, sr),
            })
            for impl, r in res.items():
                print_row(impl, *r)
            print(f"  speedup {res['loop'][0] / res['foreach'][0]:.2f}x")


def bench_chunks():
    for name, shapes in SETUPS.items():
        for dtype, sr in [(torch.float32, False), (torch.bfloat16, True)]:
            print_header(f"foreach chunk limit, {name}, {str(dtype)[6:]}, SR={sr}")
            res = interleaved({
                f"2^{c.bit_length() - 1} ({c / 1e6:.1f}M)":
                    lambda c=c, shapes=shapes, dtype=dtype, sr=sr: bench(shapes, dtype, True, sr, c)
                for c in CHUNK_SIZES
            })
            for label, r in res.items():
                print_row(label, *r)


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / ITERS * 1000, 0.0, 0.0


def bench_writeback():
    # only the fp32 -> bf16 writeback of the updated params
    shapes = SETUPS["sdxl unet LoRA r32"]
    numels = [torch.Size(s).numel() for s in shapes]
    flat = torch.randn(sum(numels), device="cuda")
    sources = [v.view(s) for v, s in zip(flat.split(numels), shapes, strict=True)]
    targets = [torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
    bf16_stochastic_rounding.set_seed(0, torch.device("cuda"))

    def per_tensor_sr():
        for t, s in zip(targets, sources, strict=True):
            bf16_stochastic_rounding.copy_stochastic_(t, s)

    def per_tensor_copy():
        for t, s in zip(targets, sources, strict=True):
            t.copy_(s)

    print(f"-- writeback only, {len(shapes)} bf16 tensors")
    res = interleaved({
        "per-tensor copy_stochastic_": lambda: time_fn(per_tensor_sr),
        "copy_stochastic_flat_": lambda: time_fn(lambda: bf16_stochastic_rounding.copy_stochastic_flat_(targets, flat)),
        "per-tensor copy_ (no SR)": lambda: time_fn(per_tensor_copy),
        "_foreach_copy_ (no SR)": lambda: time_fn(lambda: torch._foreach_copy_(targets, sources)),
    })
    for label, r in res.items():
        print(f"  {label:30s} {r[0]:8.2f} ms")


if __name__ == "__main__":
    sections = {"steps": bench_steps, "chunks": bench_chunks, "writeback": bench_writeback}
    selected = sys.argv[1:] or list(sections)
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"median of >= {MIN_ROUNDS} rotated interleaved rounds of {ITERS} timed steps after {WARMUP} warmup, beta1=None")
    for name in selected:
        sections[name]()
