"""Dual-GPU model-parallel for FLUX.2 LoRA training in OneTrainer.

Activates when ``FLUX2_DUAL_GPU=true``. Splits the underlying diffusers
``Flux2Transformer2DModel`` across two CUDA devices at the
``single_transformer_blocks`` midpoint. Enables FLUX.2 LoRA training on
pairs of 24+ GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090) —
on a single 24 GB card the transformer can't fit alongside activations
even with WDDM paging.

Companion to the validated ai-toolkit patch
(https://github.com/genno-whittlery/flux2-dual-gpu-lora) and parallel
ports for musubi-tuner and HuggingFace diffusers. OneTrainer's port is
small because:

- ``Flux2Model.transformer_to(device)`` is a single canonical hook.
- The transformer is the diffusers ``Flux2Transformer2DModel``, so the
  same forward pre-hook strategy used in the diffusers helper applies.
- ``LoRAModule.orig_module`` is preserved through ``hook_to_module``,
  so per-LoRA device routing reads ``orig_module.weight.device``
  directly — no snapshot needed.

Env vars:
    FLUX2_DUAL_GPU=true                    enable dual-GPU path
    FLUX2_DUAL_GPU_SPLIT_AT=24             override split index
                                           (default: num_single // 2)
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def is_dual_gpu_enabled() -> bool:
    """True iff ``FLUX2_DUAL_GPU=true`` in the environment."""
    return os.getenv("FLUX2_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_single_blocks: int) -> int:
    """Single-blocks split index. Override via ``FLUX2_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("FLUX2_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_single_blocks // 2


def assert_no_layer_offload(transformer_offload_conductor: Any) -> None:
    """Layer-offload and dual-GPU are mutually exclusive.

    OneTrainer's ``LayerOffloadConductor`` moves layers RAM↔single-GPU;
    the dual-GPU path splits layers statically across two GPUs. The two
    strategies make different assumptions about layer residence and
    don't compose.
    """
    if not is_dual_gpu_enabled():
        return
    if transformer_offload_conductor is None:
        return
    if transformer_offload_conductor.layer_offload_activated():
        raise RuntimeError(
            "FLUX2_DUAL_GPU=true is incompatible with layer offloading. "
            "The dual-GPU split already distributes the transformer "
            "across two GPUs; layer offloading pages layers to system "
            "RAM, which fights the split. Either disable layer "
            "offloading or unset FLUX2_DUAL_GPU."
        )


def distribute_flux2_transformer(transformer: nn.Module) -> None:
    """Distribute the diffusers Flux2Transformer2DModel across cuda:0 / cuda:1.

    Layout:
        cuda:0  — x_embedder, context_embedder, time_guidance_embed,
                  pos_embed, *_modulation*, all transformer_blocks,
                  single_transformer_blocks[0:split_at], norm_out,
                  proj_out
        cuda:1  — single_transformer_blocks[split_at:]

    Pre-hooks bridge the activation across the boundary in both
    directions. Output layers (norm_out, proj_out) stay on cuda:0 so
    the returned tensor matches the caller's expected device.
    """
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"FLUX2_DUAL_GPU=true requires ≥2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    num_single = len(transformer.single_transformer_blocks)
    split_at = get_split_at(num_single)
    if not 0 < split_at < num_single:
        raise RuntimeError(
            f"FLUX2_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(transformer has {num_single} single blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    transformer.x_embedder.to(cuda0)
    transformer.context_embedder.to(cuda0)
    transformer.time_guidance_embed.to(cuda0)
    transformer.pos_embed.to(cuda0)
    transformer.double_stream_modulation_img.to(cuda0)
    transformer.double_stream_modulation_txt.to(cuda0)
    transformer.single_stream_modulation.to(cuda0)
    for block in transformer.transformer_blocks:
        block.to(cuda0)
    for block in transformer.single_transformer_blocks[:split_at]:
        block.to(cuda0)
    for block in transformer.single_transformer_blocks[split_at:]:
        block.to(cuda1)
    transformer.norm_out.to(cuda0)
    transformer.proj_out.to(cuda0)

    # Per-block pre-hook: every single_transformer_block on cuda:1
    # needs its args moved (temb is shared across the loop, so only
    # hooking the boundary block leaves temb on cuda:0 for subsequent
    # blocks).
    for block in transformer.single_transformer_blocks[split_at:]:
        block.register_forward_pre_hook(
            _make_device_bridge_hook(cuda1), with_kwargs=True
        )
    transformer.norm_out.register_forward_pre_hook(
        _make_device_bridge_hook(cuda0), with_kwargs=True
    )

    transformer._flux2_dual_gpu_split_at = split_at


def route_lora_to_wrapped_devices(transformer_lora: Any) -> None:
    """Place each LoRA submodule on the device of its wrapped layer.

    OneTrainer's ``LoRAModule`` retains ``orig_module`` after
    ``hook_to_module()``, so the wrapped layer's device is directly
    queryable. No snapshot needed.

    A no-op when devices already match. Should be called after the
    transformer has been distributed; ``Flux2Model.transformer_to``
    invokes this when ``FLUX2_DUAL_GPU=true``.
    """
    if transformer_lora is None:
        return
    for lora in transformer_lora.lora_modules.values():
        try:
            wrapped_device = lora.orig_module.weight.device
        except (AttributeError, AssertionError):
            continue
        lora.to(wrapped_device)


# ─── Internals ──────────────────────────────────────────────────────────────

def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested tuple/list/dict to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _make_device_bridge_hook(target_device: torch.device):
    """Forward pre-hook that moves all tensor inputs to ``target_device``."""
    def hook(module, args, kwargs):
        return (
            _move_to_device(args, target_device),
            _move_to_device(kwargs, target_device),
        )
    return hook
