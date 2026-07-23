from collections.abc import Callable

from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util.enum.DataType import DataType
from modules.util.torch_util import torch_gc

import torch
from torch import nn

# A streamed sub-module keeps its base weights frozen (LoRA training streams too -- only the adapter trains, so the
# streamed base weights never diverge from disk; a fully fine-tuned part cannot stream, its in-place updates would be
# discarded). It is loaded as a meta skeleton and its real weights are streamed straight from the checkpoint to the
# compute device and quantized the first time it is used -- so the full unquantized module never lands in system RAM.
# Both load paths share
# this materialize step and differ only in how they evict the weights off the compute device afterwards, selected by
# cache_in_ram:
#   - cache_in_ram off: discard the weights to meta; re-materialize by re-streaming from the checkpoint. Frees both
#     VRAM and RAM. Lossless because the module is frozen -- its weights never diverge from disk.
#   - cache_in_ram on: keep the streamed+quantized weights resident on the temp device; re-materialize by moving them
#     back to the compute device. Frees VRAM only, but avoids re-reading the checkpoint on every use.


def _is_evicted(module: nn.Module) -> bool:
    # the skeleton is fully on meta between uses; a single real parameter means it is currently materialized
    for parameter in module.parameters():
        return parameter.is_meta
    return True


def _current_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("meta")


def evict_to_meta(module: nn.Module):
    for sub_module in module.modules():
        for name, parameter in list(sub_module.named_parameters(recurse=False)):
            if parameter.is_meta:
                continue
            if name == "weight" and isinstance(sub_module, QuantizedLinearMixin):
                # a quantized weight is stored in a packed layout (nf4 packs to a flat [N, 1] tensor); reset it to a
                # meta tensor of the original unpacked shape so the next materialize can stream the checkpoint weight
                # back into it and re-quantize. Its dtype is irrelevant (the stream overwrites it), so keep the current.
                sub_module.register_parameter(name, nn.Parameter(
                    torch.empty(sub_module.original_weight_shape(), dtype=parameter.dtype, device="meta"),
                    requires_grad=False))
            else:
                sub_module.register_parameter(
                    name, nn.Parameter(parameter.detach().to("meta"), requires_grad=False))
        for name, buffer in list(sub_module._buffers.items()):
            # non-persistent buffers (e.g. rotary inv_freq) are config-derived constants, not disk weights;
            # keep them resident rather than evict and re-derive them.
            if name in sub_module._non_persistent_buffers_set:
                continue
            if buffer is not None and not buffer.is_meta:
                sub_module._buffers[name] = buffer.to("meta")
        # let the next materialize() re-quantize the freshly streamed weights
        if isinstance(sub_module, QuantizedLinearMixin):
            sub_module.mark_needs_requantization()


def stream_module_to(
        module: nn.Module,
        device: torch.device,
        materialize_fn: Callable[[nn.Module, torch.device, DataType], None],
        train_dtype: DataType,
        cache_in_ram: bool,
        name: str,
        temp_device: torch.device,
):
    # module.to()-style entry point for a materialize-on-demand component; see the module-level comment for the
    # materialize/evict semantics. Idempotent; train_dtype is used only when materializing.
    if device.type not in ("meta", temp_device.type):
        # target is the compute device -> materialize the module onto it
        current = _current_device(module)
        try:
            if current.type == "meta":
                # cold: stream+quantize the weights from the checkpoint onto the compute device
                materialize_fn(module, device, train_dtype, part_name=name)
            elif current.type == temp_device.type:
                # warm (cache_in_ram): the quantized weights are staged resident on the temp device, move them back to
                # the compute device. Dispatch on device *type* (not equality) so a module already on the compute
                # device isn't dragged through module.to(), which would raise on the non-persistent buffers left on meta.
                module.to(device=device)
        except Exception:
            # a materialize that fails partway (typically OOM) leaves already-streamed weights resident on the compute
            # device -- live model state torch_gc can't reclaim, which can cascade into a second OOM. Roll back along the
            # inverse of the failed move: a meta origin re-streams next time (drop the partial fill back to meta), a cpu
            # origin keeps its RAM copy (move back to the temp device).
            if current.type == "meta":
                evict_to_meta(module)
                # reclaim the partial fill now: this rollback runs under BaseModel.materialize, which (unlike
                # evict) has no trailing torch_gc, so the stranded VRAM would otherwise survive into the re-raise.
                torch_gc()
            else:
                module.to(device=current)
            raise
    elif not cache_in_ram:
        if not _is_evicted(module):
            evict_to_meta(module)
    else:
        # cache_in_ram: stage the resident quantized weights on the temp device. Only when currently on the compute
        # device -- a module still on meta (never materialized) has nothing resident to stage and .to() can't move meta,
        # so it stays a no-op here and streams from the checkpoint on its first materialize.
        if _current_device(module).type not in (device.type, "meta"):
            module.to(device=device)
