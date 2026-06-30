import functools
from collections.abc import Mapping
from typing import Any

from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util.quantization_util import get_unquantized_weight

import torch
from torch import Tensor, nn
from torch.nn import Parameter

import parse


class _FusedLinear(nn.Linear, QuantizedLinearMixin):
    # Synthetic "fused qkv" Linear handed to a single, unmodified peft module so it can train one
    # fused adapter over the diffusers-split q/k/v(/mlp) base Linears (step 3, ORIGINAL/COMFY output).
    # It owns no weight of its own: in/out_features describe the fused layer, forward delegates to the
    # real leaves' (pre-hook) base forwards, and unquantized_weight concatenates theirs -- so nothing
    # about the (possibly quantized) base is copied. Only the recompose peft types (DoRA/LoKr-decompose)
    # ever read the weight; additive types (LoRA/LoHa) only use forward.
    def __init__(self, leaves: list[nn.Linear]):
        in_features = leaves[0].in_features
        out_features = sum(leaf.out_features for leaf in leaves)
        device = leaves[0].weight.device
        super().__init__(in_features, out_features, bias=False, device="meta")
        self._leaves = leaves
        # captured before the leaves are hooked, so forward never re-enters the per-leaf slicers.
        self._leaf_forwards = [leaf.forward for leaf in leaves]
        # a 0-size real weight gives the right .device for adapter creation without materializing the
        # [out, in] fused weight (get_weight_shape reads in/out_features, not this tensor).
        self.weight = Parameter(torch.empty(0, device=device), requires_grad=False)

    def reset_parameters(self):
        # nn.Linear.__init__ calls this; we own no real weight (a 0-size stub set just after
        # super().__init__) and no real bias (a property over the leaves), so there is nothing to init.
        pass

    def forward(self, x, *args, **kwargs):
        return torch.cat([f(x) for f in self._leaf_forwards], dim=-1)

    @property
    def bias(self):
        # Derived live from the leaves (like unquantized_weight below), not materialized. The recompose
        # forwards (DoRA/OFT/LoKr-decompose) call op(x, W, orig_module.bias) and so need the fused q/k/v
        # bias; additive types delegate to the leaf forwards and never read it. Reading the leaves live
        # keeps the bias in sync if they move device/dtype and avoids duplicating their storage.
        leaf_biases = [leaf.bias for leaf in self._leaves]
        if all(b is None for b in leaf_biases):
            return None
        ref = next(b for b in leaf_biases if b is not None)
        return torch.cat([
            b if b is not None
            else torch.zeros(leaf.out_features, device=ref.device, dtype=ref.dtype)
            for leaf, b in zip(self._leaves, leaf_biases, strict=True)
        ])

    def original_weight_shape(self) -> tuple[int, ...]:
        return (self.out_features, self.in_features)

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.cat([get_unquantized_weight(leaf, dtype, device) for leaf in self._leaves], dim=0)


class FusedModuleGroup:
    # Type-independent fused-qkv adapter. One inner peft module of the configured type is built on a
    # _FusedLinear over the split q/k/v(/mlp) base Linears; each real split Linear is hooked to return
    # its slice of the inner module's fused output (computed once per input and cached across the
    # group's leaves). Construction and forward are peft-type-agnostic -- the inner module never knows
    # it is fused. Used only when the chosen output format is fused (ORIGINAL/COMFY); its state_dict
    # emits the single fused module, so ORIGINAL/COMFY export is a pure namespace rename.
    def __init__(self, prefix: str, leaves: list[nn.Module], klass, additional_args: list, additional_kwargs: dict):
        self.leaves = leaves
        self.synthetic = _FusedLinear(leaves)
        self.module = klass(prefix, self.synthetic, *additional_args, **additional_kwargs)
        # wire the inner module's base forward to the fused base without using the single-module hook
        # machinery (we hook the leaves ourselves below). check_initialized only requires orig_forward.
        self.module.orig_forward = self.synthetic.forward
        self.module.is_applied = True
        self.prefix = self.module.prefix

        offsets = []
        start = 0
        for leaf in leaves:
            offsets.append((start, start + leaf.out_features))
            start += leaf.out_features
        self.slices = offsets

        self.is_applied = False

    def _leaf_forward(self, leaf_index: int, x, *args, **kwargs):
        # Each split leaf returns its slice of the single fused module's output. This is stateless on
        # purpose: an earlier version cached the fused forward across the group's leaves, but mutating
        # instance state inside the forward breaks under torch.compile + activation checkpointing
        # ("HOP: Unsafe side effect"). The fused forward (split base + one adapter) is therefore
        # recomputed once per leaf -- correct and compile-safe, at the cost of recomputing the qkv base
        # projection per leaf (only on the ORIGINAL/COMFY fused path; the split formats are unaffected).
        start, end = self.slices[leaf_index]
        return self.module.forward(x)[..., start:end]

    def hook_to_module(self):
        if not self.is_applied:
            for leaf_index, leaf in enumerate(self.leaves):
                leaf.forward = functools.partial(self._leaf_forward, leaf_index)
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.is_applied:
            for leaf, orig_forward in zip(self.leaves, self.synthetic._leaf_forwards, strict=True):
                leaf.forward = orig_forward
            self.is_applied = False

    def state_dict(self, prefix: str = "") -> dict:
        return self.module.state_dict(prefix=prefix)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)

    def parameters(self) -> list[Parameter]:
        return list(self.module.parameters())

    def modules(self) -> list[nn.Module]:
        return list(self.module.modules())

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'FusedModuleGroup':
        self.module.to(device, dtype)
        return self

    def requires_grad_(self, requires_grad: bool):
        self.module.requires_grad_(requires_grad)

    @property
    def dropout(self):
        return self.module.dropout


def discover_fused_groups(fusion_spec: list[tuple] | None, selected_modules: dict[str, nn.Module],
                          fuse: bool = False) -> list[tuple[str, list[str], list]]:
    # Match the per-model fusion_spec against the concrete selected Linear names, capturing the
    # block index {i} via parse. A group fires only when all its leaves are present and selected
    # for the same block and share in_features; otherwise those leaves stay individual modules
    # (never fuse a partial group). Returns the discovered groups as (fused name, leaf names, leaf
    # modules) -- independent of whether they'll be built fused (LoRAModuleWrapper.fuse); a split
    # wrapper keeps the same list to recognise an incompatible fused file on load (__check_fusion_match).
    #
    # fuse mirrors LoRAModuleWrapper.fuse: True when this wrapper will actually BUILD fused modules (an
    # output format that needs qkv fusion). A PARTIALLY-selected group -- some of a group's projections
    # trained, the rest excluded by the layer filter -- cannot be fused: fusion needs every leaf, and the
    # fused formats (ORIGINAL/KOHYA/COMFY) have no split keys to fall back on. When fuse, raise here at
    # setup with an actionable message instead of leaking split keys into the saver, which fails much later
    # with a cryptic "No conversion found". When NOT fusing (split build -- e.g. Flux2 + DIFFUSERS/LEGACY)
    # a partial group is fine (split formats keep the keys), so it is not checked.
    groups = []
    if not fusion_spec:
        return groups

    consumed = set()
    for block_pattern, leaf_suffixes, fused_suffix, _original_suffix in fusion_spec:
        index_sets = []
        for leaf in leaf_suffixes:
            pattern = block_pattern + "." + leaf
            indices = {match["i"] for name in selected_modules
                       if (match := parse.parse(pattern, name)) is not None and "i" in match.named}
            index_sets.append(indices)

        complete = set.intersection(*index_sets) if index_sets else set()
        if fuse:
            # blocks where some leaves are selected but not all -> cannot fuse, cannot fall back to split.
            partial = (set.union(*index_sets) - complete) if index_sets else set()
            for i in sorted(partial):
                block = block_pattern.format(i=i)
                raise RuntimeError(
                    f"The selected output format requires fusing all of {', '.join(leaf_suffixes)} in "
                    f"{block} into a single '{fused_suffix}' module, but the layer filter trained only a "
                    f"subset. Every projection in a fusion group must be trained together. Either add the "
                    f"missing layer(s) to the layer filter, or save as DIFFUSERS or LEGACY, which keep the "
                    f"projections split.")

        for i in sorted(complete):
            block = block_pattern.format(i=i)
            leaf_names = [f"{block}.{leaf}" for leaf in leaf_suffixes]
            if any(name in consumed for name in leaf_names):
                continue
            leaves = [selected_modules[name] for name in leaf_names]
            if len({leaf.in_features for leaf in leaves}) != 1:
                continue
            groups.append((f"{block}.{fused_suffix}", leaf_names, leaves))
            consumed.update(leaf_names)

    return groups
