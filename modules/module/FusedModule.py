from collections.abc import Iterator, Mapping
from typing import Any

from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util.convert_util import match_any, matched_leaf_groups
from modules.util.quantization_util import get_unquantized_weight

import torch
from torch import Tensor, nn
from torch.nn import Parameter


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

        # cumulative offsets: leaf i's share of the fused module's output ends at slices[i] and starts
        # at slices[i-1] (or 0 for leaf 0).
        offsets = []
        total = 0
        for leaf in leaves:
            total += leaf.out_features
            offsets.append(total)
        self.slices = offsets

        self.is_applied = False

    # self.module.forward(x) recomputes the full fused base (all N leaves) on every one of the N leaf
    # calls -- N times, so N^2 real base compute for a group of N. When the peft type exposes
    # delta_forward (its own contribution only, without touching the base -- see PeftBase.delta_forward),
    # add it to this leaf's real, unfused base instead, restoring N real base computes. Peft types that
    # recompose the base weight itself (delta_forward returns None) keep going through the slower,
    # generic self.module.forward(x) path.
    def _leaf_output(self, leaf_index: int, start: int, end: int, x, *args, **kwargs):
        delta = self.module.delta_forward(x, *args, **kwargs)
        if delta is None:
            return self.module.forward(x)[..., start:end]
        return self.synthetic._leaf_forwards[leaf_index](x) + delta[..., start:end]

    # One fixed method per leaf slot (not a functools.partial built at hook time) so leaf.forward is
    # always the same function object -- torch.compile guards on that identity and would otherwise
    # recompile on every hook/unhook cycle. Capped at 4 leaves (Flux/Chroma/HunyuanVideo qkv+mlp).
    # Stateless on purpose: caching across leaves broke torch.compile + activation checkpointing.
    def _leaf_forward_0(self, x, *args, **kwargs):
        return self._leaf_output(0, 0, self.slices[0], x, *args, **kwargs)

    def _leaf_forward_1(self, x, *args, **kwargs):
        return self._leaf_output(1, self.slices[0], self.slices[1], x, *args, **kwargs)

    def _leaf_forward_2(self, x, *args, **kwargs):
        return self._leaf_output(2, self.slices[1], self.slices[2], x, *args, **kwargs)

    def _leaf_forward_3(self, x, *args, **kwargs):
        return self._leaf_output(3, self.slices[2], self.slices[3], x, *args, **kwargs)

    def _leaf_forward_for_index(self, leaf_index: int):
        match leaf_index:
            case 0:
                return self._leaf_forward_0
            case 1:
                return self._leaf_forward_1
            case 2:
                return self._leaf_forward_2
            case 3:
                return self._leaf_forward_3
            case _:
                raise ValueError(f"FusedModuleGroup supports at most 4 leaves, got leaf_index {leaf_index}")

    def hook_to_module(self):
        if not self.is_applied:
            for leaf_index, leaf in enumerate(self.leaves):
                leaf.forward = self._leaf_forward_for_index(leaf_index)
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

    def parameters(self) -> Iterator[Parameter]:
        return self.module.parameters()

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def modules(self) -> Iterator[nn.Module]:
        return self.module.modules()

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
    # Match the per-model fusion_spec against the concrete selected Linear names. A group fires only when
    # all its leaves are present and selected for the same group and share in_features; otherwise those
    # leaves stay individual modules (never fuse a partial group). Returns the discovered groups as (fused
    # name, leaf names, leaf modules) -- independent of whether they'll be built fused (LoRAModuleWrapper.fuse);
    # a split wrapper keeps the same list to recognise an incompatible fused file on load (check_fusion_match).
    #
    # fuse mirrors LoRAModuleWrapper.fuse: True when this wrapper will actually BUILD fused modules (an
    # output format that needs qkv fusion). When fuse, two conditions raise here at setup with an actionable
    # message instead of leaking split keys into the saver, which fails much later with a cryptic "No
    # conversion found": (1) a PARTIALLY-selected group -- some of a group's projections trained, the rest
    # excluded by the layer filter -- cannot be fused (fusion needs every leaf, and the fused formats
    # ORIGINAL/KOHYA/COMFY have no split keys to fall back on); (2) a group whose leaves do not share
    # in_features cannot be stacked into one weight matrix, so the fusion spec is misconfigured. When NOT
    # fusing (split build -- e.g. Flux2 + DIFFUSERS/LEGACY) a partial group is fine (split formats keep the
    # keys), so neither is checked.
    groups = []
    if not fusion_spec:
        return groups

    consumed = set()
    for group_pattern, leaf_suffixes, fused_suffix, _original_suffix in fusion_spec:
        group_sets = [matched_leaf_groups(group_pattern, leaf, selected_modules) for leaf in leaf_suffixes]

        complete = set.intersection(*group_sets) if group_sets else set()
        if fuse:
            # groups where some leaves are selected but not all -> cannot fuse, cannot fall back to split.
            partial = (set.union(*group_sets) - complete) if group_sets else set()
            for group_key in sorted(partial):
                raise RuntimeError(
                    f"The selected output format requires fusing all of {', '.join(leaf_suffixes)} in "
                    f"{group_key} into a single '{fused_suffix}' module, but the layer filter trained only a "
                    f"subset. Every projection in a fusion group must be trained together. Either add the "
                    f"missing layer(s) to the layer filter, or choose a different output format.")

        for group_key in sorted(complete):
            leaf_names = [f"{group_key}.{leaf}" for leaf in leaf_suffixes]
            if any(name in consumed for name in leaf_names):
                continue
            leaves = [selected_modules[name] for name in leaf_names]
            if len({leaf.in_features for leaf in leaves}) != 1:
                # leaves reading different input spaces cannot be stacked into one weight matrix. When
                # fusing this is a misconfigured spec (raise); a split build just keeps them separate.
                if fuse:
                    raise RuntimeError(
                        f"The selected output format requires fusing all of {', '.join(leaf_suffixes)} in "
                        f"{group_key} into a single '{fused_suffix}' module, but they do not share an input "
                        f"dimension ({', '.join(f'{name}={selected_modules[name].in_features}' for name in leaf_names)}). "
                        f"Only projections that read the same input can be fused; this fusion group is misconfigured.")
                continue
            groups.append((f"{group_key}.{fused_suffix}", leaf_names, leaves))
            consumed.update(leaf_names)

    return groups


def check_fusion_match(keys, fuse: bool, fusion_groups: list[tuple] | None):
    # A LoRA's qkv fused/split shape must match what the target needs: the wrapper's own `fuse`, on the
    # training-resume load path; or the requested output format, on the convert tool's save path (the
    # convert tool loads straight into a plain dict and never goes through the wrapper, so it needs this
    # same guard before writing a file the target format can't represent). Converting between them is
    # unsupported: fusing independent split q/k/v into one rank-r adapter is lossy (SVD/re-rank), and
    # de-fusing a fused adapter into split leaves, though exact for LoRA, is not implemented -- so a
    # mismatch is a hard error rather than a silent key drop or an invalid output file.
    #
    # `fusion_groups` is the model's raw fusion spec -- (group_pattern, leaf_suffixes, fused_suffix,
    # original_suffix), the same shape as model.fusion_groups() / LoRAModuleWrapper.fusion_spec -- no live
    # module or per-instance discovery needed, just a pattern match against the on-disk keys via match_any
    # (the leading component prefix, e.g. "transformer.", is stripped positionally since fusion groups only
    # ever apply to the single denoising component; the trailing "{suffix__}" wildcard stands in for
    # whatever value suffix -- .lora_down.weight, .alpha, hada_w1_a, ... -- follows the module name).
    if not fusion_groups:
        return
    relative_keys = [key.split(".", 1)[1] for key in keys if "." in key]
    for group_pattern, leaf_suffixes, fused_suffix, _original_suffix in fusion_groups:
        file_fused = match_any([group_pattern + "." + fused_suffix + ".{suffix__}"], relative_keys)
        file_split = match_any([group_pattern + "." + leaf + ".{suffix__}" for leaf in leaf_suffixes], relative_keys)

        if fuse and file_split:
            raise RuntimeError(
                f"LoRA file has split q/k/v ({fused_suffix}), but the selected output format needs "
                f"fused qkv. Fusing independent q/k/v adapters into one rank-r adapter is lossy "
                f"(SVD/re-rank); pick a split output format or retrain.")
        if not fuse and file_fused:
            raise RuntimeError(
                f"LoRA file has fused qkv ({fused_suffix}), but the selected output format keeps q/k/v "
                f"split. De-fusing a fused adapter on load is not supported yet; pick a fused output "
                f"format or retrain.")
