from modules.util.convert_util import (
    add_prefix,
    component_body_conversion,
    convert,
    peft_magnitude_to_dora_scale,
    remove_prefix,
    reverse_conversion,
)

import torch.nn as nn
from torch import Tensor

import parse

# Load-side LAYER 1: the model-independent universal pre-pass + feature detector.
#
# Every on-disk LoRA format decomposes into the same orthogonal dimensions (see the ecosystem
# survey). Five of those dimensions carry NO model knowledge -- they are pure prefix/suffix/value
# conventions -- so they can be normalised in one place, before any per-model namespace reverse
# (Layer 2) runs. This module owns exactly those dimensions:
#
#   - Container:  strip the peft directory wrapper's `base_model.model.` top prefix, the single-
#                 adapter `.default` infix, and the generic LyCORIS `lycoris_` wrapper prefix.
#   - Suffix:     `lora_A`/`lora_B` -> OneTrainer-native `lora_down`/`lora_up`.
#   - DoRA name:  peft `lora_magnitude_vector.weight` and ai-toolkit `magnitude` -> `dora_scale`.
#   - Alpha:      detect whether `.alpha` is present (kept) or absent (folded into lora_up, the
#                 DIFFUSERS convention); if folded, synthesise `alpha = rank` so the load-time
#                 scale alpha/rank == 1 reproduces the already-folded delta exactly (the loss-free
#                 inverse of lora_fold_alpha -- no unfolding of lora_up needed).
#
# What is NOT here, because it is model-specific (Layer 2): the namespace reverse (diffusers<->
# original<->kohya-flat name maps), the denoising-component top prefix (`diffusion_model.`,
# `lora_unet_`, ... -- the canonical component name transformer/unet/prior differs per model), and
# qkv de-fusion. Layer 2 scores the universalised dict against the model's candidate namespaces.
#
# A file already in diffusers-split names (OT DIFFUSERS / INTERNAL, and the ecosystem's diffusers-
# keyed variants) reaches the canonical namespace with this pass ALONE -- its module paths already
# carry the canonical `transformer.`/`unet.` prefix, so Layer 2 is a no-op. That is the case this
# module is independently testable on.


# The adapter param-name families OneTrainer trains: LoRA/DoRA (lora_*, alpha, dora_scale), LoHa (hada_*),
# LoKr (lokr_*), OFT (oft_*). This is the closed allowlist -- checked positively so any foreign LyCORIS
# algorithm (GLoRA, IA3, full-diff, norm, tlora, ...) is rejected with no per-algorithm table to maintain.
_SUPPORTED_PARAM_PREFIXES = ("lora_", "hada_", "lokr_", "oft_")
_SUPPORTED_PARAM_SCALARS = {"alpha", "dora_scale"}


def _reject_unsupported_algorithms(state_dict: dict[str, Tensor]) -> None:
    # raise if any key's adapter param is outside the trained set -- it can't map onto a wrapper module, so
    # it would otherwise be silently dropped. This must run POST-CONVERT, i.e. after the suffix/dora-name
    # normalization steps, NOT on the raw on-disk dict: the allowlist matches only canonical param names
    # (lora_down/up, dora_scale), so the input-only spellings those steps fold in -- lora_A/B and the DoRA
    # magnitude/lora_magnitude_vector -- are already canonical by the time we check and aren't mistaken for
    # a foreign algorithm. Checking the raw dict would wrongly reject them. The param is the last path
    # segment, or the one before a trailing .weight. bundle_emb.* is a passthrough embedding and is exempt
    # (_aux.* bookkeeping tensors are already dropped in normalize_various before this runs).
    for key in state_dict:
        if key.startswith("bundle_emb"):
            continue
        segments = key.split(".")
        param = segments[-2] if segments[-1] == "weight" and len(segments) > 1 else segments[-1]
        if param in _SUPPORTED_PARAM_SCALARS or param.startswith(_SUPPORTED_PARAM_PREFIXES):
            continue
        raise RuntimeError(
            f"this LoRA file uses an adapter type OneTrainer can't load for training (key '{key}'). Only "
            f"LoRA, DoRA, LoHa, LoKr and OFT adapters are supported.")


def normalize_various(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # apply the model-independent dimensions in order: container -> suffix -> dora-name -> alpha.
    # Every step is strict=False: keys that don't match a rule (already-native suffixes, bundle_emb,
    # LyCORIS hada_*/lokr_*/oft_* params) pass through untouched.
    # drop _aux.* bookkeeping tensors first (content-hash-named, appended by some slider tools): not an
    # adapter param or embedding, nothing downstream can use them, so remove before any conversion sees them.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_aux.")}
    state_dict = _strip_container(state_dict)
    state_dict = _normalize_suffix(state_dict)
    state_dict = _normalize_dora_name(state_dict)
    _reject_unsupported_algorithms(state_dict)

    return _synthesize_alpha(state_dict)


def _strip_container(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # peft directory container conventions. `lycoris_` is the LyCORIS trainer's flat equivalent of the
    # kohya `lora_unet_` denoising prefix (underscore-joined module path, no dot-separated component).
    # Re-prefix as `lora_unet_` so normalize_namespace() and kohya_unflatten() recognise it normally.
    state_dict = {
        ("lora_unet_" + k[len("lycoris_"):] if k.startswith("lycoris_") else k): v
        for k, v in state_dict.items()
    }
    # `base_model.model.` peft-native top prefix.
    state_dict = convert(state_dict, remove_prefix("base_model.model"), strict=False)
    # `.default` single-adapter infix: `<module>.lora_A.default.weight` -> `<module>.lora_A.weight`
    # (the adapter-name slot peft inserts between the param name and `.weight`).
    state_dict = convert(state_dict, [("{}.default.weight", "{}.weight")], strict=False)
    return state_dict


def _normalize_suffix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # accept the diffusers A/B spelling and the LoRALinearLayer .lora.down/.lora.up nesting (the submodule
    # is literally named `lora` with .down/.up Linears) regardless of the file's declared format, mapping
    # to the OneTrainer-native down/up. Files already in down/up pass through.
    return convert(state_dict, [
        ("{}.lora_A.weight", "{}.lora_down.weight"),
        ("{}.lora_B.weight", "{}.lora_up.weight"),
        ("{}.lora.down.weight", "{}.lora_down.weight"),
        ("{}.lora.up.weight", "{}.lora_up.weight"),
    ], strict=False)


def _normalize_dora_name(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # collapse the three DoRA magnitude spellings to the canonical `dora_scale`. peft's
    # lora_magnitude_vector.weight (SimpleTuner/HF), ai-toolkit's bare `magnitude`, and the native
    # `dora_scale` (kohya/Comfy/LyCORIS -- already canonical, passes through). peft's magnitude is
    # per-output ([out] / [1, out, 1, 1]); reshape it to OneTrainer's per-output dora_scale layout
    # ([out, 1] / [out, 1, 1, 1]) so it loads without a shape mismatch -- the inverse of the DIFFUSERS
    # save. ai-toolkit's `magnitude` shape is left as a plain rename (not verified, so not reshaped).
    return convert(state_dict, [
        ("{}.lora_magnitude_vector.weight", "{}.dora_scale", peft_magnitude_to_dora_scale),
        ("{}.magnitude", "{}.dora_scale"),
    ], strict=False)


def _synthesize_alpha(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # DIFFUSERS folds alpha into lora_up and drops the .alpha key. For every LoRA module (one that
    # has a lora_up) lacking an .alpha, synthesise alpha = rank so the load-time scale alpha/rank
    # == 1 reproduces the already-folded delta exactly -- the loss-free inverse of lora_fold_alpha.
    # Modules that kept their .alpha are left untouched.
    out = dict(state_dict)
    for key, tensor in state_dict.items():
        if not key.endswith(".lora_up.weight"):
            continue
        module = key.removesuffix(".lora_up.weight")
        alpha_key = module + ".alpha"
        if alpha_key not in out:
            # rank == lora_down's out dim == lora_up's in dim; read it from lora_up to avoid
            # depending on lora_down's presence.
            out[alpha_key] = tensor.new_tensor(float(tensor.shape[1]))

    return out


# Load-side LAYER 2: the per-model namespace reverse. Given the universalised dict (Layer 1 output)
# and the model's own forward save knowledge -- the denoising component name and the canonical(diffusers)
# -> native body conversion -- it inverts whichever namespace the file is in back to canonical.
#
# Approach: reuse the savers' forward conversion lists, reversed with reverse_conversion(). The body
# map (e.g. SDXL's lora_diffusers_to_original(), or None for a ships-as-diffusers identity model) is the same
# one the saver applies forward; only the top prefix and the kohya flatten differ between the named
# formats, and those are handled here generically.
#
# The on-disk namespaces are recognised as points in the namespace dimension by their top prefix; we
# never "identify the format", we invert the prefix + body that produced it. normalize_namespace returns
# one of these prefix-shape families (NOT a ModelFormat -- one shape can cover several formats):
#   - component_prefixed       : already `<component>.` / `text_encoder.` dotted -> identity (DIFFUSERS after
#                                Layer 1, and born-canonical LEGACY).
#   - unprefixed               : bare native dotted (component prefix was stripped) -> re-add prefix, reverse body.
#   - diffusion_model_prefixed : `diffusion_model.` -> swap back to the component, reverse body.
#   - kohya_flat               : `lora_unet_`/`lora_te*_` flat -> un-flatten (structure-aware), reverse the
#                                prefix rename + body.
#   - legacy_flat              : the model's distinct flat LEGACY prefix -> un-flatten, reverse the prefix rename.


def normalize_namespace(
        state_dict: dict[str, Tensor],
        component: str,
        body: list | None,
        text_encoders: list[tuple[str, str]],
        module_names: dict[str, set[str]],
        legacy_conversion: list | None,
) -> tuple[dict[str, Tensor], str]:
    # full classification of the on-disk namespace: a coarse top-prefix sniff, then a second pass that
    # reclassifies (and re-prefixes) the one case the sniff can't see. Returns the possibly-rewritten dict
    # plus the resolved family; to_canonical then only dispatches. Not a pure classifier -- the peft pass
    # below rewrites state_dict -- which is exactly why it does not live in to_canonical's dispatch.
    #
    # Coarse sniff: the component prefix is a dotted `<component>.`; the bare native (unprefixed) form has no
    # such prefix even though its first segment may share the component's word stem (e.g. `transformer_blocks.`
    # vs the `transformer.` component prefix). Detection keys ONLY off the denoising component: a text encoder
    # keeps its canonical `text_encoder.` name in ORIGINAL and COMFY too, so `text_encoder.` does not separate
    # component_prefixed from those families -- only the denoising component's prefix (present in
    # component_prefixed, stripped in unprefixed, swapped in diffusion_model_prefixed, flattened in kohya_flat)
    # does. A hypothetical TE-only file with no denoising keys falls through to "unprefixed", whose reverse
    # passes the already-canonical TE keys through unchanged -- the same result as "component_prefixed".
    #
    # legacy_denoising_prefix is the model's flat LEGACY denoising prefix (from _legacy_conversion, None if
    # the model dropped LEGACY). LEGACY is a separate family ONLY when that prefix differs from kohya's
    # lora_unet -- lora_transformer for the transformer models, lora_prior_unet for Stable Cascade. When a
    # model's LEGACY instead reuses lora_unet (SD/SDXL), it is NOT separable here and is handled in the
    # kohya_flat branch (body scoring). Checked before kohya_flat because a TE model's LEGACY also carries
    # lora_te*_ keys that the kohya prefix would otherwise match.
    legacy_denoising_prefix = legacy_conversion[0][1] if legacy_conversion is not None else None
    keys = list(state_dict)
    if any(k.startswith(component + ".") for k in keys):
        family = "component_prefixed"
    elif any(k.startswith("diffusion_model.") for k in keys):
        family = "diffusion_model_prefixed"
    elif legacy_denoising_prefix not in (None, "lora_unet") and any(k.startswith(legacy_denoising_prefix + "_") for k in keys):
        family = "legacy_flat"
    elif any(k.startswith(("lora_unet_", "lora_te")) for k in keys):
        family = "kohya_flat"
    else:
        family = "unprefixed"

    # A peft adapter saved directly on the denoising module (peft wraps the transformer itself) keeps
    # canonical diffusers module names but drops the `component.` prefix, so the sniff reads "unprefixed".
    # Tell it apart from a genuine native-body bare file by matching the bare module paths against the
    # canonical name set: native and diffusers names are disjoint, so any hit means it is canonical minus the
    # prefix. Re-add the prefix and reclassify as component_prefixed (a true native-body file matches no
    # canonical name and stays unprefixed).
    if family == "unprefixed" and body is not None:
        canonical_te_prefixes = [prefix for prefix, _kohya in text_encoders]
        denoising, passthrough = _partition_denoising(state_dict, canonical_te_prefixes)
        if {_split_value_suffix(k)[0] for k in denoising} & module_names[component]:
            state_dict = convert(denoising, add_prefix(component), strict=False) | passthrough
            family = "component_prefixed"

    return state_dict, family


def _denoising_body_conversion(component: str, top: str, body: list | None) -> list:
    # the denoising component's canonical prefix mapped to `top` (the component itself for original, "lora_unet"
    # for kohya), applying `body`. Mirrors the savers' head entry (component_body_conversion with no TE extras);
    # a multi-pass body is emitted as chained passes there, and reverse_conversion inverts it pass by pass.
    return component_body_conversion(component, top, body)


def _partition_denoising(
        state_dict: dict[str, Tensor],
        text_encoder_prefixes: list[str],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    # split a universalised dict into (denoising-component keys, text-encoder/bundle passthrough keys). In
    # ORIGINAL/COMFY only the denoising component lost its prefix (original) or was renamed to
    # "diffusion_model." (comfy); the text encoders carry their ORIGINAL-namespace prefix (the name they were
    # saved under -- their canonical `text_encoder[_n].` for most models, or a per-TE rename like SD3's
    # clip_l/clip_g/t5) and bundle_emb.* passes through. Neither must be run through the denoising body
    # reverse (which is strict and would reject them); the caller renames the TE prefixes back to canonical
    # separately. A transformer-only model passes an empty prefix list -> every key is a denoising key,
    # identical to the pre-TE behaviour.
    def is_passthrough(key: str) -> bool:
        return key.startswith("bundle_emb") or any(key.startswith(p + ".") for p in text_encoder_prefixes)

    denoising = {k: v for k, v in state_dict.items() if not is_passthrough(k)}
    passthrough = {k: v for k, v in state_dict.items() if is_passthrough(k)}
    return denoising, passthrough


def reverse_original(
        state_dict: dict[str, Tensor],
        component: str,
        body: list | None,
        text_encoder_prefixes: list[str],
) -> dict[str, Tensor]:
    # inverse of _save_original: the suffix was already reversed by Layer 1. Re-add the stripped component
    # prefix to the denoising keys, then reverse the body rename. The text-encoder / bundle_emb keys were not
    # prefixed by the denoising component and keep their canonical names in ORIGINAL, so they are partitioned
    # out (the strict denoising-body reverse would otherwise reject them) and pass through unchanged.
    denoising, passthrough = _partition_denoising(state_dict, text_encoder_prefixes)
    denoising = convert(denoising, add_prefix(component), strict=False)
    denoising = convert(denoising, reverse_conversion(_denoising_body_conversion(component, component, body)), strict=True)
    return denoising | passthrough


def reverse_comfy(
        state_dict: dict[str, Tensor],
        component: str,
        body: list | None,
        text_encoder_prefixes: list[str],
        comfy_te_prefixes: dict[str, str],
) -> dict[str, Tensor]:
    # inverse of _save_comfy: undo each trained text encoder's Comfy-native "text_encoders.<name>.transformer"
    # prefix back to its canonical prefix (comfy_te_prefixes maps canonical -> comfy, the caller's projection of
    # each TE's COMFY_LORA declaration), then swap Comfy's "diffusion_model." back to the component on the
    # denoising keys and reverse the body rename. The TE un-prefix runs first so the TE keys are canonical
    # before partitioning -- otherwise they are neither "diffusion_model." nor a canonical prefix and fall into
    # the strict denoising-body reverse, which rejects them. strict=False so diffusion_model. and bundle_emb.
    # pass through untouched. No-op for transformer-only models (empty mapping). The TE keys then pass through
    # unchanged (a TE keeps its canonical name in COMFY).
    if comfy_te_prefixes:
        state_dict = convert(state_dict, [(comfy, canonical) for canonical, comfy in comfy_te_prefixes.items()], strict=False)
    denoising, passthrough = _partition_denoising(state_dict, text_encoder_prefixes)
    denoising = convert(denoising, [("diffusion_model", component)], strict=False)
    denoising = convert(denoising, reverse_conversion(_denoising_body_conversion(component, component, body)), strict=True)
    return denoising | passthrough


def reverse_kohya(
        state_dict: dict[str, Tensor],
        conversion: list,
        native_module_keys: set[str],
) -> dict[str, Tensor]:
    # inverse of _save_kohya: un-flatten the module path using the model's known native module keys (the
    # denoising component under "lora_unet" plus each text encoder under its "lora_te*" prefix), then
    # reverse the kohya prefix renames + body in one pass. `conversion` is the same canonical->kohya tuple
    # list the saver applies forward (one entry per component). Layer 1 already handled the suffix (kohya
    # keeps lora_down/up, so that was a no-op) and the alpha.
    state_dict = kohya_unflatten(state_dict, native_module_keys)
    return convert(state_dict, reverse_conversion(conversion), strict=True)


def reverse_legacy(state_dict: dict[str, Tensor], conversion: list, native_module_keys: set[str]) -> dict[str, Tensor]:
    # Inverse of the LEGACY save encoding (the saver's convert_to_mixture and the per-model legacy
    # conversions): DIFFUSERS-split module names (NO native body, split qkv) flattened under the kohya-style
    # prefixes lora_<component> (denoising) and lora_te* (each text encoder). Same mechanics as reverse_kohya
    # (structure-aware un-flatten against the model's native module names, then reverse the prefix renames in
    # one strict pass), but every component's body is identity (diffusers == canonical) and there is no qkv
    # collapse, so the caller's `conversion` carries body None throughout. Layer 1 already handled the
    # suffix/alpha. The namespace is genuinely distinct for any fusing/renaming model; a born-canonical model
    # never reaches this code path.
    state_dict = kohya_unflatten(state_dict, native_module_keys)
    return convert(state_dict, reverse_conversion(conversion), strict=True)


def _split_value_suffix(key: str) -> tuple[str, str]:
    # split a key into (module path, value suffix). Mirrors kohya_flatten: the suffix is .alpha /
    # .dora_scale (one segment) or .<param>.weight (two segments). Keys with neither (bundle_emb.*)
    # have no module path -> empty suffix, whole key is the "module".
    if key.endswith((".alpha", ".dora_scale")):
        suffix = key[key.rfind("."):]
    elif key.endswith(".weight"):
        suffix = key[key.removesuffix(key[key.rfind("."):]).rfind("."):]
    else:
        # LoHa (hada_*), LoKR (lokr_*), OFT (oft_*) params are single-segment and don't end in .weight.
        last_seg = key[key.rfind(".") + 1:]
        if last_seg.startswith(_SUPPORTED_PARAM_PREFIXES):
            return key[:key.rfind(".")], key[key.rfind("."):]
        return key, ""
    return key.removesuffix(suffix), suffix


def kohya_unflatten(state_dict: dict[str, Tensor], native_module_keys: set[str]) -> dict[str, Tensor]:
    # structure-aware inverse of kohya_flatten. Flattening module paths ('.' -> '_') is ambiguous to
    # reverse by rule (leaf names like to_q / down_blocks contain underscores), so we don't guess: we
    # match each flattened module against the model's known native module paths (flattened the same
    # way) and rebuild the exact dotted form. Keys with no module path (bundle_emb.*) pass through.
    flat_to_dotted = {k.replace(".", "_"): k for k in native_module_keys}
    out = {}
    for key, tensor in state_dict.items():
        module, suffix = _split_value_suffix(key)
        if module in flat_to_dotted:
            out[flat_to_dotted[module] + suffix] = tensor
        else:
            out[key] = tensor
    return out


def count_key_matches(state_dict: dict[str, Tensor], candidate_keys: set[str]) -> int:
    # how many of the on-disk flattened module paths match this candidate key set (flattened the same way
    # the un-flatten matches). Used to disambiguate two kohya-flattened namespaces that share a top prefix
    # but differ in body -- SDXL KOHYA (sgm UNet names input_blocks/...) vs LEGACY (diffusers UNet names
    # down_blocks/...), both under lora_unet_. The two namespaces are disjoint, so the real format scores
    # its module count and the wrong one scores zero.
    flat = {k.replace(".", "_") for k in candidate_keys}
    return sum(1 for key in state_dict if _split_value_suffix(key)[0] in flat)


def collapse_fusion_groups(names: set[str], fusion_groups: list | None) -> set[str]:
    # Replace each complete qkv fusion group's live split leaves with the single fused module name, so the
    # canonical module set matches what a FUSED wrapper emits (and what the saver converted to the on-disk
    # fused KOHYA names). fusion_groups entries are (block_pattern, [leaf_suffixes], fused_suffix, _orig);
    # a group collapses only for blocks where ALL its leaves are present (same completeness rule the
    # wrapper's discovery uses, minus the in_features check -- this only builds candidate names). Names are
    # component-relative (no top prefix yet). With no fusion_groups this is the identity.
    if not fusion_groups:
        return set(names)

    names = set(names)
    for block_pattern, leaf_suffixes, fused_suffix, _original_suffix in fusion_groups:
        index_sets = []
        for leaf in leaf_suffixes:
            pattern = block_pattern + "." + leaf
            indices = {match["i"] for name in names
                       if (match := parse.parse(pattern, name)) is not None and "i" in match.named}
            index_sets.append(indices)

        for i in set.intersection(*index_sets) if index_sets else []:
            block = block_pattern.format(i=i)
            for leaf in leaf_suffixes:
                names.discard(f"{block}.{leaf}")
            names.add(f"{block}.{fused_suffix}")

    return names


def native_module_keys_from_names(
        names: set[str],
        component: str,
        top: str,
        body: list | None,
        fusion_groups: list | None = None,
) -> set[str]:
    # the native (post-forward-conversion) module paths the model would emit, used as the un-flatten
    # target, built from the component-relative canonical LoRA-able module names (split qkv leaves, no
    # top prefix). Collapse the qkv fusion groups (the canonical leaves are split, but a fused wrapper /
    # the saver emit one fused module per group, so the on-disk KOHYA names are fused), then apply the
    # forward prefix+body rename so the result is in the on-disk native namespace (e.g. "lora_unet.
    # <native>"). For an identity body and no fusion groups this is just the renamed canonical paths.
    names = collapse_fusion_groups(set(names), fusion_groups)
    canonical = {(f"{component}.{name}" if component else name): None for name in names}  # names only; no tensors
    renamed = convert(canonical, _denoising_body_conversion(component, top, body), strict=True)
    return set(renamed.keys())


def lora_module_names(orig_module: nn.Module) -> set[str]:
    # component-relative paths of a live base module's LoRA-able (Linear/Conv2d) submodules -- the same set
    # the wrapper builds. Split out so an offline caller (e.g. the LoRA stress harness) can supply the same
    # names from a captured base-checkpoint key list instead of a live module.
    names = set()
    for name, child in orig_module.named_modules():
        name = name.replace(".checkpoint.", ".")
        if isinstance(child, nn.Linear | nn.Conv2d):
            names.add(name)
    return names


def to_canonical(
        state_dict: dict[str, Tensor],
        component: str,
        original_conversion: list | None,
        comfy_conversion: list | None,
        fusion_groups: list | None,
        text_encoders: list[tuple[str, str]],
        legacy_conversion: list | None,
        kohya_conversion: list | None,
        module_names: dict[str, set[str]],
        comfy_te_prefixes: dict[str, str] | None = None,
) -> dict[str, Tensor]:
    # the model-INDEPENDENT namespace dispatch: detect the on-disk namespace and reverse it to canonical.
    # All per-model knowledge arrives as plain data, so any caller can drive it -- the live loader
    # (LoRALoaderMixin._to_canonical) builds these from the model; an offline caller (the LoRA stress
    # harness) builds them from a captured base-checkpoint key list. The denoising body is per-format
    # (original_conversion / comfy_conversion / kohya_conversion) -- each the reverse of the matching saver
    # body -- so a model whose COMFY (or KOHYA) layout diverges from ORIGINAL is inverted with the right one.
    # `module_names` maps each component prefix (the denoising `component` plus each text encoder's canonical
    # prefix) to its component-relative LoRA-able module names; `text_encoders` is (canonical_prefix,
    # kohya_prefix) per TE -- a TE keeps its canonical name in ORIGINAL/COMFY, so there is no separate native
    # prefix. `comfy_te_prefixes` maps each TE's canonical prefix to the Comfy-native "text_encoders.<name>"
    # prefix the COMFY saver emits (the caller's projection of each TE's COMFY_LORA declaration); only the
    # comfy reverse consults it.
    state_dict, family = normalize_namespace(state_dict, component, original_conversion, text_encoders, module_names, legacy_conversion)

    if family == "component_prefixed":
        if original_conversion is None:
            return state_dict
        # the component_prefixed family lumps two different files together: a genuine OneTrainer diffusers file, and a
        # community file that uses the canonical `component.` prefix but native body naming inside (e.g.
        # HunyuanVideo `transformer.double_blocks.*` vs the canonical `transformer.transformer_blocks.*`). Only
        # the latter needs the native->diffusers conversion. Running that conversion over already-canonical keys
        # is undefined and actively wrong: a reversed rename whose native (original-side) leaf name is a
        # segment-prefix of a canonical sibling module rewrites it -- SD3's ("pos_embed.pos_embed","pos_embed")
        # reverse doubles the canonical patch-embed `pos_embed.proj` into the non-existent `pos_embed.pos_embed.proj`.
        # So decide for the whole denoising dict: if every denoising module is already a real diffusers module it
        # is genuinely canonical and returned untouched; otherwise it is native-body and gets converted. The two
        # namespaces are disjoint, so a native-body file has no canonical key to corrupt when it does convert.
        canonical_te_prefixes = [prefix for prefix, _kohya in text_encoders]
        denoising, passthrough = _partition_denoising(state_dict, canonical_te_prefixes)
        denoising_prefix = component + "."
        denoising_modules = {_split_value_suffix(k[len(denoising_prefix):])[0]
                             for k in denoising if k.startswith(denoising_prefix)}
        if not denoising_modules.issubset(module_names[component]):
            denoising = convert(denoising, reverse_conversion(_denoising_body_conversion(component, component, original_conversion)), strict=False)
        return denoising | passthrough
    te_prefixes = [prefix for prefix, _kohya_prefix in text_encoders]
    if family == "unprefixed":
        return reverse_original(state_dict, component, original_conversion, te_prefixes)
    if family == "diffusion_model_prefixed":
        return reverse_comfy(state_dict, component, comfy_conversion, te_prefixes, comfy_te_prefixes or {})
    if family == "kohya_flat":
        # denoising component (fusion-aware un-flatten) + each TE under its kohya prefix (identity body). Two
        # candidate bodies share lora_unet_: the model's native kohya_conversion (e.g. flux's BFL names, with
        # qkv fused the way the model saves; SDXL's sgm names) and the identity body (None) -- canonical
        # diffusers names flattened directly under lora_unet_, which is the mixture form (SDXL's historical LEGACY) and
        # also what LyCORIS emits when it wraps the diffusers module itself (the `lycoris_` container, normalised
        # to lora_unet_ in _strip_container). Score the native body and the identity body against the on-disk
        # keys and take the better fit, native winning ties. SD's kohya_conversion is None (native == identity),
        # so it just uses None. The qkv fusion collapse pairs with the native body only -- a diffusers-named file
        # is split, and a model with no native body never fuses (body None implies fusion_groups None) -- so
        # fusion follows the body.
        def fusion_for(b):
            return fusion_groups if b else None
        native_hits = count_key_matches(state_dict, native_module_keys_from_names(
            module_names[component], component, "lora_unet", kohya_conversion, fusion_for(kohya_conversion)))
        identity_hits = count_key_matches(state_dict, native_module_keys_from_names(
            module_names[component], component, "lora_unet", None, None))
        kohya_body = kohya_conversion if kohya_conversion is not None and native_hits >= identity_hits else None
        conversion = [(component, "lora_unet", kohya_body) if kohya_body else (component, "lora_unet")]
        keys = native_module_keys_from_names(module_names[component], component, "lora_unet", kohya_body, fusion_for(kohya_body))
        for te_prefix, te_kohya_prefix in text_encoders:
            conversion.append((te_prefix, te_kohya_prefix))
            keys |= native_module_keys_from_names(module_names[te_prefix], te_prefix, te_kohya_prefix, None, None)
            # Handle the two single-TE kohya numbering mismatches that appear in community LoRAs:
            # (a) model uses no-digit lora_te_ but LoRA has lora_te1_ (e.g. SD1.5 trainers that number the sole TE)
            # (b) model uses lora_te1_ but LoRA has no-digit lora_te_ (e.g. PixArt LoRAs trained with SD-style prefix)
            # Only for single-TE models; multi-TE ambiguity (which TE does lora_te_ mean?) is left to fail clearly.
            if te_kohya_prefix == "lora_te" and any(k.startswith("lora_te1_") for k in state_dict):
                conversion.append((te_prefix, "lora_te1"))
                keys |= native_module_keys_from_names(module_names[te_prefix], te_prefix, "lora_te1", None, None)
            elif (te_kohya_prefix == "lora_te1" and len(text_encoders) == 1
                  and any(k.startswith("lora_te_") for k in state_dict)
                  and not any(k.startswith("lora_te1_") for k in state_dict)):
                conversion.append((te_prefix, "lora_te"))
                keys |= native_module_keys_from_names(module_names[te_prefix], te_prefix, "lora_te", None, None)
        conversion.append(("bundle_emb", "bundle_emb"))
        return reverse_kohya(state_dict, conversion, keys)
    if family == "legacy_flat":
        # LEGACY = the model's historical namespace; legacy_conversion is the exact canonical->legacy
        # map (None for models that dropped LEGACY). Union the native names from each entry (matching each
        # component to its module names), then un-flatten + reverse.
        if legacy_conversion is None:
            raise RuntimeError("this model does not support loading the LEGACY LoRA format")
        keys = set()
        for entry in legacy_conversion:
            canonical_prefix, legacy_prefix = entry[0], entry[1]
            legacy_body = entry[2] if len(entry) > 2 else None
            if canonical_prefix == "bundle_emb":
                continue
            keys |= native_module_keys_from_names(module_names[canonical_prefix], canonical_prefix, legacy_prefix, legacy_body, None)
        return reverse_legacy(state_dict, legacy_conversion, keys)
    raise RuntimeError(f"unrecognized LoRA namespace family: {family}")
