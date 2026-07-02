from modules.util.convert_util import (
    component_body_conversion,
    convert,
    dora_scale_to_peft_magnitude,
    lora_fold_alpha,
)
from modules.util.enum.ModelFormat import ModelFormat

from torch import Tensor

# Save-side LoRA output-format helpers. The canonical in-memory namespace (a
# LoRAModuleWrapper.state_dict(): diffusers-dotted module paths, native lora_down/lora_up suffix) is
# rewritten here into each on-disk output format -- KOHYA flatten, the mixture conversion, the model-derived KOHYA/ORIGINAL
# conversion maps, the lora_A/lora_B suffix rename, and the DIFFUSERS format. The load side lives in
# load_lora_util.py (normalize_various + the per-family reverse).


def kohya_flatten(
        state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    # flatten the module path ('.' -> '_'), keeping the value suffix dotted. The suffix is one
    # of: a bare single-segment param -- .alpha / .dora_scale, or LoHa/LoKr's hada_*/lokr_*
    # factors (.hada_w1_a / .lokr_w1 / .lokr_w2_a, which carry no .weight); or .<param>.weight
    # (two segments, e.g. .lora_down.weight / .oft_R.weight). Keys without a recognized value
    # suffix (bundle_emb.*) carry no module path to flatten and pass through. The flatten is a
    # separate mechanical pass because the convert tool cannot collapse arbitrary-depth dots.
    out_states = {}
    for key, tensor in state_dict.items():
        if key.endswith(('.alpha', '.dora_scale')) or key.rsplit('.', 1)[-1].startswith(('hada_', 'lokr_')):
            suffix = key[key.rfind('.'):]
        elif key.endswith('.weight'):
            suffix = key[key.removesuffix(key[key.rfind('.'):]).rfind('.'):]
        else:
            out_states[key] = tensor
            continue

        module = key.removesuffix(suffix)
        out_states[module.replace('.', '_') + suffix] = tensor

    return out_states


def lora_kohya_conversion(model) -> list:
    # canonical -> KOHYA conversion list (feeds _convert_kohya, before kohya_flatten), derived from the model.
    # The denoising model is the universal lora_unet (kohya-ss's LORA_PREFIX_UNET regardless of architecture)
    # carrying its KOHYA body (model.lora_diffusers_to_kohya() -- defaults to the shared body, None for SD whose
    # KOHYA keeps diffusers UNet names); each text encoder is a pure prefix rename to its KOHYA name. A text
    # encoder is a (live module, {ModelFormat: name} dict) tuple (model.lora_text_encoders()); its in-memory
    # source key is its DIFFUSERS_LORA name (canonical). bundle_emb (bundled embeddings live in the text
    # encoders) passes through only when there are text encoders.
    canonical = model.model_type.denoising_model_part()
    conversion = model.lora_diffusers_to_kohya()
    text_encoders = model.lora_text_encoders()
    extra = [(te_names[ModelFormat.DIFFUSERS_LORA], te_names[ModelFormat.KOHYA_LORA])
             for _te_module, te_names in text_encoders]
    if text_encoders:
        extra.append(("bundle_emb", "bundle_emb"))
    return component_body_conversion(canonical, "lora_unet", conversion, extra)


def lora_original_conversion(model, conversion) -> list:
    # canonical -> ORIGINAL/COMFY conversion list, derived from the model and the format's denoising
    # `conversion` (model.lora_diffusers_to_original() for ORIGINAL, model.lora_diffusers_to_comfy() for COMFY
    # -- the caller passes the right one). The denoising model keeps its canonical top name (the mixin strips it
    # for ORIGINAL / swaps it for "diffusion_model." for COMFY) and carries `conversion`; each text encoder
    # keeps its canonical name (see the identity passthrough below; TE handling is identical for both formats).
    # bundle_emb passes through only when there are text encoders.
    canonical = model.model_type.denoising_model_part()
    text_encoders = model.lora_text_encoders()
    # text encoders keep their canonical name in the native namespace -- an identity passthrough. The bare
    # native TE name is never persisted (ORIGINAL files are denoising-only; COMFY renames the TE to its
    # text_encoders.* prefix in the next pass, keyed by the canonical name), so there is no separate native
    # name to rename to; this entry only lets the strict convert carry the TE keys through.
    extra = [(te_names[ModelFormat.DIFFUSERS_LORA], te_names[ModelFormat.DIFFUSERS_LORA])
             for _te_module, te_names in text_encoders]
    if text_encoders:
        extra.append(("bundle_emb", "bundle_emb"))
    return component_body_conversion(canonical, canonical, conversion, extra)


def convert_to_mixture(
        state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    # Emits the "mixture" format: the canonical (diffusers) component prefix renamed
    # kohya-style (lora_transformer/lora_unet/lora_te*) and the module body flattened, split qkv kept.
    # Despite the kohya-style prefixes this is NOT a faithful kohya-ss file for fused-qkv models (split
    # body, and the denoiser prefix is lora_transformer not lora_unet) -- that is the KOHYA format. It is
    # the historical LoRA output older OneTrainer versions wrote for the models that call it. The map is
    # overdefined (a rule whose source component is absent never fires) and generic over any number of text
    # encoders via {i}; bundle_emb.* passes through so strict=True still flags an unexpected component.
    state_dict = convert(state_dict, [
        ("transformer", "lora_transformer"),
        ("unet", "lora_unet"),
        ("text_encoder", "lora_te1"),
        ("text_encoder_{i}", "lora_te{i}"),
        ("bundle_emb", "bundle_emb"),
    ], strict=True)
    return kohya_flatten(state_dict)


def convert_lora_suffix_ab(
        state_dict: dict[str, Tensor],
        peft_convention: bool,
) -> dict[str, Tensor]:
    # Rename OneTrainer's native lora_down/lora_up suffix to the diffusers lora_A/lora_B
    # convention. Shared by every A/B-suffix output format (DIFFUSERS, ORIGINAL, COMFY);
    # KOHYA keeps lora_down/lora_up and does not call this. This is a pure suffix/value pass,
    # applied uniformly to every module via {} wildcards -- the namespace is whatever the
    # caller already produced (canonical for DIFFUSERS, original/native for ORIGINAL/COMFY).
    #
    # peft_convention selects the full diffusers-PEFT convention vs the native one:
    #   peft_convention=True (DIFFUSERS): alpha is folded into lora_B and its .alpha key dropped;
    #     DoRA magnitude (dora_scale) -> peft's lora_magnitude_vector, reshaped to peft's per-output
    #     layout. diffusers' peft backend has no per-layer alpha slot; on load it sets lora_alpha == r,
    #     giving a scale of alpha/r == 1.0, so the trained delta (alpha/r * up @ down) is reproduced
    #     exactly. Lossless. The caller filters out the per-input dora_scale that peft can't read before
    #     this runs, so every dora_scale reaching here is the peft-compatible per-output one.
    #   peft_convention=False (ORIGINAL/COMFY): alpha and dora_scale are kept verbatim under the
    #     A/B names -- Comfy reads .alpha and dora_scale natively, so neither is touched.
    #
    # Only LoRA/DoRA modules are renamed. LoHa/LoKr/OFT have no lora_up sibling, so their
    # rules never fire; their hada_*/lokr_*/oft_* keys (and .alpha) pass through unchanged
    # (strict=False), as does bundle_emb.*.
    if peft_convention:
        conversions = [
            ("{}.dora_scale", "{}.lora_magnitude_vector.weight", dora_scale_to_peft_magnitude),
            ("{}.lora_down.weight", "{}.lora_A.weight"),
            (["{}.lora_up.weight", "{}.alpha"], ["{}.lora_B.weight"], lora_fold_alpha),
        ]
    else:
        conversions = [
            ("{}.lora_down.weight", "{}.lora_A.weight"),
            ("{}.lora_up.weight", "{}.lora_B.weight"),
        ]
    return convert(state_dict, conversions, strict=False)


def convert_to_diffusers(
        state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    # DIFFUSERS format: the convention HF diffusers' load_lora_weights reads (not the
    # peft library's native base_model.model. / adapter_config.json layout). The canonical
    # in-memory keys are diffusers-dotted module paths but carry OneTrainer's native
    # lora_down/lora_up suffix, so module names need no remap and no key_set is involved --
    # only the suffix/alpha convert to the diffusers lora_A/lora_B convention (peft_convention=True):
    #   - LoRA/DoRA: lora_down -> lora_A; lora_up -> lora_B with alpha folded in and the
    #     .alpha key dropped; the per-output DoRA magnitude (dora_scale) -> diffusers'
    #     lora_magnitude_vector (reshaped by dora_scale_to_peft_magnitude).
    #   - LoHa/LoKr/OFT: no lora_up sibling, so the fold rule never fires for their .alpha,
    #     which passes through. These can't be loaded by HF diffusers, so the kept .alpha is
    #     correct and we warn.
    #
    # peft reads only the per-output DoRA magnitude (1-D [out] / [1,out,1,1]); OneTrainer's default
    # per-input dora_scale ([1, in(,1,1)], shape[0]==1) is the wrong axis/shape and would hard-fail
    # the load, so split it out and keep it as dora_scale -- diffusers then loads the file as a plain
    # LoRA (dropping the magnitude), the same saved-but-unreadable handling as LoHa/LoKr/OFT. The
    # per-output ones left in state_dict are reshaped by the dora_scale conversion in convert_lora_suffix_ab.
    state_dict = dict(state_dict)
    per_input_dora = {k: state_dict.pop(k) for k in list(state_dict)
                      if k.endswith('.dora_scale') and state_dict[k].shape[0] == 1}
    out_states = convert_lora_suffix_ab(state_dict, peft_convention=True)
    out_states |= per_input_dora

    # A surviving .alpha after folding belongs to a LoHa/LoKr/OFT module.
    if any(k.endswith('.alpha') for k in out_states):
        print("Warning: the DIFFUSERS format can only be loaded by HF diffusers for LoRA and "
              "DoRA adapters. LoHa/LoKr/OFT adapters are saved but HF diffusers cannot load them.")

    # A surviving .dora_scale is a per-input DoRA magnitude that peft can't read (wrong axis): diffusers
    # loads the file as a plain LoRA and drops the magnitude. Converted (per-output) magnitudes live under
    # lora_magnitude_vector, which only HF diffusers reads -- ComfyUI loads those as plain LoRA too.
    if any(k.endswith('.dora_scale') for k in out_states):
        print("Warning: this DoRA adapter stores a per-input magnitude (dora_on_output is off), which HF "
              "diffusers cannot read. It is saved as dora_scale, so diffusers will load the file as a plain "
              "LoRA and drop the DoRA magnitude. Train with dora_on_output, or use the KOHYA/COMFY format.")
    if any(k.endswith('.lora_magnitude_vector.weight') for k in out_states):
        print("Warning: the DIFFUSERS format stores the DoRA magnitude as lora_magnitude_vector, which "
              "only HF diffusers reads. ComfyUI will load this file as a plain LoRA (the DoRA magnitude "
              "is dropped). Use the KOHYA or COMFY format to keep DoRA loadable in ComfyUI.")

    return out_states
