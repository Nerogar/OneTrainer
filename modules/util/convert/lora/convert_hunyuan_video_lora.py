from modules.util.convert.lora.convert_clip import map_clip
from modules.util.convert.lora.convert_llama import map_llama
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, convert_to_omi, map_prefix_range
from modules.util.convert_util import convert as convert_util
from modules.util.convert_util import lora_qkv_fusion, lora_qkv_mlp_fusion

import torch
from torch import Tensor


def __map_token_refiner_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("self_attn_qkv.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn_qkv.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn_qkv.2", "attn.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("self_attn_proj", "attn.to_out.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("mlp.fc0", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("mlp.fc2", "ff.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("adaLN_modulation.1", "norm_out.linear", parent=key_prefix)]
    keys += [LoraConversionKeySet("norm1", "norm1", parent=key_prefix)]
    keys += [LoraConversionKeySet("norm2", "norm2", parent=key_prefix)]

    return keys


def __map_double_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("img_attn_qkv.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn_qkv.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn_qkv.2", "attn.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("txt_attn_qkv.0", "attn.add_q_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn_qkv.1", "attn.add_k_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn_qkv.2", "attn.add_v_proj", parent=key_prefix)]

    keys += [LoraConversionKeySet("img_attn_proj", "attn.to_out.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.fc0", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.fc2", "ff.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mod.linear", "norm1.linear", parent=key_prefix)]

    keys += [LoraConversionKeySet("txt_attn_proj", "attn.to_add_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.fc0", "ff_context.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.fc2", "ff_context.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mod.linear", "norm1_context.linear", parent=key_prefix)]

    return keys


def __map_single_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("linear1.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.2", "attn.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.3", "proj_mlp", parent=key_prefix)]

    keys += [LoraConversionKeySet("linear2", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("modulation.linear", "norm.linear", parent=key_prefix)]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("txt_in.c_embedder.linear_1", "context_embedder.time_text_embed.text_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_in.c_embedder.linear_2", "context_embedder.time_text_embed.text_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_in.t_embedder.linear_1", "context_embedder.time_text_embed.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_in.t_embedder.linear_2", "context_embedder.time_text_embed.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_in.input_embedder", "context_embedder.proj_in", parent=key_prefix)]
    keys += [LoraConversionKeySet("final_layer.adaLN_modulation.1", "norm_out.linear", parent=key_prefix, swap_chunks=True)]
    keys += [LoraConversionKeySet("final_layer.linear", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("guidance_in.mlp.0", "time_text_embed.guidance_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("guidance_in.mlp.2", "time_text_embed.guidance_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("vector_in.in_layer", "time_text_embed.text_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("vector_in.out_layer", "time_text_embed.text_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.mlp.0", "time_text_embed.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.mlp.2", "time_text_embed.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_in.proj", "x_embedder.proj", parent=key_prefix)]

    for k in map_prefix_range("txt_in.individual_token_refiner.blocks", "context_embedder.token_refiner.refiner_blocks", parent=key_prefix):
        keys += __map_token_refiner_block(k)

    for k in map_prefix_range("double_blocks", "transformer_blocks", parent=key_prefix):
        keys += __map_double_transformer_block(k)

    for k in map_prefix_range("single_blocks", "single_transformer_blocks", parent=key_prefix):
        keys += __map_single_transformer_block(k)

    return keys


def convert_hunyuan_video_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("transformer", "lora_transformer"))
    keys += map_llama(LoraConversionKeySet("llama", "lora_te1"))
    keys += map_clip(LoraConversionKeySet("clip_l", "lora_te2"))

    return keys


# Attribute names in OT's OMI format whose keys carry a split-index component
# (e.g. .linear1.0., .img_attn_qkv.1.) — used to detect and merge DoRA scales
# for QKV-split layers after the main lora_qkv_fusion step.
_COMFYUI_QKV_SPLIT_ATTRS = ("linear1", "img_attn_qkv", "txt_attn_qkv", "self_attn_qkv")

_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight", ".alpha", ".dora_scale")


def _remap_to_legacy(key: str, old_prefix: str, new_prefix: str) -> str | None:
    """Convert a dot-notation OMI key to ComfyUI legacy underscore format."""
    path = key[len(old_prefix):]
    for suffix in _LORA_SUFFIXES:
        if path.endswith(suffix):
            base = path[:-len(suffix)].replace(".", "_")
            return f"{new_prefix}_{base}{suffix}"
    return None


# Conversion patterns: OT OMI → ComfyUI native HunyuanVideo paths.
# lora_qkv_fusion fuses split Q/K/V adapters (requires equal alpha per group,
# which holds for OT since rank/alpha are set per-layer, not per-component).
# lora_qkv_mlp_fusion additionally fuses proj_mlp for single blocks.
_COMFYUI_BLOCK_PATTERNS = [
    ("transformer.double_blocks.{i}", "transformer.double_blocks.{i}",
     lora_qkv_fusion("img_attn_qkv.0", "img_attn_qkv.1", "img_attn_qkv.2", "img_attn_qkv") +
     lora_qkv_fusion("txt_attn_qkv.0", "txt_attn_qkv.1", "txt_attn_qkv.2", "txt_attn_qkv") + [
         ("img_attn_proj",  "img_attn_proj"),
         ("img_mlp.fc0",    "img_mlp.fc1"),
         ("img_mlp.fc2",    "img_mlp.fc2"),
         ("img_mod.linear", "img_mod.linear"),
         ("txt_attn_proj",  "txt_attn_proj"),
         ("txt_mlp.fc0",    "txt_mlp.fc1"),
         ("txt_mlp.fc2",    "txt_mlp.fc2"),
         ("txt_mod.linear", "txt_mod.linear"),
     ]),
    ("transformer.single_blocks.{i}", "transformer.single_blocks.{i}",
     lora_qkv_mlp_fusion("linear1.0", "linear1.1", "linear1.2", "linear1.3", "linear1") + [
         ("linear2",           "linear2"),
         ("modulation.linear", "modulation.linear"),
     ]),
]


def convert_hunyuan_video_lora_to_comfyui(
        state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Convert an OT HunyuanVideo LoRA state dict to ComfyUI-compatible format.

    Only double_blocks and single_blocks are exported (community convention;
    conditioning/embedding layers are inference-setting-dependent and absent
    from all reference ComfyUI HYV LoRAs). Split Q/K/V adapters are fused
    into a single block-diagonal adapter via convert_util.lora_qkv_fusion.
    """
    # Step 1: normalize to OMI (native HunyuanVideo attribute paths)
    omi = convert_to_omi(state_dict, convert_hunyuan_video_lora_key_sets())

    # Step 2: filter to double_blocks and single_blocks only
    blocks = {k: v for k, v in omi.items()
              if ".double_blocks." in k or ".single_blocks." in k}

    # Step 3: separate DoRA scales — lora_qkv_fusion only handles lora_up/lora_down/alpha
    dora_scales = {k: v for k, v in blocks.items() if k.endswith(".dora_scale")}
    main = {k: v for k, v in blocks.items() if not k.endswith(".dora_scale")}

    # Step 4: fuse split Q/K/V(/proj_mlp) adapters and rename paths via convert_util
    result: dict[str, Tensor] = convert_util(main, _COMFYUI_BLOCK_PATTERNS, strict=True)

    # Step 5: merge DoRA scales for QKV-split attrs; pass through others with path rename
    qkv_dora: dict[str, dict[int, Tensor]] = {}
    for k, v in dora_scales.items():
        matched = False
        for attr in _COMFYUI_QKV_SPLIT_ATTRS:
            pattern = f".{attr}."
            if pattern not in k:
                continue
            p = k.index(pattern)
            idx_start = p + len(pattern)
            dot_pos = k.index(".", idx_start)
            base_key = k[:p + len(pattern) - 1]
            component_idx = int(k[idx_start:dot_pos])
            qkv_dora.setdefault(base_key, {})[component_idx] = v
            matched = True
            break
        if not matched:
            k = k.replace(".img_mlp.fc0.", ".img_mlp.fc1.")
            k = k.replace(".txt_mlp.fc0.", ".txt_mlp.fc1.")
            result[k] = v

    for base_key, components in qkv_dora.items():
        result[f"{base_key}.dora_scale"] = torch.cat(
            [components[i] for i in sorted(components.keys())], dim=0,
        )

    # Step 6: text encoders — legacy underscore format (convert_util doesn't handle dot→underscore)
    for k, v in omi.items():
        if k.startswith("clip_l."):
            new_k = _remap_to_legacy(k, "clip_l.", "lora_te1")
            if new_k:
                result[new_k] = v
        elif k.startswith("llama."):
            new_k = _remap_to_legacy(k, "llama.", "lora_llama")
            if new_k:
                result[new_k] = v

    return result
