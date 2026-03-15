from modules.util.convert.lora.convert_clip import map_clip
from modules.util.convert.lora.convert_llama import map_llama
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, convert_to_omi, map_prefix_range

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
# (e.g. .linear1.0., .img_attn_qkv.1.) that must be combined before saving
# in ComfyUI format, because ComfyUI applies LoRA to the combined weight.
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


def _combine_qkv(components: dict[int, dict[str, Tensor]]) -> dict[str, Tensor]:
    """Combine split Q/K/V/proj_mlp LoRA adapters into one block-diagonal LoRA.

    OT trains separate adapters for each slice of a combined weight (Q, K, V,
    and optionally proj_mlp for single blocks). ComfyUI applies one LoRA to the
    combined weight. This function produces an exact lossless equivalent by:
      - block-diagonal lora_up  (shape [sum(out_i), sum(r_i)])
      - concatenated scaled lora_down (shape [sum(r_i), in])
      - alpha = sum(r_i)  so  alpha/rank = 1.0 (scaling baked into down)
    """
    sorted_indices = sorted(components.keys())

    downs: list[tuple[Tensor, float]] = []
    ups: list[Tensor] = []
    dora_scales: list[Tensor] = []

    for i in sorted_indices:
        comp = components[i]
        down = comp["lora_down.weight"]
        up = comp["lora_up.weight"]
        rank = down.shape[0]
        alpha_val = comp["alpha"].item() if "alpha" in comp else float(rank)
        downs.append((down, alpha_val / rank))
        ups.append(up)
        if "dora_scale" in comp:
            dora_scales.append(comp["dora_scale"])

    total_rank = sum(d.shape[0] for d, _ in downs)

    # Vertically concatenate scaled lora_down blocks
    combined_down = torch.cat([d * s for d, s in downs], dim=0)

    # Block-diagonal lora_up
    out_dim = sum(u.shape[0] for u in ups)
    combined_up = torch.zeros(out_dim, total_rank, dtype=ups[0].dtype, device=ups[0].device)
    row, col = 0, 0
    for up, (down, _) in zip(ups, downs, strict=True):
        r = down.shape[0]
        combined_up[row:row + up.shape[0], col:col + r] = up
        row += up.shape[0]
        col += r

    # Use lora_A/lora_B naming (ComfyUI diffusers-pipe convention) with no alpha —
    # scaling is already baked into lora_A so ComfyUI applies the delta at scale 1.0.
    result: dict[str, Tensor] = {
        "lora_A.weight": combined_down,
        "lora_B.weight": combined_up,
    }
    if dora_scales:
        result["dora_scale"] = torch.cat(dora_scales, dim=0)

    return result


def convert_hunyuan_video_lora_to_comfyui(
        state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Convert an OT HunyuanVideo LoRA state dict to ComfyUI-compatible format.

    ComfyUI uses native HunyuanVideo weight names (OMI paths) for the transformer
    and legacy underscore format for CLIP-L. Split Q/K/V adapters are merged into
    a single block-diagonal adapter matching ComfyUI's combined weight layout.
    """
    # Step 1: normalize to OMI (native HunyuanVideo attribute paths)
    omi = convert_to_omi(state_dict, convert_hunyuan_video_lora_key_sets())

    # Step 2: separate split QKV keys (need combining) from everything else
    # qkv_groups[base_key][component_idx][lora_suffix] = tensor
    qkv_groups: dict[str, dict[int, dict[str, Tensor]]] = {}
    passthrough: dict[str, Tensor] = {}

    for k, v in omi.items():
        if not k.startswith("transformer."):
            passthrough[k] = v
            continue

        matched = False
        for attr in _COMFYUI_QKV_SPLIT_ATTRS:
            pattern = f".{attr}."
            if pattern not in k:
                continue
            p = k.index(pattern)
            idx_start = p + len(pattern)
            dot_pos = k.index(".", idx_start)
            base_key = k[:p + len(pattern) - 1]  # up to and including attr name
            component_idx = int(k[idx_start:dot_pos])
            suffix = k[dot_pos + 1:]
            qkv_groups.setdefault(base_key, {}).setdefault(component_idx, {})[suffix] = v
            matched = True
            break

        if not matched:
            passthrough[k] = v

    result: dict[str, Tensor] = {}

    # Step 3: combine QKV groups
    for base_key, components in qkv_groups.items():
        combined = _combine_qkv(components)
        for suffix, tensor in combined.items():
            result[f"{base_key}.{suffix}"] = tensor

    # Step 4: remap passthrough keys to ComfyUI naming
    for k, v in passthrough.items():
        if k.startswith("transformer."):
            # ComfyUI normalizes mlp.0/2 sequential indices to fc1/fc2 in its key_map
            k = k.replace(".mlp.0.", ".mlp.fc1.")
            k = k.replace(".mlp.2.", ".mlp.fc2.")
            # OT OMI uses fc0 for first MLP linear; ComfyUI key_map uses fc1
            k = k.replace(".fc0.", ".fc1.")
            result[k] = v
        elif k.startswith("clip_l."):
            # ComfyUI expects legacy underscore format under lora_te1_ prefix
            new_k = _remap_to_legacy(k, "clip_l.", "lora_te1")
            if new_k:
                result[new_k] = v
        elif k.startswith("llama."):
            # No explicit ComfyUI support; use lora_llama_ to avoid collision
            new_k = _remap_to_legacy(k, "llama.", "lora_llama")
            if new_k:
                result[new_k] = v
        else:
            result[k] = v  # bundle_emb and anything else — pass through

    return result
