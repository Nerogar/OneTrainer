from modules.util.convert.lora.convert_clip import map_clip
from modules.util.convert.lora.convert_llama import map_llama
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, convert_to_omi, map_prefix_range
from modules.util.convert_util import convert, lora_qkv_fusion, lora_qkv_mlp_fusion

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


# Conversion patterns from OT's OMI format to ComfyUI's native HunyuanVideo LoRA
# format ("Form 1": diffusion_model. prefix with the raw checkpoint attribute paths,
# e.g. double_blocks.0.img_attn.qkv). This is the format used by community/diffusion-pipe
# LoRAs and is accepted both by ComfyUI's generic loader and its HunyuanVideo-specific
# remapper (comfy/lora.py). Mirrors the approach of Flux2's diffusers_to_original, using
# the LoRA-aware fusion helpers to merge OT's split Q/K/V (and single-block Q/K/V/MLP)
# adapters into the single fused projections ComfyUI expects.
def _omi_to_comfyui_patterns() -> list:
    return [
        ("transformer.double_blocks.{i}", "diffusion_model.double_blocks.{i}",
            lora_qkv_fusion("img_attn_qkv.0", "img_attn_qkv.1", "img_attn_qkv.2", "img_attn.qkv") +
            lora_qkv_fusion("txt_attn_qkv.0", "txt_attn_qkv.1", "txt_attn_qkv.2", "txt_attn.qkv") + [
                ("img_attn_proj",  "img_attn.proj"),
                ("img_mlp.fc0",    "img_mlp.0"),
                ("img_mlp.fc2",    "img_mlp.2"),
                ("img_mod.linear", "img_mod.lin"),
                ("txt_attn_proj",  "txt_attn.proj"),
                ("txt_mlp.fc0",    "txt_mlp.0"),
                ("txt_mlp.fc2",    "txt_mlp.2"),
                ("txt_mod.linear", "txt_mod.lin"),
            ]),
        ("transformer.single_blocks.{i}", "diffusion_model.single_blocks.{i}",
            lora_qkv_mlp_fusion("linear1.0", "linear1.1", "linear1.2", "linear1.3", "linear1") + [
                ("linear2",           "linear2"),
                ("modulation.linear", "modulation.lin"),
            ]),
    ]


def convert_hunyuan_video_lora_to_comfyui(
        state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Convert an OT HunyuanVideo LoRA state dict to ComfyUI-compatible format.

    Only double_blocks and single_blocks are exported: conditioning/embedder layers
    are inference-setting dependent and are absent from reference ComfyUI HYV LoRAs.
    Split Q/K/V (and single-block Q/K/V/MLP) adapters are fused into a single
    block-diagonal adapter per layer via convert_util's lora_qkv(_mlp)_fusion.
    """
    # normalize whatever source format (legacy/diffusers/omi) to OMI native paths
    omi = convert_to_omi(state_dict, convert_hunyuan_video_lora_key_sets())

    dora_scales = [k for k in omi if k.endswith(".dora_scale")]
    if dora_scales:
        raise NotImplementedError(
            "ComfyUI HunyuanVideo LoRA export does not support DoRA "
            f"(found {len(dora_scales)} .dora_scale tensors)"
        )

    # keep only the transformer blocks ComfyUI loads; drop embedders/conditioning/text encoders
    blocks = {k: v for k, v in omi.items()
              if ".double_blocks." in k or ".single_blocks." in k}

    return convert(blocks, _omi_to_comfyui_patterns(), strict=True)
