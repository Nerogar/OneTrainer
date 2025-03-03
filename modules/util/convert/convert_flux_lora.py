from modules.util.convert.convert_lora_util import LoraConversionKeySet, combine, map_prefix_range


def __map_double_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_attn.qkv.0"), combine(key_prefix.diffusers_prefix, "attn.to_q"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_attn.qkv.1"), combine(key_prefix.diffusers_prefix, "attn.to_k"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_attn.qkv.2"), combine(key_prefix.diffusers_prefix, "attn.to_v"))]

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_attn.qkv.0"), combine(key_prefix.diffusers_prefix, "attn.add_q_proj"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_attn.qkv.1"), combine(key_prefix.diffusers_prefix, "attn.add_k_proj"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_attn.qkv.2"), combine(key_prefix.diffusers_prefix, "attn.add_v_proj"))]

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_attn.proj"), combine(key_prefix.diffusers_prefix, "attn.to_out.0"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_mlp.0"), combine(key_prefix.diffusers_prefix, "ff.net.0.proj"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_mlp.2"), combine(key_prefix.diffusers_prefix, "ff.net.2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_mod.lin"), combine(key_prefix.diffusers_prefix, "norm1.linear"))]

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_attn.proj"), combine(key_prefix.diffusers_prefix, "attn.to_add_out"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_mlp.0"), combine(key_prefix.diffusers_prefix, "ff_context.net.0.proj"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_mlp.2"), combine(key_prefix.diffusers_prefix, "ff_context.net.2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_mod.lin"), combine(key_prefix.diffusers_prefix, "norm1_context.linear"))]

    return keys


def __map_single_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "linear1.0"), combine(key_prefix.diffusers_prefix, "attn.to_q"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "linear1.1"), combine(key_prefix.diffusers_prefix, "attn.to_k"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "linear1.2"), combine(key_prefix.diffusers_prefix, "attn.to_v"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "linear1.3"), combine(key_prefix.diffusers_prefix, "proj_mlp"))]

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "linear2"), combine(key_prefix.diffusers_prefix, "proj_out"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "modulation.lin"), combine(key_prefix.diffusers_prefix, "norm.linear"))]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "txt_in"), combine(key_prefix.diffusers_prefix, "context_embedder"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "final_layer.adaLN_modulation.1"), combine(key_prefix.diffusers_prefix, "norm_out.linear"), swap_chunks=True)]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "final_layer.linear"), combine(key_prefix.diffusers_prefix, "proj_out"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "guidance_in.in_layer"), combine(key_prefix.diffusers_prefix, "time_text_embed.guidance_embedder.linear_1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "guidance_in.out_layer"), combine(key_prefix.diffusers_prefix, "time_text_embed.guidance_embedder.linear_2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "vector_in.in_layer"), combine(key_prefix.diffusers_prefix, "time_text_embed.text_embedder.linear_1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "vector_in.out_layer"), combine(key_prefix.diffusers_prefix, "time_text_embed.text_embedder.linear_2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "time_in.in_layer"), combine(key_prefix.diffusers_prefix, "time_text_embed.timestep_embedder.linear_1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "time_in.out_layer"), combine(key_prefix.diffusers_prefix, "time_text_embed.timestep_embedder.linear_2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "img_in.proj"), combine(key_prefix.diffusers_prefix, "x_embedder"))]

    for k in map_prefix_range(combine(key_prefix.omi_prefix, "double_blocks"), combine(key_prefix.diffusers_prefix, "transformer_blocks")):
        keys += __map_double_transformer_block(k)

    for k in map_prefix_range(combine(key_prefix.omi_prefix, "single_blocks"), combine(key_prefix.diffusers_prefix, "single_transformer_blocks")):
        keys += __map_single_transformer_block(k)

    return keys


def convert_flux_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("lora_transformer", "lora_transformer"))

    return keys
