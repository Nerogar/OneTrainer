from modules.util.convert.convert_lora_util import LoraConversionKeySet, map_prefix_range


def __map_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("x_block.attn.qkv.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("x_block.attn.qkv.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("x_block.attn.qkv.2", "attn.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("context_block.attn.qkv.0", "attn.add_q_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("context_block.attn.qkv.1", "attn.add_k_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("context_block.attn.qkv.2", "attn.add_v_proj", parent=key_prefix)]

    keys += [LoraConversionKeySet("x_block.attn.proj", "attn.to_out.0", parent=key_prefix)]

    keys += [LoraConversionKeySet("context_block.attn.proj", "attn.to_add_out", parent=key_prefix)]

    keys += [LoraConversionKeySet("x_block.adaLN_modulation.1", "norm1.linear", parent=key_prefix)]

    keys += [LoraConversionKeySet("context_block.adaLN_modulation.1", "norm1_context.linear", parent=key_prefix, filter_is_last=False)]
    keys += [LoraConversionKeySet("context_block.adaLN_modulation.1", "norm1_context.linear", parent=key_prefix, swap_chunks=True, filter_is_last=False)]

    keys += [LoraConversionKeySet("x_block.mlp.fc1", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("x_block.mlp.fc2", "ff.net.2", parent=key_prefix)]

    keys += [LoraConversionKeySet("x_block.attn2.qkv.0", "attn2.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("x_block.attn2.qkv.1", "attn2.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("x_block.attn2.qkv.2", "attn2.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("x_block.attn2.proj", "attn2.to_out.0", parent=key_prefix)]

    keys += [LoraConversionKeySet("context_block.mlp.fc1", "ff_context.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("context_block.mlp.fc2", "ff_context.net.2", parent=key_prefix)]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("x_embedder.proj", "pos_embed.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("context_embedder", "context_embedder", parent=key_prefix)]
    keys += [LoraConversionKeySet("final_layer.adaLN_modulation.1", "norm_out.linear", parent=key_prefix, swap_chunks=True)]
    keys += [LoraConversionKeySet("final_layer.linear", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("y_embedder.mlp.0", "time_text_embed.text_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("y_embedder.mlp.2", "time_text_embed.text_embedder.linear_2", parent=key_prefix)]

    for k in map_prefix_range("joint_blocks", "transformer_blocks", parent=key_prefix):
        keys += __map_transformer_block(k)

    return keys


def convert_sd3_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("transformer", "lora_transformer"))
    keys += [LoraConversionKeySet("clip_l", "lora_te1")]
    keys += [LoraConversionKeySet("clip_g", "lora_te2")]
    keys += [LoraConversionKeySet("t5", "lora_te3")]

    return keys
