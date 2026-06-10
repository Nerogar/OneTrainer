from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range
from modules.util.convert.lora.convert_t5 import map_t5


def __map_transformer_attention_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("attn.qkv.0", "attn1.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("attn.qkv.1", "attn1.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("attn.qkv.2", "attn1.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("attn.proj", "attn1.to_out.0", parent=key_prefix)]

    keys += [LoraConversionKeySet("cross_attn.q_linear", "attn2.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.kv_linear.0", "attn2.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.kv_linear.1", "attn2.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.proj", "attn2.to_out.0", parent=key_prefix)]

    keys += [LoraConversionKeySet("mlp.fc1", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("mlp.fc2", "ff.net.2", parent=key_prefix)]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("ar_embedder.mlp.0", "adaln_single.emb.aspect_ratio_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("ar_embedder.mlp.2", "adaln_single.emb.aspect_ratio_embedder.linear_2", parent=key_prefix)]

    keys += [LoraConversionKeySet("csize_embedder.mlp.0", "adaln_single.emb.resolution_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("csize_embedder.mlp.2", "adaln_single.emb.resolution_embedder.linear_2", parent=key_prefix)]

    keys += [LoraConversionKeySet("y_embedder.y_proj.fc1", "caption_projection.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("y_embedder.y_proj.fc2", "caption_projection.linear_2", parent=key_prefix)]

    keys += [LoraConversionKeySet("x_embedder.proj", "pos_embed.proj", parent=key_prefix)]

    keys += [LoraConversionKeySet("t_embedder.mlp.0", "adaln_single.emb.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("t_embedder.mlp.2", "adaln_single.emb.timestep_embedder.linear_2", parent=key_prefix)]

    keys += [LoraConversionKeySet("t_block.1", "adaln_single.linear", parent=key_prefix)]

    for k in map_prefix_range("blocks", "transformer_blocks", parent=key_prefix):
        keys += __map_transformer_attention_block(k)

    keys += [LoraConversionKeySet("final_layer.linear", "proj_out", parent=key_prefix)]

    return keys


def convert_pixart_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("transformer", "lora_transformer"))
    keys += map_t5(LoraConversionKeySet("t5", "lora_te"))

    return keys
