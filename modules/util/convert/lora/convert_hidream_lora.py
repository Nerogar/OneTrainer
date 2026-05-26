from modules.util.convert.lora.convert_clip import map_clip
from modules.util.convert.lora.convert_llama import map_causal_llama
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range
from modules.util.convert.lora.convert_t5 import map_t5


def __map_caption_projection_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("linear", "linear", parent=key_prefix)]

    return keys


def __map_double_stream_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("block.adaLN_modulation.1", "block.adaLN_modulation.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_k", "block.attn1.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_k_t", "block.attn1.to_k_t", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_out", "block.attn1.to_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_out_t", "block.attn1.to_out_t", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_q", "block.attn1.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_q_t", "block.attn1.to_q_t", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_v", "block.attn1.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_v_t", "block.attn1.to_v_t", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.0.w1", "block.ff_i.experts.0.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.0.w2", "block.ff_i.experts.0.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.0.w3", "block.ff_i.experts.0.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.1.w1", "block.ff_i.experts.1.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.1.w2", "block.ff_i.experts.1.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.1.w3", "block.ff_i.experts.1.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.2.w1", "block.ff_i.experts.2.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.2.w2", "block.ff_i.experts.2.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.2.w3", "block.ff_i.experts.2.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.3.w1", "block.ff_i.experts.3.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.3.w2", "block.ff_i.experts.3.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.3.w3", "block.ff_i.experts.3.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.shared_experts.w1", "block.ff_i.shared_experts.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.shared_experts.w2", "block.ff_i.shared_experts.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.shared_experts.w3", "block.ff_i.shared_experts.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_t.w1", "block.ff_t.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_t.w2", "block.ff_t.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_t.w3", "block.ff_t.w3", parent=key_prefix)]

    return keys


def __map_single_stream_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("block.adaLN_modulation.1", "block.adaLN_modulation.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_k", "block.attn1.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_out", "block.attn1.to_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_q", "block.attn1.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.attn1.to_v", "block.attn1.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.0.w1", "block.ff_i.experts.0.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.0.w2", "block.ff_i.experts.0.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.0.w3", "block.ff_i.experts.0.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.1.w1", "block.ff_i.experts.1.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.1.w2", "block.ff_i.experts.1.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.1.w3", "block.ff_i.experts.1.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.2.w1", "block.ff_i.experts.2.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.2.w2", "block.ff_i.experts.2.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.2.w3", "block.ff_i.experts.2.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.3.w1", "block.ff_i.experts.3.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.3.w2", "block.ff_i.experts.3.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.experts.3.w3", "block.ff_i.experts.3.w3", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.shared_experts.w1", "block.ff_i.shared_experts.w1", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.shared_experts.w2", "block.ff_i.shared_experts.w2", parent=key_prefix)]
    keys += [LoraConversionKeySet("block.ff_i.shared_experts.w3", "block.ff_i.shared_experts.w3", parent=key_prefix)]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("final_layer.adaLN_modulation.1", "final_layer.adaLN_modulation.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("final_layer.linear", "final_layer.linear", parent=key_prefix)]
    keys += [LoraConversionKeySet("p_embedder.pooled_embedder.linear_1", "p_embedder.pooled_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("p_embedder.pooled_embedder.linear_2", "p_embedder.pooled_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("t_embedder.timestep_embedder.linear_1", "t_embedder.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("t_embedder.timestep_embedder.linear_2", "t_embedder.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("x_embedder.proj", "x_embedder.proj", parent=key_prefix)]

    for k in map_prefix_range("caption_projection", "caption_projection", parent=key_prefix):
        keys += __map_caption_projection_block(k)

    for k in map_prefix_range("double_stream_blocks", "double_stream_blocks", parent=key_prefix):
        keys += __map_double_stream_block(k)

    for k in map_prefix_range("single_stream_blocks", "single_stream_blocks", parent=key_prefix):
        keys += __map_single_stream_block(k)

    return keys


def convert_hidream_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("transformer", "lora_transformer"))
    keys += map_clip(LoraConversionKeySet("clip_l", "lora_te1"))
    keys += map_clip(LoraConversionKeySet("clip_g", "lora_te2"))
    keys += map_t5(LoraConversionKeySet("t5", "lora_te3"))
    keys += map_causal_llama(LoraConversionKeySet("llama", "lora_te4"))

    return keys
