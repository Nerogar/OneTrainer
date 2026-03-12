from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range


def map_llama(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    for k in map_prefix_range("language_model.model.layers", "language_model.model.layers", parent=key_prefix):
        keys += [LoraConversionKeySet("mlp.down_proj", "mlp.down_proj", parent=k)]
        keys += [LoraConversionKeySet("mlp.gate_proj", "mlp.gate_proj", parent=k)]
        keys += [LoraConversionKeySet("mlp.up_proj", "mlp.up_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.k_proj", "self_attn.k_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.o_proj", "self_attn.o_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.q_proj", "self_attn.q_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.v_proj", "self_attn.v_proj", parent=k)]

    return keys


def map_causal_llama(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("lm_head", "lm_head", parent=key_prefix)]
    for k in map_prefix_range("model.layers", "model.layers", parent=key_prefix):
        keys += [LoraConversionKeySet("mlp.down_proj", "mlp.down_proj", parent=k)]
        keys += [LoraConversionKeySet("mlp.gate_proj", "mlp.gate_proj", parent=k)]
        keys += [LoraConversionKeySet("mlp.up_proj", "mlp.up_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.k_proj", "self_attn.k_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.o_proj", "self_attn.o_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.q_proj", "self_attn.q_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.v_proj", "self_attn.v_proj", parent=k)]

    return keys
