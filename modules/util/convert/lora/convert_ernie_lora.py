from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range


def convert_ernie_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    root = LoraConversionKeySet(
        "transformer",
        "lora_unet",
    )

    for layer in map_prefix_range("layers", "layers", parent=root):
        keys += [LoraConversionKeySet("mlp.gate_proj", "mlp.gate_proj", parent=layer)]
        keys += [LoraConversionKeySet("mlp.linear_fc2", "mlp.linear_fc2", parent=layer)]
        keys += [LoraConversionKeySet("mlp.up_proj", "mlp.up_proj", parent=layer)]
        keys += [LoraConversionKeySet("self_attention.to_k", "self_attention.to_k", parent=layer)]
        keys += [LoraConversionKeySet("self_attention.to_out.0", "self_attention.to_out.0", parent=layer)]
        keys += [LoraConversionKeySet("self_attention.to_q", "self_attention.to_q", parent=layer)]
        keys += [LoraConversionKeySet("self_attention.to_v", "self_attention.to_v", parent=layer)]

    return keys