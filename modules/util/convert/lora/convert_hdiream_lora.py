from modules.util.convert.convert_lora_util import LoraConversionKeySet


def convert_hidream_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += [LoraConversionKeySet("transformer", "lora_transformer")]
    keys += [LoraConversionKeySet("clip_l", "lora_te1")]
    keys += [LoraConversionKeySet("clip_g", "lora_te2")]
    keys += [LoraConversionKeySet("t5", "lora_te3")]
    keys += [LoraConversionKeySet("llama", "lora_te4")]

    return keys
