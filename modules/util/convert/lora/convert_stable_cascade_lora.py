from modules.util.convert.lora.convert_clip import map_clip
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range


def __map_unet_blocks(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    for k in map_prefix_range("", "", parent=key_prefix):
        keys += [LoraConversionKeySet("0.1", "0.attentions.0", parent=k)]

        # resblock
        keys += [LoraConversionKeySet("channelwise.0", "channelwise.0", parent=k)]
        keys += [LoraConversionKeySet("channelwise.4", "channelwise.4", parent=k)]
        keys += [LoraConversionKeySet("depthwise", "depthwise", parent=k)]

        # timestep block
        keys += [LoraConversionKeySet("mapper", "mapper", parent=k)]
        keys += [LoraConversionKeySet("mapper_crp", "mapper_crp", parent=k)]
        keys += [LoraConversionKeySet("mapper_sca", "mapper_sca", parent=k)]

        # attention block
        keys += [LoraConversionKeySet("kv_mapper.1", "kv_mapper.1", parent=k)]
        keys += [LoraConversionKeySet("attention.attn.out_proj", "attention.to_out.0", legacy_diffusers_prefix="attention_attn_out_proj", parent=k)]
        keys += [LoraConversionKeySet("attention.attn.in_proj.0", "attention.to_q", legacy_diffusers_prefix="attention_attn_to_q", parent=k)]
        keys += [LoraConversionKeySet("attention.attn.in_proj.1", "attention.to_k", legacy_diffusers_prefix="attention_attn_to_k", parent=k)]
        keys += [LoraConversionKeySet("attention.attn.in_proj.2", "attention.to_v", legacy_diffusers_prefix="attention_attn_to_v", parent=k)]

    return keys


def __map_prior(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("clf.1", "clf.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("clip_img_mapper", "clip_img_mapper", parent=key_prefix)]
    keys += [LoraConversionKeySet("clip_txt_mapper", "clip_txt_mapper", parent=key_prefix)]
    keys += [LoraConversionKeySet("clip_txt_pooled_mapper", "clip_txt_pooled_mapper", parent=key_prefix)]
    keys += [LoraConversionKeySet("down_downscalers.1.1.blocks.0", "down_downscalers.1.1.blocks.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("embedding.1", "embedding.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("up_upscalers.0.1.blocks.1", "up_upscalers.0.1.blocks.1", parent=key_prefix)]

    keys += __map_unet_blocks(LoraConversionKeySet("down_blocks.0", "down_blocks.0", parent=key_prefix))
    keys += __map_unet_blocks(LoraConversionKeySet("down_blocks.1", "down_blocks.1", parent=key_prefix))
    keys += __map_unet_blocks(LoraConversionKeySet("up_blocks.0", "up_blocks.0", parent=key_prefix))
    keys += __map_unet_blocks(LoraConversionKeySet("up_blocks.1", "up_blocks.1", parent=key_prefix))

    return keys


def convert_stable_cascade_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_prior(LoraConversionKeySet( "unet", "lora_prior_unet"))
    keys += map_clip(LoraConversionKeySet("clip_g", "lora_prior_te"))

    return keys
