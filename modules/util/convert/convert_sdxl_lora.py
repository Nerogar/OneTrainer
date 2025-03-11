from modules.util.convert.convert_lora_util import LoraConversionKeySet, combine


def __map_unet_resnet_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "emb_layers.1"), combine(key_prefix.diffusers_prefix, "time_emb_proj"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "in_layers.2"), combine(key_prefix.diffusers_prefix, "conv1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "out_layers.3"), combine(key_prefix.diffusers_prefix, "conv2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "skip_connection"), combine(key_prefix.diffusers_prefix, "conv_shortcut"))]

    return keys


def __map_unet_down_blocks(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "1.0"), combine(key_prefix.diffusers_prefix, "0.resnets.0")))
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "2.0"), combine(key_prefix.diffusers_prefix, "0.resnets.1")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "3.0.op"), combine(key_prefix.diffusers_prefix, "0.downsamplers.0.conv"))]

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "4.0"), combine(key_prefix.diffusers_prefix, "1.resnets.0")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "4.1"), combine(key_prefix.diffusers_prefix, "1.attentions.0"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "5.0"), combine(key_prefix.diffusers_prefix, "1.resnets.1")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "5.1"), combine(key_prefix.diffusers_prefix, "1.attentions.1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "6.0.op"), combine(key_prefix.diffusers_prefix, "1.downsamplers.0.conv"))]

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "7.0"), combine(key_prefix.diffusers_prefix, "2.resnets.0")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "7.1"), combine(key_prefix.diffusers_prefix, "2.attentions.0"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "8.0"), combine(key_prefix.diffusers_prefix, "2.resnets.1")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "8.1"), combine(key_prefix.diffusers_prefix, "2.attentions.1"))]

    return keys


def __map_unet_mid_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "0"), combine(key_prefix.diffusers_prefix, "resnets.0")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "1"), combine(key_prefix.diffusers_prefix, "attentions.0"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "2"), combine(key_prefix.diffusers_prefix, "resnets.1")))

    return keys


def __map_unet_up_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "0.0"), combine(key_prefix.diffusers_prefix, "0.resnets.0")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "0.1"), combine(key_prefix.diffusers_prefix, "0.attentions.0"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "1.0"), combine(key_prefix.diffusers_prefix, "0.resnets.1")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "1.1"), combine(key_prefix.diffusers_prefix, "0.attentions.1"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "2.0"), combine(key_prefix.diffusers_prefix, "0.resnets.2")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "2.1"), combine(key_prefix.diffusers_prefix, "0.attentions.2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "2.2.conv"), combine(key_prefix.diffusers_prefix, "0.upsamplers.0.conv"))]

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "3.0"), combine(key_prefix.diffusers_prefix, "1.resnets.0")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "3.1"), combine(key_prefix.diffusers_prefix, "1.attentions.0"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "4.0"), combine(key_prefix.diffusers_prefix, "1.resnets.1")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "4.1"), combine(key_prefix.diffusers_prefix, "1.attentions.1"))]
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "5.0"), combine(key_prefix.diffusers_prefix, "1.resnets.2")))
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "5.1"), combine(key_prefix.diffusers_prefix, "1.attentions.2"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "5.2.conv"), combine(key_prefix.diffusers_prefix, "1.upsamplers.0.conv"))]

    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "6.0"), combine(key_prefix.diffusers_prefix, "2.resnets.0")))
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "7.0"), combine(key_prefix.diffusers_prefix, "2.resnets.1")))
    keys += __map_unet_resnet_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "8.0"), combine(key_prefix.diffusers_prefix, "2.resnets.2")))

    return keys


def __map_unet(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "input_blocks.0.0"), combine(key_prefix.diffusers_prefix, "conv_in"))]

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "time_embed.0"), combine(key_prefix.diffusers_prefix, "time_embedding.linear_1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "time_embed.2"), combine(key_prefix.diffusers_prefix, "time_embedding.linear_2"))]

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "label_emb.0.0"), combine(key_prefix.diffusers_prefix, "add_embedding.linear_1"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "label_emb.0.2"), combine(key_prefix.diffusers_prefix, "add_embedding.linear_2"))]

    keys += __map_unet_down_blocks(LoraConversionKeySet(combine(key_prefix.omi_prefix, "input_blocks"), combine(key_prefix.diffusers_prefix, "down_blocks")))
    keys += __map_unet_mid_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "middle_block"), combine(key_prefix.diffusers_prefix, "mid_block")))
    keys += __map_unet_up_block(LoraConversionKeySet(combine(key_prefix.omi_prefix, "output_blocks"), combine(key_prefix.diffusers_prefix, "up_blocks")))

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "out.0"), combine(key_prefix.diffusers_prefix, "conv_norm_out"))]
    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, "out.2"), combine(key_prefix.diffusers_prefix, "conv_out"))]

    return keys


def __map_text_encoder_1(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, ""), combine(key_prefix.diffusers_prefix, ""))]

    return keys


def __map_text_encoder_2(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet(combine(key_prefix.omi_prefix, ""), combine(key_prefix.diffusers_prefix, ""))]

    return keys


def convert_sdxl_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_unet(LoraConversionKeySet( "lora_unet", "lora_unet"))
    keys += __map_text_encoder_1(LoraConversionKeySet("lora_clip_l", "lora_te1"))
    keys += __map_text_encoder_2(LoraConversionKeySet("lora_clip_g", "lora_te2"))

    return keys
