from modules.util.convert.convert_lora_util import LoraConversionKeySet


def __map_unet_resnet_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("emb_layers.1", "time_emb_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("in_layers.2", "conv1", parent=key_prefix)]
    keys += [LoraConversionKeySet("out_layers.3", "conv2", parent=key_prefix)]
    keys += [LoraConversionKeySet("skip_connection", "conv_shortcut", parent=key_prefix)]

    return keys


def __map_unet_down_blocks(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet("1.0", "0.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("1.1", "0.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("2.0", "0.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("2.1", "0.attentions.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("3.0.op", "0.downsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("4.0", "1.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("4.1", "1.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("5.0", "1.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("5.1", "1.attentions.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("6.0.op", "1.downsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("7.0", "2.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("7.1", "2.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("8.0", "2.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("8.1", "2.attentions.1", parent=key_prefix)]
    keys += [LoraConversionKeySet("9.0.op", "2.downsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("10.0", "3.resnets.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("11.0", "3.resnets.1", parent=key_prefix))

    return keys


def __map_unet_mid_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet("0", "resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("1", "attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("2", "resnets.1", parent=key_prefix))

    return keys


def __map_unet_up_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet("0.0", "0.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("0.1", "0.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("1.0", "0.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("1.1", "0.attentions.1", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("2.0", "0.resnets.2", parent=key_prefix))
    keys += [LoraConversionKeySet("2.1", "0.attentions.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("2.2.conv", "0.upsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("3.0", "1.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("3.1", "1.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("4.0", "1.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("4.1", "1.attentions.1", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("5.0", "1.resnets.2", parent=key_prefix))
    keys += [LoraConversionKeySet("5.1", "1.attentions.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("5.2.conv", "1.upsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("6.0", "2.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("6.1", "2.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("7.0", "2.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("7.1", "2.attentions.1", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("8.0", "2.resnets.2", parent=key_prefix))
    keys += [LoraConversionKeySet("8.1", "2.attentions.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("8.2.conv", "2.upsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("9.0", "3.resnets.0", parent=key_prefix))
    keys += [LoraConversionKeySet("9.1", "3.attentions.0", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("10.0", "3.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("10.1", "3.attentions.1", parent=key_prefix)]
    keys += __map_unet_resnet_block(LoraConversionKeySet("11.0", "3.resnets.2", parent=key_prefix))
    keys += [LoraConversionKeySet("11.1", "3.attentions.2", parent=key_prefix)]

    return keys


def __map_unet(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("input_blocks.0.0", "conv_in", parent=key_prefix)]

    keys += [LoraConversionKeySet("time_embed.0", "time_embedding.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_embed.2", "time_embedding.linear_2", parent=key_prefix)]

    keys += __map_unet_down_blocks(LoraConversionKeySet("input_blocks", "down_blocks", parent=key_prefix))
    keys += __map_unet_mid_block(LoraConversionKeySet("middle_block", "mid_block", parent=key_prefix))
    keys += __map_unet_up_block(LoraConversionKeySet("output_blocks", "up_blocks", parent=key_prefix))

    keys += [LoraConversionKeySet("out.0", "conv_norm_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("out.2", "conv_out", parent=key_prefix)]

    return keys


def __map_text_encoder(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("", "", parent=key_prefix)]

    return keys


def convert_sd_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_unet(LoraConversionKeySet( "unet", "lora_unet"))
    keys += __map_text_encoder(LoraConversionKeySet("clip_l", "lora_te"))

    return keys
