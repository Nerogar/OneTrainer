import torch
from diffusers import DDIMScheduler
from torch import Tensor

from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.ModelType import ModelType


def __combine(left: str, right: str) -> str:
    if left == "":
        return right
    elif right == "":
        return left
    else:
        return left + "." + right


def __map_wb(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[__combine(out_prefix, "weight")] = in_states[__combine(in_prefix, "weight")]
    out_states[__combine(out_prefix, "bias")] = in_states[__combine(in_prefix, "bias")]

    return out_states


def __map_vae_resnet_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_wb(in_states, __combine(out_prefix, "norm1"), __combine(in_prefix, "norm1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "conv1"), __combine(in_prefix, "conv1"))

    out_states |= __map_wb(in_states, __combine(out_prefix, "norm2"), __combine(in_prefix, "norm2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "conv2"), __combine(in_prefix, "conv2"))

    if __combine(in_prefix, "conv_shortcut.weight") in in_states:
        out_states[__combine(out_prefix, "nin_shortcut.weight")] = in_states[__combine(in_prefix, "conv_shortcut.weight")]
    if __combine(in_prefix, "conv_shortcut.bias") in in_states:
        out_states[__combine(out_prefix, "nin_shortcut.bias")] = in_states[__combine(in_prefix, "conv_shortcut.bias")]

    return out_states


def __reshape_vae_attention_weight(weight: Tensor) -> Tensor:
    return torch.reshape(weight, shape=[weight.shape[0], weight.shape[1], 1, 1])


def __map_vae_attention_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_wb(in_states, __combine(out_prefix, "norm"), __combine(in_prefix, "group_norm"))

    # support for both deprecated and current attention block names
    if __combine(in_prefix, "query") in in_states.keys():
        out_states |= __map_wb(in_states, __combine(out_prefix, "q"), __combine(in_prefix, "query"))  # deprecated
    else:
        out_states |= __map_wb(in_states, __combine(out_prefix, "q"), __combine(in_prefix, "to_q"))  # current

    if __combine(in_prefix, "key") in in_states.keys():
        out_states |= __map_wb(in_states, __combine(out_prefix, "k"), __combine(in_prefix, "key"))  # deprecated
    else:
        out_states |= __map_wb(in_states, __combine(out_prefix, "k"), __combine(in_prefix, "to_k"))  # current

    if __combine(in_prefix, "value") in in_states.keys():
        out_states |= __map_wb(in_states, __combine(out_prefix, "v"), __combine(in_prefix, "value"))  # deprecated
    else:
        out_states |= __map_wb(in_states, __combine(out_prefix, "v"), __combine(in_prefix, "to_v"))  # current

    if __combine(in_prefix, "proj_attn") in in_states.keys():
        out_states |= __map_wb(in_states, __combine(out_prefix, "proj_out"), __combine(in_prefix, "proj_attn"))  # deprecated
    else:
        out_states |= __map_wb(in_states, __combine(out_prefix, "proj_out"), __combine(in_prefix, "to_out.0"))  # current

    out_states[__combine(out_prefix, "q.weight")] = __reshape_vae_attention_weight(out_states[__combine(out_prefix, "q.weight")])
    out_states[__combine(out_prefix, "k.weight")] = __reshape_vae_attention_weight(out_states[__combine(out_prefix, "k.weight")])
    out_states[__combine(out_prefix, "v.weight")] = __reshape_vae_attention_weight(out_states[__combine(out_prefix, "v.weight")])
    out_states[__combine(out_prefix, "proj_out.weight")] = __reshape_vae_attention_weight(out_states[__combine(out_prefix, "proj_out.weight")])

    return out_states


def __map_vae_encoder(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[__combine(out_prefix, "conv_in.weight")] = in_states[__combine(in_prefix, "conv_in.weight")]
    out_states[__combine(out_prefix, "conv_in.bias")] = in_states[__combine(in_prefix, "conv_in.bias")]

    # down blocks
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.0.block.0"), __combine(in_prefix, "down_blocks.0.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.0.block.1"), __combine(in_prefix, "down_blocks.0.resnets.1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "down.0.downsample.conv"), __combine(in_prefix, "down_blocks.0.downsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.1.block.0"), __combine(in_prefix, "down_blocks.1.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.1.block.1"), __combine(in_prefix, "down_blocks.1.resnets.1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "down.1.downsample.conv"), __combine(in_prefix, "down_blocks.1.downsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.2.block.0"), __combine(in_prefix, "down_blocks.2.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.2.block.1"), __combine(in_prefix, "down_blocks.2.resnets.1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "down.2.downsample.conv"), __combine(in_prefix, "down_blocks.2.downsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.3.block.0"), __combine(in_prefix, "down_blocks.3.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "down.3.block.1"), __combine(in_prefix, "down_blocks.3.resnets.1"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "mid.block_1"), __combine(in_prefix, "mid_block.resnets.0"))
    out_states |= __map_vae_attention_block(in_states, __combine(out_prefix, "mid.attn_1"), __combine(in_prefix, "mid_block.attentions.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "mid.block_2"), __combine(in_prefix, "mid_block.resnets.1"))

    # conv out
    out_states[__combine(out_prefix, "norm_out.weight")] = in_states[__combine(in_prefix, "conv_norm_out.weight")]
    out_states[__combine(out_prefix, "norm_out.bias")] = in_states[__combine(in_prefix, "conv_norm_out.bias")]
    out_states[__combine(out_prefix, "conv_out.weight")] = in_states[__combine(in_prefix, "conv_out.weight")]
    out_states[__combine(out_prefix, "conv_out.bias")] = in_states[__combine(in_prefix, "conv_out.bias")]

    return out_states


def __map_vae_quant_conv(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[__combine(out_prefix, "weight")] = in_states[__combine(in_prefix, "weight")]
    out_states[__combine(out_prefix, "bias")] = in_states[__combine(in_prefix, "bias")]

    return out_states


def __map_vae_post_quant_conv(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[__combine(out_prefix, "weight")] = in_states[__combine(in_prefix, "weight")]
    out_states[__combine(out_prefix, "bias")] = in_states[__combine(in_prefix, "bias")]

    return out_states


def __map_vae_decoder(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[__combine(out_prefix, "conv_in.weight")] = in_states[__combine(in_prefix, "conv_in.weight")]
    out_states[__combine(out_prefix, "conv_in.bias")] = in_states[__combine(in_prefix, "conv_in.bias")]

    # down blocks
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "mid.block_1"), __combine(in_prefix, "mid_block.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "mid.block_2"), __combine(in_prefix, "mid_block.resnets.1"))
    out_states |= __map_vae_attention_block(in_states, __combine(out_prefix, "mid.attn_1"), __combine(in_prefix, "mid_block.attentions.0"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.3.block.0"), __combine(in_prefix, "up_blocks.0.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.3.block.1"), __combine(in_prefix, "up_blocks.0.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.3.block.2"), __combine(in_prefix, "up_blocks.0.resnets.2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "up.3.upsample.conv"), __combine(in_prefix, "up_blocks.0.upsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.2.block.0"), __combine(in_prefix, "up_blocks.1.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.2.block.1"), __combine(in_prefix, "up_blocks.1.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.2.block.2"), __combine(in_prefix, "up_blocks.1.resnets.2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "up.2.upsample.conv"), __combine(in_prefix, "up_blocks.1.upsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.1.block.0"), __combine(in_prefix, "up_blocks.2.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.1.block.1"), __combine(in_prefix, "up_blocks.2.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.1.block.2"), __combine(in_prefix, "up_blocks.2.resnets.2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "up.1.upsample.conv"), __combine(in_prefix, "up_blocks.2.upsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.0.block.0"), __combine(in_prefix, "up_blocks.3.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.0.block.1"), __combine(in_prefix, "up_blocks.3.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, __combine(out_prefix, "up.0.block.2"), __combine(in_prefix, "up_blocks.3.resnets.2"))

    # conv out
    out_states[__combine(out_prefix, "norm_out.weight")] = in_states[__combine(in_prefix, "conv_norm_out.weight")]
    out_states[__combine(out_prefix, "norm_out.bias")] = in_states[__combine(in_prefix, "conv_norm_out.bias")]
    out_states[__combine(out_prefix, "conv_out.weight")] = in_states[__combine(in_prefix, "conv_out.weight")]
    out_states[__combine(out_prefix, "conv_out.bias")] = in_states[__combine(in_prefix, "conv_out.bias")]

    return out_states


def __map_vae(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_vae_encoder(in_states, __combine(out_prefix, "encoder"), __combine(in_prefix, "encoder"))
    out_states |= __map_vae_quant_conv(in_states, __combine(out_prefix, "quant_conv"), __combine(in_prefix, "quant_conv"))
    out_states |= __map_vae_post_quant_conv(in_states, __combine(out_prefix, "post_quant_conv"), __combine(in_prefix, "post_quant_conv"))
    out_states |= __map_vae_decoder(in_states, __combine(out_prefix, "decoder"), __combine(in_prefix, "decoder"))

    return out_states


def __map_unet_resnet_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[__combine(out_prefix, "in_layers.0.weight")] = in_states[__combine(in_prefix, "norm1.weight")]
    out_states[__combine(out_prefix, "in_layers.0.bias")] = in_states[__combine(in_prefix, "norm1.bias")]
    out_states[__combine(out_prefix, "in_layers.2.weight")] = in_states[__combine(in_prefix, "conv1.weight")]
    out_states[__combine(out_prefix, "in_layers.2.bias")] = in_states[__combine(in_prefix, "conv1.bias")]

    out_states[__combine(out_prefix, "emb_layers.1.weight")] = in_states[__combine(in_prefix, "time_emb_proj.weight")]
    out_states[__combine(out_prefix, "emb_layers.1.bias")] = in_states[__combine(in_prefix, "time_emb_proj.bias")]

    out_states[__combine(out_prefix, "out_layers.0.weight")] = in_states[__combine(in_prefix, "norm2.weight")]
    out_states[__combine(out_prefix, "out_layers.0.bias")] = in_states[__combine(in_prefix, "norm2.bias")]
    out_states[__combine(out_prefix, "out_layers.3.weight")] = in_states[__combine(in_prefix, "conv2.weight")]
    out_states[__combine(out_prefix, "out_layers.3.bias")] = in_states[__combine(in_prefix, "conv2.bias")]

    if __combine(in_prefix, "conv_shortcut.weight") in in_states:
        out_states[__combine(out_prefix, "skip_connection.weight")] = in_states[__combine(in_prefix, "conv_shortcut.weight")]
    if __combine(in_prefix, "conv_shortcut.bias") in in_states:
        out_states[__combine(out_prefix, "skip_connection.bias")] = in_states[__combine(in_prefix, "conv_shortcut.bias")]

    return out_states


def __map_unet_attention_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[__combine(out_prefix, "norm.weight")] = in_states[__combine(in_prefix, "norm.weight")]
    out_states[__combine(out_prefix, "norm.bias")] = in_states[__combine(in_prefix, "norm.bias")]
    out_states[__combine(out_prefix, "proj_in.weight")] = in_states[__combine(in_prefix, "proj_in.weight")]
    out_states[__combine(out_prefix, "proj_in.bias")] = in_states[__combine(in_prefix, "proj_in.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn1.to_q.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn1.to_q.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn1.to_k.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn1.to_k.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn1.to_v.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn1.to_v.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn1.to_out.0.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn1.to_out.0.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn1.to_out.0.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn1.to_out.0.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.ff.net.0.proj.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.ff.net.0.proj.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.ff.net.0.proj.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.ff.net.0.proj.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.ff.net.2.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.ff.net.2.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.ff.net.2.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.ff.net.2.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn2.to_q.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn2.to_q.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn2.to_k.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn2.to_k.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn2.to_v.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn2.to_v.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn2.to_out.0.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn2.to_out.0.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.attn2.to_out.0.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.attn2.to_out.0.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.norm1.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.norm1.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.norm1.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.norm1.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.norm2.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.norm2.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.norm2.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.norm2.bias")]
    out_states[__combine(out_prefix, "transformer_blocks.0.norm3.weight")] = in_states[__combine(in_prefix, "transformer_blocks.0.norm3.weight")]
    out_states[__combine(out_prefix, "transformer_blocks.0.norm3.bias")] = in_states[__combine(in_prefix, "transformer_blocks.0.norm3.bias")]
    out_states[__combine(out_prefix, "proj_out.weight")] = in_states[__combine(in_prefix, "proj_out.weight")]
    out_states[__combine(out_prefix, "proj_out.bias")] = in_states[__combine(in_prefix, "proj_out.bias")]

    return out_states


def __map_unet_down_blocks(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "1.0"), __combine(in_prefix, "0.resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "1.1"), __combine(in_prefix, "0.attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "2.0"), __combine(in_prefix, "0.resnets.1"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "2.1"), __combine(in_prefix, "0.attentions.1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "3.0.op"), __combine(in_prefix, "0.downsamplers.0.conv"))

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "4.0"), __combine(in_prefix, "1.resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "4.1"), __combine(in_prefix, "1.attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "5.0"), __combine(in_prefix, "1.resnets.1"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "5.1"), __combine(in_prefix, "1.attentions.1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "6.0.op"), __combine(in_prefix, "1.downsamplers.0.conv"))

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "7.0"), __combine(in_prefix, "2.resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "7.1"), __combine(in_prefix, "2.attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "8.0"), __combine(in_prefix, "2.resnets.1"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "8.1"), __combine(in_prefix, "2.attentions.1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "9.0.op"), __combine(in_prefix, "2.downsamplers.0.conv"))

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "10.0"), __combine(in_prefix, "3.resnets.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "11.0"), __combine(in_prefix, "3.resnets.1"))

    return out_states


def __map_unet_mid_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "0"), __combine(in_prefix, "resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "1"), __combine(in_prefix, "attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "2"), __combine(in_prefix, "resnets.1"))

    return out_states


def __map_unet_up_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "0.0"), __combine(in_prefix, "0.resnets.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "1.0"), __combine(in_prefix, "0.resnets.1"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "2.0"), __combine(in_prefix, "0.resnets.2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "2.1.conv"), __combine(in_prefix, "0.upsamplers.0.conv"))

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "3.0"), __combine(in_prefix, "1.resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "3.1"), __combine(in_prefix, "1.attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "4.0"), __combine(in_prefix, "1.resnets.1"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "4.1"), __combine(in_prefix, "1.attentions.1"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "5.0"), __combine(in_prefix, "1.resnets.2"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "5.1"), __combine(in_prefix, "1.attentions.2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "5.2.conv"), __combine(in_prefix, "1.upsamplers.0.conv"))

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "6.0"), __combine(in_prefix, "2.resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "6.1"), __combine(in_prefix, "2.attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "7.0"), __combine(in_prefix, "2.resnets.1"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "7.1"), __combine(in_prefix, "2.attentions.1"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "8.0"), __combine(in_prefix, "2.resnets.2"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "8.1"), __combine(in_prefix, "2.attentions.2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "8.2.conv"), __combine(in_prefix, "2.upsamplers.0.conv"))

    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "9.0"), __combine(in_prefix, "3.resnets.0"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "9.1"), __combine(in_prefix, "3.attentions.0"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "10.0"), __combine(in_prefix, "3.resnets.1"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "10.1"), __combine(in_prefix, "3.attentions.1"))
    out_states |= __map_unet_resnet_block(in_states, __combine(out_prefix, "11.0"), __combine(in_prefix, "3.resnets.2"))
    out_states |= __map_unet_attention_block(in_states, __combine(out_prefix, "11.1"), __combine(in_prefix, "3.attentions.2"))

    return out_states


def __map_unet(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_wb(in_states, __combine(out_prefix, "input_blocks.0.0"), __combine(in_prefix, "conv_in"))

    out_states |= __map_wb(in_states, __combine(out_prefix, "time_embed.0"), __combine(in_prefix, "time_embedding.linear_1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "time_embed.2"), __combine(in_prefix, "time_embedding.linear_2"))

    out_states |= __map_unet_down_blocks(in_states, __combine(out_prefix, "input_blocks"), __combine(in_prefix, "down_blocks"))
    out_states |= __map_unet_mid_block(in_states, __combine(out_prefix, "middle_block"), __combine(in_prefix, "mid_block"))
    out_states |= __map_unet_up_block(in_states, __combine(out_prefix, "output_blocks"), __combine(in_prefix, "up_blocks"))

    out_states |= __map_wb(in_states, __combine(out_prefix, "out.0"), __combine(in_prefix, "conv_norm_out"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "out.2"), __combine(in_prefix, "conv_out"))

    return out_states


def __map_text_encoder_resblock(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    in_proj_weight = torch.cat([
        in_states[__combine(in_prefix, "self_attn.q_proj.weight")],
        in_states[__combine(in_prefix, "self_attn.k_proj.weight")],
        in_states[__combine(in_prefix, "self_attn.v_proj.weight")],
    ], 0)

    in_proj_bias = torch.cat([
        in_states[__combine(in_prefix, "self_attn.q_proj.bias")],
        in_states[__combine(in_prefix, "self_attn.k_proj.bias")],
        in_states[__combine(in_prefix, "self_attn.v_proj.bias")],
    ], 0)

    out_states[__combine(out_prefix, "attn.in_proj_weight")] = in_proj_weight
    out_states[__combine(out_prefix, "attn.in_proj_bias")] = in_proj_bias

    out_states |= __map_wb(in_states, __combine(out_prefix, "attn.out_proj"), __combine(in_prefix, "self_attn.out_proj"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "ln_1"), __combine(in_prefix, "layer_norm1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "ln_2"), __combine(in_prefix, "layer_norm2"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "mlp.c_fc"), __combine(in_prefix, "mlp.fc1"))
    out_states |= __map_wb(in_states, __combine(out_prefix, "mlp.c_proj"), __combine(in_prefix, "mlp.fc2"))

    return out_states


def __map_text_encoder(in_states: dict, out_prefix: str, in_prefix: str, is_v2: bool) -> dict:
    out_states = {}

    if is_v2:
        dtype = in_states[__combine(in_prefix, "embeddings.position_embedding.weight")].dtype
        device = in_states[__combine(in_prefix, "embeddings.position_embedding.weight")].device

        out_states |= __map_wb(in_states, __combine(out_prefix, "model.ln_final"), __combine(in_prefix, "final_layer_norm"))
        out_states[__combine(out_prefix, "model.positional_embedding")] = in_states[__combine(in_prefix, "embeddings.position_embedding.weight")]
        out_states[__combine(out_prefix, "model.token_embedding.weight")] = in_states[__combine(in_prefix, "embeddings.token_embedding.weight")]
        out_states[__combine(out_prefix, "model.text_projection")] = torch.ones((1024, 1024), dtype=dtype, device=device)
        out_states[__combine(out_prefix, "model.logit_scale")] = torch.tensor(1, dtype=dtype, device=device)

        for i in range(0, 23, 1):
            out_states |= __map_text_encoder_resblock(in_states, __combine(out_prefix, f"model.transformer.resblocks.{str(i)}"), __combine(in_prefix, f"encoder.layers.{str(i)}"))
    else:
        for (key, value) in in_states.items():
            out_states[__combine(__combine(out_prefix, ".transformer"), key)] = value

    return out_states


def __map_noise_scheduler(noise_scheduler: DDIMScheduler) -> dict:
    out_states = {}

    coefficients = DiffusionScheduleCoefficients.from_betas(noise_scheduler.betas)

    out_states["betas"] = coefficients.betas
    out_states["alphas_cumprod"] = coefficients.alphas_cumprod
    out_states["alphas_cumprod_prev"] = coefficients.alphas_cumprod_prev
    out_states["sqrt_alphas_cumprod"] = coefficients.sqrt_alphas_cumprod
    out_states["sqrt_one_minus_alphas_cumprod"] = coefficients.sqrt_one_minus_alphas_cumprod
    out_states["log_one_minus_alphas_cumprod"] = coefficients.log_one_minus_alphas_cumprod
    out_states["sqrt_recip_alphas_cumprod"] = coefficients.sqrt_recip_alphas_cumprod
    out_states["sqrt_recipm1_alphas_cumprod"] = coefficients.sqrt_recipm1_alphas_cumprod
    out_states["posterior_variance"] = coefficients.posterior_variance
    out_states["posterior_log_variance_clipped"] = coefficients.posterior_log_variance_clipped
    out_states["posterior_mean_coef1"] = coefficients.posterior_mean_coef1
    out_states["posterior_mean_coef2"] = coefficients.posterior_mean_coef2

    return out_states


def convert_sd_diffusers_to_ckpt(
        model_type: ModelType,
        vae_state_dict: dict,
        unet_state_dict: dict,
        text_encoder_state_dict: dict,
        noise_scheduler: DDIMScheduler,
) -> dict:
    states = {}

    states |= __map_vae(vae_state_dict, "first_stage_model", "")
    states |= __map_unet(unet_state_dict, "model.diffusion_model", "")
    states |= __map_text_encoder(text_encoder_state_dict, "cond_stage_model", "text_model", model_type.is_sd_v2())
    states |= __map_noise_scheduler(noise_scheduler)

    state_dict = {'state_dict': states}

    return state_dict
