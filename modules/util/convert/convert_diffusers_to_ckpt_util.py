from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients

import torch
from torch import Tensor

from diffusers import DDIMScheduler


def combine(left: str, right: str) -> str:
    if left == "":
        return right
    elif right == "":
        return left
    else:
        return left + "." + right


def map_wb(in_states: dict[str, Tensor], out_prefix: str, in_prefix: str) -> dict[str, Tensor]:
    out_states = {}

    out_states[combine(out_prefix, "weight")] = in_states[combine(in_prefix, "weight")]
    out_states[combine(out_prefix, "bias")] = in_states[combine(in_prefix, "bias")]

    return out_states


def map_prefix(in_states: dict[str, Tensor], out_prefix: str, in_prefix: str) -> dict[str, Tensor]:
    out_states = {}

    for key in in_states:
        if key.startswith(in_prefix):
            out_key = out_prefix + key.removeprefix(in_prefix)
            out_states[out_key] = in_states[key]

    return out_states


def pop_prefix(in_states: dict[str, Tensor], in_prefix: str):
    keys = list(in_states.keys())

    for key in keys:
        if key.startswith(in_prefix):
            in_states.pop(key)


def map_noise_scheduler(noise_scheduler: DDIMScheduler) -> dict:
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



def __map_vae_resnet_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= map_wb(in_states, combine(out_prefix, "norm1"), combine(in_prefix, "norm1"))
    out_states |= map_wb(in_states, combine(out_prefix, "conv1"), combine(in_prefix, "conv1"))

    out_states |= map_wb(in_states, combine(out_prefix, "norm2"), combine(in_prefix, "norm2"))
    out_states |= map_wb(in_states, combine(out_prefix, "conv2"), combine(in_prefix, "conv2"))

    if combine(in_prefix, "conv_shortcut.weight") in in_states:
        out_states[combine(out_prefix, "nin_shortcut.weight")] = in_states[combine(in_prefix, "conv_shortcut.weight")]
    if combine(in_prefix, "conv_shortcut.bias") in in_states:
        out_states[combine(out_prefix, "nin_shortcut.bias")] = in_states[combine(in_prefix, "conv_shortcut.bias")]

    return out_states


def __reshape_vae_attention_weight(weight: Tensor) -> Tensor:
    return torch.reshape(weight, shape=[weight.shape[0], weight.shape[1], 1, 1])


def __map_vae_attention_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= map_wb(in_states, combine(out_prefix, "norm"), combine(in_prefix, "group_norm"))

    # support for both deprecated and current attention block names
    if combine(in_prefix, "query") in in_states:
        out_states |= map_wb(in_states, combine(out_prefix, "q"), combine(in_prefix, "query"))  # deprecated
    else:
        out_states |= map_wb(in_states, combine(out_prefix, "q"), combine(in_prefix, "to_q"))  # current

    if combine(in_prefix, "key") in in_states:
        out_states |= map_wb(in_states, combine(out_prefix, "k"), combine(in_prefix, "key"))  # deprecated
    else:
        out_states |= map_wb(in_states, combine(out_prefix, "k"), combine(in_prefix, "to_k"))  # current

    if combine(in_prefix, "value") in in_states:
        out_states |= map_wb(in_states, combine(out_prefix, "v"), combine(in_prefix, "value"))  # deprecated
    else:
        out_states |= map_wb(in_states, combine(out_prefix, "v"), combine(in_prefix, "to_v"))  # current

    if combine(in_prefix, "proj_attn") in in_states:
        out_states |= map_wb(in_states, combine(out_prefix, "proj_out"), combine(in_prefix, "proj_attn"))  # deprecated
    else:
        out_states |= map_wb(in_states, combine(out_prefix, "proj_out"), combine(in_prefix, "to_out.0"))  # current

    out_states[combine(out_prefix, "q.weight")] = __reshape_vae_attention_weight(out_states[combine(out_prefix, "q.weight")])
    out_states[combine(out_prefix, "k.weight")] = __reshape_vae_attention_weight(out_states[combine(out_prefix, "k.weight")])
    out_states[combine(out_prefix, "v.weight")] = __reshape_vae_attention_weight(out_states[combine(out_prefix, "v.weight")])
    out_states[combine(out_prefix, "proj_out.weight")] = __reshape_vae_attention_weight(out_states[combine(out_prefix, "proj_out.weight")])

    return out_states


def __map_vae_encoder(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[combine(out_prefix, "conv_in.weight")] = in_states[combine(in_prefix, "conv_in.weight")]
    out_states[combine(out_prefix, "conv_in.bias")] = in_states[combine(in_prefix, "conv_in.bias")]

    # down blocks
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.0.block.0"), combine(in_prefix, "down_blocks.0.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.0.block.1"), combine(in_prefix, "down_blocks.0.resnets.1"))
    out_states |= map_wb(in_states, combine(out_prefix, "down.0.downsample.conv"), combine(in_prefix, "down_blocks.0.downsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.1.block.0"), combine(in_prefix, "down_blocks.1.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.1.block.1"), combine(in_prefix, "down_blocks.1.resnets.1"))
    out_states |= map_wb(in_states, combine(out_prefix, "down.1.downsample.conv"), combine(in_prefix, "down_blocks.1.downsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.2.block.0"), combine(in_prefix, "down_blocks.2.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.2.block.1"), combine(in_prefix, "down_blocks.2.resnets.1"))
    out_states |= map_wb(in_states, combine(out_prefix, "down.2.downsample.conv"), combine(in_prefix, "down_blocks.2.downsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.3.block.0"), combine(in_prefix, "down_blocks.3.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "down.3.block.1"), combine(in_prefix, "down_blocks.3.resnets.1"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "mid.block_1"), combine(in_prefix, "mid_block.resnets.0"))
    out_states |= __map_vae_attention_block(in_states, combine(out_prefix, "mid.attn_1"), combine(in_prefix, "mid_block.attentions.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "mid.block_2"), combine(in_prefix, "mid_block.resnets.1"))

    # conv out
    out_states[combine(out_prefix, "norm_out.weight")] = in_states[combine(in_prefix, "conv_norm_out.weight")]
    out_states[combine(out_prefix, "norm_out.bias")] = in_states[combine(in_prefix, "conv_norm_out.bias")]
    out_states[combine(out_prefix, "conv_out.weight")] = in_states[combine(in_prefix, "conv_out.weight")]
    out_states[combine(out_prefix, "conv_out.bias")] = in_states[combine(in_prefix, "conv_out.bias")]

    return out_states


def __map_vae_quant_conv(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[combine(out_prefix, "weight")] = in_states[combine(in_prefix, "weight")]
    out_states[combine(out_prefix, "bias")] = in_states[combine(in_prefix, "bias")]

    return out_states


def __map_vae_post_quant_conv(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[combine(out_prefix, "weight")] = in_states[combine(in_prefix, "weight")]
    out_states[combine(out_prefix, "bias")] = in_states[combine(in_prefix, "bias")]

    return out_states


def __map_vae_decoder(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    # conv in
    out_states[combine(out_prefix, "conv_in.weight")] = in_states[combine(in_prefix, "conv_in.weight")]
    out_states[combine(out_prefix, "conv_in.bias")] = in_states[combine(in_prefix, "conv_in.bias")]

    # down blocks
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "mid.block_1"), combine(in_prefix, "mid_block.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "mid.block_2"), combine(in_prefix, "mid_block.resnets.1"))
    out_states |= __map_vae_attention_block(in_states, combine(out_prefix, "mid.attn_1"), combine(in_prefix, "mid_block.attentions.0"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.3.block.0"), combine(in_prefix, "up_blocks.0.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.3.block.1"), combine(in_prefix, "up_blocks.0.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.3.block.2"), combine(in_prefix, "up_blocks.0.resnets.2"))
    out_states |= map_wb(in_states, combine(out_prefix, "up.3.upsample.conv"), combine(in_prefix, "up_blocks.0.upsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.2.block.0"), combine(in_prefix, "up_blocks.1.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.2.block.1"), combine(in_prefix, "up_blocks.1.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.2.block.2"), combine(in_prefix, "up_blocks.1.resnets.2"))
    out_states |= map_wb(in_states, combine(out_prefix, "up.2.upsample.conv"), combine(in_prefix, "up_blocks.1.upsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.1.block.0"), combine(in_prefix, "up_blocks.2.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.1.block.1"), combine(in_prefix, "up_blocks.2.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.1.block.2"), combine(in_prefix, "up_blocks.2.resnets.2"))
    out_states |= map_wb(in_states, combine(out_prefix, "up.1.upsample.conv"), combine(in_prefix, "up_blocks.2.upsamplers.0.conv"))

    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.0.block.0"), combine(in_prefix, "up_blocks.3.resnets.0"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.0.block.1"), combine(in_prefix, "up_blocks.3.resnets.1"))
    out_states |= __map_vae_resnet_block(in_states, combine(out_prefix, "up.0.block.2"), combine(in_prefix, "up_blocks.3.resnets.2"))

    # conv out
    out_states[combine(out_prefix, "norm_out.weight")] = in_states[combine(in_prefix, "conv_norm_out.weight")]
    out_states[combine(out_prefix, "norm_out.bias")] = in_states[combine(in_prefix, "conv_norm_out.bias")]
    out_states[combine(out_prefix, "conv_out.weight")] = in_states[combine(in_prefix, "conv_out.weight")]
    out_states[combine(out_prefix, "conv_out.bias")] = in_states[combine(in_prefix, "conv_out.bias")]

    return out_states


def map_vae(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_vae_encoder(in_states, combine(out_prefix, "encoder"), combine(in_prefix, "encoder"))
    if combine(in_prefix, "quant_conv.weight") in in_states:
        out_states |= __map_vae_quant_conv(in_states, combine(out_prefix, "quant_conv"), combine(in_prefix, "quant_conv"))
    if combine(in_prefix, "post_quant_conv.weight") in in_states:
        out_states |= __map_vae_post_quant_conv(in_states, combine(out_prefix, "post_quant_conv"), combine(in_prefix, "post_quant_conv"))
    out_states |= __map_vae_decoder(in_states, combine(out_prefix, "decoder"), combine(in_prefix, "decoder"))

    return out_states


def map_unet_resnet_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= map_wb(in_states, combine(out_prefix, "in_layers.0"), combine(in_prefix, "norm1"))
    out_states |= map_wb(in_states, combine(out_prefix, "in_layers.2"), combine(in_prefix, "conv1"))

    out_states |= map_wb(in_states, combine(out_prefix, "emb_layers.1"), combine(in_prefix, "time_emb_proj"))

    out_states |= map_wb(in_states, combine(out_prefix, "out_layers.0"), combine(in_prefix, "norm2"))
    out_states |= map_wb(in_states, combine(out_prefix, "out_layers.3"), combine(in_prefix, "conv2"))

    if combine(in_prefix, "conv_shortcut.weight") in in_states:
        out_states |= map_wb(in_states, combine(out_prefix, "skip_connection"), combine(in_prefix, "conv_shortcut"))

    return out_states


def __map_unet_transformer_attention_block(in_states: dict, out_prefix: str, in_prefix: str):
    out_states = {}

    out_states[combine(out_prefix, "to_q.weight")] = in_states[combine(in_prefix, "to_q.weight")]
    out_states[combine(out_prefix, "to_k.weight")] = in_states[combine(in_prefix, "to_k.weight")]
    out_states[combine(out_prefix, "to_v.weight")] = in_states[combine(in_prefix, "to_v.weight")]
    out_states |= map_wb(in_states, combine(out_prefix, "to_out.0"), combine(in_prefix, "to_out.0"))

    return out_states

def __map_unet_transformer_ff_block(in_states: dict, out_prefix: str, in_prefix: str):
    out_states = {}

    out_states |= map_wb(in_states, combine(out_prefix, "0.proj"), combine(in_prefix, "0.proj"))
    out_states |= map_wb(in_states, combine(out_prefix, "2"), combine(in_prefix, "2"))

    return out_states

def __map_unet_transformer_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_unet_transformer_attention_block(in_states, combine(out_prefix, "attn1"), combine(in_prefix, "attn1"))
    out_states |= __map_unet_transformer_ff_block(in_states, combine(out_prefix, "ff.net"), combine(in_prefix, "ff.net"))
    out_states |= __map_unet_transformer_attention_block(in_states, combine(out_prefix, "attn2"), combine(in_prefix, "attn2"))
    out_states |= map_wb(in_states, combine(out_prefix, "norm1"), combine(in_prefix, "norm1"))
    out_states |= map_wb(in_states, combine(out_prefix, "norm2"), combine(in_prefix, "norm2"))
    out_states |= map_wb(in_states, combine(out_prefix, "norm3"), combine(in_prefix, "norm3"))

    return out_states


def map_unet_transformer(in_states: dict, out_prefix: str, in_prefix: str, num_transformer_blocks: int) -> dict:
    out_states = {}

    out_states |= map_wb(in_states, combine(out_prefix, "norm"), combine(in_prefix, "norm"))
    out_states |= map_wb(in_states, combine(out_prefix, "proj_in"), combine(in_prefix, "proj_in"))

    for i in range(num_transformer_blocks):
        out_states |= __map_unet_transformer_block(in_states, combine(out_prefix, f"transformer_blocks.{i}"), combine(in_prefix, f"transformer_blocks.{i}"))

    out_states |= map_wb(in_states, combine(out_prefix, "proj_out"), combine(in_prefix, "proj_out"))

    return out_states
