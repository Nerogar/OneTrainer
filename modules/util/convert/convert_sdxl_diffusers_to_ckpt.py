import modules.util.convert.convert_diffusers_to_ckpt_util as util

import torch

from diffusers import DDIMScheduler


def __map_unet_down_blocks(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "1.0"), util.combine(in_prefix, "0.resnets.0"))
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "2.0"), util.combine(in_prefix, "0.resnets.1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "3.0.op"), util.combine(in_prefix, "0.downsamplers.0.conv"))

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "4.0"), util.combine(in_prefix, "1.resnets.0"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "4.1"), util.combine(in_prefix, "1.attentions.0"), 2)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "5.0"), util.combine(in_prefix, "1.resnets.1"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "5.1"), util.combine(in_prefix, "1.attentions.1"), 2)
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "6.0.op"), util.combine(in_prefix, "1.downsamplers.0.conv"))

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "7.0"), util.combine(in_prefix, "2.resnets.0"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "7.1"), util.combine(in_prefix, "2.attentions.0"), 10)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "8.0"), util.combine(in_prefix, "2.resnets.1"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "8.1"), util.combine(in_prefix, "2.attentions.1"), 10)

    return out_states


def __map_unet_mid_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "0"), util.combine(in_prefix, "resnets.0"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "1"), util.combine(in_prefix, "attentions.0"), 10)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "2"), util.combine(in_prefix, "resnets.1"))

    return out_states


def __map_unet_up_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "0.0"), util.combine(in_prefix, "0.resnets.0"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "0.1"), util.combine(in_prefix, "0.attentions.0"), 10)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "1.0"), util.combine(in_prefix, "0.resnets.1"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "1.1"), util.combine(in_prefix, "0.attentions.1"), 10)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "2.0"), util.combine(in_prefix, "0.resnets.2"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "2.1"), util.combine(in_prefix, "0.attentions.2"), 10)
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "2.2.conv"), util.combine(in_prefix, "0.upsamplers.0.conv"))

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "3.0"), util.combine(in_prefix, "1.resnets.0"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "3.1"), util.combine(in_prefix, "1.attentions.0"), 2)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "4.0"), util.combine(in_prefix, "1.resnets.1"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "4.1"), util.combine(in_prefix, "1.attentions.1"), 2)
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "5.0"), util.combine(in_prefix, "1.resnets.2"))
    out_states |= util.map_unet_transformer(in_states, util.combine(out_prefix, "5.1"), util.combine(in_prefix, "1.attentions.2"), 2)
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "5.2.conv"), util.combine(in_prefix, "1.upsamplers.0.conv"))

    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "6.0"), util.combine(in_prefix, "2.resnets.0"))
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "7.0"), util.combine(in_prefix, "2.resnets.1"))
    out_states |= util.map_unet_resnet_block(in_states, util.combine(out_prefix, "8.0"), util.combine(in_prefix, "2.resnets.2"))

    return out_states


def __map_unet(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "input_blocks.0.0"), util.combine(in_prefix, "conv_in"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "time_embed.0"), util.combine(in_prefix, "time_embedding.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "time_embed.2"), util.combine(in_prefix, "time_embedding.linear_2"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "label_emb.0.0"), util.combine(in_prefix, "add_embedding.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "label_emb.0.2"), util.combine(in_prefix, "add_embedding.linear_2"))

    out_states |= __map_unet_down_blocks(in_states, util.combine(out_prefix, "input_blocks"), util.combine(in_prefix, "down_blocks"))
    out_states |= __map_unet_mid_block(in_states, util.combine(out_prefix, "middle_block"), util.combine(in_prefix, "mid_block"))
    out_states |= __map_unet_up_block(in_states, util.combine(out_prefix, "output_blocks"), util.combine(in_prefix, "up_blocks"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "out.0"), util.combine(in_prefix, "conv_norm_out"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "out.2"), util.combine(in_prefix, "conv_out"))

    return out_states


def __map_text_encoder_1(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    for (key, value) in in_states.items():
        out_states[util.combine(util.combine(out_prefix, ""), key)] = value

    return out_states


def __map_text_encoder_resblock(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    in_proj_weight = torch.cat([
        in_states[util.combine(in_prefix, "self_attn.q_proj.weight")],
        in_states[util.combine(in_prefix, "self_attn.k_proj.weight")],
        in_states[util.combine(in_prefix, "self_attn.v_proj.weight")],
    ], 0)

    in_proj_bias = torch.cat([
        in_states[util.combine(in_prefix, "self_attn.q_proj.bias")],
        in_states[util.combine(in_prefix, "self_attn.k_proj.bias")],
        in_states[util.combine(in_prefix, "self_attn.v_proj.bias")],
    ], 0)

    out_states[util.combine(out_prefix, "attn.in_proj_weight")] = in_proj_weight
    out_states[util.combine(out_prefix, "attn.in_proj_bias")] = in_proj_bias

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "attn.out_proj"), util.combine(in_prefix, "self_attn.out_proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "ln_1"), util.combine(in_prefix, "layer_norm1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "ln_2"), util.combine(in_prefix, "layer_norm2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "mlp.c_fc"), util.combine(in_prefix, "mlp.fc1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "mlp.c_proj"), util.combine(in_prefix, "mlp.fc2"))

    return out_states


def __map_text_encoder_2(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    dtype = in_states[util.combine(in_prefix, "embeddings.position_embedding.weight")].dtype
    device = in_states[util.combine(in_prefix, "embeddings.position_embedding.weight")].device

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "model.ln_final"), util.combine(in_prefix, "final_layer_norm"))
    out_states[util.combine(out_prefix, "model.positional_embedding")] = in_states[util.combine(in_prefix, "embeddings.position_embedding.weight")]
    out_states[util.combine(out_prefix, "model.token_embedding.weight")] = in_states[util.combine(in_prefix, "embeddings.token_embedding.weight")]
    out_states[util.combine(out_prefix, "model.text_projection")] = in_states["text_projection.weight"]
    out_states[util.combine(out_prefix, "model.logit_scale")] = torch.tensor(1, dtype=dtype, device=device)

    for i in range(0, 32, 1):
        out_states |= __map_text_encoder_resblock(in_states, util.combine(out_prefix, f"model.transformer.resblocks.{str(i)}"), util.combine(in_prefix, f"encoder.layers.{str(i)}"))

    return out_states

def __map_vpred(noise_scheduler: DDIMScheduler) -> dict:
    out_states = {}

    if noise_scheduler.config.prediction_type == 'v_prediction':
        out_states["v_pred"] = torch.tensor([])

    return out_states

def convert_sdxl_diffusers_to_ckpt(
        vae_state_dict: dict,
        unet_state_dict: dict,
        text_encoder_1_state_dict: dict,
        text_encoder_2_state_dict: dict,
        noise_scheduler: DDIMScheduler,
) -> dict:
    state_dict = {}

    state_dict |= util.map_vae(vae_state_dict, "first_stage_model", "")
    state_dict |= __map_unet(unet_state_dict, "model.diffusion_model", "")
    state_dict |= __map_text_encoder_1(text_encoder_1_state_dict, "conditioner.embedders.0.transformer", "")
    state_dict |= __map_text_encoder_2(text_encoder_2_state_dict, "conditioner.embedders.1", "text_model")
    state_dict |= util.map_noise_scheduler(noise_scheduler)
    state_dict |= __map_vpred(noise_scheduler)

    return state_dict
