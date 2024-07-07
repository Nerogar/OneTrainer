import torch
from diffusers import DDIMScheduler
from torch import Tensor

import modules.util.convert.convert_diffusers_to_ckpt_util as util
from modules.util.enum.ModelType import ModelType

def __swap_chunks(tensor:Tensor) -> Tensor:
    chunk_0, chunk_1 = tensor.chunk(2, dim=0)
    return torch.cat([chunk_1, chunk_0], dim=0)

def __map_transformer_block(in_states: dict, out_prefix: str, in_prefix: str, is_last:bool) -> dict:
    out_states = {}

    out_states[util.combine(out_prefix, "x_block.attn.qkv.weight")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.weight")],
        in_states[util.combine(in_prefix, "attn.to_k.weight")],
        in_states[util.combine(in_prefix, "attn.to_v.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "x_block.attn.qkv.bias")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.bias")],
        in_states[util.combine(in_prefix, "attn.to_k.bias")],
        in_states[util.combine(in_prefix, "attn.to_v.bias")],
    ], 0)

    out_states[util.combine(out_prefix, "context_block.attn.qkv.weight")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.add_q_proj.weight")],
        in_states[util.combine(in_prefix, "attn.add_k_proj.weight")],
        in_states[util.combine(in_prefix, "attn.add_v_proj.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "context_block.attn.qkv.bias")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.add_q_proj.bias")],
        in_states[util.combine(in_prefix, "attn.add_k_proj.bias")],
        in_states[util.combine(in_prefix, "attn.add_v_proj.bias")],
    ], 0)

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "x_block.attn.proj"), util.combine(in_prefix, "attn.to_out.0"))

    if not is_last:
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "context_block.attn.proj"), util.combine(in_prefix, "attn.to_add_out"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "x_block.adaLN_modulation.1"), util.combine(in_prefix, "norm1.linear"))

    if not is_last:
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "context_block.adaLN_modulation.1"), util.combine(in_prefix, "norm1_context.linear"))
    else:
        out_states[util.combine(out_prefix, "context_block.adaLN_modulation.1.weight")] = __swap_chunks(in_states[util.combine(in_prefix, "norm1_context.linear.weight")])
        out_states[util.combine(out_prefix, "context_block.adaLN_modulation.1.bias")] = __swap_chunks(in_states[util.combine(in_prefix, "norm1_context.linear.bias")])

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "x_block.mlp.fc1"), util.combine(in_prefix, "ff.net.0.proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "x_block.mlp.fc2"), util.combine(in_prefix, "ff.net.2"))

    if not is_last:
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "context_block.mlp.fc1"), util.combine(in_prefix, "ff_context.net.0.proj"))
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "context_block.mlp.fc2"), util.combine(in_prefix, "ff_context.net.2"))

    return out_states


def __map_transformer(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[util.combine(out_prefix, "pos_embed")] = in_states[util.combine(in_prefix, "pos_embed.pos_embed")]
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "x_embedder.proj"), util.combine(in_prefix, "pos_embed.proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "context_embedder"), util.combine(in_prefix, "context_embedder"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "final_layer.adaLN_modulation.1"), util.combine(in_prefix, "norm_out.linear"))
    out_states[util.combine(out_prefix, "final_layer.adaLN_modulation.1.weight")] = __swap_chunks(in_states[util.combine(in_prefix, "norm_out.linear.weight")])
    out_states[util.combine(out_prefix, "final_layer.adaLN_modulation.1.bias")] = __swap_chunks(in_states[util.combine(in_prefix, "norm_out.linear.bias")])
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "final_layer.linear"), util.combine(in_prefix, "proj_out"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "t_embedder.mlp.0"), util.combine(in_prefix, "time_text_embed.timestep_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "t_embedder.mlp.2"), util.combine(in_prefix, "time_text_embed.timestep_embedder.linear_2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "y_embedder.mlp.0"), util.combine(in_prefix, "time_text_embed.text_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "y_embedder.mlp.2"), util.combine(in_prefix, "time_text_embed.text_embedder.linear_2"))

    num_layers = 24
    for i in range(num_layers):
        is_last = i == (num_layers - 1)
        out_states |= __map_transformer_block(in_states, util.combine(out_prefix, f"joint_blocks.{i}"), util.combine(in_prefix, f"transformer_blocks.{i}"), is_last)

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


def __map_clip_text_encoder(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    for (key, value) in in_states.items():
        out_states[util.combine(out_prefix, key)] = value

    return out_states

def __map_t5_text_encoder(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    for (key, value) in in_states.items():
        out_states[util.combine(out_prefix, key)] = value

    # this keeps compatibility with the original safetensors file.
    # there is no good reason to duplicate the key.
    out_states[util.combine(out_prefix, "encoder.embed_tokens.weight")] = in_states[util.combine(in_prefix, "encoder.embed_tokens.weight")].clone()

    return out_states


def convert_sd3_diffusers_to_ckpt(
        vae_state_dict: dict,
        transformer_state_dict: dict,
        text_encoder_1_state_dict: dict,
        text_encoder_2_state_dict: dict,
        text_encoder_3_state_dict: dict,
) -> dict:
    state_dict = {}

    state_dict |= util.map_vae(vae_state_dict, "first_stage_model", "")
    state_dict |= __map_transformer(transformer_state_dict, "model.diffusion_model", "")
    if text_encoder_1_state_dict is not None:
        state_dict |= __map_clip_text_encoder(text_encoder_1_state_dict, "text_encoders.clip_l.transformer", "")
    if text_encoder_2_state_dict is not None:
        state_dict |= __map_clip_text_encoder(text_encoder_2_state_dict, "text_encoders.clip_g.transformer", "")
    if text_encoder_3_state_dict is not None:
        state_dict |= __map_t5_text_encoder(text_encoder_3_state_dict, "text_encoders.t5xxl.transformer", "")

    return state_dict
