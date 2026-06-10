import modules.util.convert.convert_diffusers_to_ckpt_util as util

import torch
from torch import Tensor


def __swap_chunks(tensor: Tensor) -> Tensor:
    chunk_0, chunk_1 = tensor.chunk(2, dim=0)
    return torch.cat([chunk_1, chunk_0], dim=0)

def __map_token_refiner_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[util.combine(out_prefix, "self_attn.qkv.weight")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.weight")],
        in_states[util.combine(in_prefix, "attn.to_k.weight")],
        in_states[util.combine(in_prefix, "attn.to_v.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "self_attn.qkv.bias")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.bias")],
        in_states[util.combine(in_prefix, "attn.to_k.bias")],
        in_states[util.combine(in_prefix, "attn.to_v.bias")],
    ], 0)

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "self_attn.proj"), util.combine(in_prefix, "attn.to_out.0"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "mlp.0"), util.combine(in_prefix, "ff.net.0.proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "mlp.2"), util.combine(in_prefix, "ff.net.2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "adaLN_modulation.1"), util.combine(in_prefix, "norm_out.linear"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "norm1"), util.combine(in_prefix, "norm1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "norm2"), util.combine(in_prefix, "norm2"))

    return out_states


def __map_double_transformer_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[util.combine(out_prefix, "img_attn.qkv.weight")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.weight")],
        in_states[util.combine(in_prefix, "attn.to_k.weight")],
        in_states[util.combine(in_prefix, "attn.to_v.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "img_attn.qkv.bias")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.bias")],
        in_states[util.combine(in_prefix, "attn.to_k.bias")],
        in_states[util.combine(in_prefix, "attn.to_v.bias")],
    ], 0)

    out_states[util.combine(out_prefix, "txt_attn.qkv.weight")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.add_q_proj.weight")],
        in_states[util.combine(in_prefix, "attn.add_k_proj.weight")],
        in_states[util.combine(in_prefix, "attn.add_v_proj.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "txt_attn.qkv.bias")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.add_q_proj.bias")],
        in_states[util.combine(in_prefix, "attn.add_k_proj.bias")],
        in_states[util.combine(in_prefix, "attn.add_v_proj.bias")],
    ], 0)

    out_states[util.combine(out_prefix, "img_attn.norm.key_norm.scale")] = in_states[util.combine(in_prefix, "attn.norm_k.weight")]
    out_states[util.combine(out_prefix, "img_attn.norm.query_norm.scale")] = in_states[util.combine(in_prefix, "attn.norm_q.weight")]
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "img_attn.proj"), util.combine(in_prefix, "attn.to_out.0"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "img_mlp.0"), util.combine(in_prefix, "ff.net.0.proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "img_mlp.2"), util.combine(in_prefix, "ff.net.2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "img_mod.lin"), util.combine(in_prefix, "norm1.linear"))

    out_states[util.combine(out_prefix, "txt_attn.norm.key_norm.scale")] = in_states[util.combine(in_prefix, "attn.norm_added_k.weight")]
    out_states[util.combine(out_prefix, "txt_attn.norm.query_norm.scale")] = in_states[util.combine(in_prefix, "attn.norm_added_q.weight")]
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_attn.proj"), util.combine(in_prefix, "attn.to_add_out"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_mlp.0"), util.combine(in_prefix, "ff_context.net.0.proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_mlp.2"), util.combine(in_prefix, "ff_context.net.2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_mod.lin"), util.combine(in_prefix, "norm1_context.linear"))

    return out_states


def __map_single_transformer_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states[util.combine(out_prefix, "linear1.weight")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.weight")],
        in_states[util.combine(in_prefix, "attn.to_k.weight")],
        in_states[util.combine(in_prefix, "attn.to_v.weight")],
        in_states[util.combine(in_prefix, "proj_mlp.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "linear1.bias")] = torch.cat([
        in_states[util.combine(in_prefix, "attn.to_q.bias")],
        in_states[util.combine(in_prefix, "attn.to_k.bias")],
        in_states[util.combine(in_prefix, "attn.to_v.bias")],
        in_states[util.combine(in_prefix, "proj_mlp.bias")],
    ], 0)

    out_states[util.combine(out_prefix, "norm.key_norm.scale")] = in_states[util.combine(in_prefix, "attn.norm_k.weight")]
    out_states[util.combine(out_prefix, "norm.query_norm.scale")] = in_states[util.combine(in_prefix, "attn.norm_q.weight")]
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "linear2"), util.combine(in_prefix, "proj_out"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "modulation.lin"), util.combine(in_prefix, "norm.linear"))

    return out_states


def __map_transformer(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_in.c_embedder.in_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.text_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_in.c_embedder.out_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.text_embedder.linear_2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_in.t_embedder.in_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.timestep_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_in.t_embedder.out_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.timestep_embedder.linear_2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "txt_in.input_embedder"), util.combine(in_prefix, "context_embedder.proj_in"))
    out_states[util.combine(out_prefix, "final_layer.adaLN_modulation.1.weight")] = __swap_chunks(in_states[util.combine(in_prefix, "norm_out.linear.weight")])
    out_states[util.combine(out_prefix, "final_layer.adaLN_modulation.1.bias")] = __swap_chunks(in_states[util.combine(in_prefix, "norm_out.linear.bias")])
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "final_layer.linear"), util.combine(in_prefix, "proj_out"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "guidance_in.in_layer"), util.combine(in_prefix, "time_text_embed.guidance_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "guidance_in.out_layer"), util.combine(in_prefix, "time_text_embed.guidance_embedder.linear_2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "vector_in.in_layer"), util.combine(in_prefix, "time_text_embed.text_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "vector_in.out_layer"), util.combine(in_prefix, "time_text_embed.text_embedder.linear_2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "time_in.in_layer"), util.combine(in_prefix, "time_text_embed.timestep_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "time_in.out_layer"), util.combine(in_prefix, "time_text_embed.timestep_embedder.linear_2"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "img_in.proj"), util.combine(in_prefix, "x_embedder.proj"))

    i = 0
    while any(key.startswith(util.combine(in_prefix, f"context_embedder.token_refiner.refiner_blocks.{i}")) for key in in_states):
        out_states |= __map_token_refiner_block(in_states, util.combine(out_prefix, f"txt_in.individual_token_refiner.blocks.{i}"), util.combine(in_prefix, f"context_embedder.token_refiner.refiner_blocks.{i}"))
        i += 1

    i = 0
    while any(key.startswith(util.combine(in_prefix, f"transformer_blocks.{i}")) for key in in_states):
        out_states |= __map_double_transformer_block(in_states, util.combine(out_prefix, f"double_blocks.{i}"), util.combine(in_prefix, f"transformer_blocks.{i}"))
        i += 1

    i = 0
    while any(key.startswith(util.combine(in_prefix, f"single_transformer_blocks.{i}")) for key in in_states):
        out_states |= __map_single_transformer_block(in_states, util.combine(out_prefix, f"single_blocks.{i}"), util.combine(in_prefix, f"single_transformer_blocks.{i}"))
        i += 1

    return out_states


def convert_hunyuan_video_diffusers_to_ckpt(
        transformer_state_dict: dict,
) -> dict:
    state_dict = {}

    state_dict |= __map_transformer(transformer_state_dict, "model.model", "")

    return state_dict
