import torch
from diffusers.models.embeddings import PatchEmbed
from torch import Tensor

import modules.util.convert.convert_diffusers_to_ckpt_util as util
from modules.util.enum.ModelType import ModelType


def __map_transformer_attention_block(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    attn_qkv_bias = torch.cat([
        in_states[util.combine(in_prefix, "attn1.to_q.bias")],
        in_states[util.combine(in_prefix, "attn1.to_k.bias")],
        in_states[util.combine(in_prefix, "attn1.to_v.bias")],
    ], 0)

    attn_qkv_weight = torch.cat([
        in_states[util.combine(in_prefix, "attn1.to_q.weight")],
        in_states[util.combine(in_prefix, "attn1.to_k.weight")],
        in_states[util.combine(in_prefix, "attn1.to_v.weight")],
    ], 0)

    cross_attn_kv_linear_bias = torch.cat([
        in_states[util.combine(in_prefix, "attn2.to_k.bias")],
        in_states[util.combine(in_prefix, "attn2.to_v.bias")],
    ], 0)

    cross_attn_kv_linear_weight = torch.cat([
        in_states[util.combine(in_prefix, "attn2.to_k.weight")],
        in_states[util.combine(in_prefix, "attn2.to_v.weight")],
    ], 0)

    out_states[util.combine(out_prefix, "attn.qkv.bias")] = attn_qkv_bias
    out_states[util.combine(out_prefix, "attn.qkv.weight")] = attn_qkv_weight
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "attn.proj"), util.combine(in_prefix, "attn1.to_out.0"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "cross_attn.q_linear"), util.combine(in_prefix, "attn2.to_q"))
    out_states[util.combine(out_prefix, "cross_attn.kv_linear.bias")] = cross_attn_kv_linear_bias
    out_states[util.combine(out_prefix, "cross_attn.kv_linear.weight")] = cross_attn_kv_linear_weight
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "cross_attn.proj"), util.combine(in_prefix, "attn2.to_out.0"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "mlp.fc1"), util.combine(in_prefix, "ff.net.0.proj"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "mlp.fc2"), util.combine(in_prefix, "ff.net.2"))
    out_states[util.combine(out_prefix, "scale_shift_table")] = in_states[util.combine(in_prefix, "scale_shift_table")]

    return out_states


def __generate_pos_embed() -> Tensor:
    return PatchEmbed(
        height=128,
        width=128,
        patch_size=2,
        in_channels=4,
        embed_dim=16*72,
        interpolation_scale=1.0,
    ).pos_embed


def __generate_y_embedding() -> Tensor:
    pass
    # TODO: generate y_embedding
    # return PixArtAlphaTextProjection(
    #     in_features=4096,
    #     hidden_size=16*72,
    # )

def __map_transformer(in_states: dict, out_prefix: str, in_prefix: str, model_type: ModelType) -> dict:
    out_states = {}

    if model_type.is_pixart_alpha():
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "ar_embedder.mlp.0"), util.combine(in_prefix, "adaln_single.emb.aspect_ratio_embedder.linear_1"))
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "ar_embedder.mlp.2"), util.combine(in_prefix, "adaln_single.emb.aspect_ratio_embedder.linear_2"))

        out_states |= util.map_wb(in_states, util.combine(out_prefix, "csize_embedder.mlp.0"), util.combine(in_prefix, "adaln_single.emb.resolution_embedder.linear_1"))
        out_states |= util.map_wb(in_states, util.combine(out_prefix, "csize_embedder.mlp.2"), util.combine(in_prefix, "adaln_single.emb.resolution_embedder.linear_2"))

    #out_states[util.combine(out_prefix, "y_embedder.y_embedding")] = __generate_y_embedding() #TODO
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "y_embedder.y_proj.fc1"), util.combine(in_prefix, "caption_projection.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "y_embedder.y_proj.fc2"), util.combine(in_prefix, "caption_projection.linear_2"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "x_embedder.proj"), util.combine(in_prefix, "pos_embed.proj"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "t_embedder.mlp.0"), util.combine(in_prefix, "adaln_single.emb.timestep_embedder.linear_1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "t_embedder.mlp.2"), util.combine(in_prefix, "adaln_single.emb.timestep_embedder.linear_2"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "t_block.1"), util.combine(in_prefix, "adaln_single.linear"))

    out_states["pos_embed"] = __generate_pos_embed()

    for i in range(28):
        out_states |= __map_transformer_attention_block(in_states, util.combine(out_prefix, f"blocks.{i}"), util.combine(in_prefix, f"transformer_blocks.{i}"))

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "final_layer.linear"), util.combine(in_prefix, "proj_out"))
    out_states[util.combine(out_prefix, "final_layer.scale_shift_table")] = in_states[util.combine(in_prefix, "scale_shift_table")]

    return out_states


def convert_pixart_diffusers_to_ckpt(
        model_type: ModelType,
        transformer_state_dict: dict,
) -> dict:
    state_dict = {}

    state_dict |= __map_transformer(transformer_state_dict, "", "", model_type)

    return state_dict
