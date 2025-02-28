import modules.util.convert.convert_omi_util as util


def __map_token_refiner_block(in_states: dict, map_back: bool, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "self_attn.qkv.0"), util.combine(in_prefix, "attn.to_q"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "self_attn.qkv.1"), util.combine(in_prefix, "attn.to_k"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "self_attn.qkv.2"), util.combine(in_prefix, "attn.to_v"))

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "self_attn.proj"), util.combine(in_prefix, "attn.to_out.0"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "mlp.0"), util.combine(in_prefix, "ff.net.0.proj"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "mlp.2"), util.combine(in_prefix, "ff.net.2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "adaLN_modulation.1"), util.combine(in_prefix, "norm_out.linear"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "norm1"), util.combine(in_prefix, "norm1"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "norm2"), util.combine(in_prefix, "norm2"))

    return out_states


def __map_double_transformer_block(in_states: dict, map_back: bool, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_attn.qkv.0"), util.combine(in_prefix, "attn.to_q"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_attn.qkv.1"), util.combine(in_prefix, "attn.to_k"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_attn.qkv.2"), util.combine(in_prefix, "attn.to_v"))

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_attn.qkv.0"), util.combine(in_prefix, "attn.add_q_proj"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_attn.qkv.1"), util.combine(in_prefix, "attn.add_k_proj"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_attn.qkv.2"), util.combine(in_prefix, "attn.add_v_proj"))

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_attn.proj"), util.combine(in_prefix, "attn.to_out.0"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_mlp.0"), util.combine(in_prefix, "ff.net.0.proj"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_mlp.2"), util.combine(in_prefix, "ff.net.2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_mod.lin"), util.combine(in_prefix, "norm1.linear"))

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_attn.proj"), util.combine(in_prefix, "attn.to_add_out"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_mlp.0"), util.combine(in_prefix, "ff_context.net.0.proj"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_mlp.2"), util.combine(in_prefix, "ff_context.net.2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_mod.lin"), util.combine(in_prefix, "norm1_context.linear"))

    return out_states


def __map_single_transformer_block(in_states: dict, map_back: bool, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "linear1.0"), util.combine(in_prefix, "attn.to_q"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "linear1.1"), util.combine(in_prefix, "attn.to_k"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "linear1.2"), util.combine(in_prefix, "attn.to_v"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "linear1.3"), util.combine(in_prefix, "proj_mlp"))

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "linear2"), util.combine(in_prefix, "proj_out"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "modulation.lin"), util.combine(in_prefix, "norm.linear"))

    return out_states


def __map_transformer(in_states: dict, map_back: bool, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_in.c_embedder.in_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.text_embedder.linear_1"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_in.c_embedder.out_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.text_embedder.linear_2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_in.t_embedder.in_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.timestep_embedder.linear_1"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_in.t_embedder.out_layer"), util.combine(in_prefix, "context_embedder.time_text_embed.timestep_embedder.linear_2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "txt_in.input_embedder"), util.combine(in_prefix, "context_embedder.proj_in"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "final_layer.adaLN_modulation.1"), util.combine(in_prefix, "norm_out.linear"), swap_chunks=True)
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "final_layer.linear"), util.combine(in_prefix, "proj_out"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "guidance_in.in_layer"), util.combine(in_prefix, "time_text_embed.guidance_embedder.linear_1"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "guidance_in.out_layer"), util.combine(in_prefix, "time_text_embed.guidance_embedder.linear_2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "vector_in.in_layer"), util.combine(in_prefix, "time_text_embed.text_embedder.linear_1"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "vector_in.out_layer"), util.combine(in_prefix, "time_text_embed.text_embedder.linear_2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "time_in.in_layer"), util.combine(in_prefix, "time_text_embed.timestep_embedder.linear_1"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "time_in.out_layer"), util.combine(in_prefix, "time_text_embed.timestep_embedder.linear_2"))
    out_states |= util.map_prefix(in_states, map_back, util.combine(out_prefix, "img_in.proj"), util.combine(in_prefix, "x_embedder.proj"))

    for i in util.map_prefix_range(in_states, map_back, util.combine(out_prefix, "txt_in.individual_token_refiner.blocks"), util.combine(in_prefix, "context_embedder.token_refiner.refiner_blocks")):
        out_states |= __map_token_refiner_block(in_states, map_back, util.combine(out_prefix, f"txt_in.individual_token_refiner.blocks.{i}"), util.combine(in_prefix, f"context_embedder.token_refiner.refiner_blocks.{i}"))

    for i in util.map_prefix_range(in_states, map_back, util.combine(out_prefix, "double_blocks"), util.combine(in_prefix, "transformer_blocks")):
        out_states |= __map_double_transformer_block(in_states, map_back, util.combine(out_prefix, f"double_blocks.{i}"), util.combine(in_prefix, f"transformer_blocks.{i}"))

    for i in util.map_prefix_range(in_states, map_back, util.combine(out_prefix, "single_blocks"), util.combine(in_prefix, "single_transformer_blocks")):
        out_states |= __map_single_transformer_block(in_states, map_back, util.combine(out_prefix, f"single_blocks.{i}"), util.combine(in_prefix, f"single_transformer_blocks.{i}"))

    return out_states


def convert_hunyuan_video_lora_to_omi(
        transformer_state_dict: dict,
        map_back: bool
) -> dict:
    state_dict = {}

    state_dict |= util.map_prefix(transformer_state_dict, map_back, "bundle_emb", "bundle_emb")
    state_dict |= __map_transformer(transformer_state_dict, map_back, "lora_transformer", "lora_transformer")

    return state_dict
