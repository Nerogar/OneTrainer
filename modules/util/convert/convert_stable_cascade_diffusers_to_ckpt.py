import modules.util.convert.convert_diffusers_to_ckpt_util as util

import torch


def __map_unet_blocks(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    i = 0
    while any(key.startswith(util.combine(in_prefix, f"{i}")) for key in in_states):
        if i % 3 == 0:
            # resblock
            out_states[util.combine(out_prefix, f"{i}.channelwise.0.weight")] = in_states[util.combine(in_prefix, f"{i}.channelwise.0.weight")]
            out_states[util.combine(out_prefix, f"{i}.channelwise.0.bias")] = in_states[util.combine(in_prefix, f"{i}.channelwise.0.bias")]
            out_states[util.combine(out_prefix, f"{i}.channelwise.2.beta")] = in_states[util.combine(in_prefix, f"{i}.channelwise.2.beta")]
            out_states[util.combine(out_prefix, f"{i}.channelwise.2.gamma")] = in_states[util.combine(in_prefix, f"{i}.channelwise.2.gamma")]
            out_states[util.combine(out_prefix, f"{i}.channelwise.4.weight")] = in_states[util.combine(in_prefix, f"{i}.channelwise.4.weight")]
            out_states[util.combine(out_prefix, f"{i}.channelwise.4.bias")] = in_states[util.combine(in_prefix, f"{i}.channelwise.4.bias")]
            out_states[util.combine(out_prefix, f"{i}.depthwise.weight")] = in_states[util.combine(in_prefix, f"{i}.depthwise.weight")]
            out_states[util.combine(out_prefix, f"{i}.depthwise.bias")] = in_states[util.combine(in_prefix, f"{i}.depthwise.bias")]
        elif i % 3 == 1:
            # timestep block
            out_states[util.combine(out_prefix, f"{i}.mapper.weight")] = in_states[util.combine(in_prefix, f"{i}.mapper.weight")]
            out_states[util.combine(out_prefix, f"{i}.mapper.bias")] = in_states[util.combine(in_prefix, f"{i}.mapper.bias")]
            out_states[util.combine(out_prefix, f"{i}.mapper_crp.weight")] = in_states[util.combine(in_prefix, f"{i}.mapper_crp.weight")]
            out_states[util.combine(out_prefix, f"{i}.mapper_crp.bias")] = in_states[util.combine(in_prefix, f"{i}.mapper_crp.bias")]
            out_states[util.combine(out_prefix, f"{i}.mapper_sca.weight")] = in_states[util.combine(in_prefix, f"{i}.mapper_sca.weight")]
            out_states[util.combine(out_prefix, f"{i}.mapper_sca.bias")] = in_states[util.combine(in_prefix, f"{i}.mapper_sca.bias")]
        elif i % 3 == 2:
            # attention block
            out_states[util.combine(out_prefix, f"{i}.kv_mapper.1.weight")] = in_states[util.combine(in_prefix, f"{i}.kv_mapper.1.weight")]
            out_states[util.combine(out_prefix, f"{i}.kv_mapper.1.bias")] = in_states[util.combine(in_prefix, f"{i}.kv_mapper.1.bias")]
            out_states[util.combine(out_prefix, f"{i}.attention.attn.out_proj.weight")] = in_states[util.combine(in_prefix, f"{i}.attention.to_out.0.weight")]
            out_states[util.combine(out_prefix, f"{i}.attention.attn.out_proj.bias")] = in_states[util.combine(in_prefix, f"{i}.attention.to_out.0.bias")]

            q_weight = in_states[util.combine(in_prefix, f"{i}.attention.to_q.weight")]
            q_bias = in_states[util.combine(in_prefix, f"{i}.attention.to_q.bias")]
            k_weight = in_states[util.combine(in_prefix, f"{i}.attention.to_k.weight")]
            k_bias = in_states[util.combine(in_prefix, f"{i}.attention.to_k.bias")]
            v_weight = in_states[util.combine(in_prefix, f"{i}.attention.to_v.weight")]
            v_bias = in_states[util.combine(in_prefix, f"{i}.attention.to_v.bias")]

            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

            out_states[util.combine(out_prefix, f"{i}.attention.attn.in_proj_weight")] = qkv_weight
            out_states[util.combine(out_prefix, f"{i}.attention.attn.in_proj_bias")] = qkv_bias

        i += 1

    return out_states


def __map_prior(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= util.map_wb(in_states, util.combine(out_prefix, "clf.1"), util.combine(in_prefix, "clf.1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "clip_img_mapper"), util.combine(in_prefix, "clip_img_mapper"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "clip_txt_mapper"), util.combine(in_prefix, "clip_txt_mapper"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "clip_txt_pooled_mapper"), util.combine(in_prefix, "clip_txt_pooled_mapper"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "down_downscalers.1.1.blocks.0"), util.combine(in_prefix, "down_downscalers.1.1.blocks.0"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "embedding.1"), util.combine(in_prefix, "embedding.1"))
    out_states |= util.map_wb(in_states, util.combine(out_prefix, "up_upscalers.0.1.blocks.1"), util.combine(in_prefix, "up_upscalers.0.1.blocks.1"))

    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "down_blocks.0"), util.combine(in_prefix, "down_blocks.0"))
    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "down_blocks.1"), util.combine(in_prefix, "down_blocks.1"))
    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "up_blocks.0"), util.combine(in_prefix, "up_blocks.0"))
    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "up_blocks.1"), util.combine(in_prefix, "up_blocks.1"))

    return out_states


def convert_stable_cascade_diffusers_to_ckpt(
        prior_state_dict: dict,
) -> dict:
    state_dict = {}

    state_dict |= __map_prior(prior_state_dict, "", "")

    return state_dict
