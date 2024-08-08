import modules.util.convert.convert_diffusers_to_ckpt_util as util


def __map_unet_blocks(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    i = 2
    while any(key.startswith(in_prefix + f"_{i}") for key in in_states):
        # attention block
        out_states |= util.map_prefix(in_states, out_prefix + f"_{i}_attention_to_q", in_prefix + f"_{i}_attention_attn_to_q")
        out_states |= util.map_prefix(in_states, out_prefix + f"_{i}_attention_to_k", in_prefix + f"_{i}_attention_attn_to_k")
        out_states |= util.map_prefix(in_states, out_prefix + f"_{i}_attention_to_v", in_prefix + f"_{i}_attention_attn_to_v")
        out_states |= util.map_prefix(in_states, out_prefix + f"_{i}_attention_to_out_0", in_prefix + f"_{i}_attention_attn_out_proj")
        util.pop_prefix(in_states, in_prefix + f"_{i}_attention_attn")

        i += 3

    return out_states


def __map_prior(in_states: dict, out_prefix: str, in_prefix: str) -> dict:
    out_states = {}

    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "lora_prior_unet_down_blocks_0"), util.combine(in_prefix, "lora_prior_unet_down_blocks_0"))
    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "lora_prior_unet_down_blocks_1"), util.combine(in_prefix, "lora_prior_unet_down_blocks_1"))
    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "lora_prior_unet_up_blocks_0"), util.combine(in_prefix, "lora_prior_unet_up_blocks_0"))
    out_states |= __map_unet_blocks(in_states, util.combine(out_prefix, "lora_prior_unet_up_blocks_1"), util.combine(in_prefix, "lora_prior_unet_up_blocks_1"))

    out_states |= util.map_prefix(in_states, out_prefix, in_prefix)

    return out_states


def convert_stable_cascade_lora_ckpt_to_diffusers(
        prior_state_dict: dict,
) -> dict:
    state_dict = {}

    state_dict |= __map_prior(prior_state_dict, "", "")

    return state_dict
