import modules.util.convert.convert_diffusers_to_ckpt_util as util
from modules.util.convert_util import add_prefix, convert
from modules.util.enum.ModelType import ModelType

import torch

from diffusers import DDIMScheduler


def __map_unet(in_states: dict, out_prefix: str, conversion: list) -> dict:
    # the diffusers -> original/sgm UNet key map is provided by the caller
    # (model.checkpoint_diffusers_to_original(), == the LoRA body for SD since there is no fusion).
    # Verified byte-identical to the old hand-rolled __map_unet_* helpers.
    sgm_state_dict = convert(in_states, conversion, strict=True)
    return convert(sgm_state_dict, add_prefix(out_prefix), strict=False)


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


def __map_text_encoder(in_states: dict, out_prefix: str, in_prefix: str, is_v2: bool) -> dict:
    out_states = {}

    if is_v2:
        dtype = in_states[util.combine(in_prefix, "embeddings.position_embedding.weight")].dtype
        device = in_states[util.combine(in_prefix, "embeddings.position_embedding.weight")].device

        out_states |= util.map_wb(in_states, util.combine(out_prefix, "model.ln_final"), util.combine(in_prefix, "final_layer_norm"))
        out_states[util.combine(out_prefix, "model.positional_embedding")] = in_states[util.combine(in_prefix, "embeddings.position_embedding.weight")]
        out_states[util.combine(out_prefix, "model.token_embedding.weight")] = in_states[util.combine(in_prefix, "embeddings.token_embedding.weight")]
        out_states[util.combine(out_prefix, "model.text_projection")] = torch.ones((1024, 1024), dtype=dtype, device=device)
        out_states[util.combine(out_prefix, "model.logit_scale")] = torch.tensor(1, dtype=dtype, device=device)

        for i in range(0, 23, 1):
            out_states |= __map_text_encoder_resblock(in_states, util.combine(out_prefix, f"model.transformer.resblocks.{str(i)}"), util.combine(in_prefix, f"encoder.layers.{str(i)}"))
    else:
        for (key, value) in in_states.items():
            out_states[util.combine(util.combine(out_prefix, "transformer"), key)] = value

    return out_states


def convert_sd_diffusers_to_ckpt(
        model_type: ModelType,
        vae_state_dict: dict,
        unet_state_dict: dict,
        text_encoder_state_dict: dict,
        noise_scheduler: DDIMScheduler,
        conversion: list,
) -> dict:
    state_dict = {}

    state_dict |= util.map_vae(vae_state_dict, "first_stage_model", "")
    state_dict |= __map_unet(unet_state_dict, "model.diffusion_model", conversion)
    state_dict |= __map_text_encoder(text_encoder_state_dict, "cond_stage_model", "text_model", model_type.is_sd_v2())
    state_dict |= util.map_noise_scheduler(noise_scheduler)

    return state_dict
