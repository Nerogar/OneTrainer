import modules.util.convert.convert_diffusers_to_ckpt_util as util
from modules.util.convert_util import add_prefix, convert


def __map_transformer(in_states: dict, out_prefix: str, conversion: list) -> dict:
    # the diffusers -> original/MMDiT key map (qkv fusion + adaLN chunk swaps) is provided by the caller
    # (StableDiffusion3Model.checkpoint_diffusers_to_original()), also driving LoRA KOHYA/ORIGINAL/COMFY.
    # The last joint block (context-pre-only, chunk-swapped context modulation) is resolved inside the
    # model method from the live transformer, so the conversion is fully self-contained here.
    # Verified byte-identical to the old hand-rolled converter.
    mmdit_state_dict = convert(in_states, conversion, strict=True)
    return convert(mmdit_state_dict, add_prefix(out_prefix), strict=False)


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
        conversion: list,
) -> dict:
    state_dict = {}

    state_dict |= util.map_vae(vae_state_dict, "first_stage_model", "")
    state_dict |= __map_transformer(transformer_state_dict, "model.diffusion_model", conversion)
    if text_encoder_1_state_dict is not None:
        state_dict |= __map_clip_text_encoder(text_encoder_1_state_dict, "text_encoders.clip_l.transformer", "")
    if text_encoder_2_state_dict is not None:
        state_dict |= __map_clip_text_encoder(text_encoder_2_state_dict, "text_encoders.clip_g.transformer", "")
    if text_encoder_3_state_dict is not None:
        state_dict |= __map_t5_text_encoder(text_encoder_3_state_dict, "text_encoders.t5xxl.transformer", "")

    return state_dict
