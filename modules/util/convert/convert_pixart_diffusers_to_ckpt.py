from modules.util.convert_util import convert
from modules.util.enum.ModelType import ModelType

from torch import Tensor

from diffusers.models.embeddings import PatchEmbed


def __generate_pos_embed() -> Tensor:
    return PatchEmbed(
        height=128,
        width=128,
        patch_size=2,
        in_channels=4,
        embed_dim=16*72,
        interpolation_scale=1.0,
    ).pos_embed


def convert_pixart_diffusers_to_ckpt(
        model_type: ModelType,
        transformer_state_dict: dict,
        conversion: list,
) -> dict:
    # The diffusers -> original/PixArt transformer key map (incl. the qkv/kv full-weight fusion) is provided
    # by the caller (PixArtAlphaModel.checkpoint_diffusers_to_original()), shared with the LoRA converter.
    # model_type is unused: the alpha-only ar/csize embedder rules are overdefined and simply don't fire for
    # sigma (those keys are absent), so one map serves both.
    #
    # Two inputs are not key-mapped, matching the historical converter: caption_projection.y_embedding (a
    # learned constant the old converter left unmapped, TODO) is dropped here so strict=True still guards
    # completeness; pos_embed is generated, not copied from the diffusers state dict.
    sd = {k: v for k, v in transformer_state_dict.items() if k != "caption_projection.y_embedding"}

    state_dict = convert(sd, conversion, strict=True)
    state_dict["pos_embed"] = __generate_pos_embed()

    return state_dict
