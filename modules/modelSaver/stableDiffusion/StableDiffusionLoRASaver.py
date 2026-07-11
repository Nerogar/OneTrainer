from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class StableDiffusionLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _convert_legacy(self, model: StableDiffusionModel, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # SD's KOHYA already keeps diffusers UNet names (unlike SDXL's sgm body), so LEGACY is byte-identical
        # to KOHYA -- just reuse it.
        return self._convert_kohya(model, state_dict)

    def _get_state_dict(
            self,
            model: StableDiffusionModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.text_encoder_lora is not None:
            state_dict |= model.text_encoder_lora.state_dict()
        if model.unet_lora is not None:
            state_dict |= model.unet_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_embedding.placeholder

                if embedding.text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l"] = embedding.text_encoder_embedding.vector
                if embedding.text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l_out"] = embedding.text_encoder_embedding.output_vector

        return state_dict
