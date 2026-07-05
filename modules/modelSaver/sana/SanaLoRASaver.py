from modules.model.SanaModel import SanaModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class SanaLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_state_dict(
            self,
            model: SanaModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.text_encoder_lora is not None:
            state_dict |= model.text_encoder_lora.state_dict()
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_embedding.placeholder

                if embedding.text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.gemma"] = embedding.text_encoder_embedding.vector
                if embedding.text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.gemma_out"] = embedding.text_encoder_embedding.output_vector

        return state_dict
