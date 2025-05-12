from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet
from omi_model_standards.convert.lora.convert_stable_cascade_lora import convert_stable_cascade_lora_key_sets


class WuerstchenLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: WuerstchenModel) -> list[LoraConversionKeySet] | None:
        if model.model_type.is_stable_cascade():
            return convert_stable_cascade_lora_key_sets()
        return None

    def _get_state_dict(
            self,
            model: WuerstchenModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.prior_text_encoder_lora is not None:
            state_dict |= model.prior_text_encoder_lora.state_dict()
        if model.prior_prior_lora is not None:
            state_dict |= model.prior_prior_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.prior_text_encoder_embedding.placeholder

                if embedding.prior_text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_g"] = embedding.prior_text_encoder_embedding.vector
                if embedding.prior_text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_g_out"] = embedding.prior_text_encoder_embedding.output_vector

        return state_dict

    def save(
            self,
            model: WuerstchenModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)
