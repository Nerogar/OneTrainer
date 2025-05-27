from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from omi_model_standards.convert.lora.convert_hunyuan_video_lora import convert_hunyuan_video_lora_key_sets
from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet


class HunyuanVideoLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: HunyuanVideoModel) -> list[LoraConversionKeySet] | None:
        return convert_hunyuan_video_lora_key_sets()

    def _get_state_dict(
            self,
            model: HunyuanVideoModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.text_encoder_1_lora is not None:
            state_dict |= model.text_encoder_1_lora.state_dict()
        if model.text_encoder_2_lora is not None:
            state_dict |= model.text_encoder_2_lora.state_dict()
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_1_embedding.placeholder

                if embedding.text_encoder_1_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.llama"] = embedding.text_encoder_1_embedding.vector
                if embedding.text_encoder_2_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l"] = embedding.text_encoder_2_embedding.vector
                if embedding.text_encoder_1_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.llama_out"] = embedding.text_encoder_1_embedding.output_vector
                if embedding.text_encoder_2_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l_out"] = embedding.text_encoder_2_embedding.output_vector

        return state_dict

    def save(
            self,
            model: HunyuanVideoModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)
