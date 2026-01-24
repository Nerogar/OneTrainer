
from modules.model.Flux2Model import Flux2Model
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor


class Flux2LoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: Flux2Model) -> list[LoraConversionKeySet] | None:
        return None

    def _get_state_dict(
            self,
            model: Flux2Model,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        return state_dict

    def save(
            self,
            model: Flux2Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)
