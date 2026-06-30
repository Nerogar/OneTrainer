from modules.model.IdeogramModel import IdeogramModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor


class IdeogramLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_state_dict(
            self,
            model: IdeogramModel,
    ) -> dict[str, Tensor]:
        # only the conditional transformer is trained; the unconditional transformer never sees the concept
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict

    def save(
            self,
            model: IdeogramModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)
