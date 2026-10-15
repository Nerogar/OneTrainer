from modules.model.ZImageModel import ZImageModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class ZImageLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _convert_legacy(self, model: ZImageModel, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # Older OneTrainer versions saved this model's LoRA unconverted (canonical / diffusers-dotted),
        # so identity reproduces that output.
        return dict(state_dict)

    def _get_state_dict(
            self,
            model: ZImageModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict
