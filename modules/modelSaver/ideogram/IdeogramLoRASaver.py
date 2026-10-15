from modules.model.IdeogramModel import IdeogramModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

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
