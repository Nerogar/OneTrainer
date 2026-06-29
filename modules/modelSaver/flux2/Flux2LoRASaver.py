from modules.model.Flux2Model import Flux2Model
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert_lora_util import convert_to_mixture

from torch import Tensor


class Flux2LoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _convert_legacy(self, model: Flux2Model, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_to_mixture(state_dict)

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
