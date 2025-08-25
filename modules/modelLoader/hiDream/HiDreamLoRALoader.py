from modules.model.BaseModel import BaseModel
from modules.model.HiDreamModel import HiDreamModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames

from omi_model_standards.convert.lora.convert_hidream_lora import convert_hidream_lora_key_sets
from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet


class HiDreamLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_hidream_lora_key_sets()

    def load(
            self,
            model: HiDreamModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
