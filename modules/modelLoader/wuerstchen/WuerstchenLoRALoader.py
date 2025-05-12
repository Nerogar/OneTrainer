from modules.model.BaseModel import BaseModel
from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames

from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet
from omi_model_standards.convert.lora.convert_stable_cascade_lora import convert_stable_cascade_lora_key_sets


class WuerstchenLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        if model.model_type.is_stable_cascade():
            return convert_stable_cascade_lora_key_sets()
        return None

    def load(
            self,
            model: WuerstchenModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
