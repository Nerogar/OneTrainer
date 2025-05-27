from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames

from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet
from omi_model_standards.convert.lora.convert_sd3_lora import convert_sd3_lora_key_sets


class StableDiffusion3LoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_sd3_lora_key_sets()

    def load(
            self,
            model: StableDiffusion3Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
