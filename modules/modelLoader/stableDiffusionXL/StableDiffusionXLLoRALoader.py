from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.convert_lora_util import LoraConversionKeySet
from modules.util.convert.lora.convert_sdxl_lora import convert_sdxl_lora_key_sets
from modules.util.ModelNames import ModelNames


class StableDiffusionXLLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_sdxl_lora_key_sets()

    def load(
            self,
            model: StableDiffusionXLModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
