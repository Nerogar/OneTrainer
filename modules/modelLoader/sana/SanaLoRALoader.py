from modules.model.BaseModel import BaseModel
from modules.model.SanaModel import SanaModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class SanaLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None

    def load(
            self,
            model: SanaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
