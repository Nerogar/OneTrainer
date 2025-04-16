from modules.model.BaseModel import BaseModel
from modules.model.HiDreamModel import HiDreamModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class HiDreamLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None  # TODO: not yet implemented

    def load(
            self,
            model: HiDreamModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
