from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusion3LoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _legacy_conversion(self, model: StableDiffusion3Model) -> list | None:
        return self._mixture_legacy_conversion(model)

    def load(
            self,
            model: StableDiffusion3Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
