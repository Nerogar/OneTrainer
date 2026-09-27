from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: StableDiffusionModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
