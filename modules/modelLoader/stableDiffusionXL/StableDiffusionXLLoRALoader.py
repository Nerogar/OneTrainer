from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionXLLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def load(
            self,
            model: StableDiffusionXLModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
