from modules.model.MossModel import MossModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class MossLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: MossModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
