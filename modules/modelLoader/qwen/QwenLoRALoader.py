from modules.model.QwenModel import QwenModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class QwenLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def load(
            self,
            model: QwenModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
