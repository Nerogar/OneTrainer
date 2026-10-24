from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class Krea2LoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def load(
            self,
            model: Krea2Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
