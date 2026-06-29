from modules.model.SanaModel import SanaModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class SanaLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def _legacy_conversion(self, model: SanaModel) -> list | None:
        # Sana dropped LEGACY: its only output was a never-loadable dotted format (see the saver).
        return None

    def load(
            self,
            model: SanaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
