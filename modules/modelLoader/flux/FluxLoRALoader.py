from modules.model.FluxModel import FluxModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class FluxLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _legacy_conversion(self, model: FluxModel) -> list | None:
        return self._mixture_legacy_conversion(model)

    def load(
            self,
            model: FluxModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
