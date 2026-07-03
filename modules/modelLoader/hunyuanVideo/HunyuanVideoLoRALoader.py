from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class HunyuanVideoLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _legacy_conversion(self, model: HunyuanVideoModel) -> list | None:
        return self._mixture_legacy_conversion(model)

    def load(
            self,
            model: HunyuanVideoModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
