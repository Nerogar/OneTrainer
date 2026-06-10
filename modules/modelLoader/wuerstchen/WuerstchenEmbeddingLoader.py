from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class WuerstchenEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: WuerstchenModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
