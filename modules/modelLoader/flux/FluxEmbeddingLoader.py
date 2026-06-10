from modules.model.FluxModel import FluxModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class FluxEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: FluxModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
