from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class PixArtAlphaEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: PixArtAlphaModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
