from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionXLEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: StableDiffusionXLModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
