from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: StableDiffusionModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
