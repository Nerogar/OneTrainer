from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusion3EmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: StableDiffusion3Model,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
