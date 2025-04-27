from modules.model.SanaModel import SanaModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class SanaEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: SanaModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
