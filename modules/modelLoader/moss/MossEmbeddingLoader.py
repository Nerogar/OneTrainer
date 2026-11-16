from modules.model.MossModel import MossModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class MossEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: MossModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
