from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class ChromaEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: ChromaModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
