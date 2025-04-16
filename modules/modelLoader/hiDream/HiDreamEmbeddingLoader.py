from modules.model.HiDreamModel import HiDreamModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class HiDreamEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: HiDreamModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
