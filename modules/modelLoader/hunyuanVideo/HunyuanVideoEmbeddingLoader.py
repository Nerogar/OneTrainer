from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class HunyuanVideoEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: HunyuanVideoModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
