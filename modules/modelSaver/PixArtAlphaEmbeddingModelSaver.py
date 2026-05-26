from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.pixartAlpha.PixArtAlphaEmbeddingSaver import PixArtAlphaEmbeddingSaver
from modules.util.enum.ModelType import ModelType

PixArtAlphaEmbeddingModelSaver = make_embedding_model_saver(
    [ModelType.PIXART_ALPHA, ModelType.PIXART_SIGMA],
    model_class=PixArtAlphaModel,
    embedding_saver_class=PixArtAlphaEmbeddingSaver,
)
