from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.pixartAlpha.PixArtAlphaEmbeddingSaver import PixArtAlphaEmbeddingSaver

PixArtAlphaEmbeddingModelSaver = make_embedding_model_saver(
    model_class=PixArtAlphaModel,
    embedding_saver_class=PixArtAlphaEmbeddingSaver,
)
