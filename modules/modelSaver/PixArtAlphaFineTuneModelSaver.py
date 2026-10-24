from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.pixartAlpha.PixArtAlphaEmbeddingSaver import PixArtAlphaEmbeddingSaver
from modules.modelSaver.pixartAlpha.PixArtAlphaModelSaver import PixArtAlphaModelSaver
from modules.util.enum.ModelType import ModelType

PixArtAlphaFineTuneModelSaver = make_fine_tune_model_saver(
    [ModelType.PIXART_ALPHA, ModelType.PIXART_SIGMA],
    model_class=PixArtAlphaModel,
    model_saver_class=PixArtAlphaModelSaver,
    embedding_saver_class=PixArtAlphaEmbeddingSaver,
)
