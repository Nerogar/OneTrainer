from modules.model.HiDreamModel import HiDreamModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver
from modules.util.enum.ModelType import ModelType

HiDreamEmbeddingModelSaver = make_embedding_model_saver(
    ModelType.HI_DREAM_FULL,
    model_class=HiDreamModel,
    embedding_saver_class=HiDreamEmbeddingSaver,
)
