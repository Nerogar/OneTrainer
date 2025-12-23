from modules.model.HiDreamModel import HiDreamModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver

HiDreamEmbeddingModelSaver = make_embedding_model_saver(
    model_class=HiDreamModel,
    embedding_saver_class=HiDreamEmbeddingSaver,
)
