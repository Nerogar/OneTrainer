from modules.model.MossModel import MossModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.moss.MossEmbeddingSaver import MossEmbeddingSaver
from modules.util.enum.ModelType import ModelType

MossEmbeddingModelSaver = make_embedding_model_saver(
    model_types=ModelType.MOSS,
    model_class=MossModel,
    embedding_saver_class=MossEmbeddingSaver,
)
