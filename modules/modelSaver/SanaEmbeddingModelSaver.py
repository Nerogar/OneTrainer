from modules.model.SanaModel import SanaModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.sana.SanaEmbeddingSaver import SanaEmbeddingSaver
from modules.util.enum.ModelType import ModelType

SanaEmbeddingModelSaver = make_embedding_model_saver(
    ModelType.SANA,
    model_class=SanaModel,
    embedding_saver_class=SanaEmbeddingSaver,
)
