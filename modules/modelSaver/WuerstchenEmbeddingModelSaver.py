from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.util.enum.ModelType import ModelType

WuerstchenEmbeddingModelSaver = make_embedding_model_saver(
    [ModelType.WUERSTCHEN_2, ModelType.STABLE_CASCADE_1],
    model_class=WuerstchenModel,
    embedding_saver_class=WuerstchenEmbeddingSaver,
)
