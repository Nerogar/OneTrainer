from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver

WuerstchenEmbeddingModelSaver = make_embedding_model_saver(
    model_class=WuerstchenModel,
    embedding_saver_class=WuerstchenEmbeddingSaver,
)
