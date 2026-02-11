from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.wuerstchen.WuerstchenEmbeddingLoader import WuerstchenEmbeddingLoader
from modules.modelLoader.wuerstchen.WuerstchenModelLoader import WuerstchenModelLoader
from modules.util.enum.ModelType import ModelType

WuerstchenEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={
        ModelType.WUERSTCHEN_2: "resources/sd_model_spec/wuerstchen_2.0-embedding.json",
        ModelType.STABLE_CASCADE_1: "resources/sd_model_spec/stable_cascade_1.0-embedding.json",
    },
    model_class=WuerstchenModel,
    model_loader_class=WuerstchenModelLoader,
    embedding_loader_class=WuerstchenEmbeddingLoader,
)
