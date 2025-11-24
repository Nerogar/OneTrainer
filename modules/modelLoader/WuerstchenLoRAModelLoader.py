from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.wuerstchen.WuerstchenEmbeddingLoader import WuerstchenEmbeddingLoader
from modules.modelLoader.wuerstchen.WuerstchenLoRALoader import WuerstchenLoRALoader
from modules.modelLoader.wuerstchen.WuerstchenModelLoader import WuerstchenModelLoader
from modules.util.enum.ModelType import ModelType

WuerstchenLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.WUERSTCHEN_2: "resources/sd_model_spec/wuerstchen_2.0-lora.json",
        ModelType.STABLE_CASCADE_1: "resources/sd_model_spec/stable_cascade_1.0-lora.json",
    },
    model_class=WuerstchenModel,
    model_loader_class=WuerstchenModelLoader,
    embedding_loader_class=WuerstchenEmbeddingLoader,
    lora_loader_class=WuerstchenLoRALoader,
)
