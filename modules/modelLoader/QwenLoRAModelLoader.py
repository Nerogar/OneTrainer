from modules.model.QwenModel import QwenModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.qwen.QwenLoRALoader import QwenLoRALoader
from modules.modelLoader.qwen.QwenModelLoader import QwenModelLoader
from modules.util.enum.ModelType import ModelType

QwenLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.QWEN: "resources/sd_model_spec/qwen-lora.json"},
    model_class=QwenModel,
    model_loader_class=QwenModelLoader,
    embedding_loader_class=None,
    lora_loader_class=QwenLoRALoader,
)
