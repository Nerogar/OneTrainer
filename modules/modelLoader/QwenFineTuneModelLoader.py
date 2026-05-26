from modules.model.QwenModel import QwenModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.qwen.QwenModelLoader import QwenModelLoader
from modules.util.enum.ModelType import ModelType

QwenFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.QWEN: "resources/sd_model_spec/qwen.json"},
    model_class=QwenModel,
    model_loader_class=QwenModelLoader,
    embedding_loader_class=None,
)
