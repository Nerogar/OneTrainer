from modules.model.MossModel import MossModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.moss.MossLoRALoader import MossLoRALoader
from modules.modelLoader.moss.MossModelLoader import MossModelLoader
from modules.util.enum.ModelType import ModelType

MossLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.MOSS: "resources/sd_model_spec/moss-lora.json"},
    model_class=MossModel,
    model_loader_class=MossModelLoader,
    embedding_loader_class=None,
    lora_loader_class=MossLoRALoader,
)
