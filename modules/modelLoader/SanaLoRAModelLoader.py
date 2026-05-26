from modules.model.SanaModel import SanaModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.sana.SanaEmbeddingLoader import SanaEmbeddingLoader
from modules.modelLoader.sana.SanaLoRALoader import SanaLoRALoader
from modules.modelLoader.sana.SanaModelLoader import SanaModelLoader
from modules.util.enum.ModelType import ModelType

SanaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.SANA: "resources/sd_model_spec/sana-lora.json"},
    model_class=SanaModel,
    model_loader_class=SanaModelLoader,
    embedding_loader_class=SanaEmbeddingLoader,
    lora_loader_class=SanaLoRALoader,
)
