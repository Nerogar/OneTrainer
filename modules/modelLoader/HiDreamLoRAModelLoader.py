from modules.model.HiDreamModel import HiDreamModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.hiDream.HiDreamEmbeddingLoader import HiDreamEmbeddingLoader
from modules.modelLoader.hiDream.HiDreamLoRALoader import HiDreamLoRALoader
from modules.modelLoader.hiDream.HiDreamModelLoader import HiDreamModelLoader
from modules.util.enum.ModelType import ModelType

HiDreamLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.HI_DREAM_FULL: "resources/sd_model_spec/hi_dream_full-lora.json"},
    model_class=HiDreamModel,
    model_loader_class=HiDreamModelLoader,
    embedding_loader_class=HiDreamEmbeddingLoader,
    lora_loader_class=HiDreamLoRALoader,
)
