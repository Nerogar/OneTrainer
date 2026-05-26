from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLEmbeddingLoader import StableDiffusionXLEmbeddingLoader
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLLoRALoader import StableDiffusionXLLoRALoader
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from modules.util.enum.ModelType import ModelType

StableDiffusionXLLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.STABLE_DIFFUSION_XL_10_BASE: "resources/sd_model_spec/sd_xl_base_1.0-lora.json",
        ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING: "resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json",
    },
    model_class=StableDiffusionXLModel,
    model_loader_class=StableDiffusionXLModelLoader,
    embedding_loader_class=StableDiffusionXLEmbeddingLoader,
    lora_loader_class=StableDiffusionXLLoRALoader,
)
