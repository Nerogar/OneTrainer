from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import StableDiffusionXLEmbeddingSaver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLLoRASaver import StableDiffusionXLLoRASaver
from modules.util.enum.ModelType import ModelType

StableDiffusionXLLoRAModelSaver = make_lora_model_saver(
    [ModelType.STABLE_DIFFUSION_XL_10_BASE, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING],
    model_class=StableDiffusionXLModel,
    lora_saver_class=StableDiffusionXLLoRASaver,
    embedding_saver_class=StableDiffusionXLEmbeddingSaver,
)
