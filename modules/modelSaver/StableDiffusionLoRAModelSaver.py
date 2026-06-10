from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionLoRASaver import StableDiffusionLoRASaver
from modules.util.enum.ModelType import ModelType

StableDiffusionLoRAModelSaver = make_lora_model_saver(
    [ModelType.STABLE_DIFFUSION_15, ModelType.STABLE_DIFFUSION_15_INPAINTING, ModelType.STABLE_DIFFUSION_20, ModelType.STABLE_DIFFUSION_20_BASE,
     ModelType.STABLE_DIFFUSION_20_INPAINTING, ModelType.STABLE_DIFFUSION_20_DEPTH, ModelType.STABLE_DIFFUSION_21, ModelType.STABLE_DIFFUSION_21_BASE],
    model_class=StableDiffusionModel,
    lora_saver_class=StableDiffusionLoRASaver,
    embedding_saver_class=StableDiffusionEmbeddingSaver,
)
