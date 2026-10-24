from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.stableDiffusion3.StableDiffusion3EmbeddingSaver import StableDiffusion3EmbeddingSaver
from modules.modelSaver.stableDiffusion3.StableDiffusion3LoRASaver import StableDiffusion3LoRASaver
from modules.util.enum.ModelType import ModelType

StableDiffusion3LoRAModelSaver = make_lora_model_saver(
    [ModelType.STABLE_DIFFUSION_3, ModelType.STABLE_DIFFUSION_35],
    model_class=StableDiffusion3Model,
    lora_saver_class=StableDiffusion3LoRASaver,
    embedding_saver_class=StableDiffusion3EmbeddingSaver,
)
