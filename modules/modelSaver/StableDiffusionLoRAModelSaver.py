from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionLoRASaver import StableDiffusionLoRASaver

StableDiffusionLoRAModelSaver = make_lora_model_saver(
    model_class=StableDiffusionModel,
    lora_saver_class=StableDiffusionLoRASaver,
    embedding_saver_class=StableDiffusionEmbeddingSaver,
)
