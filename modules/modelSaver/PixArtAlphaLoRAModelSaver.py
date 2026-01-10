from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.pixartAlpha.PixArtAlphaEmbeddingSaver import PixArtAlphaEmbeddingSaver
from modules.modelSaver.pixartAlpha.PixArtAlphaLoRASaver import PixArtAlphaLoRASaver

PixArtAlphaLoRAModelSaver = make_lora_model_saver(
    model_class=PixArtAlphaModel,
    lora_saver_class=PixArtAlphaLoRASaver,
    embedding_saver_class=PixArtAlphaEmbeddingSaver,
)
