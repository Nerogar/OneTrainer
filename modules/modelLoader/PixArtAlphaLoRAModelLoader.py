from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.pixartAlpha.PixArtAlphaEmbeddingLoader import PixArtAlphaEmbeddingLoader
from modules.modelLoader.pixartAlpha.PixArtAlphaLoRALoader import PixArtAlphaLoRALoader
from modules.modelLoader.pixartAlpha.PixArtAlphaModelLoader import PixArtAlphaModelLoader
from modules.util.enum.ModelType import ModelType

PixArtAlphaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.PIXART_ALPHA: "resources/sd_model_spec/pixart_alpha_1.0-lora.json",
        ModelType.PIXART_SIGMA: "resources/sd_model_spec/pixart_sigma_1.0-lora.json",
    },
    model_class=PixArtAlphaModel,
    model_loader_class=PixArtAlphaModelLoader,
    embedding_loader_class=PixArtAlphaEmbeddingLoader,
    lora_loader_class=PixArtAlphaLoRALoader,
)
