from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.stableDiffusion.StableDiffusionEmbeddingLoader import StableDiffusionEmbeddingLoader
from modules.modelLoader.stableDiffusion.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.util.enum.ModelType import ModelType

StableDiffusionEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={
        ModelType.STABLE_DIFFUSION_15: "resources/sd_model_spec/sd_1.5-embedding.json",
        ModelType.STABLE_DIFFUSION_15_INPAINTING: "resources/sd_model_spec/sd_1.5_inpainting-embedding.json",
        ModelType.STABLE_DIFFUSION_20: "resources/sd_model_spec/sd_2.0-embedding.json",
        ModelType.STABLE_DIFFUSION_20_BASE: "resources/sd_model_spec/sd_2.0-embedding.json",
        ModelType.STABLE_DIFFUSION_20_INPAINTING: "resources/sd_model_spec/sd_2.0_inpainting-embedding.json",
        ModelType.STABLE_DIFFUSION_20_DEPTH: "resources/sd_model_spec/sd_2.0_depth-embedding.json",
        ModelType.STABLE_DIFFUSION_21: "resources/sd_model_spec/sd_2.1-embedding.json",
        ModelType.STABLE_DIFFUSION_21_BASE: "resources/sd_model_spec/sd_2.1-embedding.json",
    },
    model_class=StableDiffusionModel,
    model_loader_class=StableDiffusionModelLoader,
    embedding_loader_class=StableDiffusionEmbeddingLoader,
)
