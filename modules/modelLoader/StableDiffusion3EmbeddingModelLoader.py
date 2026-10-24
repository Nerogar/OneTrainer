from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.stableDiffusion3.StableDiffusion3EmbeddingLoader import StableDiffusion3EmbeddingLoader
from modules.modelLoader.stableDiffusion3.StableDiffusion3ModelLoader import StableDiffusion3ModelLoader
from modules.util.enum.ModelType import ModelType

StableDiffusion3EmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={
        ModelType.STABLE_DIFFUSION_3: "resources/sd_model_spec/sd_3_2b_1.0-embedding.json",
        ModelType.STABLE_DIFFUSION_35: "resources/sd_model_spec/sd_3.5_1.0-embedding.json",
    },
    model_class=StableDiffusion3Model,
    model_loader_class=StableDiffusion3ModelLoader,
    embedding_loader_class=StableDiffusion3EmbeddingLoader,
)
