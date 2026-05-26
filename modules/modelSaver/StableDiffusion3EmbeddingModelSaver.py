from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.stableDiffusion3.StableDiffusion3EmbeddingSaver import StableDiffusion3EmbeddingSaver
from modules.util.enum.ModelType import ModelType

StableDiffusion3EmbeddingModelSaver = make_embedding_model_saver(
    [ModelType.STABLE_DIFFUSION_3, ModelType.STABLE_DIFFUSION_35],
    model_class=StableDiffusion3Model,
    embedding_saver_class=StableDiffusion3EmbeddingSaver,
)
