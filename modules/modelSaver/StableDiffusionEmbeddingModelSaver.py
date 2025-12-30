from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.util.enum.ModelType import ModelType

StableDiffusionEmbeddingModelSaver = make_embedding_model_saver(
    [ModelType.STABLE_DIFFUSION_15, ModelType.STABLE_DIFFUSION_15_INPAINTING, ModelType.STABLE_DIFFUSION_20, ModelType.STABLE_DIFFUSION_20_BASE,
     ModelType.STABLE_DIFFUSION_20_INPAINTING, ModelType.STABLE_DIFFUSION_20_DEPTH, ModelType.STABLE_DIFFUSION_21, ModelType.STABLE_DIFFUSION_21_BASE],
    model_class=StableDiffusionModel,
    embedding_saver_class=StableDiffusionEmbeddingSaver,
)
