from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import StableDiffusionXLEmbeddingSaver
from modules.util.enum.ModelType import ModelType

StableDiffusionXLEmbeddingModelSaver = make_embedding_model_saver(
    [ModelType.STABLE_DIFFUSION_XL_10_BASE, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING],
    model_class=StableDiffusionXLModel,
    embedding_saver_class=StableDiffusionXLEmbeddingSaver,
)
