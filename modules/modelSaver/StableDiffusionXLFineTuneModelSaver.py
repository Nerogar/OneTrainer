from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import StableDiffusionXLEmbeddingSaver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLModelSaver import StableDiffusionXLModelSaver
from modules.util.enum.ModelType import ModelType

StableDiffusionXLFineTuneModelSaver = make_fine_tune_model_saver(
    [ModelType.STABLE_DIFFUSION_XL_10_BASE, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING],
    model_class=StableDiffusionXLModel,
    model_saver_class=StableDiffusionXLModelSaver,
    embedding_saver_class=StableDiffusionXLEmbeddingSaver,
)
