from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionModelSaver import StableDiffusionModelSaver
from modules.util import factory
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

model_types = [ModelType.STABLE_DIFFUSION_15, ModelType.STABLE_DIFFUSION_15_INPAINTING, ModelType.STABLE_DIFFUSION_20, ModelType.STABLE_DIFFUSION_20_BASE,
              ModelType.STABLE_DIFFUSION_20_INPAINTING, ModelType.STABLE_DIFFUSION_20_DEPTH, ModelType.STABLE_DIFFUSION_21, ModelType.STABLE_DIFFUSION_21_BASE],


StableDiffusionFineTuneModelSaver = make_fine_tune_model_saver(
    model_types,
    model_class=StableDiffusionModel,
    model_saver_class=StableDiffusionModelSaver,
    embedding_saver_class=StableDiffusionEmbeddingSaver,
)

#make_fine_tune_model_saver only registers for FINE_TUNE:
for model_type in model_types:
    factory.register(BaseModelSaver, StableDiffusionFineTuneModelSaver, model_type, TrainingMethod.FINE_TUNE_VAE)
