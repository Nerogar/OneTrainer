from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.stableDiffusion3.StableDiffusion3EmbeddingSaver import StableDiffusion3EmbeddingSaver
from modules.modelSaver.stableDiffusion3.StableDiffusion3ModelSaver import StableDiffusion3ModelSaver
from modules.util.enum.ModelType import ModelType

StableDiffusion3FineTuneModelSaver = make_fine_tune_model_saver(
    [ModelType.STABLE_DIFFUSION_3, ModelType.STABLE_DIFFUSION_35],
    model_class=StableDiffusion3Model,
    model_saver_class=StableDiffusion3ModelSaver,
    embedding_saver_class=StableDiffusion3EmbeddingSaver,
)
