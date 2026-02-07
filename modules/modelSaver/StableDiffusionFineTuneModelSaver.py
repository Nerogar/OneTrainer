from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionModelSaver import StableDiffusionModelSaver
from modules.util import factory
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch


class StableDiffusionFineTuneModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        base_model_saver = StableDiffusionModelSaver()
        embedding_model_saver = StableDiffusionEmbeddingSaver()

        base_model_saver.save(model, model_type, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)


model_types = [ModelType.STABLE_DIFFUSION_15, ModelType.STABLE_DIFFUSION_15_INPAINTING, ModelType.STABLE_DIFFUSION_20, ModelType.STABLE_DIFFUSION_20_BASE,
              ModelType.STABLE_DIFFUSION_20_INPAINTING, ModelType.STABLE_DIFFUSION_20_DEPTH, ModelType.STABLE_DIFFUSION_21, ModelType.STABLE_DIFFUSION_21_BASE]

for model_type in model_types:
    factory.register(BaseModelSaver, StableDiffusionFineTuneModelSaver, model_type, TrainingMethod.FINE_TUNE)
    factory.register(BaseModelSaver, StableDiffusionFineTuneModelSaver, model_type, TrainingMethod.FINE_TUNE_VAE)
