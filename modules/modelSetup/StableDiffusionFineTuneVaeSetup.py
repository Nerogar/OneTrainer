from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class StableDiffusionFineTuneVaeSetup(
    BaseStableDiffusionSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        parameter_group_collection.add_group(NamedParameterGroup(
            unique_name="vae",
            parameters=model.vae.decoder.parameters(),
            learning_rate=config.learning_rate,
        ))

        return parameter_group_collection

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.vae.decoder.requires_grad_(True)
        model.unet.requires_grad_(False)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.to(self.temp_device)
        model.vae.to(self.train_device)
        model.unet.to(self.temp_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.temp_device)

        model.text_encoder.eval()
        model.vae.train()
        model.unet.eval()

    def predict(
            self,
            model: StableDiffusionModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        latent_image = batch['latent_image']
        image = batch['image']

        predicted_image = model.vae.decode(latent_image, return_dict=True).sample

        model_output_data = {
            'loss_type': 'target',
            'predicted': predicted_image,
            'target': image,
        }

        if config.debug_mode:
            with torch.no_grad():
                # image
                self._save_image(image, config.debug_dir + "/training_batches", "1-image", train_progress.global_step)

                # predicted image
                predicted_image_clamped = predicted_image.clamp(-1, 1)
                self._save_image(
                    predicted_image_clamped, config.debug_dir + "/training_batches", "2-predicted_image",
                    train_progress.global_step
                )

        model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type
        return model_output_data

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        pass

factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_15, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_15_INPAINTING, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_20, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_20_BASE, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_20_INPAINTING, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_20_DEPTH, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_21, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSetup, StableDiffusionFineTuneVaeSetup, ModelType.STABLE_DIFFUSION_21_BASE, TrainingMethod.FINE_TUNE_VAE)
