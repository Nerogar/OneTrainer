from collections.abc import Callable

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.VideoFormat import VideoFormat

import torch
from torchvision.transforms import transforms

from PIL import Image


class StableDiffusionVaeSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: StableDiffusionModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        # TODO: this is reusing the prompt parameters as the image path, think of a better solution
        image = Image.open(sample_config.prompt)
        image = image.convert("RGB")
        # TODO: figure out better set of transformations for resize and/or implement way to configure them as per-sample toggle

        scale = sample_config.width if sample_config.width > sample_config.height else sample_config.height

        t_in = transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop([sample_config.height, sample_config.width]),
            transforms.ToTensor()
        ])
        image_tensor = t_in(image).to(device=self.train_device, dtype=self.model.vae.dtype)
        image_tensor = image_tensor * 2 - 1

        self.model.vae_to(self.train_device)

        with torch.no_grad():
            latent_image_tensor = self.model.vae.encode(image_tensor.unsqueeze(0)).latent_dist.mean
            image_tensor = self.model.vae.decode(latent_image_tensor).sample.squeeze()

        self.model.vae_to(self.temp_device)

        image_tensor = (image_tensor + 1) * 0.5
        image_tensor = image_tensor.clamp(0, 1)

        t_out = transforms.ToPILImage()
        sampler_output = ModelSamplerOutput(
            file_type=FileType.IMAGE,
            data=t_out(image_tensor.float()),
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_15, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_15_INPAINTING, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_20, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_20_BASE, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_20_INPAINTING, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_20_DEPTH, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_21, TrainingMethod.FINE_TUNE_VAE)
factory.register(BaseModelSampler, StableDiffusionVaeSampler, ModelType.STABLE_DIFFUSION_21_BASE, TrainingMethod.FINE_TUNE_VAE)
