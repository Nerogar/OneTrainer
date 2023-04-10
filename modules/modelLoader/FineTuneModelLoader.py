import os

import torch
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from transformers import CLIPTokenizer, CLIPTextModel, DPTImageProcessor, DPTForDepthEstimation

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.util.enum.ModelType import ModelType


class FineTuneModelLoader(BaseModelLoader):
    def __init__(self):
        super(FineTuneModelLoader, self).__init__()

    @staticmethod
    def __load_diffusers_model(base_model_name: str, model_type: ModelType, dtype: torch.dtype) -> StableDiffusionModel | None:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer"
            )

            noise_scheduler = DDPMScheduler.from_config(
                base_model_name,
                subfolder="scheduler"
            )

            text_encoder = CLIPTextModel.from_pretrained(
                base_model_name,
                subfolder="text_encoder",
                torch_dtype=dtype,
            )

            vae = AutoencoderKL.from_pretrained(
                base_model_name,
                subfolder="vae",
                torch_dtype=dtype,
            )

            unet = UNet2DConditionModel.from_pretrained(
                base_model_name,
                subfolder="unet",
                torch_dtype=dtype,
            )

            image_depth_processor = DPTImageProcessor.from_pretrained(
                base_model_name,
                subfolder="feature_extractor"
            ) if model_type.has_depth_input() else None

            depth_estimator = DPTForDepthEstimation.from_pretrained(
                base_model_name,
                subfolder="depth_estimator"
            ) if model_type.has_depth_input() else None

            return StableDiffusionModel(
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                image_depth_processor=image_depth_processor,
                depth_estimator=depth_estimator,
            )
        except:
            return None

    @staticmethod
    def __default_yaml_name(model_type: ModelType) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/diffusers_model_config/v1-inference.yaml"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/diffusers_model_config/v1-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/diffusers_model_config/v2-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/diffusers_model_config/v2-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/diffusers_model_config/v2-midas-inference.yaml"
            case _:
                return None

    @staticmethod
    def __load_ckpt_model(base_model_name: str, model_type: ModelType, dtype: torch.dtype) -> StableDiffusionModel | None:
        try:
            yaml_name = os.path.splitext(base_model_name)[0] + '.yaml'
            if not os.path.exists(yaml_name):
                yaml_name = os.path.splitext(base_model_name)[0] + '.yml'
                if not os.path.exists(yaml_name):
                    yaml_name = FineTuneModelLoader.__default_yaml_name(model_type)

            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                original_config_file=yaml_name,
            )

            return StableDiffusionModel(
                tokenizer=pipeline.tokenizer,
                noise_scheduler=pipeline.scheduler,
                text_encoder=pipeline.text_encoder.to(dtype=dtype),
                vae=pipeline.vae.to(dtype=dtype),
                unet=pipeline.unet.to(dtype=dtype),
                image_depth_processor=None,  # TODO
                depth_estimator=None,  # TODO
            )
        except:
            return None

    def load(self, base_model_name: str, model_type: ModelType, dtype: torch.dtype) -> StableDiffusionModel | None:
        model = self.__load_diffusers_model(base_model_name, model_type, dtype)
        if model is not None:
            return model

        model = self.__load_ckpt_model(base_model_name, model_type, dtype)
        if model is not None:
            return model

        return None
