import json
import os

import torch
import yaml
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from transformers import CLIPTokenizer, CLIPTextModel, DPTImageProcessor, DPTForDepthEstimation

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionModelLoader(BaseModelLoader):
    def __init__(self):
        super(StableDiffusionModelLoader, self).__init__()

    @staticmethod
    def __default_yaml_name(model_type: ModelType) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/diffusers_model_config/v1-inference.yaml"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/diffusers_model_config/v1-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/diffusers_model_config/v2-inference-v.yaml"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/diffusers_model_config/v2-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/diffusers_model_config/v2-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/diffusers_model_config/v2-midas-inference.yaml"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/diffusers_model_config/v2-inference-v.yaml"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/diffusers_model_config/v2-inference.yaml"
            case _:
                return None

    @staticmethod
    def __load_internal(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionModel | None:
        try:
            with open(os.path.join(base_model_name, "meta.json"), "r") as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )

            # base model
            model = StableDiffusionModelLoader.__load_diffusers(model_type, weight_dtype, base_model_name)

            # optimizer
            try:
                model.optimizer_state_dict = torch.load(os.path.join(base_model_name, "optimizer", "optimizer.pt"))
            except FileNotFoundError:
                pass

            # ema
            try:
                model.ema_state_dict = torch.load(os.path.join(base_model_name, "ema", "ema.pt"))
            except FileNotFoundError:
                pass

            with open(StableDiffusionModelLoader.__default_yaml_name(model_type), "r") as f:
                model.sd_config = yaml.safe_load(f)

            # meta
            model.train_progress = train_progress

            return model
        except:
            return None

    @staticmethod
    def __load_diffusers(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionModel | None:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            )

            noise_scheduler = DDIMScheduler.from_pretrained(
                base_model_name,
                subfolder="scheduler",
            )

            text_encoder = CLIPTextModel.from_pretrained(
                base_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtype,
            )

            vae = AutoencoderKL.from_pretrained(
                base_model_name,
                subfolder="vae",
                torch_dtype=weight_dtype,
            )

            unet = UNet2DConditionModel.from_pretrained(
                base_model_name,
                subfolder="unet",
                torch_dtype=weight_dtype,
            )

            image_depth_processor = DPTImageProcessor.from_pretrained(
                base_model_name,
                subfolder="feature_extractor",
            ) if model_type.has_depth_input() else None

            depth_estimator = DPTForDepthEstimation.from_pretrained(
                base_model_name,
                subfolder="depth_estimator",
                torch_dtype=weight_dtype,
            ) if model_type.has_depth_input() else None

            with open(StableDiffusionModelLoader.__default_yaml_name(model_type), "r") as f:
                sd_config = yaml.safe_load(f)

            return StableDiffusionModel(
                model_type=model_type,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                image_depth_processor=image_depth_processor,
                depth_estimator=depth_estimator,
                sd_config=sd_config,
            )
        except:
            return None

    @staticmethod
    def __load_ckpt(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionModel | None:
        try:
            yaml_name = os.path.splitext(base_model_name)[0] + '.yaml'
            if not os.path.exists(yaml_name):
                yaml_name = os.path.splitext(base_model_name)[0] + '.yml'
                if not os.path.exists(yaml_name):
                    yaml_name = StableDiffusionModelLoader.__default_yaml_name(model_type)

            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                original_config_file=yaml_name,
                load_safety_checker=False,
            ).to(torch_dtype=weight_dtype)

            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="epsilon",
            )

            with open(yaml_name, "r") as f:
                sd_config = yaml.safe_load(f)

            return StableDiffusionModel(
                model_type=model_type,
                tokenizer=pipeline.tokenizer,
                noise_scheduler=noise_scheduler,
                text_encoder=pipeline.text_encoder.to(dtype=torch.float32),
                vae=pipeline.vae.to(dtype=torch.float32),
                unet=pipeline.unet.to(dtype=torch.float32),
                image_depth_processor=None,  # TODO
                depth_estimator=None,  # TODO
                sd_config=sd_config,
            )
        except:
            return None

    @staticmethod
    def __load_safetensors(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionModel | None:
        try:
            yaml_name = os.path.splitext(base_model_name)[0] + '.yaml'
            if not os.path.exists(yaml_name):
                yaml_name = os.path.splitext(base_model_name)[0] + '.yml'
                if not os.path.exists(yaml_name):
                    yaml_name = StableDiffusionModelLoader.__default_yaml_name(model_type)

            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                original_config_file=yaml_name,
                load_safety_checker=False,
                from_safetensors=True,
            ).to(torch_dtype=weight_dtype)

            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="epsilon",
            )

            with open(yaml_name, "r") as f:
                sd_config = yaml.safe_load(f)

            return StableDiffusionModel(
                model_type=model_type,
                tokenizer=pipeline.tokenizer,
                noise_scheduler=noise_scheduler,
                text_encoder=pipeline.text_encoder.to(dtype=torch.float32),
                vae=pipeline.vae.to(dtype=torch.float32),
                unet=pipeline.unet.to(dtype=torch.float32),
                image_depth_processor=None,  # TODO
                depth_estimator=None,  # TODO
                sd_config=sd_config,
            )
        except:
            return None

    def load(
            self,
            model_type: ModelType,
            weight_dtype: torch.dtype,
            base_model_name: str,
            extra_model_name: str | None
    ) -> StableDiffusionModel | None:
        model = self.__load_internal(model_type, weight_dtype, base_model_name)
        if model is not None:
            return model

        model = self.__load_diffusers(model_type, weight_dtype, base_model_name)
        if model is not None:
            return model

        model = self.__load_safetensors(model_type, weight_dtype, base_model_name)
        if model is not None:
            return model

        model = self.__load_ckpt(model_type, weight_dtype, base_model_name)
        if model is not None:
            return model

        raise Exception("could not load model: " + base_model_name)
