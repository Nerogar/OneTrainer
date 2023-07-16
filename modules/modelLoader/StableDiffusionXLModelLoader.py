import json
import os
import traceback

import torch
import yaml
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionXLModelLoader(BaseModelLoader):
    def __init__(self):
        super(StableDiffusionXLModelLoader, self).__init__()

    @staticmethod
    def __default_yaml_name(model_type: ModelType) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/diffusers_model_config/sd_xl_base.yaml"
            case _:
                return None

    @staticmethod
    def __load_internal(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionXLModel | None:
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
            model = StableDiffusionXLModelLoader.__load_diffusers(model_type, weight_dtype, base_model_name)

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

            with open(StableDiffusionXLModelLoader.__default_yaml_name(model_type), "r") as f:
                model.sd_config = yaml.safe_load(f)

            # meta
            model.train_progress = train_progress

            return model
        except:
            traceback.print_exc()
            return None

    @staticmethod
    def __load_diffusers(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionXLModel | None:
        try:
            tokenizer_1 = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            )

            tokenizer_2 = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer_2",
            )

            noise_scheduler = DDIMScheduler.from_pretrained(
                base_model_name,
                subfolder="scheduler",
            )

            text_encoder_1 = CLIPTextModel.from_pretrained(
                base_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtype,
            )

            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                base_model_name,
                subfolder="text_encoder_2",
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

            with open(StableDiffusionXLModelLoader.__default_yaml_name(model_type), "r") as f:
                sd_config = yaml.safe_load(f)

            return StableDiffusionXLModel(
                model_type=model_type,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                vae=vae,
                unet=unet,
                sd_config=sd_config,
            )
        except:
            traceback.print_exc()
            return None

    @staticmethod
    def __load_ckpt(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionXLModel | None:
        try:
            yaml_name = os.path.splitext(base_model_name)[0] + '.yaml'
            if not os.path.exists(yaml_name):
                yaml_name = os.path.splitext(base_model_name)[0] + '.yml'
                if not os.path.exists(yaml_name):
                    yaml_name = StableDiffusionXLModelLoader.__default_yaml_name(model_type)

            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                original_config_file=yaml_name,
                load_safety_checker=False,
            ).to(torch_device=weight_dtype)

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

            return StableDiffusionXLModel(
                model_type=model_type,
                tokenizer_1=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                noise_scheduler=noise_scheduler,
                text_encoder_1=pipeline.text_encoder.to(dtype=torch.float32),
                text_encoder_2=pipeline.text_encoder_2.to(dtype=torch.float32),
                vae=pipeline.vae.to(dtype=torch.float32),
                unet=pipeline.unet.to(dtype=torch.float32),
                sd_config=sd_config,
            )
        except:
            traceback.print_exc()
            return None

    @staticmethod
    def __load_safetensors(model_type: ModelType, weight_dtype: torch.dtype, base_model_name: str) -> StableDiffusionXLModel | None:
        try:
            yaml_name = os.path.splitext(base_model_name)[0] + '.yaml'
            if not os.path.exists(yaml_name):
                yaml_name = os.path.splitext(base_model_name)[0] + '.yml'
                if not os.path.exists(yaml_name):
                    yaml_name = StableDiffusionXLModelLoader.__default_yaml_name(model_type)

            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                original_config_file=yaml_name,
                load_safety_checker=False,
                from_safetensors=True,
            ).to(torch_device=weight_dtype)

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

            return StableDiffusionXLModel(
                model_type=model_type,
                tokenizer_1=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                noise_scheduler=noise_scheduler,
                text_encoder_1=pipeline.text_encoder.to(dtype=torch.float32),
                text_encoder_2=pipeline.text_encoder_2.to(dtype=torch.float32),
                vae=pipeline.vae.to(dtype=torch.float32),
                unet=pipeline.unet.to(dtype=torch.float32),
                sd_config=sd_config,
            )
        except:
            traceback.print_exc()
            return None

    def load(
            self,
            model_type: ModelType,
            weight_dtype: torch.dtype,
            base_model_name: str,
            extra_model_name: str | None
    ) -> StableDiffusionXLModel | None:
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
