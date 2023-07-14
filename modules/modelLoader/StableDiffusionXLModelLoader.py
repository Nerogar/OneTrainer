import json
import os

import torch
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
    def __load_internal(model_type: ModelType, base_model_name: str) -> StableDiffusionXLModel | None:
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
            model = StableDiffusionXLModelLoader.__load_diffusers(model_type, base_model_name)

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

            # meta
            model.train_progress = train_progress

            return model
        except:
            return None

    @staticmethod
    def __load_diffusers(model_type: ModelType, base_model_name: str) -> StableDiffusionXLModel | None:
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
                torch_dtype=torch.float32,
            )

            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                base_model_name,
                subfolder="text_encoder_2",
                torch_dtype=torch.float32,
            )

            vae = AutoencoderKL.from_pretrained(
                base_model_name,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            unet = UNet2DConditionModel.from_pretrained(
                base_model_name,
                subfolder="unet",
                torch_dtype=torch.float32,
            )

            return StableDiffusionXLModel(
                model_type=model_type,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                vae=vae,
                unet=unet,
            )
        except:
            return None

    @staticmethod
    def __load_ckpt(model_type: ModelType, base_model_name: str) -> StableDiffusionXLModel | None:
        try:
            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                load_safety_checker=False,
            )

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

            return StableDiffusionXLModel(
                model_type=model_type,
                tokenizer_1=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                noise_scheduler=noise_scheduler,
                text_encoder_1=pipeline.text_encoder.to(dtype=torch.float32),
                text_encoder_2=pipeline.text_encoder_2.to(dtype=torch.float32),
                vae=pipeline.vae.to(dtype=torch.float32),
                unet=pipeline.unet.to(dtype=torch.float32),
            )
        except:
            return None

    @staticmethod
    def __load_safetensors(model_type: ModelType, base_model_name: str) -> StableDiffusionXLModel | None:
        try:
            pipeline = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=base_model_name,
                load_safety_checker=False,
                from_safetensors=True,
            )

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

            return StableDiffusionXLModel(
                model_type=model_type,
                tokenizer_1=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                noise_scheduler=noise_scheduler,
                text_encoder_1=pipeline.text_encoder.to(dtype=torch.float32),
                text_encoder_2=pipeline.text_encoder_2.to(dtype=torch.float32),
                vae=pipeline.vae.to(dtype=torch.float32),
                unet=pipeline.unet.to(dtype=torch.float32),
            )
        except:
            return None

    def load(
            self,
            model_type: ModelType,
            base_model_name: str,
            extra_model_name: str | None
    ) -> StableDiffusionXLModel | None:
        model = self.__load_internal(model_type, base_model_name)
        if model is not None:
            return model

        model = self.__load_diffusers(model_type, base_model_name)
        if model is not None:
            return model

        model = self.__load_safetensors(model_type, base_model_name)
        if model is not None:
            return model

        model = self.__load_ckpt(model_type, base_model_name)
        if model is not None:
            return model

        raise Exception("could not load model: " + base_model_name)
