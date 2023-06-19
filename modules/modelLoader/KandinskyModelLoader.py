import json
import os

import torch
from diffusers import DDIMScheduler, PriorTransformer, UNet2DConditionModel, VQModel, UnCLIPScheduler
from diffusers.pipelines.kandinsky import MultilingualCLIP
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, \
    CLIPImageProcessor, XLMRobertaTokenizerFast

from modules.model.KandinskyModel import KandinskyModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class KandinskyModelLoader(BaseModelLoader):
    def __init__(self):
        super(KandinskyModelLoader, self).__init__()

    @staticmethod
    def __load_internal(model_type: ModelType, base_model_name: str) -> KandinskyModel | None:
        try:
            with open(os.path.join(base_model_name, "meta.json"), "r") as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )

            prior_model_name = os.path.join(base_model_name, "prior_model")
            diffusion_model_name = os.path.join(base_model_name, "diffusion_model")

            # base model
            model = KandinskyModelLoader.__load_diffusers(model_type, prior_model_name, diffusion_model_name)

            # optimizer
            try:
                model.optimizer_state_dict = torch.load(os.path.join(base_model_name, "optimizer", "optimizer.pt"))
            except FileNotFoundError:
                pass

            # meta
            model.train_progress = train_progress

            return model
        except:
            return None

    @staticmethod
    def __load_diffusers(model_type: ModelType, prior_model_name: str, diffusion_model_name: str) -> KandinskyModel | None:
        try:
            # prior
            prior_tokenizer = CLIPTokenizer.from_pretrained(
                prior_model_name,
                subfolder="tokenizer",
            )

            prior_text_encoder = CLIPTextModelWithProjection.from_pretrained(
                prior_model_name,
                subfolder="text_encoder",
                torch_dtype=torch.float32,
            )

            prior_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                prior_model_name,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
            )

            prior_prior = PriorTransformer.from_pretrained(
                prior_model_name,
                subfolder="prior",
                torch_dtype=torch.float32,
            )

            prior_noise_scheduler = UnCLIPScheduler.from_pretrained(
                prior_model_name,
                subfolder="scheduler",
            )

            prior_image_processor = CLIPImageProcessor.from_pretrained(
                prior_model_name,
                subfolder="image_processor",
            )

            # diffusion model
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                diffusion_model_name,
                subfolder="tokenizer",
            )

            text_encoder = MultilingualCLIP.from_pretrained(
                diffusion_model_name,
                subfolder="text_encoder",
                torch_dtype=torch.float32,
            )

            unet = UNet2DConditionModel.from_pretrained(
                diffusion_model_name,
                subfolder="unet",
                torch_dtype=torch.float32,
            )

            noise_scheduler = DDIMScheduler.from_pretrained(
                diffusion_model_name,
                subfolder="scheduler",
            )

            movq = VQModel.from_pretrained(
                diffusion_model_name,
                subfolder="movq",
                torch_dtype=torch.float32,
            )

            return KandinskyModel(
                model_type=model_type,
                prior_tokenizer=prior_tokenizer,
                prior_text_encoder=prior_text_encoder,
                prior_image_encoder=prior_image_encoder,
                prior_prior=prior_prior,
                prior_noise_scheduler=prior_noise_scheduler,
                prior_image_processor=prior_image_processor,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                noise_scheduler=noise_scheduler,
                movq=movq,
            )
        except Exception as e:
            return None

    @staticmethod
    def __load_ckpt(model_type: ModelType, base_model_name: str) -> KandinskyModel | None:
        return None

    @staticmethod
    def __load_safetensors(model_type: ModelType, base_model_name: str) -> KandinskyModel | None:
        return None

    def load(self, model_type: ModelType, base_model_name: str, extra_model_name: str | None) -> KandinskyModel | None:
        split_base_model_name = base_model_name.split(';')
        if len(split_base_model_name) == 2:
            prior_model_name, diffusion_model_name = split_base_model_name

            model = self.__load_diffusers(model_type, prior_model_name, diffusion_model_name)
            if model is not None:
                return model
        else:
            model = self.__load_internal(model_type, base_model_name)
            if model is not None:
                return model

            model = self.__load_safetensors(model_type, base_model_name)
            if model is not None:
                return model

            model = self.__load_ckpt(model_type, base_model_name)
            if model is not None:
                return model

        raise Exception("could not load model: " + base_model_name)
