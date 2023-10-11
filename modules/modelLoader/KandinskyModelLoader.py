import json
import os
import traceback

import torch
from diffusers import PriorTransformer, UNet2DConditionModel, VQModel, UnCLIPScheduler, DDPMScheduler
from diffusers.pipelines.kandinsky import MultilingualCLIP
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, \
    CLIPImageProcessor, XLMRobertaTokenizerFast

from modules.model.KandinskyModel import KandinskyModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.util.TrainProgress import TrainProgress
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class KandinskyModelLoader(BaseModelLoader):
    def __init__(self):
        super(KandinskyModelLoader, self).__init__()

    def __load_internal(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> KandinskyModel | None:
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
        model = self.__load_diffusers(
            model_type, weight_dtypes, prior_model_name, diffusion_model_name
        )

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

    def __load_diffusers(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            diffusion_model_name: str,
    ) -> KandinskyModel | None:
        # prior
        prior_tokenizer = CLIPTokenizer.from_pretrained(
            prior_model_name,
            subfolder="tokenizer",
        )

        prior_text_encoder = CLIPTextModelWithProjection.from_pretrained(
            prior_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )

        prior_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            prior_model_name,
            subfolder="image_encoder",
            torch_dtype=weight_dtypes.unet.torch_dtype(),
        )

        prior_prior = PriorTransformer.from_pretrained(
            prior_model_name,
            subfolder="prior",
            torch_dtype=weight_dtypes.unet.torch_dtype(),
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
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )

        unet = UNet2DConditionModel.from_pretrained(
            diffusion_model_name,
            subfolder="unet",
            torch_dtype=weight_dtypes.unet.torch_dtype(),
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            diffusion_model_name,
            subfolder="ddpm_scheduler",
        )

        movq = VQModel.from_pretrained(
            diffusion_model_name,
            subfolder="movq",
            torch_dtype=weight_dtypes.vae.torch_dtype(),
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

    def __load_ckpt(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> KandinskyModel | None:
        return None

    def __load_safetensors(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> KandinskyModel | None:
        return None

    def load(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str | None,
            extra_model_name: str | None
    ) -> KandinskyModel | None:
        stacktraces = []

        split_base_model_name = base_model_name.split(';')
        if len(split_base_model_name) == 2:
            prior_model_name, diffusion_model_name = split_base_model_name

            try:
                model = self.__load_diffusers(model_type, weight_dtypes, prior_model_name, diffusion_model_name)
                if model is not None:
                    return model
            except:
                stacktraces.append(traceback.format_exc())

        else:
            try:
                model = self.__load_internal(model_type, weight_dtypes, base_model_name)
                if model is not None:
                    return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                model = self.__load_safetensors(model_type, weight_dtypes, base_model_name)
                if model is not None:
                    return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                model = self.__load_ckpt(model_type, weight_dtypes, base_model_name)
                if model is not None:
                    return model
            except:
                stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + base_model_name)
