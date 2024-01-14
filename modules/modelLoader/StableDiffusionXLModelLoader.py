import json
import os
import traceback

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionXLPipeline
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.modelLoader.mixin.ModelLoaderSDConfigMixin import ModelLoaderSDConfigMixin
from modules.util import create
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler


class StableDiffusionXLModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin, ModelLoaderSDConfigMixin):
    def __init__(self):
        super(StableDiffusionXLModelLoader, self).__init__()

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/diffusers_model_config/sd_xl_base.yaml"
            case ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING:  # TODO: find the actual yml file
                return "resources/diffusers_model_config/sd_xl_base.yaml"
            case _:
                return None

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/sd_model_spec/sd_xl_base_1.0.json"
            case ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
                return "resources/sd_model_spec/sd_xl_base_1.0_inpainting.json"
            case _:
                return None

    def __load_internal(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ) -> StableDiffusionXLModel | None:
        with open(os.path.join(base_model_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )

        # base model
        model = self.__load_diffusers(model_type, weight_dtypes, base_model_name, vae_model_name)

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

        model.sd_config = self._load_sd_config(model_type)

        # meta
        model.train_progress = train_progress

        # model spec
        model.model_spec = self._load_default_model_spec(model_type)

        return model

    def __load_diffusers(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ) -> StableDiffusionXLModel | None:
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
        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=noise_scheduler,
        )

        text_encoder_1 = CLIPTextModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_model_name,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtypes.text_encoder_2.torch_dtype(),
        )

        if vae_model_name:
            vae = AutoencoderKL.from_pretrained(
                vae_model_name,
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                base_model_name,
                subfolder="vae",
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )

        unet = UNet2DConditionModel.from_pretrained(
            base_model_name,
            subfolder="unet",
            torch_dtype=weight_dtypes.unet.torch_dtype(),
        )

        sd_config = self._load_sd_config(model_type)

        model_spec = self._load_default_model_spec(model_type)

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
            model_spec=model_spec,
        )

    def __load_ckpt(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ) -> StableDiffusionXLModel | None:
        sd_config_name = self._get_sd_config_name(model_type, base_model_name)

        pipeline = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            original_config_file=sd_config_name,
            load_safety_checker=False,
        )

        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=pipeline.scheduler,
        )

        if vae_model_name:
            pipeline.vae = AutoencoderKL.from_pretrained(
                vae_model_name,
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )

        sd_config = self._load_sd_config(model_type, base_model_name)

        model_spec = self._load_default_model_spec(model_type)

        return StableDiffusionXLModel(
            model_type=model_type,
            tokenizer_1=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            noise_scheduler=noise_scheduler,
            text_encoder_1=pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype()),
            text_encoder_2=pipeline.text_encoder_2.to(dtype=weight_dtypes.text_encoder_2.torch_dtype()),
            vae=pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype()),
            unet=pipeline.unet.to(dtype=weight_dtypes.unet.torch_dtype()),
            sd_config=sd_config,
            model_spec=model_spec,
        )

    def __load_safetensors(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ) -> StableDiffusionXLModel | None:
        sd_config_name = self._get_sd_config_name(model_type, base_model_name)

        pipeline = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            original_config_file=sd_config_name,
            load_safety_checker=False,
            use_safetensors=True,
        )

        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=pipeline.scheduler,
        )

        if vae_model_name:
            pipeline.vae = AutoencoderKL.from_pretrained(
                vae_model_name,
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )

        sd_config = self._load_sd_config(model_type, base_model_name)

        model_spec = self._load_default_model_spec(model_type, base_model_name)

        return StableDiffusionXLModel(
            model_type=model_type,
            tokenizer_1=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            noise_scheduler=noise_scheduler,
            text_encoder_1=pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype()),
            text_encoder_2=pipeline.text_encoder_2.to(dtype=weight_dtypes.text_encoder_2.torch_dtype()),
            vae=pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype()),
            unet=pipeline.unet.to(dtype=weight_dtypes.unet.torch_dtype()),
            sd_config=sd_config,
            model_spec=model_spec,
        )

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> StableDiffusionXLModel | None:
        stacktraces = []

        try:
            model = self.__load_internal(model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_diffusers(model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_safetensors(model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_ckpt(model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
