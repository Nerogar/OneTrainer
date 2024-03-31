import json
import os
import traceback

import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.models import StableCascadeUNet
from diffusers.pipelines.wuerstchen import WuerstchenDiffNeXt, PaellaVQModel, WuerstchenPrior
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenEfficientNetEncoder
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.convert.convert_stable_cascade_ckpt_to_diffusers import convert_stable_cascade_ckpt_to_diffusers
from modules.util.enum.ModelType import ModelType


class WuerstchenModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin):
    def __init__(self):
        super(WuerstchenModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.WUERSTCHEN_2:
                return "resources/sd_model_spec/wuerstchen_2.0.json"
            case ModelType.STABLE_CASCADE_1:
                return "resources/sd_model_spec/stable_cascade_1.0.json"
            case _:
                return None

    def __load_internal(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            prior_prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ) -> WuerstchenModel | None:
        with open(os.path.join(prior_model_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )

        # base model
        model = self.__load(
            model_type,
            weight_dtypes,
            prior_model_name,
            "",  # pass an empty prior name, so it's always loaded from the backup
            effnet_encoder_model_name,
            decoder_model_name,
        )

        # optimizer
        try:
            model.optimizer_state_dict = torch.load(os.path.join(prior_model_name, "optimizer", "optimizer.pt"))
        except FileNotFoundError:
            pass

        # ema
        try:
            model.ema_state_dict = torch.load(os.path.join(prior_model_name, "ema", "ema.pt"))
        except FileNotFoundError:
            pass

        # meta
        model.train_progress = train_progress

        # model spec
        model.model_spec = self._load_default_model_spec(model_type)

        return model

    def __load(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            prior_prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ) -> WuerstchenModel | None:
        if model_type.is_wuerstchen_v2():
            decoder_tokenizer = CLIPTokenizer.from_pretrained(
                decoder_model_name,
                subfolder="tokenizer",
            )
        if model_type.is_stable_cascade():
            decoder_tokenizer = None

        decoder_noise_scheduler = DDPMWuerstchenScheduler.from_pretrained(
            decoder_model_name,
            subfolder="scheduler",
        )

        if model_type.is_wuerstchen_v2():
            decoder_text_encoder = CLIPTextModel.from_pretrained(
                decoder_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.decoder_text_encoder.torch_dtype(),
            )
            decoder_text_encoder.text_model.embeddings.to(
                dtype=weight_dtypes.text_encoder.torch_dtype(supports_fp8=False))
        if model_type.is_stable_cascade():
            decoder_text_encoder = None

        if model_type.is_wuerstchen_v2():
            decoder_decoder = WuerstchenDiffNeXt.from_pretrained(
                decoder_model_name,
                subfolder="decoder",
                torch_dtype=weight_dtypes.decoder.torch_dtype(),
            )
        elif model_type.is_stable_cascade():
            decoder_decoder = StableCascadeUNet.from_pretrained(
                decoder_model_name,
                subfolder="decoder",
                torch_dtype=weight_dtypes.decoder.torch_dtype(),
            )

        decoder_vqgan = PaellaVQModel.from_pretrained(
            decoder_model_name,
            subfolder="vqgan",
            torch_dtype=weight_dtypes.decoder_vqgan.torch_dtype(),
        )

        if model_type.is_wuerstchen_v2():
            effnet_encoder = WuerstchenEfficientNetEncoder.from_pretrained(
                effnet_encoder_model_name,
                torch_dtype=weight_dtypes.effnet_encoder.torch_dtype(),
            )
        elif model_type.is_stable_cascade():
            # TODO: this is a temporary workaround until the effnet weights are available in diffusers format
            effnet_encoder = WuerstchenEfficientNetEncoder(affine_batch_norm=False)
            effnet_encoder.load_state_dict(load_file(effnet_encoder_model_name))
            effnet_encoder.to(dtype=weight_dtypes.effnet_encoder.torch_dtype())

        if model_type.is_wuerstchen_v2():
            prior_prior = WuerstchenPrior.from_pretrained(
                prior_model_name,
                subfolder="prior",
                torch_dtype=weight_dtypes.prior.torch_dtype(),
            )
        elif model_type.is_stable_cascade():
            if prior_prior_model_name:
                with safe_open(prior_prior_model_name, framework="pt") as f:
                    if any(key.startswith("down_blocks.0.23") for key in f.keys()):
                        config_filename = "resources/model_config/stable_cascade/stable_cascade_prior_3.6b.json"
                    else:
                        config_filename = "resources/model_config/stable_cascade/stable_cascade_prior_1.0b.json"
                    with open(config_filename, "r") as config_file:
                        prior_config = json.load(config_file)
                prior_prior = StableCascadeUNet(**prior_config)
                prior_prior.load_state_dict(convert_stable_cascade_ckpt_to_diffusers(load_file(prior_prior_model_name)))
                prior_prior.to(dtype=weight_dtypes.prior.torch_dtype())
            else:
                prior_prior = StableCascadeUNet.from_pretrained(
                    prior_model_name,
                    subfolder="prior",
                    torch_dtype=weight_dtypes.prior.torch_dtype(),
                )

        prior_tokenizer = CLIPTokenizer.from_pretrained(
            prior_model_name,
            subfolder="tokenizer",
        )

        if model_type.is_wuerstchen_v2():
            prior_text_encoder = CLIPTextModel.from_pretrained(
                prior_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
            )
            prior_text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
        elif model_type.is_stable_cascade():
            prior_text_encoder = CLIPTextModelWithProjection.from_pretrained(
                prior_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
            )
            prior_text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))

        prior_noise_scheduler = DDPMWuerstchenScheduler.from_pretrained(
            prior_model_name,
            subfolder="scheduler",
        )

        model_spec = self._load_default_model_spec(model_type)

        return WuerstchenModel(
            model_type=model_type,
            decoder_tokenizer=decoder_tokenizer,
            decoder_noise_scheduler=decoder_noise_scheduler,
            decoder_text_encoder=decoder_text_encoder,
            decoder_decoder=decoder_decoder,
            decoder_vqgan=decoder_vqgan,
            effnet_encoder=effnet_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_text_encoder=prior_text_encoder,
            prior_noise_scheduler=prior_noise_scheduler,
            prior_prior=prior_prior,
            model_spec=model_spec,
        )

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> WuerstchenModel | None:
        stacktraces = []

        prior_model_name = model_names.base_model
        prior_prior_model_name = model_names.prior_model
        effnet_encoder_model_name = model_names.effnet_encoder_model
        decoder_model_name = model_names.decoder_model

        try:
            model = self.__load_internal(
                model_type,
                weight_dtypes,
                prior_model_name,
                prior_prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load(
                model_type,
                weight_dtypes,
                prior_model_name,
                prior_prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception(
            "could not load model: " + prior_model_name + ", " + effnet_encoder_model_name + ", " + decoder_model_name
        )
