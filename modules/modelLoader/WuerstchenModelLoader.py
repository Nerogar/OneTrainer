import json
import os
import traceback

import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenDiffNeXt, PaellaVQModel, WuerstchenPrior
from transformers import CLIPTokenizer, CLIPTextModel

from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenEfficientNetEncoder
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.modelLoader.mixin.ModelLoaderSDConfigMixin import ModelLoaderSDConfigMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class WuerstchenModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin, ModelLoaderSDConfigMixin):
    def __init__(self):
        super(WuerstchenModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        # TODO: replace with actual Wuerstchen config
        match model_type:
            case ModelType.WUERSTCHEN_2:
                return "resources/sd_model_spec/sd_xl_base_1.0.json"
            case _:
                return None

    def __load_internal(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
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
        model = self.__load_diffusers(
            model_type,
            weight_dtypes,
            prior_model_name,
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

    def __load_diffusers(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ) -> WuerstchenModel | None:
        decoder_tokenizer = CLIPTokenizer.from_pretrained(
            decoder_model_name,
            subfolder="tokenizer",
        )

        decoder_noise_scheduler = DDPMWuerstchenScheduler.from_pretrained(
            decoder_model_name,
            subfolder="scheduler",
        )

        decoder_text_encoder = CLIPTextModel.from_pretrained(
            decoder_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.decoder_text_encoder.torch_dtype(),
        )

        decoder_decoder = WuerstchenDiffNeXt.from_pretrained(
            decoder_model_name,
            subfolder="decoder",
            torch_dtype=weight_dtypes.decoder.torch_dtype(),
        )

        decoder_vqgan = PaellaVQModel.from_pretrained(
            decoder_model_name,
            subfolder="vqgan",
            torch_dtype=weight_dtypes.decoder_vqgan.torch_dtype(),
        )

        effnet_encoder = WuerstchenEfficientNetEncoder.from_pretrained(
            effnet_encoder_model_name,
            torch_dtype=weight_dtypes.effnet_encoder.torch_dtype(),
        )

        prior_prior = WuerstchenPrior.from_pretrained(
            prior_model_name,
            subfolder="prior",
            torch_dtype=weight_dtypes.prior.torch_dtype(),
        )

        prior_tokenizer = CLIPTokenizer.from_pretrained(
            prior_model_name,
            subfolder="tokenizer",
        )

        prior_text_encoder = CLIPTextModel.from_pretrained(
            prior_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )

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

    def __load_ckpt(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ) -> WuerstchenModel | None:
        pass

    def __load_safetensors(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ) -> WuerstchenModel | None:
        pass

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> WuerstchenModel | None:
        stacktraces = []

        prior_model_name = model_names.base_model
        effnet_encoder_model_name = model_names.effnet_encoder_model
        decoder_model_name = model_names.decoder_model

        try:
            model = self.__load_diffusers(
                model_type,
                weight_dtypes,
                prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_internal(
                model_type,
                weight_dtypes,
                prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_safetensors(
                model_type,
                weight_dtypes,
                prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_ckpt(
                model_type,
                weight_dtypes,
                prior_model_name,
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
