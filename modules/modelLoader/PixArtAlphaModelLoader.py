import json
import os
import traceback

import torch
from diffusers import DDIMScheduler, AutoencoderKL, Transformer2DModel
from transformers import T5Tokenizer, T5EncoderModel

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class PixArtAlphaModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin):
    def __init__(self):
        super(PixArtAlphaModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.PIXART_ALPHA:
                return "resources/sd_model_spec/pixart_alpha_1.0.json"
            case _:
                return None

    def __load_internal(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> PixArtAlphaModel | None:
        with open(os.path.join(base_model_name, "meta.json"), "r") as meta_file:
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
            base_model_name
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

        # model spec
        model.model_spec = self._load_default_model_spec(model_type)

        return model

    def __load_diffusers(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> PixArtAlphaModel | None:
        tokenizer = T5Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = DDIMScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = T5EncoderModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )

        vae = AutoencoderKL.from_pretrained(
            base_model_name,
            subfolder="vae",
            torch_dtype=weight_dtypes.vae.torch_dtype(),
        )

        transformer = Transformer2DModel.from_pretrained(
            base_model_name,
            subfolder="transformer",
            torch_dtype=weight_dtypes.prior.torch_dtype(),
        )

        model_spec = self._load_default_model_spec(model_type)

        return PixArtAlphaModel(
            model_type=model_type,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            text_encoder=text_encoder,
            vae = vae,
            transformer=transformer,
            model_spec=model_spec,
        )

    def __load_ckpt(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> PixArtAlphaModel | None:
        pass

    def __load_safetensors(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
    ) -> PixArtAlphaModel | None:
        pass

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> PixArtAlphaModel | None:
        stacktraces = []

        base_model_name = model_names.base_model

        try:
            model = self.__load_internal(
                model_type,
                weight_dtypes,
                base_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_diffusers(
                model_type,
                weight_dtypes,
                base_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_safetensors(
                model_type,
                weight_dtypes,
                base_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            model = self.__load_ckpt(
                model_type,
                weight_dtypes,
                base_model_name,
            )
            if model is not None:
                return model
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception(
            "could not load model: " + base_model_name
        )
