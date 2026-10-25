import os
import traceback

import torch

from modules.model.MossModel import MossModel
from modules.model.moss.pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import AutoencoderKLFlux2
from transformers import Qwen2Tokenizer, Qwen3_5Model


class MossModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: MossModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                quantization, stream_from_disk,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: MossModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        model.tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        model.noise_scheduler = MOSSImageScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Qwen3_5Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        model.vae = self._load_vae(
            AutoencoderKLFlux2,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            MOSSImageTransformer,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
            stream_from_disk=stream_from_disk,
        )

    def __load_safetensors(
            self,
            model: MossModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError("Loading of single file MOSS transformer not supported. Use the diffusers model instead.")

    def load(
            self,
            model: MossModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
