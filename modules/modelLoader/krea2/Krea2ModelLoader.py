import os
import traceback

from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Transformer2DModel,
)
from transformers import Qwen2Tokenizer, Qwen3VLModel


class Krea2ModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_text_encoder(
            Qwen3VLModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        vae = self._load_vae(
            AutoencoderKLQwenImage,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        transformer = self._load_transformer(
            Krea2Transformer2DModel,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
        )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError("Loading of single file Krea 2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    def load( #TODO share code between models
            self,
            model: Krea2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
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
