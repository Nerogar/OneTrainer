import os
import traceback

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast


class AnimaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: AnimaModel,
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
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
                stream_from_disk,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: AnimaModel,
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

        model.t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        # conditioner is always bfloat16 — small adapter, no user dtype control
        model.text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        model.vae = self._load_vae(
            AutoencoderKLQwenImage,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            CosmosTransformer3DModel,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
            stream_from_disk=stream_from_disk,
        )

    def load( #TODO share code between models
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
                stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
                stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class AnimaLoRALoader(
    LoRALoaderMixin,
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: AnimaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
    lora_loader_class=AnimaLoRALoader,
)

AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
