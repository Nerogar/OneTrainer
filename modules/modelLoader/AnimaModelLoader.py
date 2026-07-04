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
    GGUFQuantizationConfig,
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
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
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
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        # conditioner is always bfloat16 — small adapter, no user dtype control
        text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        if vae_model_name: #TODO simplify
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            transformer = CosmosTransformer3DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                #avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                CosmosTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.text_conditioner = text_conditioner
        model.vae = vae
        model.transformer = transformer

    def load( #TODO share code between models
            self,
            model: AnimaModel,
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
