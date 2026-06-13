import logging
import os
import traceback

from modules.model.BaseModel import BaseModel
from modules.model.ErnieModel import ErnieModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKLFlux2,
    ErnieImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
)
from transformers import Mistral3Model, MistralConfig, PreTrainedTokenizerFast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class ErnieModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: ErnieModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: ErnieModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        # transformers < 5.x doesn't register "ministral3"; patch it so Mistral3Config can parse its text_config
        if "ministral3" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("ministral3", MistralConfig)

        diffusers_sub = []
        transformers_sub = ["text_encoder"]
        if not transformer_model_name:
            diffusers_sub.append("transformer")
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=transformers_sub,
        )

        if transformer_model_name:
            transformer = ErnieImageTransformer2DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                ErnieImageTransformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        # TokenizersBackend is the Rust tokenizers library backend, not a transformers class — warning is a false alarm
        tokenization_logger = logging.getLogger("transformers.tokenization_utils_base")
        prev_level = tokenization_logger.level
        tokenization_logger.setLevel(logging.ERROR)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )
        tokenization_logger.setLevel(prev_level)

        text_encoder = self._load_transformers_sub_module(
            Mistral3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: ErnieModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError(
            "Loading single-file safetensors for Ernie is not supported. Use the diffusers model instead. "
            "Transformer-only safetensor files can be loaded by overriding the transformer."
        )

    def load(
            self,
            model: ErnieModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class ErnieLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None

    def load(
            self,
            model: ErnieModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


ErnieLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.ERNIE: "resources/sd_model_spec/ernie-lora.json",
    },
    model_class=ErnieModel,
    model_loader_class=ErnieModelLoader,
    lora_loader_class=ErnieLoRALoader,
    embedding_loader_class=None,
)

ErnieFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.ERNIE: "resources/sd_model_spec/ernie.json",
    },
    model_class=ErnieModel,
    model_loader_class=ErnieModelLoader,
    embedding_loader_class=None,
)
