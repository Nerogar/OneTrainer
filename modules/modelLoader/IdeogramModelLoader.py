import os
import traceback

from modules.model.IdeogramModel import IdeogramModel
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
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Ideogram4Transformer2DModel,
)
from transformers import AutoTokenizer, Qwen3VLModel


class IdeogramModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __make_rotary_autocast_safe(transformer: Ideogram4Transformer2DModel):
        # Under bfloat16 autocast, Ideogram4MRoPE's matmul-based rotary frequencies collapse at the large
        # image-position offsets, losing all spatial information (flat-color output). Force float32 here.
        # https://github.com/huggingface/diffusers/issues/13920
        rotary = transformer.rotary_emb
        orig_forward = rotary.forward

        def forward(position_ids: torch.Tensor):
            with torch.autocast(device_type=position_ids.device.type, enabled=False):
                return orig_forward(position_ids)

        rotary.forward = forward

    def __load_internal(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, include_unconditional_transformer, quantization)
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
    ):
        transformer = self._load_diffusers_sub_module(
            Ideogram4Transformer2DModel,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            "transformer",
            quantization,
        )
        # the unconditional transformer is frozen and only used for the negative branch of the dual-network CFG at
        # sampling, so it has its own weight dtype independent of the trainable transformer's. It is optional: if not
        # loaded, only cfg_scale<=1 sampling is possible.
        if include_unconditional_transformer:
            unconditional_transformer = self._load_diffusers_sub_module(
                Ideogram4Transformer2DModel,
                weight_dtypes.unconditional_transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "unconditional_transformer",
                quantization,
            )
            self.__make_rotary_autocast_safe(unconditional_transformer)
        else:
            unconditional_transformer = None

        self.__make_rotary_autocast_safe(transformer)

        text_encoder = self._load_transformers_sub_module(
            Qwen3VLModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

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
        model.unconditional_transformer = unconditional_transformer

    def __load_safetensors(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError(
            "Loading single-file safetensors for Ideogram is not supported. Use the diffusers model instead."
        )

    def load(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.include_unconditional_transformer, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.include_unconditional_transformer, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class IdeogramLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: IdeogramModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


IdeogramLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.IDEOGRAM_4: "resources/sd_model_spec/ideogram_4-lora.json",
    },
    model_class=IdeogramModel,
    model_loader_class=IdeogramModelLoader,
    lora_loader_class=IdeogramLoRALoader,
    embedding_loader_class=None,
)

IdeogramFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.IDEOGRAM_4: "resources/sd_model_spec/ideogram_4.json",
    },
    model_class=IdeogramModel,
    model_loader_class=IdeogramModelLoader,
    embedding_loader_class=None,
)
