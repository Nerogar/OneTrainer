import os
import traceback

from modules.model.BaseModel import BaseModel
from modules.model.ZImageModel import ZImageModel
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
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    ZImageTransformer2DModel,
)
from transformers import (
    Qwen2Tokenizer,
    Qwen3ForCausalLM,
)


class ZImageModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: ZImageModel,
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
            model: ZImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
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

        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3ForCausalLM,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        #TODO this is a tied weight. The dtype conversion code in _load_transformers_sub_module
        #currently does not support tied weights. Reconstruct but clone, because the quantization code
        #doesn't support tied weights either:
        text_encoder.lm_head.weight = type(text_encoder.lm_head.weight)(text_encoder.model.embed_tokens.weight)


        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            transformer = ZImageTransformer2DModel.from_single_file(
                transformer_model_name,
                #avoid loading the transformer in float32:
                torch_dtype = torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                ZImageTransformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: ZImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        #no single file .safetensors for Qwen available at the time of writing this code
        raise NotImplementedError("Loading of single file Z-Image models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    def load(
            self,
            model: ZImageModel,
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



class ZImageLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None

    def load(
            self,
            model: ZImageModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


ZImageLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.Z_IMAGE: "resources/sd_model_spec/z_image-lora.json",
    },
    model_class=ZImageModel,
    model_loader_class=ZImageModelLoader,
    lora_loader_class=ZImageLoRALoader,
    embedding_loader_class=None,
)

ZImageFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.Z_IMAGE: "resources/sd_model_spec/z_image.json",
    },
    model_class=ZImageModel,
    model_loader_class=ZImageModelLoader,
    embedding_loader_class=None,
)
