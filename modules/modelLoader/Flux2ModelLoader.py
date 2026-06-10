import os
import traceback

from modules.model.BaseModel import BaseModel
from modules.model.Flux2Model import Flux2Model
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
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
    GGUFQuantizationConfig,
)
from transformers import (
    Mistral3ForConditionalGeneration,
    PixtralProcessor,
    Qwen2Tokenizer,
    Qwen3ForCausalLM,
)


class Flux2ModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: Flux2Model,
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
            model: Flux2Model,
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

        if transformer_model_name:
            transformer = Flux2Transformer2DModel.from_single_file(
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
                Flux2Transformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        if transformer.config.num_attention_heads == 48: #Flux2.Dev
            tokenizer = PixtralProcessor.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            ).tokenizer

            text_encoder = self._load_transformers_sub_module(
                Mistral3ForConditionalGeneration,
                weight_dtypes.text_encoder,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder",
            )
        else: #Flux2.Klein
            tokenizer = Qwen2Tokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
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
            model: Flux2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        #no single file .safetensors for Qwen available at the time of writing this code
        raise NotImplementedError("Loading of single file Flux2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    def load(
            self,
            model: Flux2Model,
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



class Flux2LoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None #TODO
        #return convert_flux_lora_key_sets()

    def load(
            self,
            model: Flux2Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


Flux2LoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.FLUX_2: "resources/sd_model_spec/flux_2.0-lora.json",
    },
    model_class=Flux2Model,
    model_loader_class=Flux2ModelLoader,
    lora_loader_class=Flux2LoRALoader,
    embedding_loader_class=None,
)

Flux2FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.FLUX_2: "resources/sd_model_spec/flux_2.0.json",
    },
    model_class=Flux2Model,
    model_loader_class=Flux2ModelLoader,
    embedding_loader_class=None,
)
