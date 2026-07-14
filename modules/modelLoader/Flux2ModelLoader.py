import os
import traceback

from modules.model.Flux2Model import Flux2Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
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
        transformer = self._load_transformer(
            Flux2Transformer2DModel,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
        )

        if transformer.config.num_attention_heads == 48: #Flux2.Dev
            tokenizer = PixtralProcessor.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            ).tokenizer

            text_encoder = self._load_text_encoder(
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
            text_encoder = self._load_text_encoder(
                Qwen3ForCausalLM,
                weight_dtypes.text_encoder,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder",
            )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        vae = self._load_vae(
            AutoencoderKLFlux2,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
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

    def _legacy_conversion(self, model: Flux2Model) -> list | None:
        return self._mixture_legacy_conversion(model)

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
