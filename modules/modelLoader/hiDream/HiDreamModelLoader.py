import copy
import os
import traceback

from modules.model.HiDreamModel import HiDreamModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    HiDreamImagePipeline,
    HiDreamImageTransformer2DModel,
)
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    T5EncoderModel,
    T5Tokenizer,
)


class HiDreamModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: HiDreamModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            text_encoder_4_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
            include_text_encoder_4: bool,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, text_encoder_4_model_name, vae_model_name,
                include_text_encoder_1, include_text_encoder_2, include_text_encoder_3, include_text_encoder_4, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: HiDreamModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            text_encoder_4_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
            include_text_encoder_4: bool,
            quantization: QuantizationConfig,
    ):
        diffusers_sub = []
        transformers_sub = []

        diffusers_sub.append("transformer")
        if include_text_encoder_1:
            transformers_sub.append("text_encoder")
        if include_text_encoder_2:
            transformers_sub.append("text_encoder_2")
        if include_text_encoder_3:
            transformers_sub.append("text_encoder_3")
        if include_text_encoder_4:
            if text_encoder_4_model_name:
                self._prepare_sub_modules(
                    text_encoder_4_model_name,
                    transformers_modules=[""],
                    diffusers_modules=[],
                )
            else:
                transformers_sub.append("text_encoder_4")
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=transformers_sub,
        )

        tokenizer_1 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        ) if include_text_encoder_1 else None

        tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        ) if include_text_encoder_2 else None

        tokenizer_3 = T5Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_3",
        ) if include_text_encoder_3 else None

        tokenizer_4 = LlamaTokenizerFast.from_pretrained(
            text_encoder_4_model_name,
        ) if include_text_encoder_1 else None

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if include_text_encoder_1:
            text_encoder_1 = self._load_transformers_sub_module(
                CLIPTextModelWithProjection,
                weight_dtypes.text_encoder,
                weight_dtypes.train_dtype,
                base_model_name,
                "text_encoder",
            )
        else:
            text_encoder_1 = None

        if include_text_encoder_2:
            text_encoder_2 = self._load_transformers_sub_module(
                CLIPTextModelWithProjection,
                weight_dtypes.text_encoder_2,
                weight_dtypes.train_dtype,
                base_model_name,
                "text_encoder_2",
            )
        else:
            text_encoder_2 = None

        if include_text_encoder_3:
            text_encoder_3 = self._load_transformers_sub_module(
                T5EncoderModel,
                weight_dtypes.text_encoder_3,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder_3",
            )
        else:
            text_encoder_3 = None

        if include_text_encoder_4:
            if text_encoder_4_model_name:
                text_encoder_4 = self._load_transformers_sub_module(
                    LlamaForCausalLM,
                    weight_dtypes.text_encoder_4,
                    weight_dtypes.train_dtype,
                    text_encoder_4_model_name,
                )
            else:
                text_encoder_4 = self._load_transformers_sub_module(
                    LlamaForCausalLM,
                    weight_dtypes.text_encoder_4,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "text_encoder_4",
                )

        else:
            text_encoder_4 = None

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

        transformer = self._load_diffusers_sub_module(
            HiDreamImageTransformer2DModel,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            "transformer",
            quantization,
        )

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.tokenizer_3 = tokenizer_3
        model.tokenizer_4 = tokenizer_4
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.text_encoder_3 = text_encoder_3
        model.text_encoder_4 = text_encoder_4
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: HiDreamModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            text_encoder_4_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
            include_text_encoder_4: bool,
            quantization: QuantizationConfig,
    ):
        pipeline = HiDreamImagePipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            safety_checker=None,
        )

        if include_text_encoder_2:
            # replace T5TokenizerFast with T5Tokenizer, loaded from the same repository
            pipeline.tokenizer_2 = T5Tokenizer.from_pretrained(
                pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
                subfolder="tokenizer_2",
            )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(
                pipeline.vae, weight_dtypes.vae, weight_dtypes.train_dtype
            )

        if pipeline.text_encoder is not None and include_text_encoder_1:
            text_encoder_1 = self._convert_transformers_sub_module_to_dtype(
                pipeline.text_encoder, weight_dtypes.text_encoder, weight_dtypes.train_dtype
            )
            tokenizer_1 = pipeline.tokenizer
        else:
            text_encoder_1 = None
            tokenizer_1 = None
            print("text encoder 1 (clip l) not loaded, continuing without it")

        if pipeline.text_encoder_2 is not None and include_text_encoder_2:
            text_encoder_2 = self._convert_transformers_sub_module_to_dtype(
                pipeline.text_encoder_2, weight_dtypes.text_encoder_2, weight_dtypes.fallback_train_dtype
            )
            tokenizer_2 = pipeline.tokenizer_2
        else:
            text_encoder_2 = None
            tokenizer_2 = None
            print("text encoder 2 (t5) not loaded, continuing without it")

        transformer = self._convert_diffusers_sub_module_to_dtype(
            pipeline.transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
        )

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.transformer = transformer

    def __after_load(self, model: HiDreamModel):
        if model.tokenizer_4 is not None:
            model.tokenizer_4.pad_token = model.tokenizer_4.eos_token
        model.orig_tokenizer_1 = copy.deepcopy(model.tokenizer_1)
        model.orig_tokenizer_2 = copy.deepcopy(model.tokenizer_2)
        model.orig_tokenizer_3 = copy.deepcopy(model.tokenizer_3)
        model.orig_tokenizer_4 = copy.deepcopy(model.tokenizer_4)

    def load(
            self,
            model: HiDreamModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.text_encoder_4, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3, model_names.include_text_encoder_4, quantization,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.text_encoder_4, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3, model_names.include_text_encoder_4, quantization,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.text_encoder_4, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3, model_names.include_text_encoder_4, quantization,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
