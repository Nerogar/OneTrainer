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
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, text_encoder_4_model_name, vae_model_name,
                include_text_encoder_1, include_text_encoder_2, include_text_encoder_3, include_text_encoder_4,
                quantization, stream_from_disk,
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
            stream_from_disk: bool,
    ):
        model.tokenizer_1 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        ) if include_text_encoder_1 else None

        model.tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_2",
        ) if include_text_encoder_2 else None

        model.tokenizer_3 = T5Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_3",
        ) if include_text_encoder_3 else None

        model.tokenizer_4 = LlamaTokenizerFast.from_pretrained(
            text_encoder_4_model_name,
        ) if include_text_encoder_4 else None

        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if include_text_encoder_1:
            model.text_encoder_1, model.materialize_fn["text_encoder_1"] = self._load_text_encoder(
                CLIPTextModelWithProjection,
                weight_dtypes.text_encoder,
                weight_dtypes.train_dtype,
                base_model_name,
                "text_encoder",
                stream_from_disk=stream_from_disk,
            )
        else:
            model.text_encoder_1 = None

        if include_text_encoder_2:
            model.text_encoder_2, model.materialize_fn["text_encoder_2"] = self._load_text_encoder(
                CLIPTextModelWithProjection,
                weight_dtypes.text_encoder_2,
                weight_dtypes.train_dtype,
                base_model_name,
                "text_encoder_2",
                stream_from_disk=stream_from_disk,
            )
        else:
            model.text_encoder_2 = None

        if include_text_encoder_3:
            model.text_encoder_3, model.materialize_fn["text_encoder_3"] = self._load_text_encoder(
                T5EncoderModel,
                weight_dtypes.text_encoder_3,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder_3",
                stream_from_disk=stream_from_disk,
            )
        else:
            model.text_encoder_3 = None

        if include_text_encoder_4:
            if text_encoder_4_model_name:
                # override repo holds text_encoder_4 at its root, not in a base-model subfolder, so it bypasses
                # _load_text_encoder (which always loads from a base-repo subfolder) and loads directly.
                # _load_transformers_sub_module returns a (module, materialize_fn) pair only when streaming; a bare
                # module otherwise.
                if stream_from_disk:
                    model.text_encoder_4, model.materialize_fn["text_encoder_4"] = self._load_transformers_sub_module(
                        LlamaForCausalLM,
                        weight_dtypes.text_encoder_4,
                        weight_dtypes.train_dtype,
                        text_encoder_4_model_name,
                        stream_from_disk=True,
                    )
                else:
                    model.text_encoder_4 = self._load_transformers_sub_module(
                        LlamaForCausalLM,
                        weight_dtypes.text_encoder_4,
                        weight_dtypes.train_dtype,
                        text_encoder_4_model_name,
                    )
            else:
                model.text_encoder_4, model.materialize_fn["text_encoder_4"] = self._load_text_encoder(
                    LlamaForCausalLM,
                    weight_dtypes.text_encoder_4,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "text_encoder_4",
                    stream_from_disk=stream_from_disk,
                )

        else:
            model.text_encoder_4 = None

        model.vae = self._load_vae(
            AutoencoderKL,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            HiDreamImageTransformer2DModel,
            weight_dtypes,
            base_model_name,
            "",
            quantization,
            stream_from_disk=stream_from_disk,
        )

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
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.text_encoder_4, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3, model_names.include_text_encoder_4, quantization,
                stream_from_disk,
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
                stream_from_disk,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        if stream_from_disk:
            # the single-file loader below builds a full pipeline via from_single_file, which can't stream; fall
            # back to loading it fully into RAM.
            print(f"Warning: 'stream from disk' is not supported for single-file {model_type}; loading fully into RAM.")

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
