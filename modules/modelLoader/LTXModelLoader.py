import os
import traceback

from modules.model.LTXModel import LTXModel
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
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from transformers import (
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    AutoConfig,
    AutoTokenizer,
)

import huggingface_hub


class LTXModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: LTXModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            low_noise_transformer_model_name: str,
            include_low_noise_transformer: bool,
            text_encoder_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name,
                low_noise_transformer_model_name, include_low_noise_transformer, text_encoder_model_name, vae_model_name,
                quantization, stream_from_disk,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: LTXModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            low_noise_transformer_model_name: str,
            include_low_noise_transformer: bool,
            text_encoder_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        # LTX 2.5 ships two DiTs in one repo: transformer/ is the distilled one, transformer_full/ the full/SFT
        # model training wants. 2.3 has only transformer/ and it is already the full model, so preferring
        # transformer_full/ picks the trainable DiT in both generations.
        has_transformer_full = os.path.isdir(os.path.join(base_model_name, "transformer_full")) \
            if os.path.isdir(base_model_name) \
            else huggingface_hub.file_exists(base_model_name, "transformer_full/config.json")
        transformer_subfolder = "transformer_full" if has_transformer_full else "transformer"

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            LTX2VideoTransformer3DModel,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
            stream_from_disk=stream_from_disk,
            subfolder=transformer_subfolder,
        )

        # The distilled low-noise expert, from the base repo's transformer/ where the repo ships one (2.5), else
        # from whatever the user named. Optional: with neither it stays None. _load_diffusers_sub_module rather
        # than _load_transformer, which hardcodes weight_dtypes.transformer and would give this part the main
        # transformer's dtype.
        low_noise_transformer_source = low_noise_transformer_model_name or (base_model_name if has_transformer_full else None)
        if include_low_noise_transformer and low_noise_transformer_source:
            if os.path.isfile(low_noise_transformer_source):
                # single-file checkpoints load whole into RAM: streaming reads a shard map keyed by the module
                # tree, which only a diffusers folder provides. The config comes from the base repo's
                # transformer/, which the distilled DiT matches field for field.
                low_noise_transformer_dtype = weight_dtypes.low_noise_transformer.torch_dtype()
                model.low_noise_transformer = LTX2VideoTransformer3DModel.from_single_file(
                    low_noise_transformer_source,
                    config=base_model_name,
                    subfolder="transformer",
                    # avoid loading the expert in float32:
                    torch_dtype=torch.bfloat16 if low_noise_transformer_dtype is None else low_noise_transformer_dtype,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
                    if weight_dtypes.low_noise_transformer.is_gguf() else None,
                )
                model.low_noise_transformer = self._convert_diffusers_sub_module_to_dtype(
                    model.low_noise_transformer, weight_dtypes.low_noise_transformer, weight_dtypes.train_dtype, quantization,
                )
            else:
                model.low_noise_transformer, model.materialize_fn["low_noise_transformer"] = self._load_diffusers_sub_module(
                    LTX2VideoTransformer3DModel,
                    weight_dtypes.low_noise_transformer,
                    weight_dtypes.train_dtype,
                    low_noise_transformer_source,
                    "transformer",
                    quantization,
                    stream_from_disk=stream_from_disk,
                )

        model.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )
        # padding='max_length' needs one, and every LTX checkpoint's tokenizer config sets it
        assert model.tokenizer.pad_token is not None

        if text_encoder_model_name:
            # 2.3 bundles the encoder as float32 (48.7GB vs 24.4GB) for identical values, so naming a bf16 repo
            # is worth it there; 2.5 already ships bf16. ComfyUI's stock google/gemma-3-12b-it differs from the
            # bundled QAT weights by up to ~5% relative - a fidelity choice. A standalone repo has no subfolder.
            text_encoder_base, text_encoder_subfolder = text_encoder_model_name, ""
        else:
            text_encoder_base, text_encoder_subfolder = base_model_name, "text_encoder"

        # LTX 2.3 ships a Gemma 3 encoder, LTX 2.5 a Gemma 4 one, and both load through the same ModelType, so
        # the class comes from the checkpoint's config - resolved the way AutoModelForImageTextToText would.
        # The auto class itself can't be used: the loader builds a meta skeleton from config_class, which only
        # the concrete class has.
        text_encoder_config = AutoConfig.from_pretrained(
            text_encoder_base,
            subfolder=text_encoder_subfolder,
        )
        if type(text_encoder_config) not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
            raise NotImplementedError(
                f"unsupported LTX text encoder model_type '{text_encoder_config.model_type}'")

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[type(text_encoder_config)],
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            text_encoder_base,
            text_encoder_subfolder,
            stream_from_disk=stream_from_disk,
        )

        model.connectors, model.materialize_fn["connectors"] = self._load_diffusers_sub_module(
            LTX2TextConnectors,
            weight_dtypes.connectors,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "connectors",
            stream_from_disk=stream_from_disk,
        )

        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if vae_model_name:
            model.vae = self._load_diffusers_sub_module(
                AutoencoderKLLTX2Video,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            model.vae = self._load_diffusers_sub_module(
                AutoencoderKLLTX2Video,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        # the audio branch is frozen and never trained, but LTX2Pipeline requires both, so they are loaded
        # to keep a saved diffusers repo complete. They piggyback on the vae's dtype config.
        model.audio_vae = self._load_diffusers_sub_module(
            AutoencoderKLLTX2Audio,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            "audio_vae",
        )

        model.vocoder = self._load_diffusers_sub_module(
            LTX2VocoderWithBWE,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            "vocoder",
        )

        model.model_type = model_type

    def load(
            self,
            model: LTXModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.low_noise_transformer_model, model_names.include_low_noise_transformer,
                model_names.text_encoder_model, model_names.vae_model, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.low_noise_transformer_model, model_names.include_low_noise_transformer,
                model_names.text_encoder_model, model_names.vae_model, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class LTXLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def load(
            self,
            model: LTXModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


LTXLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.LTX_2: "resources/sd_model_spec/ltx_2-lora.json",
    },
    model_class=LTXModel,
    model_loader_class=LTXModelLoader,
    lora_loader_class=LTXLoRALoader,
    embedding_loader_class=None,
)

LTXFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.LTX_2: "resources/sd_model_spec/ltx_2.json",
    },
    model_class=LTXModel,
    model_loader_class=LTXModelLoader,
    embedding_loader_class=None,
)
