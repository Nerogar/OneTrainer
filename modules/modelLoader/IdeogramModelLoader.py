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

    def __load_internal(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, vae_model_name, include_unconditional_transformer,
                quantization, stream_from_disk)
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            Ideogram4Transformer2DModel,
            weight_dtypes,
            base_model_name,
            "",
            quantization,
            stream_from_disk=stream_from_disk,
        )
        # the unconditional transformer is frozen and only used for the negative branch of the dual-network CFG at
        # sampling, so it has its own weight dtype independent of the trainable transformer's. It is optional. It uses
        # _load_diffusers_sub_module directly (not _load_transformer) because of its own subfolder and dtype; in
        # streaming mode that returns a materialize closure, otherwise a plain module.
        if include_unconditional_transformer and stream_from_disk:
            model.unconditional_transformer, model.materialize_fn["unconditional_transformer"] = \
                self._load_diffusers_sub_module(
                    Ideogram4Transformer2DModel,
                    weight_dtypes.unconditional_transformer,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "unconditional_transformer",
                    quantization,
                    stream_from_disk=True,
                )
        elif include_unconditional_transformer:
            model.unconditional_transformer = self._load_diffusers_sub_module(
                Ideogram4Transformer2DModel,
                weight_dtypes.unconditional_transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "unconditional_transformer",
                quantization,
            )
        else:
            model.unconditional_transformer = None

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Qwen3VLModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        model.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.vae = self._load_vae(
            AutoencoderKLFlux2,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

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
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_unconditional_transformer, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_unconditional_transformer, quantization, stream_from_disk,
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
