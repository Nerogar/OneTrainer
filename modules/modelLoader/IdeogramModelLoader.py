import os
import traceback
import torch

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
    Ideogram4Pipeline,
    GGUFQuantizationConfig,
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
            transformer_model_name: str,
            unconditional_transformer_model_name: str,
            vae_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, 
                unconditional_transformer_model_name, vae_model_name, include_unconditional_transformer, quantization
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            unconditional_transformer_model_name: str,
            vae_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
    ):
        if transformer_model_name:
            transformer = Ideogram4Transformer2DModel.from_single_file(
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
            if unconditional_transformer_model_name:
                unconditional_transformer = Ideogram4Transformer2DModel.from_single_file(
                    unconditional_transformer_model_name,
                    config=base_model_name,
                    subfolder="unconditional_transformer",
                    torch_dtype=torch.bfloat16 if weight_dtypes.unconditional_transformer.torch_dtype() is None else weight_dtypes.unconditional_transformer.torch_dtype(),
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.unconditional_transformer.is_gguf() else None,
                )
                unconditional_transformer = self._convert_diffusers_sub_module_to_dtype(
                    unconditional_transformer, weight_dtypes.unconditional_transformer, weight_dtypes.train_dtype, quantization,
                )
            else:
                unconditional_transformer = self._load_diffusers_sub_module(
                    Ideogram4Transformer2DModel,
                    weight_dtypes.unconditional_transformer,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "unconditional_transformer",
                    quantization,
                )
        else:
            unconditional_transformer = None

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
        model.unconditional_transformer = unconditional_transformer

    def __load_safetensors(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            unconditional_transformer_model_name: str,
            vae_model_name: str,
            include_unconditional_transformer: bool,
            quantization: QuantizationConfig,
    ):
        transformer = Ideogram4Transformer2DModel.from_single_file(
            transformer_model_name if transformer_model_name else base_model_name,
            torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
        )

        kwargs = {
            "pretrained_model_link_or_path": base_model_name,
            "transformer": transformer,
        }

        if not include_unconditional_transformer:
            kwargs["unconditional_transformer"] = None
        elif unconditional_transformer_model_name:
            unconditional_transformer = Ideogram4Transformer2DModel.from_single_file(
                unconditional_transformer_model_name,
                torch_dtype=torch.bfloat16 if weight_dtypes.unconditional_transformer.torch_dtype() is None else weight_dtypes.unconditional_transformer.torch_dtype(),
            )
            kwargs["unconditional_transformer"] = unconditional_transformer

        pipeline = Ideogram4Pipeline.from_single_file(**kwargs)

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(
                pipeline.vae, weight_dtypes.vae, weight_dtypes.train_dtype
            )

        if pipeline.text_encoder is not None:
            text_encoder = self._convert_transformers_sub_module_to_dtype(
                pipeline.text_encoder, weight_dtypes.text_encoder, weight_dtypes.fallback_train_dtype
            )
            tokenizer = pipeline.tokenizer
        else:
            text_encoder = None
            tokenizer = None
            print("text encoder not loaded, continuing without it")

        transformer = self._convert_diffusers_sub_module_to_dtype(
            pipeline.transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
        )
        if include_unconditional_transformer and pipeline.unconditional_transformer is not None:
            unconditional_transformer = self._convert_diffusers_sub_module_to_dtype(
                pipeline.unconditional_transformer, weight_dtypes.unconditional_transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            unconditional_transformer = None

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.unconditional_transformer = unconditional_transformer

    def load(
            self,
            model: IdeogramModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        transformer_model = getattr(model_names, "transformer_model", None)
        unconditional_transformer_model = getattr(model_names, "unconditional_transformer_model", None)

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, transformer_model,
                unconditional_transformer_model, model_names.vae_model,
                model_names.include_unconditional_transformer, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, transformer_model,
                unconditional_transformer_model, model_names.vae_model,
                model_names.include_unconditional_transformer, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, transformer_model,
                unconditional_transformer_model, model_names.vae_model,
                model_names.include_unconditional_transformer, quantization,
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
