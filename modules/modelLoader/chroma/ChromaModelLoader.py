import os
import traceback

from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKL,
    ChromaPipeline,
    ChromaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import T5EncoderModel, T5Tokenizer


class ChromaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: ChromaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: ChromaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
    ):
        diffusers_sub = []
        if not transformer_model_name:
            diffusers_sub.append("transformer")
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=["text_encoder"],
        )

        tokenizer = T5Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            T5EncoderModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

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
            transformer = ChromaTransformer2DModel.from_single_file(
                transformer_model_name,
                #avoid loading the transformer in float32:
                torch_dtype = torch.bfloat16 if weight_dtypes.prior.torch_dtype() is None else weight_dtypes.prior.torch_dtype()
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.prior, weight_dtypes.train_dtype
            )
        else:
            transformer = self._load_diffusers_sub_module(
                ChromaTransformer2DModel,
                weight_dtypes.prior,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_ckpt(
            self,
            model: ChromaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        # TODO
        pass

    def __load_safetensors(
            self,
            model: ChromaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
    ):
        transformer = ChromaTransformer2DModel.from_single_file(
            #always load transformer separately even though ChromaPipeLine.from_single_file() could load it, to avoid loading in float32:
            transformer_model_name if transformer_model_name else base_model_name,
            torch_dtype = torch.bfloat16 if weight_dtypes.prior.torch_dtype() is None else weight_dtypes.prior.torch_dtype()
        )
        pipeline = ChromaPipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            safety_checker=None,
            transformer=transformer,
        )

        # replace T5TokenizerFast with T5Tokenizer, loaded from the same repository
        #TODO taken from Flux code. Why is this necessary? config files already use T5Tokenizer, in Flux and Chroma
        pipeline.tokenizer_2 = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path="lodestones/Chroma1-HD",
            subfolder="tokenizer",
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

        text_encoder = self._convert_transformers_sub_module_to_dtype(
            pipeline.text_encoder, weight_dtypes.text_encoder_2, weight_dtypes.fallback_train_dtype
        )
        tokenizer = pipeline.tokenizer

        transformer = self._convert_diffusers_sub_module_to_dtype(
            pipeline.transformer, weight_dtypes.prior, weight_dtypes.train_dtype
        )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def load(
            self,
            model: ChromaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.prior_model, model_names.vae_model,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.prior_model, model_names.vae_model,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.prior_model, model_names.vae_model,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
