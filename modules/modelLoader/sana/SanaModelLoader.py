import os
import traceback

from modules.model.SanaModel import SanaModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import AutoencoderDC, DPMSolverMultistepScheduler, SanaTransformer2DModel
from transformers import Gemma2Model, GemmaTokenizer


class SanaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: SanaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, vae_model_name, quantization)
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: SanaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer = GemmaTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Gemma2Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderDC,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderDC,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        transformer = self._load_diffusers_sub_module(
            SanaTransformer2DModel,
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

    def load(
            self,
            model: SanaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ) -> SanaModel | None:
        stacktraces = []

        base_model_name = model_names.base_model

        try:
            self.__load_internal(model, model_type, weight_dtypes, base_model_name, model_names.vae_model, quantization)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, model_names.vae_model, quantization)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + base_model_name)
