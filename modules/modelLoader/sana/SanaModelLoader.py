import copy
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
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, vae_model_name, quantization, stream_from_disk)
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
            stream_from_disk: bool,
    ):
        model.tokenizer = GemmaTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )
        model.orig_tokenizer = copy.deepcopy(model.tokenizer)

        model.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Gemma2Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        model.vae = self._load_vae(
            AutoencoderDC,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            SanaTransformer2DModel,
            weight_dtypes,
            base_model_name,
            "",
            quantization,
            stream_from_disk=stream_from_disk,
        )

    def load(
            self,
            model: SanaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ) -> SanaModel | None:
        stacktraces = []

        base_model_name = model_names.base_model

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, base_model_name, model_names.vae_model, quantization, stream_from_disk)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, model_names.vae_model, quantization, stream_from_disk)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + base_model_name)
