import os
import traceback

from diffusers import DDIMScheduler, AutoencoderKL, Transformer2DModel
from transformers import T5Tokenizer, T5EncoderModel

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class PixArtAlphaModelLoader:
    def __init__(self):
        super(PixArtAlphaModelLoader, self).__init__()

    def __load_internal(
            self,
            model: PixArtAlphaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, vae_model_name)
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: PixArtAlphaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        tokenizer = T5Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = DDIMScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = T5EncoderModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )
        text_encoder.encoder.embed_tokens.to(dtype=weight_dtypes.text_encoder.torch_dtype(supports_fp8=False))

        if vae_model_name:
            vae = AutoencoderKL.from_pretrained(
                vae_model_name,
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                base_model_name,
                subfolder="vae",
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )

        transformer = Transformer2DModel.from_pretrained(
            base_model_name,
            subfolder="transformer",
            torch_dtype=weight_dtypes.prior.torch_dtype(),
        )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def load(
            self,
            model: PixArtAlphaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> PixArtAlphaModel | None:
        stacktraces = []

        base_model_name = model_names.base_model

        try:
            self.__load_internal(model, model_type, weight_dtypes, base_model_name, model_names.vae_model)
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, model_names.vae_model)
            return
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + base_model_name)
