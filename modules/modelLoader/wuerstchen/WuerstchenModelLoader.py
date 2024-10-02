import json
import os.path
import traceback

from modules.model.WuerstchenModel import WuerstchenEfficientNetEncoder, WuerstchenModel
from modules.util.convert.convert_stable_cascade_ckpt_to_diffusers import convert_stable_cascade_ckpt_to_diffusers
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import DDPMWuerstchenScheduler
from diffusers.models import StableCascadeUNet
from diffusers.pipelines.wuerstchen import PaellaVQModel, WuerstchenDiffNeXt, WuerstchenPrior
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from safetensors import safe_open
from safetensors.torch import load_file


class WuerstchenModelLoader:
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: WuerstchenModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ):
        if os.path.isfile(os.path.join(prior_model_name, "meta.json")):
            self.__load_diffusers(
                model,
                model_type,
                weight_dtypes,
                prior_model_name,
                "",  # pass an empty prior name, so it's always loaded from the backup
                effnet_encoder_model_name,
                decoder_model_name,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: WuerstchenModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            prior_model_name: str,
            prior_prior_model_name: str,
            effnet_encoder_model_name: str,
            decoder_model_name: str,
    ):
        if model_type.is_wuerstchen_v2():
            decoder_tokenizer = CLIPTokenizer.from_pretrained(
                decoder_model_name,
                subfolder="tokenizer",
            )
        if model_type.is_stable_cascade():
            decoder_tokenizer = None

        decoder_noise_scheduler = DDPMWuerstchenScheduler.from_pretrained(
            decoder_model_name,
            subfolder="scheduler",
        )

        if model_type.is_wuerstchen_v2():
            decoder_text_encoder = CLIPTextModel.from_pretrained(
                decoder_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.decoder_text_encoder.torch_dtype(),
            )
            decoder_text_encoder.text_model.embeddings.to(
                dtype=weight_dtypes.text_encoder.torch_dtype(supports_quantization=False))
        if model_type.is_stable_cascade():
            decoder_text_encoder = None

        if model_type.is_wuerstchen_v2():
            decoder_decoder = WuerstchenDiffNeXt.from_pretrained(
                decoder_model_name,
                subfolder="decoder",
                torch_dtype=weight_dtypes.decoder.torch_dtype(),
            )
        elif model_type.is_stable_cascade():
            decoder_decoder = StableCascadeUNet.from_pretrained(
                decoder_model_name,
                subfolder="decoder",
                torch_dtype=weight_dtypes.decoder.torch_dtype(),
            )

        decoder_vqgan = PaellaVQModel.from_pretrained(
            decoder_model_name,
            subfolder="vqgan",
            torch_dtype=weight_dtypes.decoder_vqgan.torch_dtype(),
        )

        if model_type.is_wuerstchen_v2():
            effnet_encoder = WuerstchenEfficientNetEncoder.from_pretrained(
                effnet_encoder_model_name,
                torch_dtype=weight_dtypes.effnet_encoder.torch_dtype(),
            )
        elif model_type.is_stable_cascade():
            # TODO: this is a temporary workaround until the effnet weights are available in diffusers format
            effnet_encoder = WuerstchenEfficientNetEncoder(affine_batch_norm=False)
            effnet_encoder.load_state_dict(load_file(effnet_encoder_model_name))
            effnet_encoder.to(dtype=weight_dtypes.effnet_encoder.torch_dtype())

        if model_type.is_wuerstchen_v2():
            prior_prior = WuerstchenPrior.from_pretrained(
                prior_model_name,
                subfolder="prior",
                torch_dtype=weight_dtypes.prior.torch_dtype(),
            )
        elif model_type.is_stable_cascade():
            if prior_prior_model_name:
                with safe_open(prior_prior_model_name, framework="pt") as f:
                    if any(key.startswith("down_blocks.0.23") for key in f.keys()):  # noqa: SIM118
                        config_filename = "resources/model_config/stable_cascade/stable_cascade_prior_3.6b.json"
                    else:
                        config_filename = "resources/model_config/stable_cascade/stable_cascade_prior_1.0b.json"
                    with open(config_filename, "r") as config_file:
                        prior_config = json.load(config_file)
                prior_prior = StableCascadeUNet(**prior_config)
                prior_prior.load_state_dict(convert_stable_cascade_ckpt_to_diffusers(load_file(prior_prior_model_name)))
                prior_prior.to(dtype=weight_dtypes.prior.torch_dtype())
            else:
                prior_prior = StableCascadeUNet.from_pretrained(
                    prior_model_name,
                    subfolder="prior",
                    torch_dtype=weight_dtypes.prior.torch_dtype(),
                )

        prior_tokenizer = CLIPTokenizer.from_pretrained(
            prior_model_name,
            subfolder="tokenizer",
        )

        if model_type.is_wuerstchen_v2():
            prior_text_encoder = CLIPTextModel.from_pretrained(
                prior_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
            )
            prior_text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
        elif model_type.is_stable_cascade():
            prior_text_encoder = CLIPTextModelWithProjection.from_pretrained(
                prior_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
            )
            prior_text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))

        prior_noise_scheduler = DDPMWuerstchenScheduler.from_pretrained(
            prior_model_name,
            subfolder="scheduler",
        )

        model.model_type = model_type
        model.decoder_tokenizer = decoder_tokenizer
        model.decoder_noise_scheduler = decoder_noise_scheduler
        model.decoder_text_encoder = decoder_text_encoder
        model.decoder_decoder = decoder_decoder
        model.decoder_vqgan = decoder_vqgan
        model.effnet_encoder = effnet_encoder
        model.prior_tokenizer = prior_tokenizer
        model.prior_text_encoder = prior_text_encoder
        model.prior_noise_scheduler = prior_noise_scheduler
        model.prior_prior = prior_prior

    def load(
            self,
            model: WuerstchenModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []

        prior_model_name = model_names.base_model
        prior_prior_model_name = model_names.prior_model
        effnet_encoder_model_name = model_names.effnet_encoder_model
        decoder_model_name = model_names.decoder_model

        try:
            self.__load_internal(
                model,
                model_type,
                weight_dtypes,
                prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model,
                model_type,
                weight_dtypes,
                prior_model_name,
                prior_prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception(
            "could not load model: " + prior_model_name + ", " + effnet_encoder_model_name + ", " + decoder_model_name
        )
