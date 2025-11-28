import json
import os.path
import traceback

from modules.model.WuerstchenModel import WuerstchenEfficientNetEncoder, WuerstchenModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
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


class WuerstchenModelLoader(
    HFModelLoaderMixin,
):
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
            quantization: QuantizationConfig,
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
                quantization,
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
            quantization: QuantizationConfig,
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
            decoder_text_encoder = self._load_transformers_sub_module(
                CLIPTextModel,
                weight_dtypes.decoder_text_encoder,
                weight_dtypes.train_dtype,
                decoder_model_name,
                "text_encoder",
            )
        if model_type.is_stable_cascade():
            decoder_text_encoder = None

        if model_type.is_wuerstchen_v2():
            decoder_decoder = self._load_diffusers_sub_module(
                WuerstchenDiffNeXt,
                weight_dtypes.decoder,
                weight_dtypes.train_dtype,
                decoder_model_name,
                "decoder",
            )
        elif model_type.is_stable_cascade():
            decoder_decoder = self._load_diffusers_sub_module(
                StableCascadeUNet,
                weight_dtypes.decoder,
                weight_dtypes.train_dtype,
                decoder_model_name,
                "decoder",
            )

        decoder_vqgan = self._load_diffusers_sub_module(
            PaellaVQModel,
            weight_dtypes.decoder_vqgan,
            weight_dtypes.train_dtype,
            decoder_model_name,
            "vqgan",
        )

        if model_type.is_wuerstchen_v2():
            effnet_encoder = self._load_diffusers_sub_module(
                WuerstchenEfficientNetEncoder,
                weight_dtypes.effnet_encoder,
                weight_dtypes.fallback_train_dtype,
                effnet_encoder_model_name,
            )
        elif model_type.is_stable_cascade():
            # TODO: this is a temporary workaround until the effnet weights are available in diffusers format
            effnet_encoder = WuerstchenEfficientNetEncoder(affine_batch_norm=False)
            effnet_encoder.load_state_dict(load_file(effnet_encoder_model_name))
            effnet_encoder = self._convert_diffusers_sub_module_to_dtype(
                effnet_encoder, weight_dtypes.effnet_encoder, weight_dtypes.fallback_train_dtype
            )

        if model_type.is_wuerstchen_v2():
            prior_prior = self._load_diffusers_sub_module(
                WuerstchenPrior,
                weight_dtypes.prior,
                weight_dtypes.train_dtype,
                prior_model_name,
                "prior",
                quantization,
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
                prior_prior = self._convert_diffusers_sub_module_to_dtype(
                    prior_prior, weight_dtypes.prior, weight_dtypes.fallback_train_dtype, quantization,
                )
            else:
                prior_prior = self._load_diffusers_sub_module(
                    StableCascadeUNet,
                    weight_dtypes.prior,
                    weight_dtypes.fallback_train_dtype,
                    prior_model_name,
                    "prior",
                    quantization,
                )

        prior_tokenizer = CLIPTokenizer.from_pretrained(
            prior_model_name,
            subfolder="tokenizer",
        )

        if model_type.is_wuerstchen_v2():
            prior_text_encoder = self._load_transformers_sub_module(
                CLIPTextModel,
                weight_dtypes.text_encoder,
                weight_dtypes.train_dtype,
                prior_model_name,
                "text_encoder",
            )
        elif model_type.is_stable_cascade():
            prior_text_encoder = self._load_transformers_sub_module(
                CLIPTextModelWithProjection,
                weight_dtypes.text_encoder,
                weight_dtypes.train_dtype,
                prior_model_name,
                "text_encoder",
            )

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
            quantization: QuantizationConfig,
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
                decoder_model_name, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model,
                model_type,
                weight_dtypes,
                prior_model_name,
                prior_prior_model_name,
                effnet_encoder_model_name,
                decoder_model_name, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception(
            "could not load model: " + prior_model_name + ", " + effnet_encoder_model_name + ", " + decoder_model_name
        )
