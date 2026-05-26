import os
import traceback

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.SDConfigModelLoaderMixin import SDConfigModelLoaderMixin
from modules.util import create
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class StableDiffusionXLModelLoader(
    SDConfigModelLoaderMixin,
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/model_config/stable_diffusion_xl/sd_xl_base.yaml"
            case ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
                return "resources/model_config/stable_diffusion_xl/sd_xl_base-inpainting.yaml"
            case _:
                return None

    def __load_internal(
            self,
            model: StableDiffusionXLModel,
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
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_2",
        )

        noise_scheduler = DDIMScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )
        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=noise_scheduler,
        )

        text_encoder_1 = self._load_transformers_sub_module(
            CLIPTextModel,
            weight_dtypes.text_encoder,
            weight_dtypes.train_dtype,
            base_model_name,
            "text_encoder",
        )

        text_encoder_2 = self._load_transformers_sub_module(
            CLIPTextModelWithProjection,
            weight_dtypes.text_encoder_2,
            weight_dtypes.train_dtype,
            base_model_name,
            "text_encoder_2",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.fallback_train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "vae",
            )

        unet = self._load_diffusers_sub_module(
            UNet2DConditionModel,
            weight_dtypes.unet,
            weight_dtypes.train_dtype,
            base_model_name,
            "unet",
            quantization,
        )

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.unet = unet

    def __load_ckpt(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        pipeline = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            original_config=model.sd_config_filename,
            safety_checker=None,
        )

        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=pipeline.scheduler,
        )

        if vae_model_name:
            pipeline.vae = AutoencoderKL.from_pretrained(
                vae_model_name,
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )

        text_encoder_1 = pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype())
        text_encoder_1.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
        text_encoder_2 = pipeline.text_encoder_2.to(dtype=weight_dtypes.text_encoder_2.torch_dtype())
        text_encoder_2.text_model.embeddings.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(False))
        vae = pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype())
        unet = pipeline.unet.to(dtype=weight_dtypes.unet.torch_dtype())

        model.model_type = model_type
        model.tokenizer_1 = pipeline.tokenizer
        model.tokenizer_2 = pipeline.tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.unet = unet

    def __load_safetensors(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if model_type.has_conditioning_image_input():
            pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
                pretrained_model_link_or_path=base_model_name,
                original_config=model.sd_config_filename,
                safety_checker=None,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_single_file(
                pretrained_model_link_or_path=base_model_name,
                original_config=model.sd_config_filename,
                safety_checker=None,
                use_safetensors=True,
            )

        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=pipeline.scheduler,
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.fallback_train_dtype,
                vae_model_name,
            )
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(
                pipeline.vae, weight_dtypes.vae, weight_dtypes.fallback_train_dtype
            )

        text_encoder_1 = self._convert_transformers_sub_module_to_dtype(
            pipeline.text_encoder, weight_dtypes.text_encoder, weight_dtypes.train_dtype
        )
        text_encoder_2 = self._convert_transformers_sub_module_to_dtype(
            pipeline.text_encoder_2, weight_dtypes.text_encoder_2, weight_dtypes.train_dtype
        )
        unet = self._convert_diffusers_sub_module_to_dtype(
            pipeline.unet, weight_dtypes.unet, weight_dtypes.train_dtype, quantization,
        )

        model.model_type = model_type
        model.tokenizer_1 = pipeline.tokenizer
        model.tokenizer_2 = pipeline.tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.unet = unet

    def load(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        model.sd_config = self._load_sd_config(model_type, model_names.base_model)
        model.sd_config_filename = self._get_sd_config_name(model_type, model_names.base_model)

        try:
            self.__load_internal(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model, quantization)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model, quantization)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model, quantization)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        if model_names.base_model.endswith(".ckpt"):
            try:
                self.__load_ckpt(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
                print("Warning: Legacy code is used to load ckpt files. Some features may not be supported.")
                return
            except Exception:
                stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
