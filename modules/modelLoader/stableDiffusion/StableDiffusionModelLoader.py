import traceback

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from transformers import CLIPTokenizer, CLIPTextModel, DPTImageProcessor, DPTForDepthEstimation

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.mixin.SDConfigModelLoaderMixin import SDConfigModelLoaderMixin
from modules.util import create
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler


class StableDiffusionModelLoader(
    SDConfigModelLoaderMixin,
):
    def __init__(self):
        super(StableDiffusionModelLoader, self).__init__()

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/model_config/stable_diffusion/v1-inference.yaml"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/model_config/stable_diffusion/v1-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/model_config/stable_diffusion/v2-inference-v.yaml"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/model_config/stable_diffusion/v2-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/model_config/stable_diffusion/v2-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/model_config/stable_diffusion/v2-midas-inference.yaml"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/model_config/stable_diffusion/v2-inference-v.yaml"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/model_config/stable_diffusion/v2-inference.yaml"
            case _:
                return None

    def __load_internal(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, vae_model_name)

    def __load_diffusers(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        tokenizer = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = DDIMScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )
        noise_scheduler = create.create_noise_scheduler(
            noise_scheduler=NoiseScheduler.DDIM,
            original_noise_scheduler=noise_scheduler,
        )

        text_encoder = CLIPTextModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )
        text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(supports_fp8=False))

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

        unet = UNet2DConditionModel.from_pretrained(
            base_model_name,
            subfolder="unet",
            torch_dtype=weight_dtypes.unet.torch_dtype(),
        )

        image_depth_processor = DPTImageProcessor.from_pretrained(
            base_model_name,
            subfolder="feature_extractor",
        ) if model_type.has_depth_input() else None

        depth_estimator = DPTForDepthEstimation.from_pretrained(
            base_model_name,
            subfolder="depth_estimator",
            torch_dtype=weight_dtypes.unet.torch_dtype(),  # TODO: use depth estimator dtype
        ) if model_type.has_depth_input() else None

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.unet = unet
        model.image_depth_processor = image_depth_processor
        model.depth_estimator = depth_estimator

    def __fix_nai_model(self, state_dict: dict) -> dict:
        # fix for loading models with an empty state_dict key
        while 'state_dict' in state_dict and len(state_dict['state_dict']) > 0:
            state_dict = state_dict['state_dict']
        if 'state_dict' in state_dict:
            state_dict.pop('state_dict')

        converted_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('cond_stage_model.transformer') and not key.startswith(
                    'cond_stage_model.transformer.text_model'):
                key = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model')
            converted_state_dict[key] = value

        return converted_state_dict

    def __load_ckpt(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        state_dict = torch.load(base_model_name)
        state_dict = self.__fix_nai_model(state_dict)

        num_in_channels = 4
        if model_type.has_mask_input():
            num_in_channels += 1
        if model_type.has_conditioning_image_input():
            num_in_channels += 4

        pipeline = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=state_dict,
            original_config_file=model.sd_config_filename,
            num_in_channels=num_in_channels,
            load_safety_checker=False,
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

        text_encoder = pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype())
        text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
        vae = pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype())
        unet = pipeline.unet.to(dtype=weight_dtypes.unet.torch_dtype())

        model.model_type = model_type
        model.tokenizer = pipeline.tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.unet = unet
        model.image_depth_processor = None  # TODO
        model.depth_estimator = None  # TODO

    def __load_safetensors(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        num_in_channels = 4
        if model_type.has_mask_input():
            num_in_channels += 1
        if model_type.has_conditioning_image_input():
            num_in_channels += 4

        pipeline = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=base_model_name,
            original_config_file=model.sd_config_filename,
            num_in_channels=num_in_channels,
            load_safety_checker=False,
            from_safetensors=True,
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

        text_encoder = pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype())
        text_encoder.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
        vae = pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype())
        unet = pipeline.unet.to(dtype=weight_dtypes.unet.torch_dtype())

        model.model_type = model_type
        model.tokenizer = pipeline.tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.unet = unet
        model.image_depth_processor = None  # TODO
        model.depth_estimator = None  # TODO

    def load(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []

        model.sd_config = self._load_sd_config(model_type, model_names.base_model)
        model.sd_config_filename = self._get_sd_config_name(model_type, model_names.base_model)

        try:
            self.__load_internal(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_ckpt(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
            return
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
