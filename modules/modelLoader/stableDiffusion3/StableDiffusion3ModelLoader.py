import os
import traceback

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class StableDiffusion3ModelLoader:
    def __init__(self):
        super(StableDiffusion3ModelLoader, self).__init__()

    def __load_internal(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, vae_model_name,
                include_text_encoder_1, include_text_encoder_2, include_text_encoder_3,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
    ):
        if include_text_encoder_1:
            tokenizer_1 = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            )
        else:
            tokenizer_1 = None

        if include_text_encoder_2:
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer_2",
            )
        else:
            tokenizer_2 = None

        if include_text_encoder_3:
            tokenizer_3 = T5Tokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer_3",
            )
        else:
            tokenizer_3 = None

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if include_text_encoder_1:
            text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
                base_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
            )
            text_encoder_1.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(supports_fp8=False))
        else:
            text_encoder_1 = None

        if include_text_encoder_2:
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                base_model_name,
                subfolder="text_encoder_2",
                torch_dtype=weight_dtypes.text_encoder_2.torch_dtype(),
            )
            text_encoder_2.text_model.embeddings.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(supports_fp8=False))
        else:
            text_encoder_2 = None

        if include_text_encoder_3:
            text_encoder_3 = T5EncoderModel.from_pretrained(
                base_model_name,
                subfolder="text_encoder_3",
                torch_dtype=weight_dtypes.text_encoder_3.torch_dtype(),
            )
            text_encoder_3.encoder.embed_tokens.to(dtype=weight_dtypes.text_encoder_3.torch_dtype(supports_fp8=False))
        else:
            text_encoder_3 = None

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

        transformer = SD3Transformer2DModel.from_pretrained(
            base_model_name,
            subfolder="transformer",
            torch_dtype=weight_dtypes.prior.torch_dtype(),
        )

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.tokenizer_3 = tokenizer_3
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.text_encoder_3 = text_encoder_3
        model.vae = vae
        model.transformer = transformer

    def __load_ckpt(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
    ):
        # TODO
        pass

    def __load_safetensors(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
            include_text_encoder_3: bool,
    ):
        pipeline = StableDiffusion3Pipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            safety_checker=None,
        )

        if include_text_encoder_3:
            # replace T5TokenizerFast with T5Tokenizer, loaded from the same repository
            pipeline.tokenizer_3 = T5Tokenizer.from_pretrained(
                pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
                subfolder="tokenizer_3",
            )

        if vae_model_name:
            pipeline.vae = AutoencoderKL.from_pretrained(
                vae_model_name,
                torch_dtype=weight_dtypes.vae.torch_dtype(),
            )

        if pipeline.text_encoder is not None and include_text_encoder_1:
            text_encoder_1 = pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype())
            text_encoder_1.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
            tokenizer_1 = pipeline.tokenizer
        else:
            text_encoder_1 = None
            tokenizer_1 = None
            print("text encoder 1 (clip l) not loaded, continuing without it")

        if pipeline.text_encoder_2 is not None and include_text_encoder_2:
            text_encoder_2 = pipeline.text_encoder_2.to(dtype=weight_dtypes.text_encoder_2.torch_dtype())
            text_encoder_2.text_model.embeddings.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(False))
            tokenizer_2 = pipeline.tokenizer_2
        else:
            text_encoder_2 = None
            tokenizer_2 = None
            print("text encoder 2 (clip g) not loaded, continuing without it")

        if pipeline.text_encoder_3 is not None and include_text_encoder_3:
            text_encoder_3 = pipeline.text_encoder_3.to(dtype=weight_dtypes.text_encoder_3.torch_dtype())
            text_encoder_3.encoder.embed_tokens.to(dtype=weight_dtypes.text_encoder_3.torch_dtype(supports_fp8=False))
            tokenizer_3 = pipeline.tokenizer_3
        else:
            text_encoder_3 = None
            tokenizer_3 = None
            print("text encoder 3 (t5) not loaded, continuing without it")

        vae = pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype())
        transformer = pipeline.transformer.to(dtype=weight_dtypes.prior.torch_dtype())

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.tokenizer_3 = tokenizer_3
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.text_encoder_3 = text_encoder_3
        model.vae = vae
        model.transformer = transformer

    def load(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
                model_names.include_text_encoder_3,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        # try:
        #     self.__load_ckpt(
        #         model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
        #         model_names.include_text_encoder, model_names.include_text_encoder_2,
        #         model_names.include_text_encoder_3,
        #     )
        #     return
        # except:
        #     stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
