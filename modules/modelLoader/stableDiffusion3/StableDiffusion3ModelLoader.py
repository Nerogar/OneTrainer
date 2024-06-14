import traceback

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, T5Tokenizer, T5EncoderModel

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


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
    ):
        self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, vae_model_name)

    def __load_diffusers(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_2",
        )

        tokenizer_3 = T5Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_3",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )
        # TODO: maybe use the loaded scheduler to create another scheduler for training
        # noise_scheduler = create.create_noise_scheduler(
        #     noise_scheduler=NoiseScheduler.DDIM,
        #     original_noise_scheduler=noise_scheduler,
        # )

        text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )
        text_encoder_1.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(supports_fp8=False))

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_model_name,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtypes.text_encoder_2.torch_dtype(),
        )
        text_encoder_2.text_model.embeddings.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(supports_fp8=False))

        text_encoder_3 = T5EncoderModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder_3",
            torch_dtype=weight_dtypes.text_encoder_3.torch_dtype(),
        )
        text_encoder_3.encoder.embed_tokens.to(dtype=weight_dtypes.text_encoder_3.torch_dtype(supports_fp8=False))

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
    ):
        # TODO
        pass

    def load(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []

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
