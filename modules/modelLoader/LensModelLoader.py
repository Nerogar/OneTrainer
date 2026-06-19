import os
import traceback

from modules.model.BaseModel import BaseModel
from modules.model.LensModel import LensModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.OnDemandModule import OnDemandModule

import torch

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
)
from transformers import PreTrainedTokenizerFast

from lens.text_encoder import LensGptOssEncoder
from lens.transformer import LensTransformer2DModel


class LensModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: LensModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            text_encoder_on_demand: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization, text_encoder_on_demand,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: LensModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            text_encoder_on_demand: bool,
    ):
        if transformer_model_name:
            transformer = LensTransformer2DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                #avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                LensTransformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        #TODO verify whether the TokenizersBackend warning actually appears for Lens; if so, uncomment the log suppression below (see ErnieModelLoader for the pattern)
        #tokenization_logger = logging.getLogger("transformers.tokenization_utils_base")
        #prev_level = tokenization_logger.level
        #tokenization_logger.setLevel(logging.ERROR)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )
        #tokenization_logger.setLevel(prev_level)

        selected_layer_index = transformer.config.selected_layer_index

        def load_text_encoder():
            text_encoder = LensGptOssEncoder.from_pretrained(
                base_model_name,
                subfolder="text_encoder",
            )
            # set_selected_layers must be called before encode_layers(); the upstream does this in
            # LensPipeline.__init__ — we do it here since OneTrainer loads components separately.
            text_encoder.set_selected_layers(selected_layer_index)
            return text_encoder

        # Lens always loads on demand: its MXFP4 encoder cannot be parked on the CPU temp device, so it
        # is built straight onto the accelerator when needed and discarded afterwards (see
        # TrainConfig.text_encoder_on_demand / LensModel.materialize_text_encoder).
        text_encoder = OnDemandModule(load_text_encoder) if text_encoder_on_demand else load_text_encoder()

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        # read hidden_size from the config only (no weights), so an on-demand encoder stays unmaterialized
        model.text_encoder_hidden_size = LensGptOssEncoder.config_class.from_pretrained(
            base_model_name, subfolder="text_encoder",
        ).hidden_size
        model.vae = vae
        model.transformer = transformer

    def load(
            self,
            model: LensModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization, model_names.text_encoder_on_demand,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization, model_names.text_encoder_on_demand,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)



class LensLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None #TODO

    def load(
            self,
            model: LensModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


LensLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.LENS: "resources/sd_model_spec/lens-lora.json",
    },
    model_class=LensModel,
    model_loader_class=LensModelLoader,
    lora_loader_class=LensLoRALoader,
    embedding_loader_class=None,
)

LensFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.LENS: "resources/sd_model_spec/lens.json",
    },
    model_class=LensModel,
    model_loader_class=LensModelLoader,
    embedding_loader_class=None,
)
