import json
import os
import traceback

from modules.model.FluxModel import FluxModel
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import replace_linear_with_int8_layers, replace_linear_with_nf4_layers

from torch import nn

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

import accelerate
import huggingface_hub
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file


class FluxModelLoader:
    def __init__(self):
        super(FluxModelLoader, self).__init__()

    def __load_sub_module(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            keep_in_fp32_modules: list[str] | None,
            pretrained_model_name_or_path: str,
            subfolder: str,
            model_filename: str,
            shard_index_filename: str,
    ):
        with accelerate.init_empty_weights():
            if dtype.quantize_nf4():
                replace_linear_with_nf4_layers(sub_module, keep_in_fp32_modules)
            elif dtype.quantize_int8():
                replace_linear_with_int8_layers(sub_module, keep_in_fp32_modules)

        is_local = os.path.isdir(pretrained_model_name_or_path)

        if is_local:
            full_shard_index_filename = os.path.join(pretrained_model_name_or_path, subfolder, shard_index_filename)
            if not os.path.isfile(full_shard_index_filename):
                full_shard_index_filename = None
        else:
            try:
                full_shard_index_filename = huggingface_hub.hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=shard_index_filename,
                )
            except EntryNotFoundError:
                full_shard_index_filename = None

        is_sharded = full_shard_index_filename is not None

        if is_sharded:
            with open(full_shard_index_filename, "r") as f:
                index_file = json.loads(f.read())
                filenames = sorted(set(index_file["weight_map"].values()))
        else:
            filenames = [model_filename]

        if is_local:
            full_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in filenames]
        else:
            full_filenames = [huggingface_hub.hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                subfolder=subfolder,
                filename=f,
            ) for f in filenames]

        state_dict = {}

        for f in full_filenames:
            state_dict |= load_file(f)

        for key, value in state_dict.items():
            module = sub_module
            tensor_name = key
            key_splits = tensor_name.split(".")
            for split in key_splits[:-1]:
                module = getattr(module, split)
            tensor_name = key_splits[-1]

            is_buffer = tensor_name in module._buffers
            old_value = module._buffers[tensor_name] if is_buffer else module._parameters[tensor_name]

            old_type = type(old_value)
            new_value = old_type(value)

            if is_buffer:
                module._buffers[tensor_name] = new_value
            else:
                module._parameters[tensor_name] = new_value

        del state_dict

        return sub_module

    def load_transformers_sub_module(
            self,
            module_type,
            dtype: DataType,
            pretrained_model_name_or_path: str,
            subfolder: str,
    ):
        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
        }
        config, model_kwargs = module_type.config_class.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            user_agent=user_agent,
        )

        with accelerate.init_empty_weights():
            sub_module = module_type(config)

        return self.__load_sub_module(
            sub_module=sub_module,
            dtype=dtype,
            keep_in_fp32_modules=module_type._keep_in_fp32_modules,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            model_filename="model.safetensors",
            shard_index_filename="model.safetensors.index.json",
        )

    def load_diffusers_sub_module(
            self,
            module_type,
            dtype: DataType,
            pretrained_model_name_or_path: str,
            subfolder: str,
    ):
        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
        }
        config, unused_kwargs, commit_hash = module_type.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            user_agent=user_agent,
        )

        with accelerate.init_empty_weights():
            sub_module = module_type.from_config(config)

        return self.__load_sub_module(
            sub_module=sub_module,
            dtype=dtype,
            keep_in_fp32_modules=None,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            model_filename="diffusion_pytorch_model.safetensors",
            shard_index_filename="diffusion_pytorch_model.safetensors.index.json",
        )

    def __load_internal(
            self,
            model: FluxModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, vae_model_name,
                include_text_encoder_1, include_text_encoder_2,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: FluxModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
    ):
        if include_text_encoder_1:
            tokenizer_1 = CLIPTokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            )
        else:
            tokenizer_1 = None

        if include_text_encoder_2:
            tokenizer_2 = T5Tokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer_2",
            )
        else:
            tokenizer_2 = None

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if include_text_encoder_1:
            text_encoder_1 = CLIPTextModel.from_pretrained(
                base_model_name,
                subfolder="text_encoder",
                torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
            )
            text_encoder_1.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(
                supports_quantization=False))
        else:
            text_encoder_1 = None

        if include_text_encoder_2:
            text_encoder_2 = self.load_transformers_sub_module(
                T5EncoderModel,
                weight_dtypes.text_encoder_2,
                base_model_name,
                "text_encoder_2",
            )
            text_encoder_2.encoder.embed_tokens.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(
                supports_quantization=False))
        else:
            text_encoder_2 = None

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

        transformer = self.load_diffusers_sub_module(
            FluxTransformer2DModel,
            weight_dtypes.prior,
            base_model_name,
            "transformer",
        )

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.transformer = transformer

    def __load_ckpt(
            self,
            model: FluxModel,
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
            model: FluxModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder_1: bool,
            include_text_encoder_2: bool,
    ):
        pipeline = FluxPipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            safety_checker=None,
        )

        if include_text_encoder_2:
            # replace T5TokenizerFast with T5Tokenizer, loaded from the same repository
            pipeline.tokenizer_2 = T5Tokenizer.from_pretrained(
                pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
                subfolder="tokenizer_2",
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

            if weight_dtypes.text_encoder.quantize_nf4():
                replace_linear_with_nf4_layers(text_encoder_1)
        else:
            text_encoder_1 = None
            tokenizer_1 = None
            print("text encoder 1 (clip l) not loaded, continuing without it")

        if pipeline.text_encoder_2 is not None and include_text_encoder_2:
            text_encoder_2 = pipeline.text_encoder_2.to(dtype=weight_dtypes.text_encoder_2.torch_dtype())
            text_encoder_2.encoder.embed_tokens.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(
                supports_quantization=False))
            tokenizer_2 = pipeline.tokenizer_2

            if weight_dtypes.text_encoder_2.quantize_nf4():
                replace_linear_with_nf4_layers(text_encoder_2)
        else:
            text_encoder_2 = None
            tokenizer_2 = None
            print("text encoder 2 (t5) not loaded, continuing without it")

        vae = pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype())
        if weight_dtypes.vae.quantize_nf4():
            replace_linear_with_nf4_layers(pipeline.vae)

        transformer = pipeline.transformer.to(dtype=weight_dtypes.prior.torch_dtype())
        if weight_dtypes.prior.quantize_nf4():
            replace_linear_with_nf4_layers(pipeline.transformer)

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.transformer = transformer

    def load(
            self,
            model: FluxModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
            )
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, model_names.include_text_encoder_2,
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
