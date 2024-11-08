import json
import os
from abc import ABCMeta

from modules.util.enum.DataType import DataType
from modules.util.quantization_util import (
    is_quantized_parameter,
    replace_linear_with_fp8_layers,
    replace_linear_with_int8_layers,
    replace_linear_with_nf4_layers,
)

from torch import nn

import accelerate
import huggingface_hub
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file


class HFModelLoaderMixin(metaclass=ABCMeta):

    def __load_sub_module(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            keep_in_fp32_modules: list[str] | None,
            pretrained_model_name_or_path: str,
            subfolder: str | None,
            model_filename: str,
            shard_index_filename: str,
    ):
        if keep_in_fp32_modules is None:
            keep_in_fp32_modules = []

        with accelerate.init_empty_weights():
            if dtype.quantize_nf4():
                replace_linear_with_nf4_layers(sub_module, keep_in_fp32_modules)
            elif dtype.quantize_int8():
                replace_linear_with_int8_layers(sub_module, keep_in_fp32_modules)
            elif dtype.quantize_fp8():
                replace_linear_with_fp8_layers(sub_module, keep_in_fp32_modules)


        is_local = os.path.isdir(pretrained_model_name_or_path)

        if is_local:
            if subfolder:
                full_shard_index_filename = os.path.join(pretrained_model_name_or_path, subfolder, shard_index_filename)
            else:
                full_shard_index_filename = os.path.join(pretrained_model_name_or_path, shard_index_filename)
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
            if subfolder:
                full_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in filenames]
            else:
                full_filenames = [os.path.join(pretrained_model_name_or_path, f) for f in filenames]
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
            module_name = None
            key_splits = tensor_name.split(".")
            for split in key_splits[:-1]:
                module = getattr(module, split)
                module_name = split
            tensor_name = key_splits[-1]

            is_buffer = tensor_name in module._buffers
            old_value = module._buffers[tensor_name] if is_buffer else module._parameters[tensor_name]

            old_type = type(old_value)
            if not is_quantized_parameter(module, tensor_name):
                if dtype.is_quantized() or module_name in keep_in_fp32_modules:
                    value = value.to(dtype=train_dtype.torch_dtype())
                else:
                    value = value.to(dtype=dtype.torch_dtype())
            new_value = old_type(value)

            if is_buffer:
                module._buffers[tensor_name] = new_value
            else:
                module._parameters[tensor_name] = new_value

        del state_dict

        return sub_module


    def _load_transformers_sub_module(
            self,
            module_type,
            dtype: DataType,
            train_dtype: DataType,
            pretrained_model_name_or_path: str,
            subfolder: str | None = None,
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
            train_dtype=train_dtype,
            keep_in_fp32_modules=module_type._keep_in_fp32_modules,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            model_filename="model.safetensors",
            shard_index_filename="model.safetensors.index.json",
        )


    def _load_diffusers_sub_module(
            self,
            module_type,
            dtype: DataType,
            train_dtype: DataType,
            pretrained_model_name_or_path: str,
            subfolder: str | None = None,
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
            train_dtype=train_dtype,
            keep_in_fp32_modules=None,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            model_filename="diffusion_pytorch_model.safetensors",
            shard_index_filename="diffusion_pytorch_model.safetensors.index.json",
        )
