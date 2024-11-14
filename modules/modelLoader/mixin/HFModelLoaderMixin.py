import json
import os
from abc import ABCMeta
from itertools import repeat

from modules.util.enum.DataType import DataType
from modules.util.quantization_util import (
    is_quantized_parameter,
    replace_linear_with_fp8_layers,
    replace_linear_with_int8_layers,
    replace_linear_with_nf4_layers,
)

import torch
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
            pytorch_model_filename: str | None,
            shard_index_filename: str,
    ):
        if keep_in_fp32_modules is None:
            keep_in_fp32_modules = []

        with accelerate.init_empty_weights():
            if dtype.quantize_nf4():
                replace_linear_with_nf4_layers(sub_module, keep_in_fp32_modules, copy_parameters=False)
            elif dtype.quantize_int8():
                replace_linear_with_int8_layers(sub_module, keep_in_fp32_modules, copy_parameters=False)
            elif dtype.quantize_fp8():
                replace_linear_with_fp8_layers(sub_module, keep_in_fp32_modules, copy_parameters=False)

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

        safetensors_filenames = []

        if is_sharded:
            with open(full_shard_index_filename, "r") as f:
                index_file = json.loads(f.read())
                safetensors_filenames = sorted(set(index_file["weight_map"].values()))
        elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, model_filename)
        ):
            safetensors_filenames = [model_filename]

        state_dict = {}
        if len(safetensors_filenames) > 0:
            if is_local:
                if subfolder:
                    full_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in
                                      safetensors_filenames]
                else:
                    full_filenames = [os.path.join(pretrained_model_name_or_path, f) for f in safetensors_filenames]
            else:
                full_filenames = [huggingface_hub.hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=f,
                ) for f in safetensors_filenames]

            for f in full_filenames:
                state_dict |= load_file(f)
        else:
            # fall back to the pytorch_model_filename
            state_dict = torch.load(os.path.join(pretrained_model_name_or_path, subfolder, pytorch_model_filename),
                                    weights_only=True)
            while 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

        if hasattr(sub_module, '_convert_deprecated_attention_blocks'):
            sub_module._convert_deprecated_attention_blocks(state_dict)

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

            if torch.is_floating_point(old_value):
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
            pytorch_model_filename="pytorch_model.bin",
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
            pytorch_model_filename="diffusion_pytorch_model.bin",
            shard_index_filename="diffusion_pytorch_model.safetensors.index.json",
        )

    def __convert_sub_module_to_dtype(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            keep_in_fp32_modules: list[str] | None,
    ):
        if keep_in_fp32_modules is None:
            keep_in_fp32_modules = []

        if dtype.quantize_nf4():
            replace_linear_with_nf4_layers(sub_module, keep_in_fp32_modules, copy_parameters=True)
        elif dtype.quantize_int8():
            replace_linear_with_int8_layers(sub_module, keep_in_fp32_modules, copy_parameters=True)
        elif dtype.quantize_fp8():
            replace_linear_with_fp8_layers(sub_module, keep_in_fp32_modules, copy_parameters=True)

        for module_name, module in sub_module.named_modules():
            param_iter = [(x, y[0], y[1]) for x, y in zip(repeat(False), module._parameters.items(), strict=False)]
            buffer_iter = [(x, y[0], y[1]) for x, y in zip(repeat(True), module._buffers.items(), strict=False)]

            for is_buffer, tensor_name, value in param_iter + buffer_iter:
                if value is not None and torch.is_floating_point(value):
                    old_type = type(value)
                    if not is_quantized_parameter(module, tensor_name):
                        if dtype.is_quantized() or module_name in keep_in_fp32_modules:
                            value = value.to(dtype=train_dtype.torch_dtype())
                        else:
                            value = value.to(dtype=dtype.torch_dtype())

                        value = old_type(value)

                        if is_buffer:
                            module._buffers[tensor_name] = value
                        else:
                            module._parameters[tensor_name] = value

        return sub_module

    def _convert_transformers_sub_module_to_dtype(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
    ):
        module_type = type(sub_module)

        return self.__convert_sub_module_to_dtype(
            sub_module,
            dtype,
            train_dtype,
            module_type._keep_in_fp32_modules,
        )

    def _convert_diffusers_sub_module_to_dtype(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
    ):
        return self.__convert_sub_module_to_dtype(
            sub_module,
            dtype,
            train_dtype,
            None,
        )
