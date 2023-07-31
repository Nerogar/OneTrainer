import copy
import hashlib
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime

import torch
import safetensors.torch as safetensors
from torch import Tensor

from modules.model.BaseModel import BaseModel
from modules.util import git_util
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class BaseModelSaver(metaclass=ABCMeta):

    @staticmethod
    def _convert_state_dict_dtype(state_dict: dict, dtype: torch.dtype) -> dict:
        converted_state_dict = {}

        for (key, value) in state_dict.items():
            if isinstance(value, dict):
                converted_state_dict[key] = BaseModelSaver._convert_state_dict_dtype(value, dtype)
            else:
                converted_state_dict[key] = value.to(dtype=dtype)

        return converted_state_dict

    @staticmethod
    def _convert_state_dict_to_contiguous(state_dict: dict):
        for (key, value) in state_dict.items():
            if isinstance(value, dict):
                BaseModelSaver._convert_state_dict_to_contiguous(value)
            else:
                state_dict[key] = value.contiguous()

    @staticmethod
    def __calculate_safetensors_hash(state_dict: dict[str, Tensor] | None = None) -> str | None:
        if state_dict is None:
            return None

        sha256_hash = hashlib.sha256()

        ordered_state_dict = OrderedDict(sorted(state_dict.items()))
        for key, tensor in ordered_state_dict.items():
            data = safetensors._tobytes(tensor, key)
            sha256_hash.update(data)

        return f"0x{sha256_hash.hexdigest()}"

    @staticmethod
    def _create_safetensors_header(model: BaseModel, state_dict: dict[str, Tensor] | None = None) -> dict[str, str]:
        if model.model_spec is not None:
            model_spec = copy.deepcopy(model.model_spec)
        else:
            model_spec = ModelSpec()

        # update calculated fields
        model_spec.date = datetime.now().strftime("%Y-%m-%d")
        model_spec.hash_sha256 = BaseModelSaver.__calculate_safetensors_hash(state_dict)

        # assemble the header
        model_spec_dict = model_spec.to_dict()
        one_trainer_header = {
            "ot_branch": git_util.get_git_branch(),
            "ot_revision": git_util.get_git_revision(),
        }
        return model_spec_dict | one_trainer_header

    @abstractmethod
    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        pass
