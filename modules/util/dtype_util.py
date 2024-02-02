from contextlib import nullcontext

import torch
from torch.nn import Parameter

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType


def allow_mixed_precision(train_config: TrainConfig):
    all_dtypes = list(train_config.weight_dtypes().all_dtypes() + [train_config.train_dtype])
    all_dtypes = list(filter(lambda dtype: dtype != DataType.NONE, all_dtypes))
    all_dtypes = set(all_dtypes)

    return len(all_dtypes) != 1


def enable_grad_scaling(train_dtype: DataType, parameters: list[Parameter]):
    trainable_parameter_dtype = list(set([parameter.dtype for parameter in parameters]))
    return train_dtype == DataType.FLOAT_16 and all(dtype == torch.float32 for dtype in trainable_parameter_dtype)


def create_autocast_context(
        device: torch.device,
        train_dtype: DataType | None,
        weight_dtypes: list[DataType | None],
) -> (torch.autocast | nullcontext, DataType):
    weight_dtypes = list(weight_dtypes)
    weight_dtypes = list(filter(lambda dtype: dtype != DataType.NONE and dtype is not None, weight_dtypes))
    weight_dtypes = list(set(weight_dtypes))

    if len(weight_dtypes) == 1 and train_dtype == weight_dtypes[0]:
        return torch.autocast(device_type=device.type, enabled=False), train_dtype
    else:
        return torch.autocast(device_type=device.type, dtype=train_dtype.torch_dtype()), train_dtype


def disable_fp16_autocast_context(
        device: torch.device,
        train_dtype: DataType | None,
        fallback_train_dtype: DataType | None,
        weight_dtypes: list[DataType | None],
) -> (torch.autocast | nullcontext, DataType):
    weight_dtypes = list(filter(lambda dtype: dtype != DataType.NONE and dtype is not None, weight_dtypes))
    weight_dtypes = list(set(weight_dtypes))

    if train_dtype != DataType.FLOAT_16:
        # train dtype is not fp16 -> nothing to disable
        return nullcontext(), train_dtype

    if len(weight_dtypes) == 1 and fallback_train_dtype == weight_dtypes[0]:
        # fallback_train_dtype is the same as all weights -> disable autocast
        return torch.autocast(device_type=device.type, enabled=False), weight_dtypes[0]

    return torch.autocast(device_type=device.type, dtype=fallback_train_dtype.torch_dtype()), fallback_train_dtype


def get_autocast_dtype(
        train_dtypes: list[DataType | None],
) -> DataType:
    for dtype in reversed(train_dtypes):
        if dtype is not None and dtype != DataType.NONE:
            return dtype
